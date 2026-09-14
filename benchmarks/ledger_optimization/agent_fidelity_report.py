"""Read-only audit of frozen089 inputs, requests, answers and reported usage."""
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
import sys

from .agent_fidelity import MODEL, SYSTEM, TOOL, compact_bytes, grade, sha, validate


def independently_correct(answer, ref):
    """Mapping-based exact-fact audit, separate from the recursive field scorer."""
    try:
        validate(answer)
        if answer['pagination'] != ref['pagination']:
            return False
        if answer['global_ranking_supported'] is not False or answer['provider_calls'] != 0:
            return False
        pairs = {p['pair_index']: p for p in answer['comparisons']}
        if len(pairs) != len(answer['comparisons']) or set(pairs) != {p['pair_index'] for p in ref['comparisons']}:
            return False
        for original in ref['comparisons']:
            pair = pairs[original['pair_index']]
            if pair['recent_lifetime_disagreement'] is not original['recent_lifetime_disagreement']:
                return False
            windows = {w['window']: w for w in pair['windows']}
            if len(windows) != len(pair['windows']) or set(windows) != {w['window'] for w in original['windows']}:
                return False
            for expected in original['windows']:
                window = windows[expected['window']]
                if any(window[k] != expected[k] for k in ('matched_origins', 'n', 'start', 'end')):
                    return False
                if sorted(window['lowest_error_config_ids']) != sorted(expected['lowest_error_config_ids']):
                    return False
                scores = {s['config_id']: s['rmsle'] for s in window['scores']}
                target = {s['config_id']: s['rmsle'] for s in expected['scores']}
                if len(scores) != len(window['scores']) or set(scores) != set(target):
                    return False
                if any(abs(scores[c] - target[c]) > 1e-9 for c in target):
                    return False
        return True
    except (ValueError, TypeError, KeyError):
        return False


def report(prepared, run_root, output):
    prepared, run_root, output = Path(prepared), Path(run_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    audit = []
    def check(value, message):
        audit.append({'check': message, 'passed': bool(value)})
        if not value:
            raise AssertionError(message)
    for name in ('cases', 'schedule'):
        check(sha(prepared / (name + '.json')) == manifest[name + '_sha256'], 'Frozen ' + name)
    for name, digest in manifest['code'].items():
        check(sha(name) == digest, 'Frozen source ' + name)
    before = {str(p): sha(p) for root in (prepared, run_root) for p in root.rglob('*') if p.is_file()}
    cases = {c['case_id']: c for c in json.loads((prepared / 'cases.json').read_text())}
    schedule = json.loads((prepared / 'schedule.json').read_text())
    rows = []
    pending = []
    upstream = 0
    for job in schedule:
        folder = run_root / job['job_id']
        if not (folder / 'result.json').exists():
            pending.append({'job_id': job['job_id'], 'request_started': any(folder.glob('started-*.json'))})
            continue
        result = json.loads((folder / 'result.json').read_text())
        for key in job:
            check(result[key] == job[key], 'Job identity ' + key)
        check(1 <= result['requests'] <= 2, 'Request budget')
        check(len(list(folder.glob('started-*.json'))) == result['requests'], 'Every started request counted')
        check(len(list(folder.glob('request-*.json'))) == result['requests'], 'Every payload counted')
        case = cases[job['case_id']]
        check(grade(case['expected'], case['expected'])['all_facts_correct'], 'Valid expected answer')
        raw_usage = []
        seconds = 0
        messages = []
        protocol_models = []
        for attempt in range(result['requests']):
            payload = json.loads((folder / f'request-{attempt}.json').read_text())
            check(payload['model'] == MODEL and payload['seed'] == job['seed'], 'Model and seed fixed')
            check(payload['temperature'] == 0.2 and payload['max_tokens'] == 2048, 'Sampling budget fixed')
            check(payload['tools'] == [TOOL], 'Identical typed submission tool')
            check(payload['messages'][0] == {'role': 'system', 'content': SYSTEM}, 'Identical system instruction')
            user = json.loads(payload['messages'][1]['content'])
            check(user == {'question': case['question'], 'review': case['formats'][job['format']]},
                  'Only correct format and questions sent, no answer reference')
            if (folder / f'error-{attempt}.json').exists():
                error = json.loads((folder / f'error-{attempt}.json').read_text())
                seconds += error['wall_seconds']
                check(result['service_error'], 'Service error retained')
                continue
            raw = json.loads((folder / f'response-{attempt}.json').read_text())
            check(raw.get('model') == MODEL, 'Response reports required model')
            raw_usage.append(raw.get('usage'))
            protocol_models.append(raw.get('model'))
            messages.append(raw['choices'][0]['message'])
            seconds += json.loads((folder / f'timing-{attempt}.json').read_text())['wall_seconds']
        check(raw_usage == result['usage'], 'Usage copied exactly from responses')
        if result['schema_completed']:
            check(bool(messages), 'Completed submission has raw response')
            calls = messages[-1].get('tool_calls') or []
            check(len(calls) == 1 and calls[0]['function']['name'] == 'submit_evidence', 'One executed submission')
            answer = json.loads(calls[0]['function']['arguments'])
            check(answer == result['answer'], 'Answer matches raw tool call')
            scored = grade(answer, case['expected'])
            check(scored['checks'] == result['checks'], 'Every recorded field grade recomputed')
            independent = independently_correct(answer, case['expected'])
            check(independent == scored['all_facts_correct'] == result['all_facts_correct'],
                  'Independent exact-fact verdict agrees')
        else:
            check(result['all_facts_correct'] is False, 'No unsubmitted answer counted correct')
        usable_usage = len(raw_usage) == result['requests'] and all(
            isinstance(u, dict) and all(type(u.get(k)) is int and u[k] >= 0
                                       for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
            for u in raw_usage)
        totals = {k: sum(u[k] for u in raw_usage) if usable_usage else None
                  for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        checks = result.get('checks', [])
        rows.append({**job, 'schema_completed': result['schema_completed'],
                     'all_facts_correct': result['all_facts_correct'],
                     'service_error': result['service_error'], 'requests': result['requests'],
                     'syntax_corrections': result['syntax_corrections'], 'wall_seconds': seconds,
                     'reported_models': protocol_models, 'usage_complete': usable_usage, **totals,
                     'fields_correct': sum(c['passed'] for c in checks), 'fields_checked': len(checks),
                     'incorrect_fields': [c['field'] for c in checks if not c['passed']]})
        upstream += result['requests']
    check(upstream <= 128, 'Total upstream budget')
    formats = {}
    for name in ('original', 'brief'):
        selected = [r for r in rows if r['format'] == name]
        formats[name] = {
            'scheduled': 32, 'terminal': len(selected),
            'all_facts_correct': sum(r['all_facts_correct'] for r in selected),
            'schema_completed': sum(r['schema_completed'] for r in selected),
            'service_errors': sum(r['service_error'] for r in selected),
            'requests': sum(r['requests'] for r in selected),
            'syntax_corrections': sum(r['syntax_corrections'] for r in selected),
            'mean_wall_seconds': mean(r['wall_seconds'] for r in selected) if selected else None,
            'usage_complete_sessions': sum(r['usage_complete'] for r in selected),
            'fields_correct': sum(r['fields_correct'] for r in selected),
            'fields_checked': sum(r['fields_checked'] for r in selected),
            'incorrect_field_counts': dict(Counter(p for r in selected for p in r['incorrect_fields'])),
        }
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            formats[name][key] = sum(r[key] for r in selected) if selected and all(r['usage_complete'] for r in selected) else None
    grouped = {}
    for row in rows:
        grouped.setdefault((row['case_id'], row['seed']), {})[row['format']] = row
    paired = []
    for (case_id, seed), arms in sorted(grouped.items()):
        if set(arms) == {'original', 'brief'}:
            a, b = arms['original'], arms['brief']
            paired.append({'case_id': case_id, 'seed': seed,
                           'correctness_difference_brief_minus_original': int(b['all_facts_correct']) - int(a['all_facts_correct']),
                           'prompt_fraction_saved': 1 - b['prompt_tokens'] / a['prompt_tokens']
                             if a['usage_complete'] and b['usage_complete'] and a['prompt_tokens'] else None})
    full_usage = len(paired) == 32 and all(p['prompt_fraction_saved'] is not None for p in paired)
    saved = mean(p['prompt_fraction_saved'] for p in paired) if full_usage else None
    # Prospective gate specifies paired mean token use, so ratio of paired means
    # is primary. Average per-pair fractions above are separately descriptive.
    ratio_saved = (1 - formats['brief']['prompt_tokens'] / formats['original']['prompt_tokens']) if full_usage else None
    completed = len(rows) == 64 and not pending
    gate = completed and full_usage and formats['brief']['all_facts_correct'] >= formats['original']['all_facts_correct'] and ratio_saved >= 0.25
    check(before == {str(p): sha(p) for root in (prepared, run_root) for p in root.rglob('*') if p.is_file()},
          'All inputs and raw outputs unchanged')
    summary = {'completed': completed, 'formats': formats, 'paired_sessions': len(paired),
               'paired_mean_prompt_tokens_fraction_saved': ratio_saved, 'mean_per_pair_fraction_saved': saved,
               'adoption_screen_passed': gate, 'upstream_requests': upstream, 'pending': pending,
               'reported_models': dict(Counter(m for r in rows for m in r['reported_models'])),
               'accuracy_trial': False, 'forecast_calls': 0, 'billing_cost': None,
               'billing_cost_basis': 'API usage retained; no billing receipt available.',
               'audit_checks': len(audit), 'audit_failures': 0}
    for name, value in [('summary', summary), ('rows', rows), ('paired', paired), ('audit', audit), ('input_hashes', before)]:
        (output / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    print(json.dumps(report(*sys.argv[1:]), indent=2))
