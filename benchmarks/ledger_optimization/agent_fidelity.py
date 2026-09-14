"""Frozen auxiliary agent evidence-reading test; not a forecasting evaluation."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import urllib.request

from .agent_review import brief, compact_bytes
from .agent_review_audit import INVENTORY_SHA
from .ml_ledger_cards import compact_cards

MODEL = 'deepseek-v4.1-flash'
ENDPOINT = 'https://api.engy.ai/v1/chat/completions'
SYSTEM = '''Read the supplied historical forecast evidence accurately.
Submit the requested facts using submit_evidence. Use RMSLE, not MAE.
Compare models only within the same pair and window. Different pairs may have
different matched cohorts: the evidence does not establish a global ranking.
Resolve same_as within its card. Count all exact ties as winners. Empty support
means empty scores and no winners, not zero error. This is an evidence-reading
task; do not forecast, infer causes, or choose a model for an unseen outcome.
Use exactly the requested pair indexes and windows. Keep the answer concise.'''


def obj(properties):
    return {'type': 'object', 'additionalProperties': False,
            'required': list(properties), 'properties': properties}


NULL_STRING = {'type': ['string', 'null']}
INTEGER = {'type': 'integer'}
BOOL = {'type': 'boolean'}
SCORE = obj({'config_id': {'type': 'string'}, 'rmsle': {'type': 'number'}})
WINDOW = obj({'window': {'enum': ['last_4_origins', 'lifetime']},
              'matched_origins': INTEGER, 'n': INTEGER, 'start': NULL_STRING, 'end': NULL_STRING,
              'scores': {'type': 'array', 'items': SCORE},
              'lowest_error_config_ids': {'type': 'array', 'items': {'type': 'string'}}})
PAIR = obj({'pair_index': INTEGER, 'windows': {'type': 'array', 'items': WINDOW},
            'recent_lifetime_disagreement': {'type': ['boolean', 'null']}})
ANSWER = obj({'comparisons': {'type': 'array', 'items': PAIR},
              'pagination': obj({'total_pairs': INTEGER, 'shown_pairs': INTEGER,
                                 'all_pairs_included': BOOL, 'next_offset': {'type': ['integer', 'null']}}),
              'global_ranking_supported': BOOL, 'provider_calls': INTEGER})
TOOL = {'type': 'function', 'function': {'name': 'submit_evidence',
        'description': 'Submit the requested factual extraction from the supplied review.', 'parameters': ANSWER}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('xb') as stream:
        stream.write(compact_bytes(value) + b'\n')


def expected(full, indexes):
    comparisons = []
    for index in indexes:
        card = full['cards'][index]
        ids = [card['left_config_id'], card['right_config_id']]
        windows = []
        for name in ('last_4_origins', 'lifetime'):
            w = card['windows'][name]
            scores = {c: next(m['rmsle'] for m in w['models']
                              if m['provider'] == full['configuration_index'][c]['provider'])
                      for c in ids} if w['matched_origins'] else {}
            windows.append({'window': name, **{k: w[k] for k in ('matched_origins', 'n', 'start', 'end')},
                            'scores': [{'config_id': c, 'rmsle': scores[c]} for c in ids if c in scores],
                            'lowest_error_config_ids': [c for c in ids if scores and scores[c] == min(scores.values())]})
        comparisons.append({'pair_index': index, 'windows': windows,
                            'recent_lifetime_disagreement': card.get('recent_lifetime_disagreement')})
    return {'comparisons': comparisons,
            'pagination': {'total_pairs': full['total_pairs'], 'shown_pairs': len(full['cards']),
                           'all_pairs_included': full['offset'] == 0 and full['next_offset'] is None
                               and len(full['cards']) == full['total_pairs'],
                           'next_offset': full['next_offset']},
            'global_ranking_supported': False, 'provider_calls': full['provider_calls']}


def validate(value, schema=ANSWER, path='$'):
    """Validate structure only; never provide factual answers in correction."""
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(path + ': unsupported enum value')
    typ = schema.get('type')
    if typ is not None:
        types = typ if isinstance(typ, list) else [typ]
        valid = {'null': value is None, 'object': type(value) is dict, 'array': type(value) is list,
                 'string': type(value) is str, 'integer': type(value) is int,
                 'boolean': type(value) is bool,
                 'number': type(value) in (int, float) and math.isfinite(value)}
        if not any(valid[t] for t in types):
            raise ValueError(path + ': wrong JSON type')
    if type(value) is dict:
        props = schema['properties']
        if set(value) != set(props):
            raise ValueError(path + ': fields must be ' + ', '.join(props))
        for key, child in value.items():
            validate(child, props[key], path + '.' + key)
    elif type(value) is list:
        for index, child in enumerate(value):
            validate(child, schema['items'], f'{path}[{index}]')


def normalize(value):
    result = deepcopy(value)
    result['comparisons'].sort(key=lambda p: p['pair_index'])
    for pair in result['comparisons']:
        pair['windows'].sort(key=lambda w: w['window'])
        for window in pair['windows']:
            window['scores'].sort(key=lambda s: s['config_id'])
            window['lowest_error_config_ids'].sort()
    return result


def grade(answer, reference):
    validate(answer)
    got, wanted = normalize(answer), normalize(reference)
    checks = []
    missing = object()
    def compare(a, b, path):
        if type(b) is dict:
            for k in b:
                compare(a.get(k, missing) if type(a) is dict else missing, b[k], path + '.' + k)
        elif type(b) is list:
            checks.append({'field': path + '.length', 'passed': type(a) is list and len(a) == len(b)})
            for i, item in enumerate(b):
                compare(a[i] if type(a) is list and i < len(a) else missing, item, f'{path}[{i}]')
        else:
            ok = (type(a) in (int, float) and math.isfinite(a) and abs(a - b) <= 1e-9
                  if path.endswith('.rmsle') else a == b and type(a) is type(b))
            checks.append({'field': path, 'passed': ok})
    compare(got, wanted, '$')
    return {'all_facts_correct': all(c['passed'] for c in checks), 'checks': checks}


def prepare(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if sha(source / 'SHA256SUMS.json') != INVENTORY_SHA:
        raise ValueError('Wrong original030 inventory')
    inventory = json.loads((source / 'SHA256SUMS.json').read_text())
    access = {}
    def read(name):
        if name not in inventory or sha(source / name) != inventory[name]:
            raise ValueError('Original source hash mismatch: ' + name)
        access[name] = inventory[name]
        return json.loads((source / name).read_text())
    jobs = read('evaluation/host-jobs.json')
    cases = []
    for series, tasks in sorted(jobs.items()):
        project = f'evaluation/ledger/{series}/round-25/project'
        log_name = project + '/review-log.jsonl'
        if sha(source / log_name) != inventory[log_name]:
            raise ValueError('Review log hash mismatch')
        access[log_name] = inventory[log_name]
        logs = [json.loads(line) for line in (source / log_name).read_text().splitlines()]
        for round_ in (1, 9, 17, 25):
            task = next(t for t in tasks if t['round'] == round_)
            matches = [p for p in logs if p['task_origin'] == task['origin']]
            if len(matches) != 1:
                raise ValueError('Exactly one saved review per frozen task required')
            rel = matches[0]['full_evidence_path']
            if Path(rel).is_absolute() or '..' in Path(rel).parts:
                raise ValueError('Nonlocal evidence path')
            name = project + '/' + rel
            full = read(name)
            indexes = sorted({0, len(full['cards']) - 1})
            if not full['cards']:
                raise ValueError('Frozen case has no returned pairs')
            header = {'series_id': series, 'origin': task['origin'],
                      'horizon': task['request']['horizon'], 'unit': task['request']['unit']}
            old = compact_cards(full, rel)
            for key in old:
                if old[key] != matches[0][key]:
                    raise ValueError('Saved original summary differs')
            new = brief(full, header, rel, inventory[name])
            ref = expected(full, indexes)
            validate(ref)
            case = {'case_id': f'{series}-r{round_:02d}', 'series_id': series, 'round': round_,
                    'origin': task['origin'], 'source': name, 'source_sha256': inventory[name],
                    'question': {'pair_indexes': indexes, 'windows': ['last_4_origins', 'lifetime'],
                                 'instruction': 'Return the facts requested by submit_evidence for these pairs/windows.'},
                    'formats': {'original': old, 'brief': new}, 'expected': ref}
            cases.append(case)
    if len(cases) != 16 or len(jobs) != 4:
        raise ValueError('Frozen cohort must contain four series and16reviews')
    schedule = [{'case_id': c['case_id'], 'seed': seed, 'format': format_}
                for c in cases for seed in (7, 19) for format_ in ('original', 'brief')]
    random.Random(17).shuffle(schedule)
    for i, job in enumerate(schedule):
        job['job_id'] = f'{i:03d}-{job["case_id"]}-{job["seed"]}-{job["format"]}'
    code = {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('AGENT_FIDELITY_089.md'),
                                    Path(__file__).with_name('agent_review.py'), Path(__file__).with_name('ml_ledger_cards.py'))}
    save(output / 'cases.json', cases)
    save(output / 'schedule.json', schedule)
    save(output / 'manifest.json', {'source_inventory': INVENTORY_SHA, 'source_access': access,
          'cases_sha256': sha(output / 'cases.json'), 'schedule_sha256': sha(output / 'schedule.json'),
          'code': code, 'model': MODEL, 'max_upstream_requests': 128, 'accuracy_trial': False,
          'protected_access': False})
    for name, digest in access.items():
        if sha(source / name) != digest:
            raise ValueError('Source was modified')
    return {'cases': len(cases), 'sessions': len(schedule), 'source_files': len(access), 'api_calls': 0}


def key():
    if os.environ.get('ENGY_API_KEY'):
        return os.environ['ENGY_API_KEY']
    for line in Path('/root/Gnomon/.env').read_text().splitlines():
        if line.startswith('ENGY_API_KEY='):
            return line.split('=', 1)[1].strip().strip('"\'')
    raise ValueError('Engy credential not available')


def run(prepared, output):
    prepared, output = Path(prepared), Path(output)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    for name, digest in manifest['code'].items():
        if sha(name) != digest:
            raise ValueError('Frozen code changed')
    for name in ('cases', 'schedule'):
        if sha(prepared / (name + '.json')) != manifest[name + '_sha256']:
            raise ValueError('Frozen inputs changed')
    credential = key()
    output.mkdir(parents=True, exist_ok=False)
    cases = {c['case_id']: c for c in json.loads((prepared / 'cases.json').read_text())}
    schedule = json.loads((prepared / 'schedule.json').read_text())
    consecutive_errors = 0
    for job in schedule:
        folder = output / job['job_id']
        folder.mkdir()
        case = cases[job['case_id']]
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': compact_bytes({'question': case['question'],
                     'review': case['formats'][job['format']]}).decode()}]
        result = {**job, 'schema_completed': False, 'all_facts_correct': False,
                  'requests': 0, 'usage': [], 'service_error': False, 'syntax_corrections': 0}
        for attempt in range(2):
            payload = {'model': MODEL, 'messages': messages, 'tools': [TOOL],
                       'temperature': 0.2, 'max_tokens': 2048, 'seed': job['seed']}
            save(folder / f'request-{attempt}.json', payload)
            save(folder / f'started-{attempt}.json', {'unix_time': time.time()})
            result['requests'] += 1
            request = urllib.request.Request(ENDPOINT, data=compact_bytes(payload),
                      headers={'Authorization': 'Bearer ' + credential, 'Content-Type': 'application/json'})
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read()
                (folder / f'response-{attempt}.json').write_bytes(raw)
                response = json.loads(raw)
                result['usage'].append(response.get('usage'))
                save(folder / f'timing-{attempt}.json', {'wall_seconds': time.monotonic() - started})
                message = response['choices'][0]['message']
            except Exception as exc:
                # No credential or request headers enter diagnostics.
                result.update(service_error=True, error_category=type(exc).__name__)
                save(folder / f'error-{attempt}.json', {'category': type(exc).__name__,
                     'http_status': getattr(exc, 'code', None), 'wall_seconds': time.monotonic() - started})
                if hasattr(exc, 'read'):
                    body = exc.read().decode('utf-8', errors='replace').replace(credential, '[REDACTED]')
                    (folder / f'error-body-{attempt}.txt').write_text(body)
                consecutive_errors += 1
                break
            consecutive_errors = 0
            calls = message.get('tool_calls') or []
            try:
                if len(calls) != 1 or calls[0]['function']['name'] != 'submit_evidence':
                    raise ValueError('Use exactly one submit_evidence tool call.')
                answer = json.loads(calls[0]['function']['arguments'])
                validate(answer)
            except (ValueError, KeyError, TypeError) as exc:
                result['syntax_corrections'] += int(attempt == 0)
                if attempt == 0:
                    messages.append(message)
                    if calls:
                        for call in calls:
                            messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                             'content': 'Submission rejected: ' + str(exc)})
                    else:
                        messages.append({'role': 'user', 'content': 'Submission rejected: ' + str(exc)})
                continue
            result.update(schema_completed=True, answer=answer, **grade(answer, case['expected']))
            break
        save(folder / 'result.json', result)
        print(json.dumps({k: result[k] for k in ('job_id', 'schema_completed', 'all_facts_correct', 'requests', 'service_error')}), flush=True)
        if consecutive_errors >= 3:
            save(output / 'STOPPED.json', {'cause': 'three_consecutive_service_failures', 'last_job': job['job_id']})
            return
    save(output / 'FINISHED.json', {'sessions': len(schedule), 'model': MODEL, 'forecast_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    value = (prepare if args.operation == 'prepare' else run)(args.source, args.output)
    if value is not None:
        print(json.dumps(value))
