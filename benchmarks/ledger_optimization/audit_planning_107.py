"""Read-only audit of recipe plans against actual published worker replies.

Does not import the recipe producer, sequence planner, or compact renderer.
The existing annotation audit independently checks current CV facts. Historical
page consistency is checked here; ledger visibility additionally requires the
original whole-workflow audit (this report never substitutes for that audit).
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from .contrast_audit_100 import audit_annotations
from .paired_consistency_098 import paired_consistency


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def audit_session(folder):
    folder = Path(folder); project = folder/'project'; checks = 0; inputs = {}

    def check(label, condition):
        nonlocal checks
        checks += 1
        if not condition: raise ValueError('Recipe audit: '+label)

    def read(name, *, base=project):
        p = Path(name)
        check('relative evidence path', not p.is_absolute() and '..' not in p.parts)
        for part in [p, *p.parents]:
            check('no evidence symlink', not (base/part).is_symlink())
        path = base/p; raw = path.read_bytes(); sha = digest(raw)
        check('read is stable', path not in inputs or inputs[path] == sha)
        inputs[path] = sha
        return raw

    common = audit_annotations(folder)
    task = json.loads(read('task.json')); arm = json.loads(read('backend.json'))['arm']
    query = {k: task[k] for k in ('series_id', 'unit', 'horizon', 'origin')}
    events = [json.loads(line) for line in read('boundary-events.jsonl', base=folder).splitlines()]
    raw_log = read('experiments.jsonl')
    all_rows = [json.loads(line) for line in raw_log.splitlines()]
    log_path = project/'planning-annotations.jsonl'
    records = [json.loads(line) for line in read('planning-annotations.jsonl').splitlines()] if log_path.exists() else []
    records = [r for r in records if r['query'] == query]
    check('controls have no recipe log', arm == 'ledger' or not log_path.exists())
    requested, latest_review, reviewed, checkpoint = {}, None, False, None
    index, previous_prefix, plans, suggested, attempts = 0, 0, 0, 0, 0
    for event in events:
        if event.get('stage') == 'requested' and event.get('tool') == 'lab':
            key = event['tool_call_id']; check('unique tool call', key not in requested)
            requested[key] = json.loads(event['raw_arguments'])
        if event.get('tool') != 'lab' and event.get('stage') != 'host_completion_check': continue
        envelope = event.get('result', {})
        if envelope.get('status') != 'ok': continue
        payload = envelope.get('result', {})
        if 'stdout' not in payload: continue
        reply = json.loads(payload['stdout'])
        args = requested[event['tool_call_id']] if event.get('tool') == 'lab' else {'operation': 'status'}
        operation = args['operation']; ok = reply.get('status') == 'ok'
        if ok and arm == 'ledger' and operation == 'review':
            reviewed = True
            latest_review = reply['result'] if reply['result'].get('schema_version') == 'agent-review-088' else None
        if ok and operation in ('start', 'commit') and reply['result'].get('checkpoint_id'):
            cp_id = reply['result']['checkpoint_id']; raw_cp = read(f'checkpoints/{cp_id}.json')
            checkpoint = json.loads(raw_cp)
            selections = [r for r in all_rows if r.get('event') == 'selection' and r.get('checkpoint_id') == cp_id]
            check('published checkpoint has one retained selection', len(selections) == 1)
            check('checkpoint digest and execution match', selections[0]['checkpoint_sha256'] == digest(encoded(checkpoint))
                  and checkpoint['execution_id'] == reply['result']['execution_id'])
            check('published checkpoint belongs to current task', checkpoint['task_origin'] == query['origin']
                  and checkpoint['series_id'] == query['series_id'] and checkpoint['unit'] == query['unit']
                  and checkpoint['future_timestamps'] == task['future_timestamps'])
        expected = ok and arm == 'ledger' and reviewed and operation in ('review', 'start', 'backtest', 'status')
        check('recipe presence follows actual review and operation', ('recipe_plan' in reply) == expected)
        if not expected: continue
        check('retained annotation exists', index < len(records))
        receipt = records[index]; index += 1; view = reply['recipe_plan']; reference = view['evidence']
        check('published view exactly retained', receipt['view'] == view and receipt['operation'] == operation)
        raw = read(reference['path']); artifact = json.loads(raw)
        check('artifact address length and digest', reference == {
            'path': f'planning-evidence/{digest(raw)}.json', 'sha256': digest(raw), 'bytes': len(raw)})
        check('canonical artifact bytes', raw == encoded(artifact))
        check('artifact identity', artifact['schema_version'] == 'recipe-plan-artifact-107'
              and artifact['operation'] == operation and artifact['query'] == query)
        check('input digest coverage', set(artifact['input_sha256']) == {'task.json', 'history.csv', 'future.csv', 'backend.json'})
        for name, sha in artifact['input_sha256'].items(): check('input digest', digest(read(name)) == sha)
        budget = reply['budget']
        check('budget exactly as published', artifact['observed_budget'] == budget)
        check('common attempt budget', budget['numerical_limit'] == 60
              and budget['numerical_remaining'] == 60-budget['numerical_attempts']
              and budget['fresh_backtest_fits'] == 4 and budget['reserved_final_fits'] == 1)
        ref = artifact['execution_log_prefix']; size = ref['bytes']
        check('monotonic available prefix', type(size) is int and previous_prefix <= size <= len(raw_log))
        previous_prefix = size; prefix = raw_log[:size]
        check('prefix integrity', (not prefix or prefix.endswith(b'\n')) and digest(prefix) == ref['sha256']
              and receipt['execution_log_prefix'] == ref)
        rows = [json.loads(line) for line in prefix.splitlines()]
        rows = [r for r in rows if r.get('task_origin') == query['origin']]
        count = sum(r['event'] == 'attempt' for r in rows)
        check('exact current attempt count', count == budget['numerical_attempts'] and count >= attempts)
        # The common audit checks fit deltas across *all* replies, including
        # commit (which may fit) and replies without a recipe annotation.
        attempts = count
        check('latest published checkpoint, not later selection', artifact['checkpoint'] == checkpoint)
        if checkpoint:
            check('checkpoint selection already in prefix', any(r.get('checkpoint_id') == checkpoint['checkpoint_id']
                  and r['event'] == 'selection' for r in rows))
        state = {'query': query, 'review': latest_review}
        check('exact most recently requested review', artifact['review_state_sha256'] == digest(encoded(state)))
        check('review reference matches latest request', artifact['review_reference'] ==
              (latest_review['full_evidence'] if latest_review else None))
        check('zero extra execution and no selection', artifact['provider_calls'] == 0
              and artifact['additional_ledger_queries'] == 0 and artifact['forecast_selection_made'] is False
              and view['provider_calls'] == 0 and view['additional_ledger_queries'] == 0)
        check('view schema and optional status', view['schema_version'] == 'recipe-plan-107' and view['optional'] is True)
        plan = artifact['plan']
        if latest_review is None:
            check('cold review cannot invent a catalog', plan is None and view['status'] == 'no_historical_catalog'
                  and view['next_call'] is None and view['recipes'] == [])
            continue
        plans += 1
        history_raw = read(latest_review['full_evidence']['path'])
        paired_consistency(latest_review, history_raw)
        history = json.loads(history_raw); catalog = latest_review['configuration_index']
        # Current CV table is already independently reconstructed by audit_annotations.
        complete = {r['config_id']: r['config'] for r in reply['evidence_summary']['current_cv']['configurations']}
        executed = [{'config_id': cid, 'config': config, 'provider': config['model']+'_'+cid,
                     'revision': 'ml-lab-v1:'+cid} for cid, config in sorted(complete.items())]
        observed = {'query': query, 'budget': budget, 'checkpoint_available': checkpoint is not None,
                    'executed_configurations': executed}
        baseline = {'model': 'seasonal', 'season': 7}
        baseline_id = digest(json.dumps(baseline, sort_keys=True).encode())[:16]
        if checkpoint is None:
            observed['initial_checkpoint'] = {'config': baseline, 'max_numerical_attempts': 4}
        check('observed context from independent CV and published checkpoint', plan['observed_state'] == observed)
        check('plan task and original page', plan['query'] == query and plan['full_evidence'] == latest_review['full_evidence']
              and plan['source_pagination'] == latest_review['pagination'])
        check('planning does not create projected execution', plan['schema_version'] == 'planning-sequence-107'
              and plan['provider_calls'] == 0 and plan['numerical_attempts'] == 0
              and plan['forecast_selection_made'] is False and plan['state_projected_as_executed'] is False)
        supports = {cid: set() for cid in catalog}; pointers = {cid: [] for cid in catalog}
        for i, card in enumerate(history['cards']):
            origins = {datetime.fromisoformat(r['origin']) for r in card['windows']['lifetime']['origins']}
            for cid in (card['left_config_id'], card['right_config_id']):
                supports[cid] |= origins
                if origins: pointers[cid].append(f'/cards/{i}/windows/lifetime')
        for cid, row in catalog.items():
            check('exact versioned recipe', cid == digest(json.dumps(row['config'], sort_keys=True).encode())[:16]
                  and row['provider'] == row['config']['model']+'_'+cid and row['revision'] == 'ml-lab-v1:'+cid)
        excluded = set(complete) | ({baseline_id} if checkpoint is None else set())
        eligible = sorted((c for c in catalog if c not in excluded and len(supports[c]) >= 4), key=lambda c: (-len(supports[c]), c))
        check('support threshold and deterministic non-error order', plan['minimum_distinct_origins'] == 4
              and plan['display_limit'] == 3 and plan['eligible_recipes_on_page'] == len(eligible)
              and [r['config_id'] for r in plan['recipes']] == eligible[:3])
        remaining = budget['numerical_remaining']; phase = budget['phase']; cost = 4; reserve = 1
        if checkpoint:
            reason = 'selection_phase' if phase != 'exploration' else 'insufficient_fits_with_final_reserve' if remaining < 5 else None
            capacity = (remaining-reserve)//cost if reason is None else 0
            first = None
            check('no redundant checkpoint prerequisite', plan['prerequisite'] is None and plan['checkpoint_max_numerical_attempts'] == 0)
        else:
            req = budget['agent_requests_remaining']; exp = budget['exploration_requests_remaining']
            reason = ('selection_phase' if phase != 'exploration' else 'insufficient_interaction_budget'
                      if req < 3 or exp < 2 else 'insufficient_sequence_numerical_budget' if remaining < 9 else None)
            capacity = min((remaining-5)//4, req-2, exp-1) if reason is None else 0
            first = {'tool': 'lab', 'arguments': {'operation': 'start'}, 'valid_for_task': query,
                     'runnable': True, 'admissible_now': bool(eligible) and capacity > 0,
                     'max_numerical_attempts': 4,
                     'blocked_reason': reason or ('no_supported_untried_recipe' if not eligible else None)}
            check('exact common checkpoint prerequisite', plan['prerequisite'] == first
                  and plan['checkpoint_max_numerical_attempts'] == 4 and plan['initial_checkpoint_config_id'] == baseline_id)
        calls = []
        for row in plan['recipes']:
            cid = row['config_id']; dates = sorted(supports[cid]); identity = catalog[cid]
            call = {'tool': 'lab', 'arguments': {'operation': 'backtest', 'config': identity['config']},
                    'valid_for_task': query, 'runnable': True, 'admissible_now': checkpoint is not None and reason is None,
                    'blocked_reason': reason if checkpoint else 'checkpoint_required', 'required_fits': 4, 'final_fit_reserve': 1}
            if checkpoint is None:
                call.update(conditional_budget_feasible=capacity > 0, sequence_blocked_reason=reason,
                            requires=['successful_task_matching_checkpoint', 'fresh_time_phase_and_budget_admission'])
            check('recipe fields and union support', row == {'config_id': cid, 'provider': identity['provider'],
                  'revision': identity['revision'], 'historical_origins_with_paired_evidence': len(dates),
                  'earliest_origin': dates[0].isoformat(), 'latest_origin': dates[-1].isoformat(),
                  'evidence_pointers': pointers[cid], 'next_call': call})
            calls.append(call)
        if checkpoint and calls: first = calls[0]
        expected_next = first if first and first['admissible_now'] else None
        check('next action retains prerequisites', plan['next_call'] == expected_next and view['next_call'] == expected_next)
        capacity = min(capacity, len(calls))
        check('shared sequential capacity', plan['sequential_capacity_upper_bound'] == capacity and view['shared_budget'] == {
            'numerical_remaining': remaining, 'checkpoint_maximum': 0 if checkpoint else 4,
            'fresh_recipe_maximum': 4, 'final_reserve': 1, 'recipe_capacity': capacity,
            'capacity_is_conditional': checkpoint is None})
        check('compact recipe content', view['recipes'] == [
            {'config_id': row['config_id'], 'arguments': call['arguments'],
             'historical_origins_with_paired_evidence': row['historical_origins_with_paired_evidence'],
             'admissible_now': call['admissible_now'], 'conditional_budget_feasible': call.get('conditional_budget_feasible'),
             'blocked_reason': call['blocked_reason']} for row, call in zip(plan['recipes'], calls, strict=True)])
        check('view completion and query', view['query'] == query and view['status'] == (
            'optional_experiments_available' if expected_next else 'no_admissible_optional_experiment'))
        suggested += expected_next is not None
    check('no retained annotation omitted from published sequence', index == len(records))
    for path, sha in inputs.items(): check('audit did not change evidence', digest(path.read_bytes()) == sha)
    return {'passed': True, 'checks': checks, 'existing_cv_audit_checks': common['checks'],
            'annotations': index, 'historical_plans': plans, 'actionable_next_calls': suggested,
            'inputs': {str(p.relative_to(folder)): sha for p, sha in inputs.items()},
            'provider_calls': 0, 'ledger_queries': 0,
            'scope': 'Actual boundary order, visible current CV, cached requested review, support unions, checkpoint and budget. '
                     'Original workflow audit remains required for source/recording visibility.'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    folders = sorted(p.parent for p in args.root.glob('*/*/round-*/grade.json'))
    if not folders: raise ValueError('No retained sessions')
    results = []
    try:
        for folder in folders:
            results.append({'session': str(folder.relative_to(args.root)), **audit_session(folder)})
    finally:
        (args.output/'sessions.json').write_text(json.dumps(results, indent=2)+'\n')
    report = {'passed': True, 'sessions': len(results), 'checks': sum(r['checks'] for r in results),
              'existing_cv_audit_checks': sum(r['existing_cv_audit_checks'] for r in results),
              'annotations': sum(r['annotations'] for r in results),
              'historical_plans': sum(r['historical_plans'] for r in results),
              'actionable_next_calls': sum(r['actionable_next_calls'] for r in results),
              'provider_calls': 0, 'ledger_queries': 0, 'engy_calls': 0,
              'recipe_audit_sha256': digest(Path(__file__).read_bytes()),
              'sessions_sha256': digest((args.output/'sessions.json').read_bytes())}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report))


if __name__ == '__main__': main()
