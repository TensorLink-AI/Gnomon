"""Measure prospective recipe exposure and subsequent actions from saved calls.

Descriptive secondary analysis only. It does not establish why an agent acted,
replace the workflow/recipe audits, change scores or admit a development run.
"""
import argparse
import hashlib
import json
from pathlib import Path


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def summarize_events(events, query, grade):
    pending, returned, exposed, attempted, completed = {}, set(), {}, set(), set()
    successful, views, historical, actionable, starts, start_requests = set(), 0, 0, 0, 0, 0
    latest_next = None; traces = []
    for ordinal, event in enumerate(events):
        if event.get('tool') != 'lab': continue
        key = event['tool_call_id']
        if event['stage'] == 'requested':
            if key in pending: raise ValueError('Repeated tool request identity')
            try: args = json.loads(event['raw_arguments'])
            except (ValueError, TypeError): args = {}
            if not isinstance(args, dict): args = {}
            # Freeze exposure at request time, before its response can add a plan.
            proposal = latest_next
            pending[key] = {'args': args, 'exposed': set(exposed), 'ordinal': ordinal,
                            'checkpoint_suggested': bool(proposal and proposal.get('arguments') == {'operation': 'start'})}
            if args.get('operation') == 'backtest' and isinstance(args.get('config'), dict):
                cid = config_id(args['config'])
                if cid in exposed: attempted.add(cid)
            if args.get('operation') == 'start' and pending[key]['checkpoint_suggested']: start_requests += 1
            continue
        if event['stage'] != 'returned': continue
        if key not in pending or key in returned: raise ValueError('Unmatched or duplicate tool return')
        returned.add(key); request = pending[key]; args = request['args']
        envelope = event.get('result', {})
        if envelope.get('status') != 'ok': continue
        stdout = envelope.get('result', {}).get('stdout')
        if not isinstance(stdout, str): continue
        try: reply = json.loads(stdout)
        except ValueError: continue
        if not isinstance(reply, dict) or reply.get('status') != 'ok': continue
        result = reply.get('result', {}); operation = args.get('operation')
        if operation == 'backtest' and isinstance(result.get('config'), dict):
            cid = config_id(result['config'])
            if result.get('config_id') != cid: raise ValueError('Backtest identity inconsistent')
            successful.add(cid)
            if cid in request['exposed']:
                attempted.add(cid)
                completed.add(cid)
                traces.append({'tool_call_id': key, 'request_event': request['ordinal'], 'response_event': ordinal,
                               'config_id': cid, 'first_proposal_event': exposed[cid], 'reused': result.get('reused')})
        if operation == 'start' and result.get('checkpoint_id') and request['checkpoint_suggested']: starts += 1
        if 'recipe_plan' not in reply: continue
        view = reply['recipe_plan']
        # The compact cold-start form omits query; the separately audited
        # artifact retains it. An omitted query cannot accompany any action.
        cold = view.get('status') == 'no_historical_catalog'
        if view.get('query') != query and not (
                cold and 'query' not in view and view.get('recipes') == [] and view.get('next_call') is None):
            raise ValueError('Recipe belongs to another task')
        if grade['arm'] != 'ledger': raise ValueError('Control was exposed to ledger recipe plan')
        views += 1; historical += view['status'] != 'no_historical_catalog'
        latest_next = view.get('next_call'); actionable += latest_next is not None
        for recipe in view['recipes']:
            cid = config_id(recipe['arguments']['config'])
            if recipe['config_id'] != cid: raise ValueError('Suggested configuration identity inconsistent')
            exposed.setdefault(cid, ordinal)
    selected = config_id(grade['config']) if isinstance(grade.get('config'), dict) else None
    return {'arm': grade['arm'], 'series_id': grade['series_id'], 'round': grade['round'], 'origin': grade['origin'],
        'plan_views': views, 'historical_plan_views': historical, 'actionable_plan_views': actionable,
        'proposed_configurations': sorted(exposed), 'proposed_then_requested_configurations': sorted(attempted),
        'proposed_then_successfully_backtested_configurations': sorted(completed),
        'successful_explicit_backtest_configurations': sorted(successful),
        'suggested_checkpoint_requests': start_requests, 'suggested_checkpoints_published': starts,
        'selected_config_id': selected,
        'selected_previously_proposed_and_backtested': bool(grade.get('valid') and not grade.get('fallback_used') and selected in completed),
        'unreturned_requests': sorted(set(pending)-returned), 'execution_traces': traces,
        'scope': 'Temporal association of published plans and explicit tool calls; not causal attribution.'}


def analyze(root, output):
    root, output = Path(root).absolute(), Path(output).absolute()
    if output.exists() or output.is_symlink() or root == output or root in output.parents or output in root.parents:
        raise ValueError('Fresh analysis outside original evidence required')
    rows, inputs = [], {}
    def read(path):
        raw = path.read_bytes(); inputs[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest(); return raw
    for path in sorted(root.glob('*/*/round-*/grade.json')):
        grade = json.loads(read(path)); task = json.loads(read(path.parent/'project/task.json'))
        query = {k: task[k] for k in ('series_id', 'unit', 'horizon', 'origin')}
        if grade['series_id'] != query['series_id'] or grade['origin'] != query['origin']:
            raise ValueError('Grade and task mismatch')
        events = [json.loads(line) for line in read(path.parent/'boundary-events.jsonl').splitlines()]
        rows.append(summarize_events(events, query, grade))
    if not rows: raise ValueError('No completed sessions to analyze')
    for name, digest in inputs.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest: raise ValueError('Evidence changed during analysis')
    summary = {}
    for arm in sorted({r['arm'] for r in rows}):
        selected = [r for r in rows if r['arm'] == arm]
        summary[arm] = {'sessions': len(selected),
            'sessions_exposed_to_plans': sum(r['plan_views'] > 0 for r in selected),
            'sessions_with_suggested_recipes': sum(bool(r['proposed_configurations']) for r in selected),
            'sessions_testing_a_previously_suggested_recipe': sum(bool(r['proposed_then_successfully_backtested_configurations']) for r in selected),
            'sessions_selecting_a_previously_suggested_and_tested_recipe': sum(r['selected_previously_proposed_and_backtested'] for r in selected),
            'suggested_checkpoints_published': sum(r['suggested_checkpoints_published'] for r in selected)}
    output.mkdir(parents=True)
    for name, value in [('sessions.json', rows), ('inputs.json', inputs)]:
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    result = {'status': 'descriptive_analysis_complete', 'sessions': len(rows), 'arms': summary,
        'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'session_results_sha256': hashlib.sha256((output/'sessions.json').read_bytes()).hexdigest(),
        'input_hashes_sha256': hashlib.sha256((output/'inputs.json').read_bytes()).hexdigest(),
        'original_inputs_unchanged': True, 'engy_calls': 0, 'provider_calls': 0,
        'accuracy_gate_changed': False, 'final_gate_opened': False,
        'limitations': ['Requires separate full workflow and recipe audits.',
                       'A later matching action does not prove that the suggestion caused it.',
                       'Counts successful explicit backtest calls, not implicit work inside start/commit.',
                       'No inference from agent prose and no exclusion of failed sessions.']}
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n'); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    print(json.dumps(analyze(**vars(parser.parse_args())), indent=2))
