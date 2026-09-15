"""Optional ledger recipe plans at a host-owned, locked lab response boundary."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .contrast_annotations_100 import _atomic, _json, _project_path
from .contrast_records_100 import current_runs
from .planning_sequence_107 import planning_sequence


def compact_view(plan):
    """Compress presentation only; the full plan remains authoritative."""
    if plan is None:
        return {'schema_version': 'recipe-plan-107', 'status': 'no_historical_catalog',
                'next_call': None, 'recipes': [], 'optional': True}
    view = {'schema_version': 'recipe-plan-107', 'query': deepcopy(plan['query']),
            'status': 'optional_experiments_available' if plan['next_call'] else 'no_admissible_optional_experiment',
            'optional': True, 'next_call': deepcopy(plan['next_call']), 'recipes': [],
            'shared_budget': {'numerical_remaining': plan['observed_state']['budget']['numerical_remaining'],
                              'checkpoint_maximum': plan['checkpoint_max_numerical_attempts'],
                              'fresh_recipe_maximum': plan['observed_state']['budget']['fresh_backtest_fits'],
                              'final_reserve': plan['observed_state']['budget']['reserved_final_fits'],
                              'recipe_capacity': plan['sequential_capacity_upper_bound'],
                              'capacity_is_conditional': not plan['observed_state']['checkpoint_available']},
            'guidance': 'Optional experiments, not an accuracy ranking. Do not execute every recipe by default. '
                        'Complete any prerequisite and recheck live time/budget before each backtest. '
                        'Choose explicitly after evaluating current CV; this plan selects no forecast.'}
    for recipe in plan['recipes']:
        call = recipe['next_call']
        view['recipes'].append({'config_id': recipe['config_id'],
                                'arguments': deepcopy(call['arguments']),
                                'historical_origins_with_paired_evidence': recipe['historical_origins_with_paired_evidence'],
                                'admissible_now': call['admissible_now'],
                                'conditional_budget_feasible': call.get('conditional_budget_feasible'),
                                'blocked_reason': call['blocked_reason']})
    return view


def annotate_planning(core, operation, result, *, budget, checkpoint_getter):
    """Use only the already requested current-task review, without a new query."""
    if operation not in ('review', 'start', 'backtest', 'status'):
        return None
    if core.read('backend.json')['arm'] != 'ledger':
        return None
    task = core.read('task.json')
    query = {k: task[k] for k in ('series_id', 'unit', 'horizon', 'origin')}
    state_path = _project_path('comparison-state.json')
    if not state_path.exists(): return None
    state_raw = state_path.read_bytes(); state = json.loads(state_raw)
    if state['query'] != query: return None
    review = state['review']
    checkpoint = checkpoint_getter()
    prefix = Path('experiments.jsonl').read_bytes() if Path('experiments.jsonl').exists() else b''
    records = [json.loads(line) for line in prefix.splitlines()]
    if checkpoint is not None and (
            checkpoint['task_origin'] != query['origin'] or checkpoint['series_id'] != query['series_id']
            or checkpoint['unit'] != query['unit'] or checkpoint['future_timestamps'] != task['future_timestamps']):
        raise ValueError('Checkpoint must belong to the exact current task')
    plan = None
    if review is not None:
        if review['query'] != query: raise ValueError('Historical recipe review belongs to another task')
        raw = _project_path(review['full_evidence']['path']).read_bytes()
        expected = [core.request_at(end) for end in (688, 702, 716)]
        current = current_runs(task, records, [r for r, _ in expected], core.configuration,
                               arm='ledger', expected_actuals={r['cutoff']: a for r, a in expected})
        context = {'query': query, 'budget': deepcopy(budget), 'checkpoint_available': checkpoint is not None,
                   'executed_configurations': [{k: r[k] for k in ('config_id', 'config', 'provider', 'revision')}
                                               for r in current['runs']]}
        if checkpoint is None:
            context['initial_checkpoint'] = {'config': core.configuration({'model': 'seasonal', 'season': 7}),
                                             'max_numerical_attempts': 4}
        plan = planning_sequence(review, raw, context, core.configuration)
    artifact = {'schema_version': 'recipe-plan-artifact-107', 'operation': operation, 'query': query,
                'execution_log_prefix': {'bytes': len(prefix), 'sha256': hashlib.sha256(prefix).hexdigest()},
                'review_state_sha256': hashlib.sha256(state_raw).hexdigest(),
                'review_reference': deepcopy(review['full_evidence']) if review else None,
                'checkpoint': deepcopy(checkpoint), 'observed_budget': deepcopy(budget), 'plan': plan,
                'input_sha256': {name: hashlib.sha256(_project_path(name).read_bytes()).hexdigest()
                                 for name in ('task.json', 'history.csv', 'future.csv', 'backend.json')},
                'provider_calls': 0, 'additional_ledger_queries': 0, 'forecast_selection_made': False}
    raw = _json(artifact); digest = hashlib.sha256(raw).hexdigest()
    path = f'planning-evidence/{digest}.json'
    _atomic(path, raw, immutable=True)
    if hashlib.sha256(_project_path(path).read_bytes()).hexdigest() != digest:
        raise ValueError('Recipe plan artifact persistence failed')
    view = compact_view(plan)
    view['evidence'] = {'path': path, 'sha256': digest, 'bytes': len(raw)}
    view['provider_calls'] = 0; view['additional_ledger_queries'] = 0
    with _project_path('planning-annotations.jsonl').open('ab') as stream:
        stream.write(_json({'operation': operation, 'query': query, 'view': view,
                            'execution_log_prefix': artifact['execution_log_prefix']})+b'\n')
    if ((Path('experiments.jsonl').read_bytes() if Path('experiments.jsonl').exists() else b'') != prefix
            or state_path.read_bytes() != state_raw):
        raise ValueError('Execution or requested review state changed during recipe annotation')
    return view
