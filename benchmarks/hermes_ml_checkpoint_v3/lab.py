"""Checkpoint-first lab: bounded batches and explicit atomic selection."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import core
from numerical import configuration, config_id
from policy import phase, EXPLORATION_REQUESTS

LIMIT = 60
FOLDS = (688, 702, 716)


class Rejected(ValueError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code, self.details = code, details


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    """One atomic authoritative pointer; immutable payload already exists."""
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def current_records():
    task = core.read('task.json')
    return [r for r in core.logs() if r.get('task_origin') == task['origin']]


def tested():
    grouped = {}
    for r in current_records():
        if r['event'] == 'result' and r['kind'] == 'backtest':
            grouped.setdefault(r['config_id'], {})[r['request']['cutoff']] = r
    output = {}
    for cid, folds in grouped.items():
        if len(folds) != 3:
            continue
        rows = list(folds.values())
        output[cid] = {'config': rows[0]['config'], 'config_id': cid,
                       'folds': [{'origin': r['request']['cutoff'], 'metrics': r['metrics'],
                                  'execution_id': r['execution']['execution_id']} for r in rows],
                       'mean_rmsle': sum(r['metrics']['rmsle'] for r in rows) / 3}
    return output


def budget():
    attempts = sum(r['event'] == 'attempt' for r in current_records())
    agent = {}
    try:
        agent = core.read('agent-budget.json')
    except (OSError, ValueError):
        pass
    deadline = agent.get('deadline_epoch')
    seconds = deadline-time.time() if deadline else None
    active_phase = phase(agent.get('forwarded_requests', 0), seconds)
    return {'numerical_limit': LIMIT, 'numerical_attempts': attempts,
            'numerical_remaining': max(0, LIMIT-attempts),
            'reserved_final_fits': 1 if attempts < LIMIT else 0,
            'backtest_batches_remaining': max(0, (LIMIT-attempts-1)//3),
            'agent_requests_remaining': agent.get('remaining_requests'),
            'phase': active_phase,
            'exploration_requests_remaining': max(0, EXPLORATION_REQUESTS-agent.get('forwarded_requests', 0)) if agent else None,
            'next_action': 'commit a tested configuration; no new exploration' if active_phase == 'selection' else 'compare and commit early; last four requests are reserved',
            'seconds_remaining_approx': max(0, round(deadline-time.time(), 1)) if deadline else None,
            'agent_budget_basis': 'host metered API requests; time from parent launch'}


def checkpoint():
    if not Path('checkpoint.json').exists():
        return None
    value = core.read('checkpoint.json')
    task = core.read('task.json')
    if (value.get('task_origin') != task['origin'] or value.get('series_id') != task['series_id']
            or value.get('future_timestamps') != task['future_timestamps'] or value.get('unit') != task['unit']):
        raise Rejected('CHECKPOINT_TASK_MISMATCH', 'Checkpoint does not belong to the current task.')
    immutable = Path('checkpoints') / (value['checkpoint_id'] + '.json')
    if core.read(immutable) != value:
        raise Rejected('CHECKPOINT_CHANGED', 'Checkpoint differs from its immutable record.')
    return value


def status():
    selected = checkpoint()
    return {'budget': budget(), 'tested_configurations': list(tested().values()),
            'checkpoint': ({k: selected[k] for k in ('checkpoint_id', 'execution_id', 'config',
                             'selection_after_comparison', 'baseline_only')} if selected else None)}


def admit(cost, reserve=0):
    remaining = budget()['numerical_remaining']
    if remaining < cost + reserve:
        raise Rejected('NUMERICAL_BUDGET_RESERVED',
                       'Batch was not started. Commit a previously backtested configuration or keep the saved checkpoint.',
                       required_fits=cost, reserved_final_fits=reserve, remaining_fits=remaining,
                       numerical_calls_started=0, next_call={'argv': ['python', 'lab.py', 'status']})


def backtest(config, *, initial_baseline=False):
    cid = config_id(config)
    # Exact repetition retrieves the three stored folds; it does not refit silently.
    if cid in tested():
        return {**tested()[cid], 'reused': True}
    if budget()['phase'] == 'selection' and not initial_baseline:
        raise Rejected('SELECTION_PHASE_RESERVED',
                       'New exploration is closed. Explicitly commit a previously tested configuration.',
                       numerical_calls_started=0, next_call={'argv':['python','lab.py','status']})
    admit(3, reserve=1)
    core.append('experiments.jsonl', {'event': 'batch_admitted', 'task_origin': core.read('task.json')['origin'],
                'config_id': cid, 'fits': 3, 'remaining_before': budget()['numerical_remaining'], 'reserve': 1,
                'phase':budget()['phase'], 'initial_baseline':initial_baseline})
    rows = [core.execute(config, end, 'backtest') for end in FOLDS]
    return {'config': config, 'config_id': cid,
            'folds': [{'origin': r['request']['cutoff'], 'metrics': r['metrics'],
                       'execution_id': r['execution']['execution_id']} for r in rows],
            'mean_rmsle': sum(r['metrics']['rmsle'] for r in rows)/3, 'reused': False}


def publish(result, reason):
    task = core.read('task.json')
    request, _ = core.request_at(730)
    if result['request'] != request or result['kind'] != 'forecast' or result['actual'] is not None:
        raise Rejected('EXECUTION_TASK_MISMATCH', 'Selection must reference a forecast for this exact current task.')
    point = list(result['point'])
    if (len(point) != 14 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in point)
            or point != list(result['execution']['result']['point'])):
        raise Rejected('INVALID_RESULT', 'Result does not satisfy the numerical forecast contract.')
    evidence = tested()
    if result['config_id'] not in evidence:
        raise Rejected('CONFIG_NOT_BACKTESTED', 'Complete the three current folds for this configuration first.')
    complete = len(evidence) >= 2 and any(v['config']['model'] != 'seasonal' for v in evidence.values())
    old = checkpoint()
    value = {'schema_version': 1, 'checkpoint_id': str(uuid4()), 'task_origin': task['origin'],
             'series_id': task['series_id'], 'unit': task['unit'],
             'future_timestamps': request['future_timestamps'], 'request_fingerprint': digest(request),
             'execution_id': result['execution']['execution_id'], 'provider': result['execution']['provider'],
             'revision': result['execution']['revision'], 'point': point, 'config': result['config'],
             'config_id': result['config_id'], 'selection_after_comparison': complete,
             'baseline_only': not complete, 'compared_configurations': sorted(evidence),
             'previous_checkpoint_id': old['checkpoint_id'] if old else None, 'reason': reason}
    destination = Path('checkpoints')
    destination.mkdir(exist_ok=True)
    path = destination / (value['checkpoint_id'] + '.json')
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    # A record written before the pointer is evidence of an attempted selection;
    # only the published pointer determines the final submitted execution.
    core.append('experiments.jsonl', {'event': 'selection', 'task_origin': task['origin'],
                'checkpoint_id': value['checkpoint_id'], 'execution_id': value['execution_id'],
                'checkpoint_sha256': digest(value), 'compared_configurations': sorted(evidence),
                'selection_after_comparison': complete})
    atomic_json('checkpoint.json', value)
    return {'saved': 'checkpoint.json', 'checkpoint_id': value['checkpoint_id'],
            'execution_id': value['execution_id'], 'config': value['config'],
            'selection_after_comparison': complete, 'previous_checkpoint_preserved': old is not None}


def commit(config=None, execution_id=None, reason='explicit_agent_selection'):
    available = [r for r in current_records() if r['event'] == 'result' and r['kind'] == 'forecast']
    if execution_id:
        found = [r for r in available if r['execution']['execution_id'] == execution_id]
        if len(found) != 1:
            raise Rejected('EXECUTION_NOT_FOUND', 'Select one logged forecast execution for the current task.',
                           available_execution_ids=[r['execution']['execution_id'] for r in available])
        if config is not None and config_id(config) != found[0]['config_id']:
            raise Rejected('SELECTION_CONFLICT', 'Configuration and execution_id identify different selections.')
        return publish(found[0], reason)
    if config is None:
        raise Rejected('SELECTION_REQUIRED', 'Supply --config or --execution-id; selection is never inferred from prose.')
    cid = config_id(config)
    if cid not in tested():
        raise Rejected('CONFIG_NOT_BACKTESTED', 'Backtest this configuration on all three current folds before committing it.')
    found = [r for r in available if r['config_id'] == cid]
    if len(found) > 1:
        raise Rejected('AMBIGUOUS_EXECUTIONS', 'Multiple forecasts share this configuration; supply --execution-id.',
                       available_execution_ids=[r['execution']['execution_id'] for r in found])
    if found:
        return publish(found[0], reason)
    admit(1)
    result = core.execute(config, 730, 'forecast')
    return publish(result, reason)


def start():
    existing = checkpoint()
    if existing:
        return {'reused_checkpoint': True, 'execution_id': existing['execution_id']}
    config = configuration({'model': 'seasonal', 'season': 7})
    admit(1 if config_id(config) in tested() else 4)
    backtest(config, initial_baseline=True)
    return commit(config, reason='initial_baseline_checkpoint')


def review():
    task = core.read('task.json')
    arm = core.read('backend.json')['arm']
    if arm == 'ledger':
        value = core.review()
    else:
        values = []
        for r in core.logs():
            if r['event'] == 'result' and r.get('metrics') is not None:
                values.append({'kind': r['kind'], 'origin': r['request']['cutoff'],
                               'config': r['config'], 'metrics': r['metrics'],
                               'execution_id': r['execution']['execution_id']})
            elif r['event'] == 'matured':
                values.append({'kind': 'matured_forecast', 'origin': r['forecast_origin'],
                               'config': r['config'], 'metrics': r['metrics'], 'execution_id': r['execution_id']})
        value = {'records': values[-20:], 'record_count': len(values), 'summary_limited': len(values)>20,
                 'ledger_queries': 0, 'note': 'Raw recorded facts; complete logs remain available. Unequal-origin averages are not matched evidence.'}
    core.append('review_calls.jsonl', {'task_origin': task['origin'], 'arm': arm,
                'ledger_queries': value.get('ledger_queries', 0), 'record_count': value.get('record_count', 0)})
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog='Ridge: window90..730, lags7..56, alpha.01..10000. Random Forest: window90..730, lags7..56, depth2..16. Seasonal: season1..28. start saves a baseline; backtest compares; commit explicitly selects.')
    p.add_argument('operation', choices=['start', 'status', 'review', 'backtest', 'commit', 'forecast', 'sync'])
    p.add_argument('--config', type=json.loads)
    p.add_argument('--execution-id')
    args = p.parse_args()
    # Hold the same lock across budget admission, all folds and publication.
    # Concurrent terminal calls cannot over-admit a batch or race checkpoint updates.
    with Path('.lab.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = budget()['numerical_attempts']
        try:
            config = configuration(args.config) if args.config is not None else None
            if args.operation == 'sync':
                result = {'synced': True, **core.sync()}
            elif args.operation == 'start':
                result = start()
            elif args.operation == 'status':
                result = status()
            elif args.operation == 'review':
                result = review()
            elif args.operation == 'backtest':
                if config is None:
                    raise Rejected('CONFIG_REQUIRED', 'Supply --config for a backtest.')
                if checkpoint() is None:
                    raise Rejected('CHECKPOINT_REQUIRED', 'Run python lab.py start before exploring configurations.')
                result = backtest(config)
            else:
                result = commit(config, args.execution_id)
            answer = {'status': 'ok', 'operation': args.operation, 'result': result, 'budget': budget()}
            code = 0
        except Exception as exc:
            answer = {'status': 'error', 'operation': args.operation,
                      'error': {'code': getattr(exc, 'code', type(exc).__name__), 'cause': str(exc),
                                'details': getattr(exc, 'details', {})}, 'budget': budget(),
                      'checkpoint_file_present': Path('checkpoint.json').is_file()}
            code = 2
        answer['numerical_calls_started'] = budget()['numerical_attempts'] - before
        print(json.dumps(answer, allow_nan=False))
        raise SystemExit(code)


if __name__ == '__main__':
    main()
