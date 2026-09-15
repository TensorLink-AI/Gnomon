"""Read trusted CV records; never fit models or read production outcomes.

The host supplies the actual log prefix, canonical configuration validator and
exact requests reconstructed from the current visible input at each CV origin.
Production forecasts and other-task rows are ignored before reading their values.
"""
import hashlib
import json
import math
from statistics import mean

from .current_cv_pair_100 import _instant


def current_runs(task, records, expected_requests, canonicalize, *, arm, expected_actuals):
    """Return complete current runs and explicit incomplete-group exclusions.

    This is specific to the frozen ML lab's three folds and versioned provider
    identities. It is not a generic validator for arbitrary Gnomon studies.
    """
    if arm not in ('plain', 'gnomon', 'ledger'):
        raise ValueError('Host must supply the exact experiment arm')
    query = {k: task[k] for k in ('series_id', 'unit', 'horizon', 'origin')}
    now = _instant(query['origin'])
    horizon = query['horizon']
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Positive horizon required')
    expected = {}
    for request in expected_requests:
        origin = request['cutoff']
        times = [_instant(t) for t in request['future_timestamps']]
        if (any(request[k] != query[k] for k in ('series_id', 'unit', 'horizon'))
                or len(times) != horizon or times != sorted(set(times))
                or not _instant(origin) < times[0] <= times[-1] <= now
                or not request['timestamps'] or request['timestamps'][-1] != origin
                or len(request['timestamps']) != len(request['history']) or origin in expected):
            raise ValueError('Expected requests must be three distinct completed current-task CV folds')
        expected[origin] = request
    if len(expected) != 3:
        raise ValueError('Exactly three expected CV requests required')
    if set(expected_actuals) != set(expected):
        raise ValueError('Visible actuals required for each expected CV origin')
    grouped, metadata, execution_ids = {}, {}, set()
    for row in records:
        if (row.get('task_origin') != query['origin'] or row.get('event') != 'result'
                or row.get('kind') != 'backtest'):
            continue
        request = row['request']
        origin = request['cutoff']
        config = canonicalize(row['config'])
        cid = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        ex = row['execution']
        point, actual = row['point'], row['actual']
        eid = ex.get('execution_id')
        if (config != row['config'] or cid != row['config_id']
                or request != expected.get(origin)
                or ex.get('status', 'ok' if arm == 'plain' else None) != 'ok' or type(eid) is not str or not eid
                or eid in execution_ids
                or ex.get('provider') != config['model']+'_'+cid
                or ex.get('revision') != 'ml-lab-v1:'+cid
                or (arm != 'plain' and not {'point', 'timestamps', 'series_id', 'unit'} <= set(ex.get('result', {})))
                or ex.get('result', {}).get('point') != point
                or ex['result'].get('timestamps', request['future_timestamps'] if arm == 'plain' else None) != request['future_timestamps']
                or ex['result'].get('series_id', query['series_id'] if arm == 'plain' else None) != query['series_id']
                or ex['result'].get('unit', query['unit'] if arm == 'plain' else None) != query['unit']
                or actual != expected_actuals.get(origin)
                or len(point) != horizon or len(actual) != horizon
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in [*point, *actual])):
            raise ValueError('Current CV execution identity, request or scored pairs do not match trusted input')
        score = math.sqrt(mean((math.log1p(p)-math.log1p(a))**2 for p, a in zip(point, actual, strict=True)))
        reported = row.get('metrics', {})
        if (type(reported.get('n')) is not int or reported['n'] != horizon
                or type(reported.get('rmsle')) not in (int, float)
                or not math.isfinite(reported['rmsle']) or abs(reported['rmsle']-score) > 1e-12):
            raise ValueError('Reported current CV score disagrees with its scored pairs')
        folds = grouped.setdefault(cid, {})
        if origin in folds:
            raise ValueError('Ambiguous duplicate current CV origin')
        execution_ids.add(eid)
        metadata[cid] = {'config_id': cid, 'config': config, 'provider': ex['provider'], 'revision': ex['revision']}
        folds[origin] = {'origin': origin, 'target_end': request['future_timestamps'][-1],
                         'n': horizon, 'rmsle': score, 'execution_id': eid}
    runs, excluded = [], []
    for cid, folds in sorted(grouped.items()):
        if set(folds) == set(expected):
            runs.append({**metadata[cid], 'folds': [folds[t] for t in sorted(folds, key=_instant)]})
        else:
            excluded.append({'config_id': cid, 'reason': 'incomplete_current_cv',
                             'completed_folds': len(folds), 'required_folds': 3,
                             'missing_origins': sorted(set(expected)-set(folds), key=_instant)})
    return {'query': query, 'runs': runs, 'excluded': excluded,
            'verified_fold_executions': len(execution_ids), 'provider_calls': 0}
