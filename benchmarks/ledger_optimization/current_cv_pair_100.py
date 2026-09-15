"""Common-arm current-CV pair summary, with no historical ledger dependency.

Caller authenticates the supplied execution records and visibility. This helper
checks identities, matched complete folds, target endpoints and numeric values.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from statistics import mean


def _instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.utcoffset() is None:
        raise ValueError('Timezone-aware evidence instants required')
    return result


def current_cv_pair(current):
    """Summarize two authenticated current three-fold runs for any arm."""
    query = current['query']
    if (set(query) != {'series_id', 'unit', 'horizon', 'origin'}
            or type(query['series_id']) is not str or not query['series_id']
            or (query['unit'] is not None and (type(query['unit']) is not str or not query['unit']))
            or type(query['horizon']) is not int or query['horizon'] < 1):
        raise ValueError('An explicit series, unit, horizon and origin are required')
    now = _instant(query['origin'])
    horizon = query['horizon']
    runs = current['runs']
    if len(runs) != 2:
        raise ValueError('Exactly two current runs required')
    by_id, folds, refs, ends = {}, {}, {}, {}
    for run in runs:
        cid = run['config_id']
        if (cid in by_id or hashlib.sha256(json.dumps(run['config'], sort_keys=True).encode()).hexdigest()[:16] != cid):
            raise ValueError('Distinct canonical configuration identities required')
        if any(type(run.get(k)) is not str or not run[k] for k in ('provider', 'revision')):
            raise ValueError('Versioned current providers required')
        by_id[cid] = run
        if len(run['folds']) != 3:
            raise ValueError('Complete current three-fold runs required')
        values, execution_ids, targets = {}, {}, {}
        for fold in run['folds']:
            at = _instant(fold['origin'])
            end = _instant(fold['target_end'])
            value = fold['rmsle']
            if (not at < end <= now or at in values or type(fold['n']) is not int or fold['n'] != horizon
                    or type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or type(fold['execution_id']) is not str or not fold['execution_id']):
                raise ValueError('Distinct past complete folds with finite scores and execution references required')
            values[at] = value
            execution_ids[at] = fold['execution_id']
            targets[at] = end
        folds[cid] = values
        refs[cid] = execution_ids
        ends[cid] = targets
    ids = sorted(by_id)
    left, right = ids
    if set(folds[left]) != set(folds[right]) or ends[left] != ends[right]:
        raise ValueError('Current fold origins must match exactly')
    if len({(r['provider'], r['revision']) for r in runs}) != 2:
        raise ValueError('Distinct versioned current providers required')
    origins = sorted(folds[left])
    if len({refs[c][t] for c in ids for t in origins}) != 6:
        raise ValueError('Distinct current fold executions required')
    scores = {cid: mean(folds[cid].values()) for cid in ids}
    winners = [cid for cid in ids if scores[cid] == min(scores.values())]
    wins = {cid: sum(folds[cid][t] < folds[other][t] for t in origins)
            for cid, other in ((left, right), (right, left))}
    ties = sum(folds[left][t] == folds[right][t] for t in origins)
    result = {
        'schema_version': 'current-cv-pair-100', 'query': deepcopy(query),
        'metric': 'rmsle', 'config_ids': ids,
        'current_cv': {'matched_folds': 3, 'n': 3 * horizon, 'scores': scores,
                       'lowest_mean_config_ids': winners, 'fold_wins': wins, 'fold_ties': ties,
                       'lower_error_on_every_fold': [cid for cid in ids if wins[cid] == 3],
                       'origins': [t.isoformat() for t in origins],
                       'target_ends': [ends[left][t].isoformat() for t in origins],
                       'execution_ids': {cid: [refs[cid][t] for t in origins] for cid in ids}},
        'configuration_index': {cid: {key: deepcopy(by_id[cid][key]) for key in ('config', 'provider', 'revision')} for cid in ids},
        'provider_calls': 0, 'forecast_selection_made': False,
    }
    return result
