"""Undeployed descriptive contrast of current CV and visible paired history.

No model recommendation, new evidence query, fit, or outcome access. The caller
must supply current runs from authenticated execution records. Returned means
compare configurations within each cohort, never CV levels against production
levels. Historical file identity and all paired aggregates are checked here.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from statistics import mean

from .paired_consistency_098 import paired_consistency


def _instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.utcoffset() is None:
        raise ValueError('Timezone-aware evidence instants required')
    return result


def current_history_contrast(review, full_evidence_bytes, current):
    """Contrast exactly two current three-fold runs with their historical pair.

    current contains query and runs. Each run supplies canonical config/config_id,
    provider/revision and folds containing origin, target_end, n, rmsle and
    execution_id. The three folds must match across configurations and their
    targets must have ended by this task's origin.
    """
    paired_consistency(review, full_evidence_bytes)  # Verify the unchanged source.
    if current['query'] != review['query']:
        raise ValueError('Current and historical evidence must concern the same task')
    query = review['query']
    now = _instant(query['origin'])
    horizon = query['horizon']
    runs = current['runs']
    if len(runs) != 2:
        raise ValueError('Exactly two current runs required')
    catalog = review['configuration_index']
    by_id, folds, refs, ends = {}, {}, {}, {}
    for run in runs:
        cid = run['config_id']
        if (cid in by_id or hashlib.sha256(json.dumps(run['config'], sort_keys=True).encode()).hexdigest()[:16] != cid):
            raise ValueError('Distinct canonical configuration identities required')
        if any(type(run.get(k)) is not str or not run[k] for k in ('provider', 'revision')):
            raise ValueError('Versioned current providers required')
        if cid in catalog and any(run[k] != catalog[cid][k] for k in ('config', 'provider', 'revision')):
            raise ValueError('Current and historical provider identities disagree')
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
        'schema_version': 'current-history-contrast-100', 'query': deepcopy(query),
        'metric': 'rmsle', 'config_ids': ids,
        'current_cv': {'matched_folds': 3, 'n': 3 * horizon, 'scores': scores,
                       'lowest_mean_config_ids': winners, 'fold_wins': wins, 'fold_ties': ties,
                       'lower_error_on_every_fold': [cid for cid in ids if wins[cid] == 3],
                       'origins': [t.isoformat() for t in origins],
                       'target_ends': [ends[left][t].isoformat() for t in origins],
                       'execution_ids': {cid: [refs[cid][t] for t in origins] for cid in ids}},
        'history': {}, 'full_evidence': deepcopy(review['full_evidence']),
        'provider_calls': 0, 'forecast_selection_made': False,
        'interpretation': 'Compare model scores only within each cohort. Conflicting CV and production winners are descriptive, not a rule to override either. A single historical origin does not establish repeatability.',
    }
    cards = [c for c in review['cards'] if set(c['config_ids']) == set(ids)]
    if not cards:
        result['history_status'] = ('configuration_not_in_historical_catalog'
                                    if any(cid not in catalog for cid in ids) else 'pair_not_on_supplied_page')
        result['history_evidence_pointer'] = None
        return result
    card = cards[0]
    result['history_status'] = 'pair_on_supplied_page'
    result['history_evidence_pointer'] = card['evidence_pointer']
    for label, window in card['windows'].items():
        if 'same_as' in window:
            result['history'][label] = {'same_as': window['same_as']}
            continue
        count = window['matched_origins']
        historical_winners = window['lowest_error_config_ids']
        result['history'][label] = {
            'matched_origins': count, 'n': window['n'], 'scores': deepcopy(window['scores']),
            'lowest_mean_config_ids': list(historical_winners),
            'latest_matched_origin': window['end'],
            'days_since_latest_matched_origin': ((now - _instant(window['end'])).total_seconds() / 86400 if count else None),
            'single_origin_only': count == 1,
            'current_cv_and_history_winners_disagree': bool(set(winners).isdisjoint(historical_winners)) if count else None,
            'evidence_pointer': window['evidence_pointer'],
        }
    return result
