"""Undeployed descriptive contrast of current CV and visible paired history.

No model recommendation, new evidence query, fit, or outcome access. The caller
must supply current runs from authenticated execution records. Returned means
compare configurations within each cohort, never CV levels against production
levels. Historical file identity and all paired aggregates are checked here.
"""
from copy import deepcopy

from .paired_consistency_098 import paired_consistency
from .current_cv_pair_100 import current_cv_pair, _instant


def current_history_contrast(review, full_evidence_bytes, current):
    """Add verified historical comparisons to the same common-arm CV summary."""
    paired_consistency(review, full_evidence_bytes)
    if current['query'] != review['query']:
        raise ValueError('Current and historical evidence must concern the same task')
    result = current_cv_pair(current)
    catalog = review['configuration_index']
    for cid, run in result.pop('configuration_index').items():
        if cid in catalog and any(run[k] != catalog[cid][k] for k in ('config', 'provider', 'revision')):
            raise ValueError('Current and historical provider identities disagree')
    result.update(schema_version='current-history-contrast-100', history={},
                  full_evidence=deepcopy(review['full_evidence']),
                  interpretation='Compare model scores only within each cohort. Conflicting CV and production winners are descriptive, not a rule to override either. A single historical origin does not establish repeatability.')
    ids = result['config_ids']
    winners = result['current_cv']['lowest_mean_config_ids']
    now = _instant(result['query']['origin'])
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
