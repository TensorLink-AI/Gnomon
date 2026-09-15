"""Undeployed, read-only origin consistency for existing RMSLE ledger reviews.

Verifies the supplied full-evidence bytes against the brief's digest and checks
their internal identities/counts/scores. Upstream ledger retrieval remains the
authority for recording visibility and forecast/actual correctness. No file,
ledger, model, or network access occurs here.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from statistics import mean

WINDOWS = ('last_4_origins', 'last_12_origins', 'lifetime')


def _instant(value):
    at = datetime.fromisoformat(value)
    if at.tzinfo is None:
        raise ValueError('Explicit timezone required')
    return at


def _score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative RMSLE required')
    return value


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate evidence JSON key')
        result[key] = value
    return result


def _models(rows, providers):
    if len(rows) != 2 or {r['provider']: r['revision'] for r in rows} != providers:
        raise ValueError('Both versioned pair identities required')
    return {r['provider']: _score(r['rmsle']) for r in rows}


def _statistics(window, ids, catalog, horizon, query_at):
    origins = window['origins']
    count = window['matched_origins']
    if (type(count) is not int or count < 0 or count != len(origins)
            or type(window['n']) is not int or window['n'] != count * horizon):
        raise ValueError('Complete matched origins and horizons required')
    providers = {catalog[c]['provider']: catalog[c]['revision'] for c in ids}
    if len(providers) != 2:
        raise ValueError('Pair configurations must identify distinct providers')
    points = []
    seen = set()
    for origin in origins:
        at = _instant(origin['origin'])
        if at in seen or at >= query_at:
            raise ValueError('Distinct historical origins required')
        seen.add(at)
        if type(origin['n']) is not int or origin['n'] != horizon:
            raise ValueError('Incomplete matched origin')
        scores = _models(origin['models'], providers)
        points.append((at, origin['origin'], [scores[catalog[c]['provider']] for c in ids]))
    points.sort()
    if not count:
        if (window['models'] or window['start'] is not None or window['end'] is not None
                or window['status'] != 'insufficient_evidence'):
            raise ValueError('Empty evidence has conflicting scores or bounds')
        return {}, {'status': 'no_matched_origins', 'matched_origins': 0,
                    'left_wins': 0, 'right_wins': 0, 'ties': 0,
                    'mean_left_minus_right': None, 'difference_range': None,
                    'lowest_on_every_origin_config_ids': [],
                    'aggregate_winner_has_origin_losses': None,
                    'latest_origin': None, 'latest_lowest_error_config_ids': []}
    if (_instant(window['start']) != points[0][0] or _instant(window['end']) != points[-1][0]
            or window['status'] != 'ok'):
        raise ValueError('Evidence bounds or status disagree with origins')
    means = {c: mean(row[2][i] for row in points) for i, c in enumerate(ids)}
    declared = _models(window['models'], providers)
    if any(not math.isclose(means[c], declared[catalog[c]['provider']], rel_tol=1e-12, abs_tol=1e-12)
           for c in ids):
        raise ValueError('Aggregate scores disagree with per-origin evidence')
    differences = [row[2][0] - row[2][1] for row in points]
    wins = [sum(d < 0 for d in differences), sum(d > 0 for d in differences)]
    ties = sum(d == 0 for d in differences)
    never_lost = [c for i, c in enumerate(ids) if wins[i] + ties == count]
    lowest_mean = [c for c in ids if declared[catalog[c]['provider']] == min(declared.values())]
    latest = points[-1]
    return {c: declared[catalog[c]['provider']] for c in ids}, {
        'status': 'descriptive_matched_origins', 'matched_origins': count,
        'left_wins': wins[0], 'right_wins': wins[1], 'ties': ties,
        'mean_left_minus_right': mean(differences),
        'difference_range': [min(differences), max(differences)],
        'lowest_on_every_origin_config_ids': never_lost,
        'aggregate_winner_has_origin_losses': any(c not in never_lost for c in lowest_mean),
        'latest_origin': latest[1],
        'latest_lowest_error_config_ids': [c for i, c in enumerate(ids) if latest[2][i] == min(latest[2])],
    }


def paired_consistency(review, full_evidence_bytes, *, detailed=False):
    """Augment the exact original page, preserving scores, losses and navigation.

    Counts use exact per-origin ties and are pair/window specific. Shared windows
    retain their references. detailed=True also adds observed difference ranges
    and latest-origin results. These are descriptive, not confidence intervals,
    independent samples, causal claims, or forecast recommendations.
    """
    if (review.get('schema_version') not in ('agent-review-087', 'agent-review-088')
            or review.get('metric') != 'rmsle' or review.get('provider_calls') != 0):
        raise ValueError('Original read-only RMSLE review required')
    if type(full_evidence_bytes) is not bytes:
        raise ValueError('Exact saved evidence bytes required')
    if type(detailed) is not bool:
        raise ValueError('detailed must be a boolean')
    if hashlib.sha256(full_evidence_bytes).hexdigest() != review['full_evidence']['sha256']:
        raise ValueError('Full evidence digest mismatch')
    full = json.loads(full_evidence_bytes, object_pairs_hook=_unique_object)
    query = review['query']
    horizon = query['horizon']
    if type(horizon) is not int or horizon <= 0:
        raise ValueError('Positive task horizon required')
    query_at = _instant(query['origin'])
    if (full['metric'] != 'rmsle' or full['provider_calls'] != 0
            or full['as_of'] != query['origin']
            or full['configuration_index'] != review['configuration_index']):
        raise ValueError('Review and full evidence identity mismatch')
    page = review['pagination']
    if (page['shown_pairs'] != len(review['cards']) or len(full['cards']) != len(review['cards'])
            or page['offset'] != full['offset'] or page['total_pairs'] != full['total_pairs']):
        raise ValueError('Review and full evidence page mismatch')
    expected_next = None if full['next_offset'] is None else {
        'operation': 'review', 'arguments': {'offset': full['next_offset'], 'limit': len(full['cards'])},
        'valid_for_origin': query['origin']}
    if page['next_call'] != expected_next:
        raise ValueError('Review and full evidence navigation mismatch')
    if page['all_pairs_included'] != (full['offset'] == 0 and full['next_offset'] is None
                                     and len(full['cards']) == full['total_pairs']):
        raise ValueError('Page completeness disagrees with full evidence')
    result = deepcopy(review)
    seen_pairs = set()
    for index, (original, source) in enumerate(zip(review['cards'], full['cards'], strict=True)):
        ids = original['config_ids']
        catalog = review['configuration_index']
        pair = tuple(sorted(ids))
        if (len(ids) != 2 or len(set(ids)) != 2 or pair in seen_pairs
                or ids != [source['left_config_id'], source['right_config_id']]
                or any(c not in catalog for c in ids)
                or any(type(catalog[c].get(k)) is not str or not catalog[c][k]
                       for c in ids for k in ('provider', 'revision'))
                or {catalog[c]['provider']: catalog[c]['revision'] for c in ids} != source['providers']):
            raise ValueError('Distinct source pair identities required')
        if original.get('evidence_pointer') != f'/cards/{index}':
            raise ValueError('Pair evidence pointer mismatch')
        seen_pairs.add(pair)
        if set(original['windows']) != set(WINDOWS) or set(source['windows']) != set(WINDOWS):
            raise ValueError('All three evidence windows required')
        for label in WINDOWS:
            supplied = original['windows'][label]
            window = source['windows'][label]
            scores, stats = _statistics(window, ids, catalog, horizon, query_at)
            pointer = f'/cards/{index}/windows/{label}'
            if supplied.get('evidence_pointer') != pointer:
                raise ValueError('Window evidence pointer mismatch')
            if 'same_as' in supplied:
                target = supplied['same_as']
                if (target not in WINDOWS[:WINDOWS.index(label)] or len(supplied) != 2
                        or source['windows'][target] != window):
                    raise ValueError('Shared window does not reference identical earlier evidence')
                continue
            if (type(supplied['matched_origins']) is not int or type(supplied['n']) is not int):
                raise ValueError('Integer brief counts required')
            for value in supplied['scores'].values():
                _score(value)
            if (supplied['scores'] != scores or supplied['matched_origins'] != stats['matched_origins']
                    or supplied['n'] != window['n'] or supplied['status'] != window['status']
                    or supplied['start'] != window['start'] or supplied['end'] != window['end']
                    or supplied['lowest_error_config_ids'] != [c for c in ids if scores and scores[c] == min(scores.values())]):
                raise ValueError('Brief scores or counts disagree with full evidence')
            result['cards'][index]['windows'][label]['origin_consistency'] = (
                stats if detailed else {k: stats[k] for k in
                    ('left_wins', 'right_wins', 'ties', 'aggregate_winner_has_origin_losses')})
    result['source_schema_version'] = review['schema_version']
    result['schema_version'] = 'agent-review-paired-098'
    result['origin_consistency_semantics'] = {
        'detail': 'detailed' if detailed else 'counts',
        'orientation': 'left and right follow each card config_ids order; lower RMSLE wins',
        'ties': 'Exact per-origin score equality, independent of aggregate ties',
        'empty_evidence': 'Zero counts with a null loss flag means no matched evidence, not a tie',
        'limitation': 'A lower mean does not imply winning every origin. Origins may overlap and are not independent samples. No causal or predictive superiority is established.',
        'scope': 'Same returned pairs and windows only; no global ranking, new query, forecast or selection',
    }
    if detailed:
        result['origin_consistency_semantics']['difference_range'] = (
            'Observed minimum and maximum left-minus-right; not a confidence interval')
    return result
