"""Undeployed, read-only context cohorts over already authenticated pair evidence.

The caller authenticates files and upstream source/recording visibility. This
module checks task identity and horizon maturity; it does not establish the
trustworthiness of supplied observations or their real-world availability.
"""
from datetime import datetime
import math
from statistics import mean, pstdev


FILTERS = (
    ('sparsity', 'trend', 'promotion', 'volatility'),
    ('sparsity', 'trend', 'promotion'), ('sparsity', 'trend'), ('sparsity',), (),
)


def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.utcoffset() is None:
        raise ValueError('Explicit timezone required')
    return result


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative observations/scores required')
    return value


def context(request):
    """Original four Favorita labels; no target demand values are accepted."""
    history = request['history'][-28:]
    if len(history) != 28:
        raise ValueError('At least 28 observed values required')
    history = [number(v) for v in history]
    horizon = request['horizon']
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Positive horizon required')
    ts = [instant(t) for t in request['timestamps']]
    future = [instant(t) for t in request['future_timestamps']]
    origin = instant(request['cutoff'])
    if (len(ts) != len(request['history']) or ts != sorted(set(ts))
            or ts[-1] != origin or len(future) != horizon
            or future != sorted(set(future)) or future[0] <= origin):
        raise ValueError('Ordered, origin-bounded history and complete future dates required')
    zero = sum(v == 0 for v in history) / 28
    before, after = mean(history[:14]), mean(history[14:])
    trend = (('rising' if after else 'stable') if before == 0 else
             'falling' if after / before < .75 else
             'rising' if after / before > 4 / 3 else 'stable')
    average = mean(history)
    names = request.get('future_covariate_names', [])
    promotion = 'unknown'
    if len(names) != len(set(names)):
        raise ValueError('Unique covariate names required')
    if 'onpromotion' in names:
        values = request.get('future_covariates', [])
        if len(values) != horizon or any(len(row) != len(names) for row in values):
            raise ValueError('Complete future promotion plan required')
        values = [number(row[names.index('onpromotion')]) for row in values]
        if any(v not in (0, 1) for v in values):
            raise ValueError('Binary known promotion plan required')
        promotion = 'all' if all(values) else 'some' if any(values) else 'none'
    return {
        'sparsity': 'low' if zero < .1 else 'intermittent' if zero < .5 else 'high',
        'trend': trend,
        'volatility': 'volatile' if average and pstdev(history) / average > 1 else 'stable',
        'promotion': promotion,
    }


def cohorts(current_request, providers, episodes, current_cv_scores):
    """Pairwise nested cohorts; four origins is eligibility, not confidence.

episodes contain origin, source_as_of, recorded_as_of, and models with provider,
revision, execution_id, request and per-origin rmsle. providers maps exact
provider identities to revisions; no unexecuted or cross-pair scores are inferred.
"""
    if len(providers) != 2 or any(not k or not v for k, v in providers.items()):
        raise ValueError('Two versioned providers required')
    if set(current_cv_scores) != set(providers):
        raise ValueError('Both current CV scores required')
    for v in current_cv_scores.values():
        number(v)
    labels = context(current_request)
    query = instant(current_request['cutoff'])
    rows, seen = [], set()
    for episode in episodes:
        origin = instant(episode['origin'])
        if origin >= query or origin in seen:
            raise ValueError('Unique strictly historical origins required')
        seen.add(origin)
        if any(instant(episode[k]) > query for k in ('source_as_of', 'recorded_as_of')):
            raise ValueError('Evidence query exceeds current visibility boundary')
        models = episode['models']
        if len(models) != 2 or {m['provider'] for m in models} != set(providers):
            raise ValueError('Exact matched provider pair required')
        profiles, tails, scores, ids = [], [], {}, []
        for model in models:
            if model['revision'] != providers[model['provider']] or not model['execution_id']:
                raise ValueError('Exact versioned execution required')
            request = model['request']
            if any(request[k] != current_request[k] for k in ('series_id', 'unit', 'horizon')):
                raise ValueError('Historical task identity mismatch')
            profiles.append(context(request))
            if instant(request['cutoff']) != origin or instant(request['future_timestamps'][-1]) > query:
                raise ValueError('Complete historical horizon must mature before query')
            tails.append((request['history'][-28:],
                          [instant(t) for t in request['timestamps'][-28:]],
                          [instant(t) for t in request['future_timestamps']],
                          request.get('future_covariate_names', []),
                          request.get('future_covariates', [])))
            scores[model['provider']] = number(model['rmsle'])
            ids.append(model['execution_id'])
        if profiles[0] != profiles[1] or tails[0] != tails[1] or len(set(ids)) != 2:
            raise ValueError('Matched executions must share observable context and targets')
        rows.append({'origin': episode['origin'], 'context': profiles[0],
                     'scores': scores, 'execution_ids': ids})

    def winners(scores):
        return sorted(k for k, v in scores.items() if v == min(scores.values())) if scores else []

    results = []
    for keys in FILTERS:
        matched = [r for r in rows if all(r['context'][k] == labels[k] for k in keys)]
        scores = {p: mean(r['scores'][p] for r in matched) for p in providers} if matched else {}
        results.append({'filters': {k: labels[k] for k in keys}, 'matched_origins': len(matched),
                        'origins': [r['origin'] for r in matched], 'scores': scores,
                        'lowest_mean_providers': winners(scores), 'eligible': len(matched) >= 4})
    for r in results:
        r['disagrees_with_unfiltered'] = bool(r['scores']) and r['lowest_mean_providers'] != results[-1]['lowest_mean_providers']
        r['disagrees_with_current_cv'] = bool(r['scores']) and r['lowest_mean_providers'] != winners(current_cv_scores)
    selected = next((i for i, r in enumerate(results) if r['eligible']), None)
    return {'current_context': labels, 'cohorts': results, 'selected_filter_index': selected,
            'episodes': sorted(rows, key=lambda r: instant(r['origin'])),
            'current_cv_winners': winners(current_cv_scores),
            'provider_calls': 0, 'forecast_selection_made': False}
