"""Calculated comparison cards over complete, already-matched ledger origins.

These are descriptive summaries, not a significance test or a routing policy.
The ledger, not this module, resolves visibility, task identity and revisions.
"""

from itertools import combinations
import math
from statistics import mean

from .forecast_adapter import ForecastAdapterError


def validate_comparison_options(metric, recent_origins, negative_predictions):
    if metric not in ('mae', 'rmsle'):
        raise ForecastAdapterError('metric must be mae or rmsle', details={'rejected_fields': ['metric']})
    if type(recent_origins) is not int or not 1 <= recent_origins <= 1000:
        raise ForecastAdapterError('recent_origins must be an integer from 1 to 1000',
                                   details={'rejected_fields': ['recent_origins']})
    if negative_predictions not in ('reject', 'clip_zero'):
        raise ForecastAdapterError('negative_predictions must be reject or clip_zero',
                                   details={'rejected_fields': ['negative_predictions']})


def rmsle(pairs, negative_predictions='reject'):
    """Natural-log RMSLE; negative actuals always reject, clipping is opt-in."""
    errors, clipped = [], 0
    for prediction, actual in pairs:
        if not math.isfinite(prediction) or not math.isfinite(actual):
            raise ForecastAdapterError('rmsle_requires_finite_pairs')
        if actual < 0:
            raise ForecastAdapterError('rmsle_requires_nonnegative_actuals')
        if prediction < 0:
            if negative_predictions == 'reject':
                raise ForecastAdapterError('rmsle_negative_prediction_requires_explicit_clip_zero')
            prediction, clipped = 0, clipped + 1
        errors.append((math.log1p(prediction) - math.log1p(actual)) ** 2)
    return (math.sqrt(mean(errors)) if errors else None), clipped


def _window(origins, providers, metric):
    count = len(origins)
    if not count:
        return {'matched_origins': 0, 'n': 0, 'unique_actuals': 0, 'start': None, 'end': None,
                'ranking': [], 'ties': [], 'pairwise_differences': []}
    scores = {p: mean(next(m[metric] for m in o['models'] if m['provider'] == p)
                      for o in origins) for p in providers}
    order = sorted(providers, key=scores.__getitem__)
    ranking = [{'provider': p, 'score': scores[p],
                'rank': 1 + sum(v < scores[p] for v in scores.values())} for p in order]
    ties = [[p for p in order if scores[p] == value] for value in sorted(set(scores.values()))
            if sum(v == value for v in scores.values()) > 1]
    differences = []
    for left, right in combinations(providers, 2):
        delta = scores[left] - scores[right]
        differences.append({'left_provider': left, 'right_provider': right,
            'left_minus_right': delta,
            'left_relative_improvement_over_right': -delta / scores[right] if scores[right] else None,
            'relative_denominator': scores[right],
            'relative_status': 'defined' if scores[right] else 'undefined_zero_reference'})
    return {'matched_origins': count, 'n': sum(o['n'] for o in origins),
            'unique_actuals': len({a for o in origins for a in o['actual_ids']}),
            'start': origins[0]['origin'], 'end': origins[-1]['origin'],
            'ranking': ranking, 'ties': ties, 'pairwise_differences': differences}


def comparison_summary(origins, providers, *, metric='mae', recent_origins=4):
    """Keep all providers on one matched cohort; recent is its latest N origins."""
    origins = sorted(origins, key=lambda o: o['origin'])
    lifetime = _window(origins, providers, metric)
    recent = _window(origins[-recent_origins:], providers, metric)
    recent_ranks = {m['provider']: m['rank'] for m in recent['ranking']}
    disagreement = [{'provider': m['provider'], 'lifetime_rank': m['rank'],
                     'recent_rank': recent_ranks[m['provider']]}
                    for m in lifetime['ranking'] if m['rank'] != recent_ranks[m['provider']]]
    return {'metric': metric, 'lower_is_better': True,
            'aggregation': f'mean_{metric}_over_complete_matched_origins',
            'recent_origins_requested': recent_origins, 'lifetime': lifetime, 'recent': recent,
            'recent_lifetime_disagreement': bool(disagreement), 'rank_changes': disagreement,
            'tie_policy': 'exact_score_competition_rank_provider_input_order',
            'difference_semantics': 'left_minus_right below zero favors left; relative improvement divides by right',
            'evidence_pointer': '/origins', 'confidence_interval': None,
            'limitation': 'Descriptive matched evidence. Origins may overlap; ranks and differences do not establish statistical or causal superiority.'}
