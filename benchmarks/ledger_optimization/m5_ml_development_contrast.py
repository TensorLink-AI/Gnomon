"""Descriptive matched M5 development comparisons; no inferential interval.

The fixed development panel contains eight items in only two stores. Item-level
resampling would misrepresent the dependence relevant to final confirmation.
The separate final analysis retains its prospectively specified store/time-block
sampling. This helper does not open data or authorize either evaluation.
"""
import math
from statistics import mean


def paired_series_contrast(pairs):
    if not pairs:
        raise ValueError('At least one matched pair is required')
    seen = set()
    for treatment, control in pairs:
        key = (treatment['series_id'], treatment['round'])
        if (key != (control['series_id'], control['round'])
                or treatment['origin'] != control['origin'] or key in seen):
            raise ValueError('Require distinct matched series/origin task identities')
        seen.add(key)
        for row in (treatment, control):
            value = row['rmsle']
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('RMSLE must be finite and nonnegative, including fallback cases')
    base = mean(b['rmsle'] for a, b in pairs)
    score = mean(a['rmsle'] for a, b in pairs)
    wins = sum(a['rmsle'] < b['rmsle']-1e-12 for a, b in pairs)
    losses = sum(a['rmsle'] > b['rmsle']+1e-12 for a, b in pairs)
    return {'pairs': len(pairs), 'control_mean_rmsle': base,
            'treatment_mean_rmsle': score,
            'relative_rmsle_reduction': 1-score/base if base else None,
            'relative_rmsle_status': 'defined' if base else 'undefined_zero_control_mean',
            'absolute_rmsle_reduction': base-score,
            'wins': wins, 'losses': losses, 'ties': len(pairs)-wins-losses,
            'exploratory_series_bootstrap_95_interval': None,
            'exploratory_absolute_series_bootstrap_95_interval': None,
            'relative_interval_status': 'not_estimated_development_two_store_cohort',
            'bootstrap_draws': 0, 'undefined_relative_draws': 0,
            'target_established': False,
            'warning': 'Descriptive development comparison only. Eight fixed items '
                       'share two stores; no independent-item interval is reported. '
                       'Final confirmation requires the untouched panel and its '
                       'frozen store-cluster/time-block analysis.'}
