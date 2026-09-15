"""Descriptive progress by prior-outcome count on already audited run rows.

No learned cutoff, optional stopping, forecast execution or final-data access.
The caller authenticates its report sources. Incomplete arm groups remain
explicitly pending; invalid/fallback forecasts are never success-filtered.
"""
from collections import defaultdict
import math
from statistics import mean

ARMS = ('plain', 'gnomon', 'ledger')


def summarize(rows):
    keyed = {}
    for row in rows:
        arm, series, number = row.get('arm'), row.get('series_id'), row.get('round')
        if (arm not in ARMS or not isinstance(series, str) or not series
                or type(number) is not int or number < 0
                or type(row.get('prior_outcomes')) is not int or row['prior_outcomes'] != number):
            raise ValueError('Canonical arm/series/round and explicit prior-outcome count required')
        if any(type(row.get(k)) is not bool for k in ('valid', 'workflow_complete', 'fallback_used')):
            raise ValueError('Explicit completion and fallback fields required')
        if not isinstance(row.get('origin'), str) or not row['origin'].strip():
            raise ValueError('Explicit forecast origin required, including pending sessions')
        value = row.get('rmsle')
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Every observed forecast needs a finite nonnegative score')
        key = arm, series, number
        if key in keyed:
            raise ValueError('Duplicate session across audit batches')
        keyed[key] = row
    tasks = sorted({(s, n) for a, s, n in keyed})
    matched = [(s, n) for s, n in tasks if all((a, s, n) in keyed for a in ARMS)]
    for series, number in matched:
        if len({keyed[a, series, number].get('origin') for a in ARMS}) != 1:
            raise ValueError('Matched arms disagree about forecast origin')

    def comparison(cases):
        scores = {a: mean(keyed[a, s, n]['rmsle'] for s, n in cases) if cases else None for a in ARMS}
        base, treatment = scores['gnomon'], scores['ledger']
        return {'matched_cases': len(cases), 'mean_rmsle': scores,
                'ledger_relative_improvement': 1-treatment/base if base else None,
                'relative_status': 'no_matched_cases' if not cases else
                                   ('defined' if base else 'undefined_zero_control_mean'),
                'valid': {a: sum(keyed[a, s, n]['valid'] for s, n in cases) for a in ARMS},
                'full_workflows': {a: sum(keyed[a, s, n]['workflow_complete'] for s, n in cases) for a in ARMS},
                'fallbacks': {a: sum(keyed[a, s, n]['fallback_used'] for s, n in cases) for a in ARMS}}

    by_round, by_series = defaultdict(list), defaultdict(list)
    for series, number in matched:
        by_round[number].append((series, number))
        by_series[series].append((series, number))
    cumulative = {}
    for series, cases in sorted(by_series.items()):
        cases.sort(key=lambda c: c[1])
        cumulative[series] = [{'through_round': n, 'prior_outcomes': n,
                               **comparison(cases[:index+1])}
                              for index, (_, n) in enumerate(cases)]
    return {'observed_sessions': len(rows), 'primary_all_matched': comparison(matched),
            'by_prior_outcomes': [{'prior_outcomes': n, 'series': [s for s, _ in cases],
                                   **comparison(cases)} for n, cases in sorted(by_round.items())],
            'per_series_cumulative': cumulative,
            'pending': [{'series_id': s, 'round': n,
                         'missing_arms': [a for a in ARMS if (a, s, n) not in keyed]}
                        for s, n in tasks if (s, n) not in matched],
            'cold_mature_threshold': None, 'threshold_predeclared': False,
            'target_established': False, 'final_gate_opened': False,
            'scope': 'Descriptive reused-development monitoring only. Every completed '
                     'three-arm group is included, including failures. Reporting by '
                     'origin/prior-outcome count does not identify a causal memory '
                     'benefit: calendar conditions and series coverage also change. '
                     'No cold/mature cutoff or accuracy stopping rule is introduced.'}
