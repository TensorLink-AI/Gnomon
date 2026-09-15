"""Hindsight bounds on already executed development searches, never a policy."""
from collections import defaultdict
import math
from statistics import mean

ARMS = ('plain', 'gnomon', 'ledger')


def rmsle(point, actual):
    if (not isinstance(actual, list) or not actual or not isinstance(point, list)
            or len(point) != len(actual)):
        raise ValueError('Equal nonempty forecast and actual vectors required')
    for values in (point, actual):
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError('Finite numeric values required')
    if any(x < 0 for x in actual):
        raise ValueError('Negative actuals are inadmissible')
    return math.sqrt(mean((math.log1p(max(p, 0)) - math.log1p(a)) ** 2
                          for p, a in zip(point, actual, strict=True)))


def summarize(rows):
    """Caller authenticates executions and exact requests; this recomputes scores.

    Every observed row is retained; only complete three-arm groups enter means.
    Failed selections must use the common fallback and stay in the denominator.
    """
    keyed = {}
    for row in rows:
        a, s, n = row['arm'], row['series_id'], row['round']
        if (a not in ARMS or not isinstance(s, str) or not s
                or type(n) is not int or n < 0 or not row.get('origin')):
            raise ValueError('Explicit canonical task identity required')
        if any(type(row.get(k)) is not bool for k in ('valid', 'workflow_complete')):
            raise ValueError('Explicit workflow statuses required')
        key = s, n, a
        if key in keyed:
            raise ValueError('Duplicate arm/task')
        selected = rmsle(row['selected_point'], row['actual'])
        fallback = rmsle(row['fallback_point'], row['actual'])
        candidates = [{'choice': 'common_fallback', 'rmsle': fallback}]
        ids = set()
        for forecast in row['forecasts']:
            eid = forecast['execution_id']
            if not isinstance(eid, str) or not eid or eid in ids or not forecast['config_id']:
                raise ValueError('Unique successful execution identities required')
            ids.add(eid)
            candidates.append({'choice': eid, 'config_id': forecast['config_id'],
                               'rmsle': rmsle(forecast['point'], row['actual'])})
        if row['valid']:
            if not any(f['point'] == row['selected_point'] for f in row['forecasts']):
                raise ValueError('Valid selection must belong to an executed forecast')
        elif row['selected_point'] != row['fallback_point']:
            raise ValueError('Failed selection must retain the common fallback')
        keyed[key] = (row, selected, candidates)
    tasks = sorted({(s, n) for s, n, _ in keyed})
    cases, pending = [], []
    for s, n in tasks:
        missing = [a for a in ARMS if (s, n, a) not in keyed]
        if missing:
            pending.append({'series_id': s, 'round': n, 'missing_arms': missing})
            continue
        reference = keyed[s, n, 'plain'][0]
        if any(keyed[s, n, a][0][k] != reference[k]
               for a in ARMS for k in ('origin', 'actual', 'fallback_point')):
            raise ValueError('Matched arms must share origin, actuals and fallback')
        own = {a: min(keyed[s, n, a][2], key=lambda c: (c['rmsle'], c['choice']))
               for a in ARMS}
        cases.append({'series_id': s, 'round': n, 'origin': reference['origin'],
                      'selected_rmsle': {a: keyed[s, n, a][1] for a in ARMS},
                      'own_hindsight': own,
                      'union_hindsight_rmsle': min(v['rmsle'] for v in own.values()),
                      'configuration_counts': {a: len({f['config_id'] for f in
                          keyed[s, n, a][0]['forecasts']}) for a in ARMS},
                      'valid': {a: keyed[s, n, a][0]['valid'] for a in ARMS},
                      'full_workflows': {a: keyed[s, n, a][0]['workflow_complete'] for a in ARMS}})

    def aggregate(items):
        selected = {a: mean(c['selected_rmsle'][a] for c in items) if items else None for a in ARMS}
        own = {a: mean(c['own_hindsight'][a]['rmsle'] for c in items) if items else None for a in ARMS}
        union = mean(c['union_hindsight_rmsle'] for c in items) if items else None
        base = selected['gnomon']
        return {'matched_cases': len(items), 'selected_mean_rmsle': selected,
                'own_hindsight_mean_rmsle': own, 'union_hindsight_mean_rmsle': union,
                'own_reduction_vs_actual_no_ledger': {a: 1-v/base if base else None for a, v in own.items()},
                'union_reduction_vs_actual_no_ledger': 1-union/base if base else None,
                'valid': {a: sum(c['valid'][a] for c in items) for a in ARMS},
                'full_workflows': {a: sum(c['full_workflows'][a] for c in items) for a in ARMS}}

    by_series = defaultdict(list)
    for c in cases:
        by_series[c['series_id']].append(c)
    return {'observed_sessions': len(rows), 'all_matched': aggregate(cases),
            'by_series': {s: aggregate(cs) for s, cs in sorted(by_series.items())},
            'cases': cases, 'pending': pending, 'target_established': False,
            'final_gate_opened': False, 'executable_policy': False,
            'scope': 'Post-hoc hindsight over observed production forecasts plus the common fallback. '
                     'The union combines arm search trajectories and is not a budget-matched agent. '
                     'This neither estimates a past-only policy nor bounds untried configurations.'}
