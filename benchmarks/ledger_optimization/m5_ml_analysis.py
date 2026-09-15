"""Pure scoring analysis for the prospectively reserved M5 ML grid.

No file/network access or dispatch. Call only on independently audited per-case
scores after the final access gate permits it; synthetic fixtures can be used
before that gate. Numerical criteria alone never establish the full objective.
"""
from collections import Counter
import hashlib
import json
import math
import random
from statistics import mean

ARMS = ('plain', 'gnomon', 'ledger')
ROUNDS = 26
STORES = 8
ITEMS_PER_STORE = 3
REPLICATES = 5000
BOOTSTRAP_SEED = 20260912
BLOCK_LENGTH = 4


def _cohort(panel, seeds, rows, *, arms=ARMS):
    if len(panel) != STORES * ITEMS_PER_STORE:
        raise ValueError('Require the original 24-series panel; no replacements')
    for entry in panel:
        if any(not isinstance(entry.get(k), str) or not entry[k].strip()
               for k in ('series_id', 'store_id', 'item_id')):
            raise ValueError('Explicit series, store and item identities required')
    if len({p['series_id'] for p in panel}) != len(panel) or len({p['item_id'] for p in panel}) != len(panel):
        raise ValueError('Series and items must each be unique in the reserved panel')
    counts = Counter(p['store_id'] for p in panel)
    if len(counts) != STORES or set(counts.values()) != {ITEMS_PER_STORE}:
        raise ValueError('Require eight store clusters with three items each')
    if len(seeds) != 2 or any(type(s) is not int for s in seeds) or len(set(seeds)) != 2:
        raise ValueError('Require two distinct prospectively frozen integer agent seeds')
    expected = {(a, p['series_id'], r, s) for a in arms for p in panel
                for r in range(ROUNDS) for s in seeds}
    keyed = {}
    for row in rows:
        if type(row.get('round')) is not int or type(row.get('seed')) is not int:
            raise ValueError('Round and seed must be integers, not coerced values')
        key = (row.get('arm'), row.get('series_id'), row['round'], row['seed'])
        if key in keyed:
            raise ValueError('Duplicate decision: ' + str(key))
        if key not in expected:
            raise ValueError('Unexpected decision: ' + str(key))
        value = row.get('rmsle')
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError('Every decision, including fallback, needs finite nonnegative RMSLE')
        if any(type(row.get(k)) is not bool for k in ('valid', 'workflow_complete', 'fallback_used')):
            raise ValueError('Explicit validity, workflow completion and fallback flags required')
        if row['workflow_complete'] and not row['valid']:
            raise ValueError('A full workflow requires a valid forecast')
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError(f'Incomplete grid: {len(expected-set(keyed))} missing decisions; never success-filter')
    return keyed


def _draw(rng):
    """All arms, three items and both seeds share the same sampled coordinates."""
    stores = [rng.randrange(STORES) for _ in range(STORES)]
    origins = []
    while len(origins) < ROUNDS:
        start = rng.randrange(ROUNDS)
        origins.extend((start + offset) % ROUNDS for offset in range(BLOCK_LENGTH))
    return stores, origins[:ROUNDS]


def _percentile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def _reduction(control, treatment):
    return 1 - treatment / control if control else None


def _analyze(panel, seeds, rows, *, arms=ARMS):
    """Analyze the complete requested grid, retaining failures and cold starts.

    ``panel`` is the frozen identity metadata, not inferred from observed rows.
    ``seeds`` comes from the final freeze, not from successful completions.
    Rows must already have independently verified per-case RMSLE, including the
    predeclared fallback for invalid executions. This helper cannot verify data
    provenance, leakage, model budgets, runtime versions or source immutability.
    """
    keyed = _cohort(panel, seeds, rows, arms=arms)
    stores = sorted({p['store_id'] for p in panel})
    series = {store: sorted(p['series_id'] for p in panel if p['store_id'] == store)
              for store in stores}
    # Balanced grid: averaging the six item/seed cases inside each store-origin
    # exactly preserves equal per-case weights and keeps cluster members paired.
    cubes = {a: [[mean(keyed[a, item, r, seed]['rmsle']
                       for item in series[store] for seed in sorted(seeds))
                  for r in range(ROUNDS)] for store in stores] for a in arms}
    scores = {a: mean(row['rmsle'] for key, row in keyed.items() if key[0] == a) for a in arms}
    controls = ('gnomon', 'plain')
    if 'ledger_reference' in arms:
        controls += ('ledger_reference',)
    draws = {control: [] for control in controls}
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(REPLICATES):
        store_indices, origins = _draw(rng)
        sample = {a: mean(cubes[a][s][r] for s in store_indices for r in origins) for a in arms}
        for control in draws:
            draws[control].append(_reduction(sample[control], sample['ledger']))
    contrasts = {}
    for control, values in draws.items():
        undefined = sum(v is None for v in values)
        interval = None if undefined else [_percentile(values, .025), _percentile(values, .975)]
        reduction = _reduction(scores[control], scores['ledger'])
        contrasts['ledger_vs_' + control] = {
            'relative_rmsle_reduction': reduction,
            'absolute_rmsle_reduction': scores[control] - scores['ledger'],
            'paired_95_interval': interval,
            'zero_control_bootstrap_draws': undefined,
            # Compare means directly so exact 0.8/1.0 is not lost to the
            # rounding in 1 - 0.8 (0.19999999999999996).
            'point_target_met': scores[control] > 0 and scores['ledger'] <= .8 * scores[control],
            'interval_excludes_zero_in_favor_of_ledger': interval is not None and interval[0] > 0,
        }
    primary = contrasts['ledger_vs_gnomon']
    canonical = {'panel': sorted(({k: p[k] for k in ('series_id', 'store_id', 'item_id')} for p in panel),
                                key=lambda p: p['series_id']), 'seeds': sorted(seeds),
                 'rows': [{**dict(zip(('arm', 'series_id', 'round', 'seed'), key)),
                           **{k: keyed[key][k] for k in ('rmsle', 'valid', 'workflow_complete', 'fallback_used')}}
                          for key in sorted(keyed)]}
    return {
        'scope': 'Numerical analysis only; provenance and fairness need separate audits.',
        'decisions': len(keyed), 'matched_cases_per_arm': len(keyed) // len(arms),
        'mean_per_case_rmsle': scores, 'contrasts': contrasts,
        'completion': {a: {field: sum(row[field] for key, row in keyed.items() if key[0] == a)
                           for field in ('valid', 'workflow_complete', 'fallback_used')} for a in arms},
        'per_store_mean_rmsle': {store: {a: mean(cubes[a][s]) for a in arms}
                                 for s, store in enumerate(stores)},
        'per_origin_mean_rmsle': [{a: mean(cubes[a][s][r] for s in range(STORES)) for a in arms}
                                  for r in range(ROUNDS)],
        'uncertainty': {'replicates': REPLICATES, 'seed': BOOTSTRAP_SEED,
                        'store_clusters': STORES, 'items_per_cluster': ITEMS_PER_STORE,
                        'agent_seeds_retained_together': sorted(seeds),
                        'shared_circular_origin_block_length': BLOCK_LENGTH,
                        'percentile_interpolation': 'linear',
                        'zero_control_policy': 'undefined interval if any resample control mean is zero',
                        'limitation': 'Eight stores; seeds/items are not independent store clusters.'},
        'numerical_primary_criteria_met': (primary['point_target_met'] and
                                          primary['interval_excludes_zero_in_favor_of_ledger']),
        'target_established': False,
        'analysis_input_sha256': hashlib.sha256(json.dumps(canonical, sort_keys=True,
                                  separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
    }


def analyze(panel, seeds, rows):
    """Original three-arm numerical analysis; unchanged public result contract.

    This does not assess improvement over the prior ledger. Use the separate
    four-arm analysis for the full objective, with independently audited scores.
    """
    return _analyze(panel, seeds, rows)
