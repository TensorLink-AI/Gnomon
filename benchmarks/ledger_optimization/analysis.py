"""Paired analysis using the registered series/shared-time-block bootstrap.

This computes statistical criteria; it cannot certify an untouched holdout,
temporal validity, or fair experimental execution. Those require a separate
provenance audit before any target-achieved claim.
"""
import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import random
from statistics import mean


REPLICATES = 5000
BOOTSTRAP_SEED = 20260911
BLOCK_LENGTH = 4
TARGET = .2


def _axis(manifest, name):
    values = manifest[name]
    if not values or len(values) != len(set(values)):
        raise ValueError(f'{name} must be nonempty and unique')
    return sorted(values)


def validated_cohort(report):
    """Reject missing, duplicate or extra observations before any averaging."""
    manifest = report['manifest']
    series = _axis(manifest, 'series')
    rounds = _axis(manifest, 'rounds')
    seeds = _axis(manifest, 'seeds_requested')
    arms = _axis(manifest, 'arms')
    if not {'no_ledger', 'ledger_119'} <= set(arms) or len(arms) < 3:
        raise ValueError('Requires no-ledger, original-ledger and treatment arms')
    if rounds != list(range(rounds[0], rounds[-1]+1)):
        raise ValueError('Shared four-origin blocks require consecutive origins; sparse samples are not eligible')
    if len(rounds) < BLOCK_LENGTH:
        raise ValueError('Not enough origins for the registered block length')
    expected = set(product(series, rounds, seeds, arms))
    rows = {}
    for row in report['per_case']:
        key = (row['series_id'], row['round'], row['seed'], row['arm'])
        if key in rows:
            raise ValueError(f'Duplicate decision: {key}')
        value = row['rmsle']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Every decision must have finite nonnegative RMSLE, including fallback decisions')
        if not isinstance(row['fallback_used'], bool):
            raise ValueError('Every decision must disclose fallback status')
        rows[key] = row
    if set(rows) != expected:
        raise ValueError(f'Incomplete/mismatched cohort: missing={len(expected-set(rows))}, extra={len(set(rows)-expected)}')
    if manifest['expected_decisions'] != len(expected) or report.get('harness_failures', 0):
        raise ValueError('Manifest count mismatch or unresolved harness failures')
    return series, rounds, seeds, arms, rows


def _percentile(values, probability):
    ordered = sorted(values)
    index = (len(ordered)-1)*probability
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)


def _interval(values):
    if any(value is None for value in values):
        return None
    return [_percentile(values, .025), _percentile(values, .975)]


def _reduction(control, treatment):
    return 1-treatment/control if control else None


def _sample_indices(rng, series_count, origin_count):
    series = [rng.randrange(series_count) for _ in range(series_count)]
    origins = []
    while len(origins) < origin_count:
        start = rng.randrange(origin_count)
        origins.extend((start+i) % origin_count for i in range(BLOCK_LENGTH))
    return series, origins[:origin_count]


def analyze(report):
    series, rounds, seeds, arms, rows = validated_cohort(report)
    # Average requested seeds inside each case. They are not new independent
    # series, and all seeds are retained together in every bootstrap draw.
    cubes = {arm: [[mean(rows[s, r, seed, arm]['rmsle'] for seed in seeds)
                    for r in rounds] for s in series] for arm in arms}
    scores = {arm: mean(value for row in cube for value in row) for arm, cube in cubes.items()}
    draws = {arm: {'relative_vs_no_ledger': [], 'relative_vs_original': [],
                   'absolute_vs_no_ledger': [], 'absolute_vs_original': []}
             for arm in arms if arm not in ('no_ledger', 'ledger_119')}
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(REPLICATES):
        sampled_series, sampled_origins = _sample_indices(rng, len(series), len(rounds))
        # One shared time-block sample for every series and arm retains common
        # calendar shocks and exact treatment/control pairing.
        sample = {arm: mean(cube[i][j] for i in sampled_series for j in sampled_origins)
                  for arm, cube in cubes.items()}
        for arm, values in draws.items():
            for baseline, label in [('no_ledger', 'no_ledger'), ('ledger_119', 'original')]:
                values[f'relative_vs_{label}'].append(_reduction(sample[baseline], sample[arm]))
                values[f'absolute_vs_{label}'].append(sample[baseline]-sample[arm])
    comparisons = {}
    for arm, values in draws.items():
        intervals = {key: _interval(value) for key, value in values.items()}
        relative = _reduction(scores['no_ledger'], scores[arm])
        ci = intervals['relative_vs_no_ledger']
        comparisons[arm] = {
            'relative_reduction_vs_no_ledger': relative,
            'relative_reduction_vs_original': _reduction(scores['ledger_119'], scores[arm]),
            'absolute_reduction_vs_no_ledger': scores['no_ledger']-scores[arm],
            'absolute_reduction_vs_original': scores['ledger_119']-scores[arm],
            'paired_95_percent_intervals': intervals,
            'numerical_criteria_met': relative is not None and relative >= TARGET
                                    and ci is not None and ci[0] > 0
                                    and scores[arm] < scores['ledger_119'],
        }

    def grouped(predicate):
        selected = [row for row in rows.values() if predicate(row)]
        return {'decisions': len(selected), 'mean_case_rmsle': {
            arm: mean(row['rmsle'] for row in selected if row['arm'] == arm)
            for arm in arms} if selected else None}

    return {
        'scope': report['scope'], 'target_established': False,
        'interpretation': 'Numerical criteria only. A development result never establishes the target; '
                          'confirmation requires the frozen protocol and a separate provenance audit.',
        'metric': 'Arithmetic mean of per-case RMSLE; equal series/origin/requested-seed weights',
        'cohort': {'series': len(series), 'origins': len(rounds), 'requested_seeds': seeds,
                   'decisions': len(rows), 'matched_case_seed_pairs': len(rows)//len(arms)},
        'mean_case_rmsle': scores, 'comparisons': comparisons,
        'bootstrap': {'replicates': REPLICATES, 'random_seed': BOOTSTRAP_SEED,
                      'rng': 'Python random.Random', 'block_length_origins': BLOCK_LENGTH,
                      'series_resampled_with_replacement': True, 'shared_circular_time_blocks': True,
                      'agent_seeds_retained_together': True, 'interval': 'Percentile 2.5/97.5 with linear interpolation',
                      'zero_control_rule': 'Relative interval is null if any replicate control mean is zero; '
                                           'absolute intervals remain defined. No replicates discarded.'},
        'per_seed': {str(seed): grouped(lambda row: row['seed'] == seed) for seed in seeds},
        'per_series': {s: grouped(lambda row: row['series_id'] == s) for s in series},
        'cold_start': grouped(lambda row: 0 <= row['round'] <= 3),
        'mature_history': grouped(lambda row: row['round'] >= 4),
        'completion': {arm: {'fallbacks': sum(row['fallback_used'] for row in rows.values() if row['arm'] == arm),
                            'resolution_statuses': dict(Counter(row['resolution_status'] for row in rows.values()
                                                                if row['arm'] == arm))} for arm in arms},
        'reported_usage': report.get('completion'),
        'limitations': ['Finite-panel uncertainty; no general retail-population or model claim',
                        'Requested seed reproducibility depends on backend support',
                        'No causal-explanation or dollar-savings claim'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text()))
    result['input'] = {'path': str(args.report), 'sha256': hashlib.sha256(args.report.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'cohort': result['cohort'], 'comparisons': result['comparisons'],
                      'target_established': result['target_established']}, indent=2))


if __name__ == '__main__':
    main()
