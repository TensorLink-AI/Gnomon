"""Descriptive robustness diagnostics on a complete matched development run.

These diagnostics cannot replace the primary mean-RMSLE target or its paired
confirmation interval. Cold/mature groups preserve the original round numbers.
"""
import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean


def describe(report):
    manifest = report['manifest']
    expected = set(product(manifest['series'], manifest['rounds'], manifest['seeds_requested'], manifest['arms']))
    records = {(r['series_id'], r['round'], r['seed'], r['arm']): r for r in report['per_case']}
    if len(records) != len(report['per_case']) or set(records) != expected or report.get('harness_failures'):
        raise ValueError('Robustness requires every registered matched decision exactly once')
    arms = manifest['arms']
    by_arm = {arm: [r for r in records.values() if r['arm'] == arm] for arm in arms}
    result = {'scope': report['scope'], 'target_established': False,
              'diagnostic_only': True, 'primary_metric_unchanged': True,
              'tail_definition': 'Mean of the largest ceil(10% * decisions_per_arm) case RMSLE values; each arm may have different tail cases',
              'tail': {}, 'paired_differences': {}, 'cold_start': {}, 'mature_history': {},
              'per_origin': {}, 'provider_selection_agreement_across_seeds': {}}
    for arm, rows in by_arm.items():
        losses = sorted((r['rmsle'] for r in rows), reverse=True)
        tail_n = max(1, math.ceil(.1*len(losses)))
        result['tail'][arm] = {'worst_decile_mean_rmsle': mean(losses[:tail_n]),
                               'tail_decisions': tail_n, 'maximum_rmsle': losses[0]}
        for label, predicate in [('cold_start', lambda r: r['round'] < 4), ('mature_history', lambda r: r['round'] >= 4)]:
            selected = [r['rmsle'] for r in rows if predicate(r)]
            result[label][arm] = {'decisions': len(selected), 'mean_case_rmsle': mean(selected) if selected else None}
        for index in manifest['rounds']:
            selected = [r['rmsle'] for r in rows if r['round'] == index]
            result['per_origin'].setdefault(index, {})[arm] = mean(selected)
        providers = defaultdict(set)
        for row in rows:
            providers[row['series_id'], row['round']].add(row['provider'])
        result['provider_selection_agreement_across_seeds'][arm] = {
            'cases_with_one_provider_name': sum(len(v) == 1 for v in providers.values()), 'cases': len(providers),
            'limitation': 'Names only; distinct providers can return identical forecasts. Agreement is not accuracy.'}
        if arm != 'no_ledger':
            deltas = [r['rmsle']-records[r['series_id'], r['round'], r['seed'], 'no_ledger']['rmsle'] for r in rows]
            result['paired_differences'][arm] = {
                'sign': 'treatment_minus_control; negative favors treatment',
                'mean': mean(deltas), 'maximum_damage': max(deltas), 'maximum_gain': -min(deltas),
                'better_cases': sum(d < -1e-12 for d in deltas), 'worse_cases': sum(d > 1e-12 for d in deltas),
                'tied_cases': sum(abs(d) <= 1e-12 for d in deltas), 'tie_absolute_tolerance': 1e-12}
    result['limitation'] = 'Descriptive finite development cohort. Tail metrics and subgroup wins do not establish the 20% overall target, causality or statistical significance.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = describe(json.loads(args.report.read_text()))
    result['input'] = {'path': str(args.report), 'sha256': hashlib.sha256(args.report.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_origin',)}, indent=2))


if __name__ == '__main__':
    main()
