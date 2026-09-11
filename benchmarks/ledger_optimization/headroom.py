"""Measure fixed-portfolio headroom only after a complete live cohort.

This oracle uses future outcomes and is never an executable selection policy.
It is a lower bound on loss obtainable by choosing one existing candidate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def analyze(report, cases):
    if report['scope'] not in ('development only', 'confirmation') or report['run_completion'] != 'complete':
        raise ValueError('Requires a complete registered run')
    if report['scope'] == 'confirmation' and not report['manifest'].get('confirmation_freeze_sha256'):
        raise ValueError('Confirmation requires a frozen implementation')
    lookup = {(c['series_id'], c['round']): c for c in cases}
    rows = []
    for result in report['per_case']:
        if result['arm'] != 'no_ledger':
            continue
        case = lookup[result['series_id'], result['round']]
        scores = {}
        for provider, point in case['predictions'].items():
            if len(point) != len(case['actual']) or not point:
                raise ValueError('Candidate and actual horizon mismatch')
            scores[provider] = math.sqrt(mean(
                (math.log1p(max(p, 0))-math.log1p(a))**2
                for p, a in zip(point, case['actual'], strict=True)))
            if not math.isclose(scores[provider], case['scores'][provider], abs_tol=1e-12):
                raise ValueError('Archived score disagrees with independent calculation')
        if not math.isclose(scores[result['provider']], result['rmsle'], abs_tol=1e-12):
            raise ValueError('Live selected score disagrees with candidate cache')
        rows.append({'series_id': case['series_id'], 'round': case['round'],
                     'seed': result['seed'], 'control_rmsle': result['rmsle'],
                     'oracle_rmsle': min(scores.values())})
    control = mean(r['control_rmsle'] for r in rows)
    oracle = mean(r['oracle_rmsle'] for r in rows)
    return {'scope': report['scope'], 'matched_case_seed_pairs': len(rows),
            'control_mean_case_rmsle': control, 'oracle_mean_case_rmsle': oracle,
            'maximum_relative_improvement': 1-oracle/control,
            'target_relative_improvement': .2,
            'target_feasible_on_this_fixed_cohort': oracle <= .8*control,
            'confirmation_evaluated': report['scope'] == 'confirmation', 'target_established': False,
            'limitation': 'Future-aware bound for these fixed forecasts and this observed control only. '
                          'Not a deployable result or a bound on unseen series, other controls, '
                          'ensembles or new forecasting candidates.',
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('cases', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text()), json.loads(args.cases.read_text()))
    result['inputs'] = {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in [('report', args.report), ('cases', args.cases)]}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
