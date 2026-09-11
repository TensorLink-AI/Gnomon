"""Development screen: learn how past CV errors related to mature outcomes.

This changes only model selection, never the candidate forecasts. A live test
would have to supply the same archived CV/outcome rows to every arm. This is
not the already-running context trial, not a confidence test and not holdout
evidence. Select a rule on origins 0..17 and report 18..25 separately.
"""
import argparse
from datetime import datetime
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean


def run(cases, contexts):
    variants = list(product(('series', 'pooled'), ('all', 'same_sparsity_trend'),
                            ('additive', 'multiplicative'), (4, 12, 32)))
    names = {v: '_'.join(map(str, v)) for v in variants}
    previous, rows = [], []
    for case in sorted(cases, key=lambda c: (c['origin'], c['series_id'])):
        now = datetime.fromisoformat(case['origin'])
        current_context = contexts[case['series_id'], case['round']]
        visible = [p for p in previous if datetime.fromisoformat(p['origin']) < now
                   and datetime.fromisoformat(p['outcome_recorded_at']) <= now
                   and max(map(datetime.fromisoformat, p['future_timestamps'])) <= now]
        providers = list(case['predictions'])
        estimates = {'current_cv': {p: case['current_card'][p]['cv_rmsle'] for p in providers}}
        sample_counts = {}
        for variant in variants:
            scope, relevance, method, prior = variant
            matched = [p for p in visible if (scope == 'pooled' or p['series_id'] == case['series_id'])
                       and (relevance == 'all' or all(contexts[p['series_id'], p['round']][key] == current_context[key]
                                                     for key in ('sparsity', 'trend')))]
            name = names[variant]
            sample_counts[name] = len(matched)
            estimate = {}
            for provider in providers:
                current_cv = case['current_card'][provider]['cv_rmsle']
                if method == 'additive':
                    correction = sum(p['scores'][provider]-p['current_card'][provider]['cv_rmsle'] for p in matched)/(len(matched)+prior)
                    estimate[provider] = max(0, current_cv+correction)
                else:
                    # Zero CV does not imply zero future error. The fixed .01
                    # offset makes finite log ratios; historical extremes are
                    # capped symmetrically at factor ten, disclosed here.
                    offsets = [max(-math.log(10), min(math.log(10), math.log((p['scores'][provider]+.01)/
                               (p['current_card'][provider]['cv_rmsle']+.01)))) for p in matched]
                    ratio = math.exp(sum(offsets)/(len(matched)+prior))
                    estimate[provider] = max(0, (current_cv+.01)*ratio-.01)
            estimates[name] = estimate
        choices = {name: min(providers, key=value.__getitem__) for name, value in estimates.items()}
        rows.append({'series_id': case['series_id'], 'round': case['round'], 'visible_origins': len(visible),
                     'sample_counts': sample_counts, 'choices': choices,
                     'rmsle': {name: case['scores'][p] for name, p in choices.items()}})
        previous.append(case)
    split = {label: {name: mean(row['rmsle'][name] for row in rows if predicate(row))
                     for name in rows[0]['rmsle']}
             for label, predicate in [('train', lambda r: r['round'] < 18),
                                      ('validation', lambda r: r['round'] >= 18), ('all', lambda r: True)]}
    selected = min((name for name in split['train'] if name != 'current_cv'), key=split['train'].__getitem__)
    return {'scope': 'development selection-policy screen, not an agent or held-out result',
            'changes_forecasts': False, 'provider_calls': 0, 'confirmation_opened': False,
            'variants': len(variants), 'training_origins': '0..17', 'validation_origins': '18..25',
            'selected_on_training_only': selected,
            'selected_validation_relative_gain': 1-split['validation'][selected]/split['validation']['current_cv'],
            'scores': split, 'cases': rows, 'target_established': False,
            'warning': 'Pooled variants do not imply different series are exchangeable. '
                       'Context is available at the original origin; only mature outcomes train each decision. '
                       'Any live trial must give the same historical CV/outcome information to all arms.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output file; retain previous screens')
    contexts = {(p['series_id'], p['round']): p['current_context'] for p in json.loads(args.memory.read_text())['packets']}
    report = run(json.loads(args.cases.read_text()), contexts)
    report['inputs'] = {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in [('cases', args.cases), ('memory', args.memory)]}
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'scores')}, indent=2))


if __name__ == '__main__':
    main()
