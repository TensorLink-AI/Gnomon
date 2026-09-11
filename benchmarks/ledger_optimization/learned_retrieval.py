"""Explore pooled historical experience for selecting fixed providers.

Trees estimate provider error, not demand; no forecast is changed or combined.
Every fit uses only earlier, matured development episodes. A future live trial
would require all arms to have the same pooled historical inputs/outcomes.
This is not the registered within-series live experiment.
"""
import argparse
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from gnomon import TemporalLedger


def features(case, request, providers):
    history = request['history'][-112:]
    avg = mean(history)
    values = [np.log1p(case['current_card'][p]['cv_rmsle']) for p in providers]
    values += [np.log1p(case['current_card'][p]['forecast_total']/request['horizon'])-np.log1p(avg) for p in providers]
    for n in (14, 28, 56, 112):
        sample = history[-n:]
        values += [np.log1p(mean(sample)), np.log1p(pstdev(sample)), sum(x == 0 for x in sample)/len(sample)]
    if not all(np.isfinite(values)):
        raise ValueError('Nonfinite feature')
    return values


def run(cases, requests):
    providers = sorted(cases[0]['predictions'])
    variants = list(product((3, None), (8, 16), ('loss', 'cv_residual'), (.5, 1)))
    names = {v: '_'.join(map(str, v)) for v in variants}
    rows = []
    x = np.array([features(c, requests[c['series_id'], c['round']], providers) for c in cases])
    cv = np.array([[c['current_card'][p]['cv_rmsle'] for p in providers] for c in cases])
    losses = np.array([[c['scores'][p] for p in providers] for c in cases])
    for origin in sorted({c['origin'] for c in cases}):
        now = datetime.fromisoformat(origin)
        current = [i for i, c in enumerate(cases) if c['origin'] == origin]
        visible = [i for i, c in enumerate(cases) if datetime.fromisoformat(c['origin']) < now
                   and datetime.fromisoformat(c['outcome_recorded_at']) <= now
                   and max(map(datetime.fromisoformat, c['future_timestamps'])) <= now]
        estimated = {'current_cv': cv[current]}
        for depth, leaf, target in product((3, None), (8, 16), ('loss', 'cv_residual')):
            prediction = cv[current].copy()
            if len(visible) >= 32:
                model = ExtraTreesRegressor(n_estimators=128, max_depth=depth, min_samples_leaf=leaf,
                                            random_state=20260911, n_jobs=1)
                y = losses[visible] if target == 'loss' else losses[visible]-cv[visible]
                model.fit(x[visible], y)
                prediction = model.predict(x[current])
                if target == 'cv_residual':
                    prediction += cv[current]
                prediction = np.maximum(prediction, 0)
            for weight in (.5, 1):
                estimated[names[depth, leaf, target, weight]] = weight*prediction+(1-weight)*cv[current]
        for j, index in enumerate(current):
            choices = {name: providers[int(np.argmin(values[j]))] for name, values in estimated.items()}
            case = cases[index]
            rows.append({'series_id': case['series_id'], 'round': case['round'], 'visible_training_cases': len(visible),
                         'choices': choices, 'rmsle': {name: case['scores'][p] for name, p in choices.items()}})
    scores = {label: {name: mean(row['rmsle'][name] for row in rows if predicate(row)) for name in rows[0]['rmsle']}
              for label, predicate in [('train', lambda r: r['round'] < 18),
                                       ('validation', lambda r: r['round'] >= 18), ('all', lambda r: True)]}
    selected = min(names.values(), key=lambda name: scores['train'][name])
    return {'scope': 'pooled development selection-support prototype; not an agent or held-out result',
            'forecasts_changed': False, 'provider_calls': 0, 'confirmation_opened': False, 'target_established': False,
            'features': 'Current CV RMSLE, forecast totals normalized to current history mean, and observable history moments/zero fractions',
            'minimum_mature_training_cases': 32, 'variants': len(variants),
            'selected_on_training_only': selected,
            'selected_validation_relative_gain': 1-scores['validation'][selected]/scores['validation']['current_cv'],
            'scores': scores, 'cases': rows,
            'limitation': 'Learns selection evidence, not new demand forecasts. Pooling adds historical information '
                          'beyond the current within-series trial; any live comparator must receive the same pooled records. '
                          'No uncertainty or exchangeability claim is established by this screen.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new report path')
    cases = json.loads(args.cases.read_text())
    ledger = TemporalLedger(args.ledger, create=False)
    requests = {(c['series_id'], c['round']): ledger.execution(c['execution_ids'][0])['request'] for c in cases}
    report = run(cases, requests)
    report['inputs'] = {'cases_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                       'ledger_path': str(args.ledger), 'ledger_sha256': hashlib.sha256(args.ledger.read_bytes()).hexdigest()}
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'scores')}, indent=2))


if __name__ == '__main__':
    main()
