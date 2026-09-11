"""Past-only error calibration and retrieval screen; development data only.

Prototype selectors stay outside the runtime until independently justified.
Uses current CV vectors to retrieve similar finalized forecast episodes, or
regularized calibration of CV to subsequent RMSLE. No future labels are inputs.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import mean

import numpy as np


def features(case, providers):
    return np.log1p([case['current_card'][p]['cv_rmsle'] for p in providers])


def estimate(current, history, providers, method, strength, blend):
    cv = np.array([current['current_card'][p]['cv_rmsle'] for p in providers])
    if len(history) < 8:
        return cv
    x = np.asarray([features(h, providers) for h in history])
    y = np.asarray([[h['scores'][p] for p in providers] for h in history])
    location = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.1)
    x = (x-location)/scale
    current_x = (features(current, providers)-location)/scale
    if method == 'neighbors':
        distance = np.sqrt(np.mean((x-current_x)**2, axis=1))
        indices = np.argsort(distance, kind='stable')[:int(strength)]
        raw = np.average(y[indices], axis=0, weights=1/(0.1+distance[indices]))
    else:
        center = y.mean(axis=0)
        coefficients = np.linalg.solve(x.T@x+strength*np.eye(len(providers)), x.T@(y-center))
        raw = np.maximum(center+current_x@coefficients, 0)
    return (1-blend)*raw+blend*cv


def run(cases, train_rounds=60):
    variants = [(f'{scope}_{method}_{strength}_cv{blend}', scope, method, strength, blend)
        for scope in ('series', 'pooled')
        for method, values in [('neighbors', (3, 8, 16)), ('ridge', (1, 10, 100))]
        for strength in values for blend in (0, 0.5)]
    prior, rows = [], []
    for case in sorted(cases, key=lambda c: (c['origin'], c['series_id'])):
        providers = sorted(case['current_card'])
        now = datetime.fromisoformat(case['origin'])
        visible = [h for h in prior if datetime.fromisoformat(h['outcome_recorded_at']) <= now
                   and max(datetime.fromisoformat(t) for t in h['future_timestamps']) <= now]
        selected = {}
        for name, scope, method, strength, blend in variants:
            history = [h for h in visible if scope == 'pooled' or h['series_id'] == case['series_id']]
            estimates = estimate(case, history, providers, method, strength, blend)
            provider = providers[int(np.argmin(estimates))]
            selected[name] = {'provider': provider, 'rmsle': case['scores'][provider], 'visible_examples': len(history)}
        rows.append({'round': case['round'], 'series_id': case['series_id'], 'policies': selected})
        prior.append(case)
    train = [r for r in rows if r['round'] < train_rounds]
    valid = [r for r in rows if r['round'] >= train_rounds]
    aggregate = {name: {'development_train': mean(r['policies'][name]['rmsle'] for r in train),
                        'development_validation': mean(r['policies'][name]['rmsle'] for r in valid) if valid else None,
                        'all_development': mean(r['policies'][name]['rmsle'] for r in rows)} for name, *_ in variants}
    choice = min(aggregate, key=lambda name: aggregate[name]['development_train'])
    return {'scope': 'development hyperparameter screen; no final holdout or agent calls', 'api_calls': 0,
            'variant_count': len(variants), 'train_rounds': train_rounds, 'selected_on_training_only': choice,
            'selected_metrics': aggregate[choice], 'all_variants': aggregate, 'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-rounds', type=int, default=60)
    args = parser.parse_args()
    result = run(json.loads(args.cases.read_text()), args.train_rounds)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','all_variants')}, indent=2))


if __name__ == '__main__':
    main()
