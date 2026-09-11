"""Development-only, past-outcome support conditioned on the historical CV leader."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from math import isclose, log1p, sqrt
from pathlib import Path
from statistics import mean

INPUT_SHA = '5ee5a6dd4c346882287ef199f7640fa553a1f50062f73e9859ba33083f3a0c3f'
VARIANTS = tuple((minimum, fraction) for minimum in (4, 8) for fraction in (.5, .75))


def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Visibility timestamps must include a timezone')
    return result


def leader(case):
    return min(sorted(case['current_card']), key=lambda p: case['current_card'][p]['cv_rmsle'])


def rmsle(point, actual):
    if not actual or len(point) != len(actual) or any(a < 0 for a in actual):
        raise ValueError('Require a complete nonnegative actual horizon')
    return sqrt(mean((log1p(max(p, 0)) - log1p(a)) ** 2 for p, a in zip(point, actual)))


def visible_history(case, cases, *, conditioned):
    now, base = instant(case['origin']), leader(case)
    return sorted((c for c in cases
                   if c['series_id'] == case['series_id']
                   and instant(c['origin']) < now
                   and instant(c['outcome_recorded_at']) <= now
                   and c['future_timestamps']
                   and max(map(instant, c['future_timestamps'])) <= now
                   and (not conditioned or leader(c) == base)),
                  key=lambda c: instant(c['origin']))


def select(case, history, minimum, fraction):
    base = leader(case)
    if len(history) < minimum:
        return base
    candidates = []
    base_loss = mean(c['scores'][base] for c in history)
    for provider in sorted(case['current_card']):
        loss = mean(c['scores'][provider] for c in history)
        wins = sum(c['scores'][provider] < c['scores'][base] - 1e-12 for c in history)
        if loss < base_loss - 1e-12 and wins / len(history) >= fraction:
            candidates.append((loss, provider))
    return min(candidates)[1] if candidates else base


def run(cases):
    # Independent arithmetic verifies cached losses; never rewrite original inputs.
    cases = deepcopy(cases)
    keys = [(c['series_id'], c['round']) for c in cases]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate task identities')
    for c in cases:
        if set(c['predictions']) != set(c['current_card']) or set(c['scores']) != set(c['predictions']):
            raise ValueError('Provider cohorts must match')
        for p, point in c['predictions'].items():
            value = rmsle(point, c['actual'])
            if not isclose(value, c['scores'][p], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Cached metric disagrees with independent arithmetic')
            c['scores'][p] = value
    rows = []
    for case in sorted(cases, key=lambda c: (instant(c['origin']), c['series_id'])):
        history = visible_history(case, cases, conditioned=True)
        all_history = visible_history(case, cases, conditioned=False)
        choices = {'current_cv': leader(case), 'frozen_support': select(case, all_history, 4, .5)}
        choices.update({f'leader_context_{n}_{f}': select(case, history, n, f) for n, f in VARIANTS})
        rows.append({'series_id': case['series_id'], 'round': case['round'], 'origin': case['origin'],
                     'cv_leader': leader(case), 'matched_origins': len(history),
                     'retrieved_origins': [c['origin'] for c in history],
                     'all_visible_origins': len(all_history), 'choices': choices,
                     'rmsle': {name: case['scores'][p] for name, p in choices.items()}})
    scores = {}
    for split, predicate in [('train', lambda r: r['round'] < 18),
                             ('development_validation', lambda r: r['round'] >= 18),
                             ('all', lambda r: True)]:
        group = [r for r in rows if predicate(r)]
        if not group:
            raise ValueError('Both development slices must be present')
        scores[split] = {name: mean(r['rmsle'][name] for r in group) for name in rows[0]['rmsle']}
    selected = min((f'leader_context_{n}_{f}' for n, f in VARIANTS),
                   key=lambda name: (scores['train'][name], name))
    validation = scores['development_validation']
    return {'scope': 'iterative development only; not agent or untouched confirmation evidence',
            'target_established': False, 'provider_calls': 0, 'api_calls': 0,
            'forecasts_changed': False, 'confirmation_opened': False,
            'selected_on_training_only': selected, 'scores': scores,
            'selected_validation_gain_vs_cv': (1 - validation[selected] / validation['current_cv'])
            if validation['current_cv'] else None,
            'selected_validation_gain_vs_frozen_support': (1 - validation[selected] / validation['frozen_support'])
            if validation['frozen_support'] else None,
            'series_count': len({r['series_id'] for r in rows}), 'case_count': len(rows),
            'limitation': 'Previously used development slices, fixed deterministic policies, inherited source/recording assumptions. No uncertainty or agent-superiority claim.',
            'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite a retained experiment')
    raw = args.cases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise ValueError('Only the registered development input is allowed')
    result = run(json.loads(raw))
    result['inputs'] = {'path': str(args.cases), 'sha256': INPUT_SHA}
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['protocol_sha256'] = hashlib.sha256(Path(__file__).with_name('CV_CONTEXT_011.md').read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
