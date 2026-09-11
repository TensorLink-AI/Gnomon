"""Past-only evidence about whether historical-support overrides beat contemporaneous CV."""
import argparse
from copy import deepcopy
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import mean

from .cv_context_screen import INPUT_SHA, instant, leader, rmsle, select, visible_history

VARIANTS = tuple((minimum, window) for minimum in (2, 4) for window in (0, 4))


def prepare(cases):
    cases = deepcopy(cases)
    keys = [(c['series_id'], c['round']) for c in cases]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate task identity')
    for c in cases:
        if set(c['predictions']) != set(c['current_card']) or set(c['scores']) != set(c['predictions']):
            raise ValueError('Provider cohorts differ')
        for p, point in c['predictions'].items():
            score = rmsle(point, c['actual'])
            if not isclose(score, c['scores'][p], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Independent arithmetic disagrees')
            c['scores'][p] = score
    # The proposal for each episode is fixed using ONLY its own origin's visible data.
    for c in cases:
        c['support_provider'] = select(c, visible_history(c, cases, conditioned=False), 4, .5)
        c['cv_provider'] = leader(c)
    return cases


def gate(current, episodes, minimum, window):
    visible = visible_history(current, episodes, conditioned=False)
    overrides = [c for c in visible if c['support_provider'] != c['cv_provider']]
    matched = overrides[-window:] if window else overrides
    deltas = [c['scores'][c['support_provider']] - c['scores'][c['cv_provider']] for c in matched]
    enough = len(matched) >= minimum
    supported = enough and mean(deltas) < -1e-12 and sum(d < -1e-12 for d in deltas) / len(deltas) >= .5
    reason = 'supported_by_past_overrides' if supported else 'insufficient_past_overrides' if not enough else 'past_overrides_do_not_support_change'
    if current['support_provider'] == current['cv_provider']:
        reason = 'no_override_proposed'
    return {'provider': current['support_provider'] if supported else current['cv_provider'],
            'reason': reason, 'matched_origins': len(matched),
            'mean_paired_delta': mean(deltas) if deltas else None,
            'strict_wins': sum(d < -1e-12 for d in deltas),
            'retrieved_origins': [c['origin'] for c in matched]}


def run(cases):
    episodes = prepare(cases)
    rows = []
    for case in sorted(episodes, key=lambda c: (instant(c['origin']), c['series_id'])):
        evidence = {f'override_trust_{n}_{w}': gate(case, episodes, n, w) for n, w in VARIANTS}
        choices = {'current_cv': case['cv_provider'], 'frozen_support': case['support_provider']}
        choices.update({name: item['provider'] for name, item in evidence.items()})
        rows.append({'series_id': case['series_id'], 'round': case['round'], 'origin': case['origin'],
                     'choices': choices, 'evidence': evidence,
                     'rmsle': {name: case['scores'][p] for name, p in choices.items()}})
    scores = {}
    for name, predicate in [('train', lambda r: r['round'] < 18),
                            ('development_validation', lambda r: r['round'] >= 18), ('all', lambda r: True)]:
        group = [r for r in rows if predicate(r)]
        if not group:
            raise ValueError('Both development slices must be present')
        scores[name] = {policy: mean(r['rmsle'][policy] for r in group) for policy in rows[0]['choices']}
    selected = min((f'override_trust_{n}_{w}' for n, w in VARIANTS), key=lambda name: (scores['train'][name], name))
    validation = scores['development_validation']
    return {'scope': 'iterative development only; not agent or untouched confirmation evidence',
            'target_established': False, 'api_calls': 0, 'provider_calls': 0, 'forecasts_changed': False,
            'confirmation_opened': False, 'selected_on_training_only': selected, 'scores': scores,
            'selected_validation_relative_gain': {name: 1 - validation[selected] / validation[name]
                                                   if validation[name] else None
                                                 for name in ('current_cv', 'frozen_support')},
            'series_count': len({r['series_id'] for r in rows}), 'case_count': len(rows), 'cases': rows,
            'limitation': 'Already-used development slices and inherited visibility assumptions. Small override counts are descriptive, not confidence or causal evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite retained evidence')
    raw = args.cases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise ValueError('Only registered development cases are allowed')
    result = run(json.loads(raw))
    result['inputs'] = {'path': str(args.cases), 'sha256': INPUT_SHA}
    result['source'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__), Path(__file__).with_name('cv_context_screen.py'),
                         Path(__file__).with_name('OVERRIDE_TRUST_012.md')]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
