"""Development-only opportunity and historical-provider-linkage diagnostics."""
import argparse
from copy import deepcopy
import hashlib
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean, median

from .cv_context_screen import INPUT_SHA, instant, select, visible_history
from .override_trust_screen import prepare

SEED = 20260912
TRIALS = 256
TARGET = .2


def validated(cases):
    if not cases:
        raise ValueError('Require cases')
    providers = set(cases[0]['predictions'])
    if len(providers) < 2:
        raise ValueError('Require at least two providers')
    grouped = {}
    for c in cases:
        if set(c['predictions']) != providers:
            raise ValueError('Provider portfolio must be stable')
        values = list(c['actual']) + list(c['scores'].values())
        values += [v for point in c['predictions'].values() for v in point]
        values += [v['cv_rmsle'] for v in c['current_card'].values()]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not isfinite(v) for v in values):
            raise ValueError('Require finite numeric evidence')
        if any(v['cv_rmsle'] < 0 for v in c['current_card'].values()):
            raise ValueError('CV losses cannot be negative')
        times = list(map(instant, c['future_timestamps']))
        origin = instant(c['origin'])
        instant(c['outcome_recorded_at'])
        if (len(times) != len(c['actual']) or not times or
                times != sorted(set(times)) or times[0] <= origin):
            raise ValueError('Invalid forecast time range')
        if type(c['round']) is not int or c['round'] < 0:
            raise ValueError('Invalid round')
        grouped.setdefault(c['series_id'], []).append(c)
    for rows in grouped.values():
        rows = sorted(rows, key=lambda c: c['round'])
        if [c['round'] for c in rows] != list(range(26)):
            raise ValueError('Require complete consecutive development grid 0..25')
        origins = [instant(c['origin']) for c in rows]
        if origins != sorted(set(origins)):
            raise ValueError('Duplicate or unordered origins')
    return sorted(prepare(cases), key=lambda c: (instant(c['origin']), c['series_id']))


def choices_with_mapping(episodes, mappings):
    """Corrupt only copied historical labels; current requests/forecasts stay intact."""
    history_view = deepcopy(episodes)
    for c in history_view:
        original = c['scores']
        mapping = mappings[c['series_id']]
        if set(mapping) != set(original) or set(mapping.values()) != set(original):
            raise ValueError('Mapping must be a complete provider permutation')
        c['scores'] = {provider: original[source] for provider, source in mapping.items()}
    return [select(c, visible_history(c, history_view, conditioned=False), 4, .5) for c in episodes]


def relative_gain(reference, value):
    return 1 - value / reference if reference > 0 else None


def run(cases, *, trials=TRIALS, seed=SEED):
    if type(trials) is not int or trials < 1:
        raise ValueError('Require positive trial count')
    episodes = validated(cases)
    providers = sorted(episodes[0]['predictions'])
    series = sorted({c['series_id'] for c in episodes})
    predicates = {'all': lambda c: True, 'cold': lambda c: c['round'] < 4,
                  'mature': lambda c: c['round'] >= 4,
                  'later_development': lambda c: c['round'] >= 18}
    indices = {name: [i for i, c in enumerate(episodes) if predicate(c)]
               for name, predicate in predicates.items()}
    rng = Random(seed)
    shuffled = []
    for trial in range(trials):
        mappings = {}
        for name in series:
            permutation = providers[:]
            rng.shuffle(permutation)
            mappings[name] = dict(zip(providers, permutation, strict=True))
        choices = choices_with_mapping(episodes, mappings)
        shuffled.append({'trial': trial, 'mappings': mappings, 'choices_in_case_order': choices,
                         'mean_rmsle': {name: mean(episodes[i]['scores'][choices[i]] for i in ids)
                                        for name, ids in indices.items()}})
    summaries = {}
    for name, ids in indices.items():
        rows = [episodes[i] for i in ids]
        cv = mean(c['scores'][c['cv_provider']] for c in rows)
        support = mean(c['scores'][c['support_provider']] for c in rows)
        oracle = mean(min(c['scores'].values()) for c in rows)
        constants = {s: min(providers, key=lambda p: (mean(c['scores'][p] for c in rows if c['series_id'] == s), p))
                     for s in series}
        constant = mean(c['scores'][constants[c['series_id']]] for c in rows)
        samples = [t['mean_rmsle'][name] for t in shuffled]
        ceiling = relative_gain(cv, oracle)
        gain = relative_gain(cv, support)
        summaries[name] = {
            'cases': len(rows), 'current_cv_mean_rmsle': cv, 'frozen_support_mean_rmsle': support,
            'future_aware_case_oracle_mean_rmsle': oracle,
            'future_aware_constant_per_series_mean_rmsle': constant,
            'future_aware_constant_providers': constants,
            'case_oracle_relative_gain_vs_cv': ceiling,
            'constant_oracle_relative_gain_vs_cv': relative_gain(cv, constant),
            'support_relative_gain_vs_cv': gain,
            'oracle_opportunity_captured_by_support': (cv - support) / (cv - oracle) if cv > oracle else None,
            'required_oracle_opportunity_fraction_for_target': TARGET * cv / (cv - oracle) if cv > oracle else None,
            'corrupted_history_mean_rmsle': {'minimum': min(samples), 'median': median(samples), 'maximum': max(samples)},
            'fraction_corrupted_trials_worse_than_true_support': mean(s > support + 1e-12 for s in samples),
            'diagnostic_gates': {
                'headroom_reaches_target_against_cv_reference': ceiling is not None and ceiling >= TARGET,
                'true_support_beats_median_corrupted_history': support < median(samples) - 1e-12,
                'support_reaches_target_against_cv_reference': gain is not None and gain >= TARGET,
            },
        }
    return {'scope': 'previously used development only; diagnostic, not live-agent or confirmation evidence',
            'target_established': False, 'new_confirmation_authorized': False,
            'api_calls': 0, 'provider_calls': 0, 'forecasts_changed': False, 'confirmation_opened': False,
            'seed': seed, 'trials': trials, 'target_relative_gain': TARGET, 'summaries': summaries,
            'case_order': [{'series_id': c['series_id'], 'round': c['round'], 'origin': c['origin'],
                            'cv_provider': c['cv_provider'], 'support_provider': c['support_provider'],
                            'scores': c['scores'],
                            'visible_origins': [h['origin'] for h in visible_history(c, episodes, conditioned=False)]}
                           for c in episodes],
            'corruption_trials': shuffled,
            'limitations': [
                'Both oracles use future outcomes and cannot be deployed.',
                'The CV reference is not a matched live agent; no bound is asserted for other controls.',
                'Label-corruption distribution is descriptive, not a confidence interval or superiority p-value.',
                'All development slices were used previously. Do not select final cohorts by realized headroom.',
                'Source availability at valid time and recording at horizon close are inherited replay assumptions.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite retained evidence')
    raw = args.cases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise ValueError('Only registered development input is allowed')
    result = run(json.loads(raw))
    result['inputs'] = {'path': str(args.cases), 'sha256': INPUT_SHA}
    paths = [Path(__file__), Path(__file__).with_name('cv_context_screen.py'),
             Path(__file__).with_name('override_trust_screen.py'), Path(__file__).with_name('OPPORTUNITY_013.md')]
    result['source'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('case_order', 'corruption_trials')}, indent=2))


if __name__ == '__main__':
    main()
