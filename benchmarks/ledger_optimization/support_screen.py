"""Screen explicit historical support rules for changing a current CV choice.

No forecast is changed. The ledger must show repeated paired improvement,
enough matched origins and optionally recent confirmation before suggesting an
alternative. These are descriptive support rules, never significance claims.
Rules are selected on development origins 0..17, validated on 18..25.
"""
import argparse
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import mean


def suggest(case, packet, scope, minimum, fraction, recent, margin):
    providers = packet['raw_history']['providers']
    base = min(providers, key=lambda p: case['current_card'][p]['cv_rmsle'])
    rows = packet['raw_history']['records']
    if any(datetime.fromisoformat(r['origin']) >= datetime.fromisoformat(case['origin']) for r in rows):
        raise ValueError('Historical support cannot use the current or a future origin')
    if scope == 'context':
        filters = packet['context_retrieval']['selected_filters']
        if filters is None:
            return base
        rows = [r for r in rows if all(r['context'].get(k) == v for k, v in filters.items())]
    if len(rows) < minimum:
        return base
    baseline_index = providers.index(base)
    baseline_loss = mean(r['rmsle'][baseline_index] for r in rows)
    qualified = []
    for index, provider in enumerate(providers):
        if provider == base:
            continue
        loss = mean(r['rmsle'][index] for r in rows)
        wins = sum(r['rmsle'][index] < r['rmsle'][baseline_index]-1e-12 for r in rows)/len(rows)
        recent_delta = mean(r['rmsle'][index]-r['rmsle'][baseline_index] for r in rows[-4:])
        if loss < baseline_loss*(1-margin)-1e-12 and wins >= fraction and (not recent or recent_delta < -1e-12):
            qualified.append((loss, provider))
    return min(qualified)[1] if qualified else base


def support_packet(case, packet):
    """Frozen training-selected rule for a subsequent, separately logged trial."""
    provider = suggest(case, packet, 'all', 4, .5, False, 0)
    raw = packet['raw_history']
    providers, rows = raw['providers'], raw['records']
    baseline = min(providers, key=lambda p: case['current_card'][p]['cv_rmsle'])
    candidate_index, baseline_index = providers.index(provider), providers.index(baseline)
    deltas = [r['rmsle'][candidate_index]-r['rmsle'][baseline_index] for r in rows]
    changed = provider != baseline
    return {
        'kind': 'descriptive_historical_support_not_a_forecast',
        'provider': provider, 'current_cv_provider': baseline,
        'supports_changing_current_cv_choice': changed,
        'rule': 'All matched history; at least four origins; lower mean RMSLE and wins on at least half the origins. Among qualifying providers choose lowest mean history error; otherwise keep current CV.',
        'rule_selection': 'Selected on development origins 0..17; not confirmation validated',
        'matched_origins': len(rows), 'context_specific': False,
        'mean_historical_rmsle': mean(r['rmsle'][candidate_index] for r in rows) if rows else None,
        'baseline_mean_historical_rmsle': mean(r['rmsle'][baseline_index] for r in rows) if rows else None,
        'paired_wins': sum(d < -1e-12 for d in deltas), 'paired_losses': sum(d > 1e-12 for d in deltas),
        'paired_ties': sum(abs(d) <= 1e-12 for d in deltas), 'tie_absolute_tolerance': 1e-12,
        'candidate_current_cv_rmsle': case['current_card'][provider]['cv_rmsle'],
        'baseline_current_cv_rmsle': case['current_card'][baseline]['cv_rmsle'],
        'source_as_of': raw['source_as_of'], 'recorded_as_of': raw['recorded_as_of'],
        'provider_calls': 0, 'forecast_executed': False,
        'guidance': 'Use this calculated support with the shared raw evidence and current CV. It is not a guarantee or a mandate; execute a provider before selecting its execution ID.'}


def run(cases, packets):
    variants = list(product(('context', 'all'), (4, 8, 12), (.5, .75, 1), (False, True), (0, .05, .1)))
    names = {v: '_'.join(map(str, v)) for v in variants}
    rows = []
    for case in cases:
        packet = packets[case['series_id'], case['round']]
        base = min(case['current_card'], key=lambda p: case['current_card'][p]['cv_rmsle'])
        choices = {'current_cv': base} | {names[v]: suggest(case, packet, *v) for v in variants}
        rows.append({'series_id': case['series_id'], 'round': case['round'], 'choices': choices})
    lookup = {(c['series_id'], c['round']): c for c in cases}
    scores = {label: {name: mean(lookup[r['series_id'], r['round']]['scores'][r['choices'][name]]
                               for r in rows if predicate(r)) for name in rows[0]['choices']}
              for label, predicate in [('train', lambda r: r['round'] < 18),
                                       ('validation', lambda r: r['round'] >= 18), ('all', lambda r: True)]}
    selected = min(names.values(), key=lambda name: scores['train'][name])
    return {'scope': 'development automatic support screen; not an agent or held-out result',
            'variants': len(variants), 'selected_on_training_only': selected, 'scores': scores,
            'selected_validation_relative_gain': 1-scores['validation'][selected]/scores['validation']['current_cv'],
            'selected_changes': sum(r['choices'][selected] != r['choices']['current_cv'] for r in rows),
            'selection_rule': 'Choose lowest historical mean among candidates satisfying the explicit support rule; otherwise current CV',
            'changes_forecasts': False, 'provider_calls': 0, 'confirmation_opened': False,
            'target_established': False, 'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output path')
    packets = {(p['series_id'], p['round']): p for p in json.loads(args.memory.read_text())['packets']}
    report = run(json.loads(args.cases.read_text()), packets)
    report['inputs'] = {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in [('cases', args.cases), ('memory', args.memory)]}
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'scores')}, indent=2))


if __name__ == '__main__':
    main()
