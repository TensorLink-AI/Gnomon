"""Freeze legacy development evidence and evaluate strictly past-only selectors.

No API calls and no modifications to original ledgers. This is a policy screen,
not a matched agent experiment or a final evaluation.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import mean

from gnomon.evidence_summary import rmsle


ARMS = ('statsforecast', 'gnomon_control', 'gnomon_ledger')


def dumps(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def freeze(source, target, max_origins):
    target.mkdir(parents=True, exist_ok=False)
    manifest, cases = [], []
    for path in sorted((source / 'origins').glob('*.json'))[:max_origins]:
        raw = path.read_bytes()
        data = json.loads(raw)
        if data['status'] != 'complete':
            raise ValueError(f'Incomplete origin: {path}')
        manifest.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        updates = {u['series_id']: u for u in data['ledger_update']['updates']}
        for c in data['cases']:
            u = updates[c['series_id']]
            predictions = {p: v['point'] for p, v in u['predictions'].items()}
            actual = c['actual']
            if any(len(v) != len(actual) for v in predictions.values()):
                raise ValueError('Unequal horizons')
            scores = {p: rmsle(zip(v, actual), 'clip_zero')[0] for p, v in predictions.items()}
            for arm in ARMS:
                point = c[arm]['prediction']
                if point is None and c[arm]['used_fallback']:
                    point = predictions[c[arm]['scored_provider']]
                recalculated = rmsle(zip(point, actual), 'clip_zero')[0]
                if abs(recalculated - c[arm]['intent_to_treat_rmsle']) > 1e-10:
                    raise ValueError(f'Stored score mismatch: {path} {arm}')
            cases.append({k: c[k] for k in ('round', 'series_id', 'origin', 'future_timestamps', 'segment', 'actual', 'current_card')} | {
                'predictions': predictions, 'scores': scores,
                'mae': {p: mean(abs(y-a) for y, a in zip(v, actual)) for p, v in predictions.items()},
                'outcome_recorded_at': u['outcome_recorded_at'],
                'legacy_evidence': c['ledger_evidence'],
                'execution_ids': u['execution_ids'],
                'arms': {arm: {k: c[arm][k] for k in ('intent_to_treat_rmsle', 'used_fallback', 'scored_provider', 'candidate_conformant')}
                         for arm in ARMS}})
    (target/'cases.json').write_text(dumps(cases))
    manifest = {'scope': 'previously inspected development evidence only', 'origins': len(manifest),
                'cases': len(cases), 'source_files': manifest,
                'cases_sha256': hashlib.sha256((target/'cases.json').read_bytes()).hexdigest()}
    (target/'manifest.json').write_text(dumps(manifest))
    return cases, manifest


def select(current, prior, *, metric, window, cv_weight):
    """Prior is already availability-filtered; never inspect current outcomes."""
    providers = list(current)
    history = prior[-window:] if window else prior
    if not history:
        return min(providers, key=lambda p: current[p]['cv_rmsle'])
    field = 'scores' if metric == 'rmsle' else 'mae'
    cvfield = 'cv_rmsle' if metric == 'rmsle' else 'cv_mae'
    estimates = {p: (1-cv_weight)*mean(c[field][p] for c in history)
                   + cv_weight*current[p][cvfield] for p in providers}
    return min(providers, key=estimates.__getitem__)


def screen(cases):
    variants = [('cv_rmsle', None, None, None)] + [
        (f'{metric}_window{window or "all"}_cv{weight}', metric, window, weight)
        for metric in ('mae', 'rmsle') for window in (4, 12, 0) for weight in (0, 0.5, 0.75)]
    prior = defaultdict(list)
    results = []
    for c in sorted(cases, key=lambda c: (c['origin'], c['series_id'])):
        now = datetime.fromisoformat(c['origin'])
        visible = [p for p in prior[c['series_id']] if datetime.fromisoformat(p['outcome_recorded_at']) <= now
                   and max(datetime.fromisoformat(t) for t in p['future_timestamps']) <= now]
        choices = {}
        for name, metric, window, weight in variants:
            p = min(c['current_card'], key=lambda p: c['current_card'][p]['cv_rmsle']) if metric is None else select(
                c['current_card'], visible, metric=metric, window=window, cv_weight=weight)
            choices[name] = {'provider': p, 'rmsle': c['scores'][p]}
        results.append({'round': c['round'], 'series_id': c['series_id'], 'visible_origins': len(visible),
                        'choices': choices, 'hindsight_oracle_rmsle': min(c['scores'].values()),
                        'legacy_scores': {a: c['arms'][a]['intent_to_treat_rmsle'] for a in ARMS}})
        prior[c['series_id']].append(c)
    aggregates = {}
    for split, rows in [('all_development', results), ('development_train_0_59', [r for r in results if r['round'] < 60]),
                        ('development_validation_60_84', [r for r in results if r['round'] >= 60])]:
        if not rows:
            continue
        base = mean(r['legacy_scores']['gnomon_control'] for r in rows)
        oracle = mean(r['hindsight_oracle_rmsle'] for r in rows)
        aggregates[split] = {'cases': len(rows), 'legacy_scores': {a: mean(r['legacy_scores'][a] for r in rows) for a in ARMS},
            'hindsight_oracle_rmsle': oracle, 'oracle_max_relative_improvement': 1-oracle/base,
            'policies': {name: {'rmsle': mean(r['choices'][name]['rmsle'] for r in rows),
                              'relative_improvement_vs_legacy_control': 1-mean(r['choices'][name]['rmsle'] for r in rows)/base}
                         for name, *_ in variants}}
    return {'scope': 'offline development policy screen; no agent calls; not held-out proof',
            'api_calls': 0, 'metric': 'mean_per_case_rmsle', 'negative_predictions': 'clip_zero',
            'aggregates': aggregates, 'cases': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--max-origins', type=int, default=85)
    args = parser.parse_args()
    cases, manifest = freeze(args.source, args.snapshot, args.max_origins)
    report = screen(cases)
    report['input_manifest'] = manifest
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(dumps(report))
    overall = report['aggregates']['all_development']
    print(dumps({k: v for k, v in overall.items() if k != 'policies'}))
    print(dumps(sorted(overall['policies'].items(), key=lambda pair: pair[1]['rmsle'])[:5]))


if __name__ == '__main__':
    main()
