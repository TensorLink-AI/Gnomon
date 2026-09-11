"""Post-run independent arithmetic/visibility check; never mutates source evidence."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


def verify(report_path, reference_path):
    root = Path(__file__).resolve().parents[1]
    report_bytes = report_path.read_bytes()
    r = json.loads(report_bytes)
    raw = Path(r['inputs']['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == r['inputs']['sha256']
    cases = json.loads(raw)
    by = {(c['series_id'], c['round']): c for c in cases}
    assert len(by) == len(cases) == len(r['case_order'])
    reference = {(c['series_id'], c['round']): c['choices']['all_4_0.5_False_0']
                 for c in json.loads(reference_path.read_text())['cases']}
    ordered = [by[row['series_id'], row['round']] for row in r['case_order']]
    assert len({(c['series_id'], c['round']) for c in ordered}) == len(cases)
    trials = r['corruption_trials']
    assert len(trials) == r['trials'] == 256 and r['seed'] == 20260912
    checked = 0

    def dt(value):
        return datetime.fromisoformat(value.replace('Z', '+00:00'))

    for i, c in enumerate(ordered):
        row = r['case_order'][i]
        assert row['support_provider'] == reference[c['series_id'], c['round']]
        now = dt(c['origin'])
        history = sorted((old for old in cases if old['series_id'] == c['series_id']
                          and dt(old['origin']) < now and dt(old['outcome_recorded_at']) <= now
                          and max(dt(t) for t in old['future_timestamps']) <= now),
                         key=lambda old: dt(old['origin']))
        assert [x['origin'] for x in history] == row['visible_origins']
        cv = min(sorted(c['current_card']), key=lambda k: c['current_card'][k]['cv_rmsle'])
        assert cv == row['cv_provider']
        for provider, points in c['predictions'].items():
            score = math.sqrt(sum((math.log1p(max(0, pred)) - math.log1p(actual)) ** 2
                                  for pred, actual in zip(points, c['actual'], strict=True)) / len(points))
            assert math.isclose(score, row['scores'][provider], rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(score, c['scores'][provider], rel_tol=1e-12, abs_tol=1e-12)
        for trial in trials:
            mapping = trial['mappings'][c['series_id']]
            assert sorted(mapping) == sorted(c['predictions']) == sorted(mapping.values())
            candidates = []
            if len(history) >= 4:
                base = mean(h['scores'][mapping[cv]] for h in history)
                for provider in sorted(c['predictions']):
                    losses = [h['scores'][mapping[provider]] for h in history]
                    avg = mean(losses)
                    wins = sum(loss < h['scores'][mapping[cv]] - 1e-12
                               for loss, h in zip(losses, history, strict=True))
                    if avg < base - 1e-12 and wins / len(history) >= .5:
                        candidates.append((avg, provider))
            chosen = min(candidates)[1] if candidates else cv
            assert chosen == trial['choices_in_case_order'][i]
            checked += 1
    predicates = {'all': lambda c: True, 'cold': lambda c: c['round'] < 4,
                  'mature': lambda c: c['round'] >= 4, 'later_development': lambda c: c['round'] >= 18}
    for name, predicate in predicates.items():
        indices = [i for i, c in enumerate(ordered) if predicate(c)]
        samples = []
        for trial in trials:
            value = mean(ordered[i]['scores'][trial['choices_in_case_order'][i]] for i in indices)
            assert math.isclose(value, trial['mean_rmsle'][name], rel_tol=1e-12, abs_tol=1e-12)
            samples.append(value)
        summary = r['summaries'][name]
        assert math.isclose(median(samples), summary['corrupted_history_mean_rmsle']['median'], abs_tol=1e-12)
        oracle = mean(min(ordered[i]['scores'].values()) for i in indices)
        assert math.isclose(oracle, summary['future_aware_case_oracle_mean_rmsle'], abs_tol=1e-12)
        for series, provider in summary['future_aware_constant_providers'].items():
            group = [ordered[i] for i in indices if ordered[i]['series_id'] == series]
            expected = min(sorted(group[0]['scores']), key=lambda k: (mean(c['scores'][k] for c in group), k))
            assert expected == provider
    for filename, sha in r['source'].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == sha
    return {'status': 'passed', 'cases': len(cases), 'historical_mapping_choices_checked': checked,
            'trial_slice_aggregates_checked': len(trials) * 4,
            'incumbent_reference_choices_checked': len(cases), 'inputs_unchanged': True,
            'source_hashes_unchanged': True, 'retained_report_sha256': hashlib.sha256(report_bytes).hexdigest(),
            'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'target_established': False,
            'limitation': 'Independent diagnostic arithmetic and visibility, not forecasting superiority evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite audit evidence')
    result = verify(args.report, args.reference)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
