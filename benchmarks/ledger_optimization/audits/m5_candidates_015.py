"""Independent post-run verification of M5 requests, scores and visible-history choices."""
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
from math import cos, isclose, isfinite, log1p, pi, sin, sqrt
from pathlib import Path
from statistics import mean


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_request(req, source, index):
    history = source[max(0, index - 729):index + 1]
    times = [datetime.fromisoformat(r['ds']) for r in history]
    future = [times[-1] + timedelta(days=i) for i in range(1, 15)]
    assert req['history'] == [float(r['y']) for r in history]
    assert req['timestamps'] == [t.isoformat() for t in times]
    assert req['future_timestamps'] == [t.isoformat() for t in future]
    assert req['horizon'] == 14 and req['season'] == 7 and req['frequency'] == 'D'
    assert req['series_id'] == history[-1]['unique_id'] and req['unit'] == 'unit_sales'
    assert req['cutoff'] == req['known_time_cutoff'] == req['recorded_time_cutoff'] == times[-1].isoformat()
    for field, dates in [('past', times), ('future', future)]:
        expected = [[0.0, sin(2 * pi * (t - timedelta(days=1)).weekday() / 7),
                     cos(2 * pi * (t - timedelta(days=1)).weekday() / 7)] for t in dates]
        assert req[f'{field}_covariates'] == expected
        assert req[f'{field}_covariate_names'] == ['onpromotion', 'dow_sin', 'dow_cos']


def score(point, actual):
    assert len(point) == len(actual) == 14
    assert all(isfinite(x) and x >= 0 for x in [*point, *actual])
    return {'rmsle': sqrt(sum((log1p(p) - log1p(a)) ** 2 for p, a in zip(point, actual, strict=True)) / 14),
            'mae': sum(abs(p - a) for p, a in zip(point, actual, strict=True)) / 14}


def same(left, right):
    assert isclose(left, right, abs_tol=1e-12, rel_tol=1e-12)


def verify(root, manifest_path):
    if not __debug__:
        raise RuntimeError('Audit must run without optimized-away assertions')
    freeze = json.loads((root / 'freeze.json').read_text())
    completion = json.loads((root / 'completion.json').read_text())
    result = json.loads((root / 'screen.json').read_text())
    cases = json.loads((root / 'cases.json').read_text())
    manifest = json.loads(manifest_path.read_text())
    assert completion['status'] == 'complete' and not completion['source_changes'] and not completion['failed_series']
    assert digest(manifest_path) == freeze['panel_manifest_sha256']
    assert digest(Path(manifest['development']['path'])) == freeze['development_sha256']
    assert digest(root / 'cases.json') == result['inputs']['cases_sha256']
    assert digest(root / 'freeze.json') == result['inputs']['freeze_sha256']
    assert digest(root / 'completion.json') == result['inputs']['completion_sha256']
    assert digest(Path(freeze['recipe_path'])) == freeze['recipe_sha256']
    assert digest(Path(freeze['runtime']['request_module'])) == freeze['runtime']['request_module_sha256']
    code_root = Path(__file__).resolve().parents[1]
    for name, expected in freeze['source'].items():
        assert digest(code_root / name) == expected
    groups = {}
    with Path(manifest['development']['path']).open(newline='') as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(row['unique_id'], []).append(row)
    expected_ids = {r['series_id'] for r in manifest['splits']['development']}
    assert set(groups) == expected_ids
    assert not expected_ids & {r['series_id'] for r in manifest['splits']['reserved']}
    assert {(c['series_id'], c['round']) for c in cases} == {(sid, r) for sid in expected_ids for r in range(26)}
    assert len(cases) == 208
    screened = {(r['series_id'], r['round']): r for r in result['cases']}
    checked_forecasts = 0
    distinct_requests = {}
    for c in cases:
        data = sorted(groups[c['series_id']], key=lambda r: r['ds'])
        index = 365 + c['round'] * 14
        check_request(c['request'], data, index)
        actual = [float(r['y']) for r in data[index + 1:index + 15]]
        assert c['actual'] == actual
        assert c['origin'] == c['request']['cutoff']
        assert c['future_timestamps'] == c['request']['future_timestamps']
        assert c['outcome_recorded_at'] == c['future_timestamps'][-1]
        providers = sorted(c['predictions'])
        assert len(providers) == 8
        for p in providers:
            metrics = score(c['predictions'][p], actual)
            same(metrics['rmsle'], c['scores'][p])
            same(metrics['mae'], c['mae'][p])
            checked_forecasts += 1
            all_results = [(c['request'], c['predictions'][p], c['candidate_metadata'][p])]
            folds = c['cv_evidence'][p]
            assert len(folds) == 2
            cv_scores = []
            for offset, fold in zip((28, 14), folds, strict=True):
                check_request(fold['request'], data, index - offset)
                truth = [float(r['y']) for r in data[index - offset + 1:index - offset + 15]]
                assert fold['actual'] == truth
                assert datetime.fromisoformat(fold['request']['future_timestamps'][-1]) <= datetime.fromisoformat(c['origin'])
                computed = score(fold['point'], truth)
                for metric in ('rmsle', 'mae'):
                    same(computed[metric], fold['metrics'][metric])
                cv_scores.append(computed)
                checked_forecasts += 1
                all_results.append((fold['request'], fold['point'], fold['metadata']))
            same(mean(f['rmsle'] for f in cv_scores), c['current_card'][p]['cv_rmsle'])
            same(mean(f['mae'] for f in cv_scores), c['current_card'][p]['cv_mae'])
            for req, point, metadata in all_results:
                assert metadata['recipe'] == p
                assert type(metadata['fallback_used']) is bool
                if metadata['fallback_used']:
                    assert metadata['fallback_provider'] == 'sf_seasonal_naive_7'
                    assert metadata['model_error_type']
                key = c['series_id'], p, req['cutoff']
                stable = (req, point, {k: v for k, v in metadata.items() if k != 'elapsed_seconds'})
                if key in distinct_requests:
                    assert distinct_requests[key] == stable
                distinct_requests[key] = stable
        # Independently reconstruct the incumbent from eligible prior observations.
        now = datetime.fromisoformat(c['origin'])
        history = sorted((h for h in cases if h['series_id'] == c['series_id']
                          and datetime.fromisoformat(h['origin']) < now
                          and datetime.fromisoformat(h['outcome_recorded_at']) <= now
                          and max(map(datetime.fromisoformat, h['future_timestamps'])) <= now),
                         key=lambda h: h['origin'])
        row = screened[c['series_id'], c['round']]
        assert row['visible_origins'] == [h['origin'] for h in history]
        cv = min(providers, key=lambda p: c['current_card'][p]['cv_rmsle'])
        eligible = []
        if len(history) >= 4:
            base = mean(h['scores'][cv] for h in history)
            for p in providers:
                average = mean(h['scores'][p] for h in history)
                wins = sum(h['scores'][p] < h['scores'][cv] - 1e-12 for h in history)
                if average < base - 1e-12 and wins / len(history) >= .5:
                    eligible.append((average, p))
        support = min(eligible)[1] if eligible else cv
        mae = min(providers, key=lambda p: (mean(h['mae'][p] for h in history), p)) if len(history) >= 4 else cv
        assert row['choices'] == {'current_cv': cv, 'frozen_support': support, 'lifetime_mae_diagnostic': mae}
        for name, p in row['choices'].items():
            same(row['rmsle'][name], c['scores'][p])
    assert len(distinct_requests) == 8 * 28 * 8
    assert checked_forecasts == 208 * 8 * 3
    for counts in completion['recipe_cache_by_series'].values():
        assert counts['hits'] == 400 and counts['misses'] == 224
    for name, predicate in [('all', lambda c: True), ('cold', lambda c: c['round'] < 4),
                            ('mature', lambda c: c['round'] >= 4), ('training', lambda c: c['round'] < 18),
                            ('later_development', lambda c: c['round'] >= 18)]:
        rows = [r for r in result['cases'] if predicate(r)]
        summary = result['summaries'][name]
        assert len(rows) == summary['case_count']
        for policy in rows[0]['choices']:
            same(mean(r['rmsle'][policy] for r in rows), summary['mean_case_rmsle'][policy])
        same(mean(r['future_aware_oracle_rmsle'] for r in rows), summary['future_aware_oracle_rmsle'])
    return {'status': 'passed', 'cases_checked': len(cases), 'forecasts_and_scores_checked': checked_forecasts,
            'distinct_forecasting_requests_checked': len(distinct_requests), 'source_hashes_unchanged': True,
            'reserved_data_read': False, 'target_established': False,
            'cases_sha256': digest(root / 'cases.json'), 'screen_sha256': digest(root / 'screen.json'),
            'audit_script_sha256': digest(Path(__file__)),
            'limitation': 'Development numerical and visibility audit, not an agent or confirmation result.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--panel-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite an audit')
    result = verify(args.run, args.panel_manifest)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
