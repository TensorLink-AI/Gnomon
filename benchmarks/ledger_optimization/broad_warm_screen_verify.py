"""Independent warm-up pairs, context features and regularized-fit audit."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ORDER = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')


def audit(warmup, panel, original, result):
    warmup, panel, original, result = map(Path, (warmup, panel, original, result))
    destination = result/'verification.json'
    if destination.exists():raise FileExistsError(destination)
    here = Path(__file__).parent; checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:raise AssertionError(message)
    def near(a, b, tol=1e-10):
        check(math.isclose(a, b, rel_tol=tol, abs_tol=tol), 'numeric mismatch')
    def avg(values):return math.fsum(values)/len(values)
    def metric(point, actual):
        check(len(point) == len(actual) == 24, 'horizon')
        check(all(math.isfinite(p) and p >= 0 for p in point), 'finite predictions')
        return math.sqrt(avg([(math.log1p(p)-math.log1p(a))**2 for p, a in zip(point, actual, strict=True)]))
    manifest = json.loads((result/'manifest.json').read_text())
    for name, sha in manifest['code_sha256'].items():
        check(hashlib.sha256((here/name).read_bytes()).hexdigest() == sha, 'code changed')
    for name, sha in manifest['source_receipts_sha256'].items():
        check(hashlib.sha256((here/'evidence'/name).read_bytes()).hexdigest() == sha, 'receipt changed')
    receipt38 = json.loads((here/'evidence/broad-screen-038.json').read_text())
    receipt42 = json.loads((here/'evidence/broad-warmup-042.json').read_text())
    receipt37 = json.loads((here/'evidence/broad-panel-037.json').read_text())
    for root, names, receipt in ((warmup, ('warmup-spans.json', 'boundaries.json'), receipt42),
                                  (panel, ('development-spans.json',), receipt37)):
        for name in names:check(hashlib.sha256((root/name).read_bytes()).hexdigest() == receipt['files'][name], 'spans changed')
    spans = json.loads((warmup/'warmup-spans.json').read_text())
    development = json.loads((panel/'development-spans.json').read_text())
    boundaries = json.loads((warmup/'boundaries.json').read_text())
    check(json.loads((result/'excluded_warmup.json').read_text()) == [t for t in boundaries if not t['ready']], 'all exclusions preserved')
    originals = {}; warm_rows = {}
    for name, sha in receipt38['files'].items():
        if not name.startswith(('electricity:', 'pedestrian:')):continue
        raw = (original/name).read_bytes(); check(hashlib.sha256(raw).hexdigest() == sha, 'scored predictions changed')
        row = json.loads(raw); originals[(row['series_id'], row['round'])] = row
    for task in boundaries:
        if not task['ready']:continue
        identity, index = task['series_id'], task['round']+8
        row = json.loads((result/f'warmup-{identity}-{index:02d}.json').read_text())
        warm_rows[(identity, task['round'])] = row
        values = spans[identity]['values']; low, stop = task['history_indices']
        check(row['history_indices'] == [low, stop] and row['target_indices'] == task['target_indices'], 'warmup slices')
        check(row['origin'].replace('+00:00', '') == task['origin'], 'warmup origin')
        check(row['last_target'].replace('+00:00', '') == task['last_target'], 'warmup closure')
        check(row['actual'] == values[stop:stop+24], 'warmup targets')
        for model in ORDER:
            scores = []
            for fold, end in zip(row['folds'][model], (658, 682, 706), strict=True):
                check(fold['end'] == end and fold['actual'] == values[low+end:low+end+24], 'fold target slice')
                score = metric(fold['point'], fold['actual']); near(score, fold['rmsle']); scores.append(score)
                if model in ORDER[:3]:
                    for i, p in enumerate(fold['point']):
                        target = avg([values[low+end-168*k+i] for k in (1, 2, 3)]) if model == 'weekly_mean' else values[low+end-({'daily':24, 'weekly':168}[model])+i]
                        near(p, target)
            near(row['cv'][model], avg(scores)); near(row['scores'][model], metric(row['point'][model], row['actual']))
            if model in ORDER[:3]:
                for i, p in enumerate(row['point'][model]):
                    expected = avg([values[stop-168*k+i] for k in (1, 2, 3)]) if model == 'weekly_mean' else values[stop-({'daily':24, 'weekly':168}[model])+i]
                    near(p, expected)
    episodes = json.loads((result/'episodes.json').read_text())
    check(len(episodes) == 541 and len(warm_rows) == 125 and len(originals) == 416, 'all episodes')
    indexed = {}
    for e in episodes:
        key = e['series_id'], e['round']; check(key not in indexed, 'unique episode'); indexed[key] = e
        raw = warm_rows[key] if e['round'] < 0 else originals[key]
        for k in ('cv', 'scores', 'origin', 'last_target'):check(e[k] == raw[k], 'episode evidence linkage')
        check(e['outcome_recorded_at'] == raw['last_target'], 'disclosed recording assumption')
        check(e['domain'] == e['series_id'].split(':')[0], 'domain label')
        values = (spans if e['round'] < 0 else development)[e['series_id']]['values']
        low, high = raw['history_indices']; history = values[low:high]; z = [math.log1p(v) for v in history]
        expected = [math.log1p(raw['cv'][m]) for m in ORDER]+[avg(z), statistics.pstdev(z), history.count(0.)/730,
            avg(z[-168:])-avg(z[-336:-168]), avg([abs(z[i]-z[i-24]) for i in range(24, 730)]),
            avg([abs(z[i]-z[i-168]) for i in range(168, 730)])]
        check(len(e['features']) == 12, 'feature count')
        for a, b in zip(e['features'], expected, strict=True):near(a, b)
    report = json.loads((result/'report.json').read_text()); seen = set(); fit_count = 0
    for row in report['rows']:
        key = row['series_id'], row['round']; check(key not in seen, 'unique decision'); seen.add(key)
        current = indexed[key]; cv = current['cv']; now = datetime.fromisoformat(current['origin'])
        valid = [e for e in episodes if e['domain'] == current['domain'] and datetime.fromisoformat(e['origin']) < now
                 and datetime.fromisoformat(e['last_target']) <= now and datetime.fromisoformat(e['outcome_recorded_at']) <= now]
        last = sorted({e['origin'] for e in valid}, key=datetime.fromisoformat)[-8:]
        valid = sorted([e for e in valid if e['origin'] in last], key=lambda e: (e['origin'], e['series_id']))
        context = row['context']
        check(context['retrieved'] == [{'series_id': e['series_id'], 'origin': e['origin']} for e in valid], 'exact visible context cohort')
        check(context['records'] == len(valid) and context['distinct_origins'] == len(last), 'context counts')
        check(context['current_features'] == current['features'], 'current observed features')
        check(context['effective_source_as_of'] == context['effective_recorded_as_of'] == current['origin'], 'effective cutoffs')
        x = np.array([e['features'] for e in valid]); y = np.array([[e['scores'][m]-e['cv'][m] for m in ORDER] for e in valid])
        estimates = dict(cv)
        if len(valid) >= 16 and len(last) >= 3:
            fit_count += 1; fit = context['fit']
            location = x.mean(axis=0); scale = np.maximum(x.std(axis=0), .1); target_mean = y.mean(axis=0)
            standardized = (x-location)/scale
            # Augmented least squares verifies the original normal-equation solve.
            augmented_x = np.vstack([standardized, math.sqrt(10)*np.eye(12)])
            augmented_y = np.vstack([y-target_mean, np.zeros((12, 6))])
            coefficients = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
            for name, expected in (('location', location), ('scale', scale), ('target_mean', target_mean), ('coefficients', coefficients)):
                check(np.allclose(np.asarray(fit[name]), expected, rtol=1e-9, atol=1e-10), 'independent fit '+name)
            training_hash = hashlib.sha256(json.dumps({'x': x.tolist(), 'y': y.tolist()}, separators=(',', ':')).encode()).hexdigest()
            check(fit['training_sha256'] == training_hash, 'training matrix hash')
            residual = target_mean+((np.asarray(current['features'])-location)/scale)@coefficients
            for i, m in enumerate(ORDER):
                near(fit['predicted_residual'][i], residual[i], 1e-9)
                estimates[m] = max(0., cv[m]+.5*residual[i])
        else:check(context['fit'] is None, 'cold-start fallback')
        chosen = min(ORDER, key=lambda m: (estimates[m], cv[m], ORDER.index(m)))
        control = min(ORDER, key=lambda m: (cv[m], ORDER.index(m)))
        check(context['provider'] == chosen and context['control_provider'] == control, 'independent choice')
        for m in ORDER:near(context['estimated_rmsle'][m], estimates[m], 1e-9)
        own = sorted([e for e in episodes if e['series_id'] == current['series_id']
                      and datetime.fromisoformat(e['origin']) < now
                      and datetime.fromisoformat(e['last_target']) <= now
                      and datetime.fromisoformat(e['outcome_recorded_at']) <= now], key=lambda e: e['origin'])
        recent = own[-4:]; calibration = own[-8:]
        check(row['recent']['history_origins'] == [e['origin'] for e in recent], 'secondary recent cohort')
        past = blended = control
        if len(recent) >= 3:
            means = {m: avg([e['scores'][m] for e in recent]) for m in ORDER}
            past = min(ORDER, key=lambda m: (means[m], cv[m], ORDER.index(m)))
            blended = min(ORDER, key=lambda m: ((means[m]+cv[m])/2, cv[m], ORDER.index(m)))
        check(row['recent']['past'] == past and row['recent']['blended'] == blended, 'secondary recent selection')
        check(row['calibrated']['retrieved_origins'] == [e['origin'] for e in calibration], 'secondary calibration cohort')
        corrected = dict(cv)
        if len(calibration) >= 4:
            weight = len(calibration)/(len(calibration)+4)
            corrected = {m: max(0., cv[m]+weight*avg([e['scores'][m]-e['cv'][m] for e in calibration])) for m in ORDER}
        calibrated_choice = min(ORDER, key=lambda m: (corrected[m], cv[m], ORDER.index(m)))
        check(row['calibrated']['provider'] == calibrated_choice, 'secondary calibration selection')
        for policy, selected in (('cv', control), ('context', chosen), ('past', row['recent']['past']),
                                  ('blended', row['recent']['blended']), ('calibrated', row['calibrated']['provider'])):
            near(row['scores'][policy], current['scores'][selected])
    check(len(seen) == 416 and report['contextual_estimator_fits'] == fit_count, 'all decisions and contextual fits')
    def summary(observed, rows):
        check(observed['cases'] == len(rows), 'aggregate count')
        means = {p: avg([r['scores'][p] for r in rows]) for p in ('cv', 'context', 'past', 'blended', 'calibrated')}
        for p, value in means.items():
            near(observed['mean_rmsle'][p], value); near(observed['relative_reduction'][p], 1-value/means['cv'])
        check(observed['context_overrides'] == sum(r['context']['provider'] != r['context']['control_provider'] for r in rows), 'override count')
    summary(report['overall'], report['rows'])
    for d in ('electricity', 'pedestrian'):summary(report['domains'][d], [r for r in report['rows'] if r['series_id'].startswith(d+':')])
    for name, test in (('mature_round_ge_10', lambda r: r['round'] >= 10),
                        ('early_development_round_0_17', lambda r: r['round'] < 18),
                        ('later_development_round_18_25', lambda r: r['round'] >= 18)):
        summary(report[name], [r for r in report['rows'] if test(r)])
    check(report['development_gate_passed'] == (report['overall']['relative_reduction']['context'] >= .2 and all(v['relative_reduction']['context'] > 0 for v in report['domains'].values())), 'frozen gate')
    check(report['additional_forecast_computations'] == 3000 and report['additional_estimator_fits'] == 1500, 'warmup cost')
    check(report['total_common_forecast_computations'] == 12984 and report['api_calls'] == 0, 'total costs')
    result = {'checks': checks, 'failures': 0, 'warmup_cases': 125, 'scored_cases': 416,
        'report_sha256': hashlib.sha256((result/'report.json').read_bytes()).hexdigest(),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'All source hashes, warmup scored pairs and baselines, observed-only features, context cohorts, independent augmented least-squares fits, primary/secondary selections and aggregates. Warmup regression forecasts not independently refitted.'}
    destination.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('warmup', 'panel', 'original', 'result'):parser.add_argument(name)
    args = parser.parse_args()
    print(json.dumps(audit(args.warmup, args.panel, args.original, args.result), indent=2))
