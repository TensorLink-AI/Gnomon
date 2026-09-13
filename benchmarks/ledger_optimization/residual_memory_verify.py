"""Independent array audit of shared residual corrections and strong-control guard."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')


def audit(original, warm, control, contexts, directory):
    original, warm, control, contexts, directory = map(Path, (original, warm, control, contexts, directory))
    destination = directory/'verification.json'
    if destination.exists():raise FileExistsError(destination)
    here = Path(__file__).parent; checks = 0
    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:raise AssertionError(message)
    def near(a, b, tolerance=1e-9):
        check(bool(np.allclose(a, b, rtol=tolerance, atol=tolerance)), 'independent numerical mismatch')
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, sha in manifest['code_sha256'].items():
        check(hashlib.sha256((here/name).read_bytes()).hexdigest() == sha, 'code hash')
    for name, sha in manifest['source_receipts_sha256'].items():
        check(hashlib.sha256((here/'evidence'/name).read_bytes()).hexdigest() == sha, 'receipt hash')
    def load(root, receipt_name, prefixes):
        receipt = json.loads((here/'evidence'/receipt_name).read_text()); values = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefixes):continue
            raw = (root/name).read_bytes()
            check(hashlib.sha256(raw).hexdigest() == sha, 'frozen source artifact')
            values.append(json.loads(raw))
        return values
    scored = load(original, 'broad-screen-038.json', ('electricity:', 'pedestrian:'))
    earlier = load(warm, 'broad-warm-screen-043.json', ('warmup-electricity:', 'warmup-pedestrian:'))
    episodes = load(warm, 'broad-warm-screen-043.json', ('episodes.json',))[0]
    controls = load(control, 'broad-ensemble-045.json', ('electricity:', 'pedestrian:'))
    context_rows = load(contexts, 'broad-context-ensemble-047.json', ('electricity:', 'pedestrian:'))
    check(tuple(map(len, (scored, earlier, episodes, controls, context_rows))) == (416, 125, 541, 416, 416), 'complete cohorts')
    def index(rows):
        result = {(r['series_id'], r['origin']): r for r in rows}
        check(len(result) == len(rows), 'unique identity')
        return result
    history = index(scored+earlier); metadata = index(episodes); controls = index(controls); contexts = index(context_rows)
    report_rows = []; counts = {'cv_correction': 0, 'ledger_correction': 0}; fits = 0
    for source in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
        key = source['series_id'], source['origin']; origin = datetime.fromisoformat(key[1])
        name = f'{key[0]}-{source["round"]:02d}.json'
        row = json.loads((directory/name).read_text())
        check((row['series_id'], row['origin'], row['round']) == (*key, source['round']), 'unchanged task')
        check(row['source_execution_case'] == name and row['derived_forecasts'] is True, 'execution provenance')
        check(row['actual'] == source['actual'], 'original actuals')
        w = np.array(controls[key]['control_fit']['weights'])
        check(row['weights'] == w.tolist() and row['point']['uncorrected'] == controls[key]['point']['cv_ensemble'], 'same base for both corrections and fixed guard')
        refs = [{'series_id': r['series_id'], 'origin': r['origin']} for r in contexts[key]['retrieval']['selected']]
        check(row['neighbors'] == refs and len(refs) == 16, 'exact unchanged retrieval')
        past = []
        for ref in refs:
            identity = ref['series_id'], ref['origin']; meta = metadata[identity]
            at, close, recorded = (datetime.fromisoformat(meta[field]) for field in ('origin', 'last_target', 'outcome_recorded_at'))
            check(all(t.tzinfo is not None for t in (origin, at, close, recorded)), 'explicit clock')
            check(at < origin and close <= origin and recorded <= origin, 'visible evidence')
            check(meta['domain'] == key[0].split(':')[0] and at.timetz() == origin.timetz(), 'domain and lead phase')
            past.append(history[identity])
        cv = []
        for i in range(3):
            ends = [source['folds'][m][i]['end'] for m in MODELS]
            check(len(set(ends)) == 1 and (730-ends[0]) % 24 == 0, 'CV same lead phase')
            check(all(source['folds'][m][i]['actual'] == source['folds'][MODELS[0]][i]['actual'] for m in MODELS), 'matched CV actuals')
            cv.append({'point': {m: source['folds'][m][i]['point'] for m in MODELS}, 'actual': source['folds'][MODELS[0]][i]['actual']})
        pairs = {'cv_correction': cv, 'ledger_correction': cv+[{'point': r['point'], 'actual': r['actual']} for r in past]}
        masses = {'cv_correction': [1/3]*3, 'ledger_correction': [1/6]*3+[.5/16]*16}
        for arm in pairs:
            inputs = pairs[arm]; q = np.array(masses[arm]); fitted = row['fits'][arm]
            expected_hash = hashlib.sha256(json.dumps({'pairs': inputs, 'masses': masses[arm], 'weights': w.tolist()}, separators=(',', ':')).encode()).hexdigest()
            check(fitted['input_sha256'] == expected_hash, 'fit input hash')
            check(fitted['weights'] == w.tolist() and fitted['masses'] == masses[arm], 'shared primitive inputs')
            forecasts = np.log1p(np.array([[pair['point'][m] for m in MODELS] for pair in inputs])).transpose(0, 2, 1)@w
            residuals = np.log1p(np.array([pair['actual'] for pair in inputs]))-forecasts
            # Solve each lead's unit-penalty quadratic via its normal equation.
            correction = np.linalg.solve(2*np.eye(24), residuals.T@q)
            empirical = float(q@np.mean((residuals-correction)**2, axis=1))
            penalty = float(np.mean(correction**2)); gradient = (4*correction-2*(q@residuals))/24
            near(fitted['residuals'], residuals); near(fitted['correction'], correction)
            near(fitted['empirical_loss'], empirical); near(fitted['penalty'], penalty)
            near(fitted['objective'], empirical+penalty); near(fitted['max_absolute_gradient'], np.max(np.abs(gradient)))
            check(np.max(np.abs(gradient)) < 1e-10, 'stationary unique convex solution')
            # A perturbed solution must increase this strictly convex objective.
            perturbed = correction+.01
            check(float(q@np.mean((residuals-perturbed)**2, axis=1)+np.mean(perturbed**2)) > empirical+penalty, 'positive curvature')
            current_logs = np.log1p(np.array([source['point'][m] for m in MODELS]).T)@w+correction
            clipped = np.flatnonzero(current_logs < 0).tolist()
            check(row['clipped_leads'][arm] == clipped, 'exact clipping')
            near(row['point'][arm], np.expm1(np.maximum(current_logs, 0.)))
            counts[arm] += len(clipped); fits += 1
        for arm, point in row['point'].items():
            check(len(point) == 24 and all(np.isfinite(p) and p >= 0 for p in point), 'valid derived output')
            near(row['scores'][arm], np.sqrt(np.mean((np.log1p(point)-np.log1p(source['actual']))**2)))
        report_rows.append({k: row[k] for k in ('series_id', 'round', 'scores')})
    report = json.loads((directory/'report.json').read_text())
    check(report['rows'] == report_rows and len(report_rows) == 416, 'all reported tasks')
    def summary(observed, rows):
        check(observed['cases'] == len(rows), 'aggregate count')
        means = {arm: float(np.mean([r['scores'][arm] for r in rows])) for arm in ('uncorrected', 'cv_correction', 'ledger_correction')}
        for arm, value in means.items():near(observed['mean_rmsle'][arm], value)
        near(observed['primary_reduction'], 1-means['ledger_correction']/means['cv_correction'])
        near(observed['reduction_vs_uncorrected_guard'], 1-means['ledger_correction']/means['uncorrected'])
        near(observed['control_correction_reduction_vs_uncorrected'], 1-means['cv_correction']/means['uncorrected'])
    summary(report['overall'], report_rows)
    for d in ('electricity', 'pedestrian'):summary(report['domains'][d], [r for r in report_rows if r['series_id'].startswith(d+':')])
    check(report['vector_fits'] == fits == 832 and report['clipped_leads'] == counts, 'fit and clipping counts')
    check(report['api_calls'] == report['provider_fits'] == 0 and report['inherited_forecast_computations'] == 12984, 'cost provenance')
    expected_gate = report['overall']['primary_reduction'] >= .2 and report['overall']['reduction_vs_uncorrected_guard'] >= .2 and all(v['primary_reduction'] > 0 and v['reduction_vs_uncorrected_guard'] > 0 for v in report['domains'].values())
    check(report['development_gate_passed'] == expected_gate, 'matched comparator and strong guard')
    result = {'checks': checks, 'failures': 0, 'cases': 416, 'verified_vector_fits': fits,
        'report_sha256': hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Independent array residuals, normal-equation corrections, objectives, stationarity, clipping, forecasts and metrics; exact frozen sources, time/phase matching and both promotion comparators. No provider refits or reserved access.'}
    destination.write_text(json.dumps(result, indent=2)+'\n'); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'warm', 'control', 'contexts', 'directory'):p.add_argument(name)
    print(json.dumps(audit(**vars(p.parse_args())), indent=2))
