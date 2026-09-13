"""Independent derived forecasts, convex certificates and ensemble comparison."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

ORDER = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')


def audit(original, warm, directory):
    original, warm, directory = map(Path, (original, warm, directory))
    output = directory/'verification.json'
    if output.exists():raise FileExistsError(output)
    here = Path(__file__).parent; checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:raise AssertionError(message)
    def near(a, b, tolerance=1e-9):
        check(math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), 'numeric mismatch')
    def average(xs):return math.fsum(xs)/len(xs)
    def metric(p, y):return math.sqrt(average([(math.log1p(a)-math.log1p(b))**2 for a, b in zip(p, y, strict=True)]))
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, sha in manifest['code_sha256'].items():
        check(hashlib.sha256((here/name).read_bytes()).hexdigest() == sha, 'code changed')
    for name, sha in manifest['source_receipts_sha256'].items():
        check(hashlib.sha256((here/'evidence'/name).read_bytes()).hexdigest() == sha, 'receipt changed')
    refinement = json.loads((directory/'refinement.json').read_text())
    check(refinement['ftol'] == 1e-12 and refinement['convex_gap_threshold'] == 1e-5, 'frozen refinement')
    check(hashlib.sha256((here/'ensemble_refinement.py').read_bytes()).hexdigest() == refinement['code_sha256'], 'wrapper changed')
    check(hashlib.sha256((here/'BROAD_ENSEMBLE_045.md').read_bytes()).hexdigest() == refinement['protocol_sha256'], 'refinement protocol changed')
    def load(root, receipt_name, prefixes):
        receipt = json.loads((here/'evidence'/receipt_name).read_text()); rows = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefixes):continue
            raw = (root/name).read_bytes(); check(hashlib.sha256(raw).hexdigest() == sha, 'original forecast changed')
            rows.append(json.loads(raw))
        return rows
    scored = load(original, 'broad-screen-038.json', ('electricity:', 'pedestrian:'))
    history = load(warm, 'broad-warm-screen-043.json', ('warmup-electricity:', 'warmup-pedestrian:'))+scored
    report = json.loads((directory/'report.json').read_text()); checked_rows = []; iterations = 0

    def certificate(fit, pairs, masses):
        w = fit['weights']; check(len(w) == 6 and all(math.isfinite(v) and v >= 0 for v in w), 'feasible weights')
        near(math.fsum(w), 1., 1e-10)
        check(fit['success'], 'solver success')
        check(fit['input_sha256'] == hashlib.sha256(json.dumps({'pairs': pairs, 'masses': masses}, separators=(',', ':')).encode()).hexdigest(), 'training input hash')
        gradient = [2e-6*(a-1/6) for a in w]
        loss = 1e-6*math.fsum((a-1/6)**2 for a in w); initial = 0.; omitted = 0.
        for pair, mass in zip(pairs, masses, strict=True):
            logs = [[math.log1p(pair['point'][m][i]) for m in ORDER] for i in range(24)]
            actual = [math.log1p(v) for v in pair['actual']]
            error = [math.fsum(w[j]*logs[i][j] for j in range(6))-actual[i] for i in range(24)]
            norm = math.sqrt(average([e*e for e in error])); loss += mass*norm
            initial += mass*math.sqrt(average([(average(logs[i])-actual[i])**2 for i in range(24)]))
            if norm <= 1e-8:omitted += mass*norm
            else:
                for j in range(6):gradient[j] += mass*math.fsum(logs[i][j]*error[i] for i in range(24))/(24*norm)
        gap = math.fsum(a*b for a, b in zip(w, gradient, strict=True))-min(gradient)+omitted
        near(fit['objective'], loss); near(fit['initial_objective'], initial)
        near(fit['convex_gap_bound'], gap, 1e-8); near(fit['near_zero_loss_bound'], omitted)
        check(gap <= 1e-5+1e-10 and loss <= initial+1e-8, 'certified objective')

    for source in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
        row = json.loads((directory/f'{source["series_id"]}-{source["round"]:02d}.json').read_text())
        check(row['series_id'] == source['series_id'] and row['round'] == source['round'], 'identity')
        check(row['origin'] == source['origin'] and row['actual'] == source['actual'], 'task unchanged')
        check(row['derived_forecasts'] is True, 'derived forecast label')
        now = datetime.fromisoformat(source['origin'])
        eligible = sorted([r for r in history if r['series_id'] == source['series_id']
            and datetime.fromisoformat(r['origin']) < now and datetime.fromisoformat(r['last_target']) <= now], key=lambda r: datetime.fromisoformat(r['origin']))[-4:]
        check(row['retrieved'] == [{'origin': r['origin'], 'round': r['round']} for r in eligible], 'visible matched cohort')
        pairs = [{'point': {m: source['folds'][m][i]['point'] for m in ORDER},
                  'actual': source['folds'][ORDER[0]][i]['actual']} for i in range(3)]
        certificate(row['control_fit'], pairs, [1/3]*3)
        check(len(eligible) >= 3, 'warm-up supports planned ledger fit')
        certificate(row['ledger_fit'], pairs+[{'point': r['point'], 'actual': r['actual']} for r in eligible],
                    [1/6]*3+[.5/len(eligible)]*len(eligible))
        iterations += row['control_fit']['iterations']+row['ledger_fit']['iterations']
        for name, weights in (('cv_ensemble', row['control_fit']['weights']), ('ledger_ensemble', row['ledger_fit']['weights']), ('uniform', [1/6]*6)):
            check(len(row['point'][name]) == 24, 'derived horizon')
            for i, p in enumerate(row['point'][name]):
                expected = math.expm1(math.fsum(weights[j]*math.log1p(source['point'][m][i]) for j, m in enumerate(ORDER)))
                near(p, expected)
        check(row['point']['hard_cv'] == source['point'][source['selection']['cv']], 'unchanged hard CV')
        for name, p in row['point'].items():near(row['scores'][name], metric(p, source['actual']))
        checked_rows.append({k: row[k] for k in ('series_id', 'round', 'scores')})
    check(report['rows'] == checked_rows and len(checked_rows) == 416, 'all reported cases')
    def summary(observed, rows):
        check(observed['cases'] == len(rows), 'aggregate count')
        means = {k: average([r['scores'][k] for r in rows]) for k in rows[0]['scores']}
        for k, v in means.items():near(observed['mean_rmsle'][k], v)
        near(observed['primary_ledger_reduction'], 1-means['ledger_ensemble']/means['cv_ensemble'])
        near(observed['control_ensemble_reduction_vs_hard_cv'], 1-means['cv_ensemble']/means['hard_cv'])
        near(observed['ledger_reduction_vs_hard_cv'], 1-means['ledger_ensemble']/means['hard_cv'])
    summary(report['overall'], checked_rows)
    for d in ('electricity', 'pedestrian'):summary(report['domains'][d], [r for r in checked_rows if r['series_id'].startswith(d+':')])
    summary(report['mature_round_ge_10'], [r for r in checked_rows if r['round'] >= 10])
    summary(report['later_development_round_18_25'], [r for r in checked_rows if r['round'] >= 18])
    check(report['weight_fits'] == 832 and report['optimizer_iterations'] == iterations, 'fit costs')
    check(report['development_gate_passed'] == (report['overall']['primary_ledger_reduction'] >= .2 and all(v['primary_ledger_reduction'] > 0 for v in report['domains'].values())), 'primary gate')
    result = {'checks': checks, 'failures': 0, 'cases': 416, 'verified_weight_fits': 832,
        'report_sha256': hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Independent scalar arithmetic for fit inputs/objectives/convex bounds, temporal cohorts, derived forecasts, all scores and primary comparison. No refitting original providers or accessing reserved data.'}
    output.write_text(json.dumps(result, indent=2)+'\n'); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('original'); p.add_argument('warm'); p.add_argument('directory'); a = p.parse_args()
    print(json.dumps(audit(a.original, a.warm, a.directory), indent=2))
