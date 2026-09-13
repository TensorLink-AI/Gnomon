"""Independent scalar convex lower-bound and feasible-forecast verification."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ORDER = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')


def audit(source, comparison, directory):
    source, comparison, directory = map(Path, (source, comparison, directory))
    destination = directory/'verification.json'
    if destination.exists():raise FileExistsError(destination)
    here = Path(__file__).parent; checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:raise AssertionError(message)
    def near(a, b, tol=1e-9):check(math.isclose(a, b, rel_tol=tol, abs_tol=tol), 'numeric mismatch')
    def average(xs):return math.fsum(xs)/len(xs)
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, sha in manifest['code_sha256'].items():
        check(hashlib.sha256((here/name).read_bytes()).hexdigest() == sha, 'frozen code changed')
    for name, sha in manifest['receipts_sha256'].items():
        check(hashlib.sha256((here/'evidence'/name).read_bytes()).hexdigest() == sha, 'receipt changed')
    r38 = json.loads((here/'evidence/broad-screen-038.json').read_text())
    r45 = json.loads((here/'evidence/broad-ensemble-045.json').read_text())
    check(hashlib.sha256((comparison/'report.json').read_bytes()).hexdigest() == r45['files']['report.json'], 'control changed')
    controls = {(r['series_id'], r['round']): r['scores']['cv_ensemble'] for r in json.loads((comparison/'report.json').read_text())['rows']}
    report = json.loads((directory/'report.json').read_text()); seen = set(); iterations = 0
    for row in report['rows']:
        key = row['series_id'], row['round']; check(key not in seen, 'duplicate case'); seen.add(key)
        name = f'{key[0]}-{key[1]:02d}.json'; raw = (source/name).read_bytes()
        check(hashlib.sha256(raw).hexdigest() == r38['files'][name], 'source changed')
        original = json.loads(raw); fit = row['fit']; w = fit['weights']
        check(len(w) == 6 and all(math.isfinite(a) and a >= 0 for a in w), 'simplex')
        near(sum(w), 1., 1e-10); near(row['control_rmsle'], controls[key])
        pair = {'point': original['point'], 'actual': original['actual']}
        check(fit['input_sha256'] == hashlib.sha256(json.dumps({'pairs': [pair], 'masses': [1.]}, separators=(',', ':')).encode()).hexdigest(), 'fit pair hash')
        logs = [[math.log1p(original['point'][m][i]) for m in ORDER] for i in range(24)]
        y = [math.log1p(v) for v in original['actual']]
        prediction = [math.fsum(w[j]*logs[i][j] for j in range(6)) for i in range(24)]
        error = [p-a for p, a in zip(prediction, y, strict=True)]
        norm = math.sqrt(average([e*e for e in error])); penalty = 1e-6*math.fsum((a-1/6)**2 for a in w)
        near(fit['objective'], norm+penalty)
        initial = math.sqrt(average([(average(logs[i])-y[i])**2 for i in range(24)]))
        near(fit['initial_objective'], initial)
        gradient = [2e-6*(a-1/6) for a in w]; omitted = norm if norm <= 1e-8 else 0.
        if norm > 1e-8:
            for j in range(6):gradient[j] += math.fsum(logs[i][j]*error[i] for i in range(24))/(24*norm)
        gap = math.fsum(a*b for a, b in zip(w, gradient, strict=True))-min(gradient)+omitted
        near(fit['convex_gap_bound'], gap, 1e-8); near(fit['near_zero_loss_bound'], omitted)
        check(gap <= 1e-5+1e-10 and fit['success'], 'numerical optimality certificate')
        near(row['lower_bound'], max(0., norm+penalty-gap-(5/6)*1e-6), 1e-8)
        near(row['upper_bound'], norm)
        check(row['lower_bound'] <= row['upper_bound']+1e-10, 'ordered interval')
        for i, p in enumerate(row['point']):near(p, math.expm1(prediction[i]))
        iterations += fit['iterations']
    check(len(seen) == 416 and report['weight_fits'] == 416, 'complete cohort')
    check(iterations == report['iterations'], 'iteration accounting')
    def summary(observed, rows):
        control = average([r['control_rmsle'] for r in rows])
        low = average([r['lower_bound'] for r in rows]); high = average([r['upper_bound'] for r in rows])
        near(observed['control_rmsle'], control)
        for a, b in zip(observed['hindsight_minimum_mean_rmsle_bounds'], [low, high], strict=True):near(a, b)
        for a, b in zip(observed['hindsight_improvement_bounds'], [1-high/control, 1-low/control], strict=True):near(a, b)
        near(observed['target_rmsle'], .8*control)
        check(observed['twenty_percent_ruled_out'] == (low > .8*control), 'feasibility interpretation')
    summary(report['overall'], report['rows'])
    for domain in ('electricity', 'pedestrian'):
        summary(report['domains'][domain], [r for r in report['rows'] if r['series_id'].startswith(domain+':')])
    result = {'checks': checks, 'failures': 0, 'cases': 416,
        'report_sha256': hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Independent scalar objectives, feasible forecasts, convex gap bounds, maximum-regularizer correction, source hashes and aggregates. Floating-point tolerances apply; hindsight diagnostic only.'}
    destination.write_text(json.dumps(result, indent=2)+'\n'); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source'); p.add_argument('comparison'); p.add_argument('directory'); a = p.parse_args()
    print(json.dumps(audit(a.source, a.comparison, a.directory), indent=2))
