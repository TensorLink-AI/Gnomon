"""Independent scalar audit of 047 neighborhoods, weights and reported results."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

ORDER = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')


def audit(original, warm, control, directory):
    original, warm, control, directory = map(Path, (original, warm, control, directory))
    destination = directory/'verification.json'
    if destination.exists():raise FileExistsError(destination)
    here = Path(__file__).parent; checks = 0
    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:raise AssertionError(message)
    def near(a, b, tol=1e-9):
        check(math.isclose(a, b, rel_tol=tol, abs_tol=tol), f'{a} != {b}')
    def avg(values):return math.fsum(values)/len(values)
    def metric(point, actual):
        return math.sqrt(avg([(math.log1p(p)-math.log1p(y))**2 for p, y in zip(point, actual, strict=True)]))
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, sha in manifest['code_sha256'].items():
        check(hashlib.sha256((here/name).read_bytes()).hexdigest() == sha, 'code hash')
    for name, sha in manifest['source_receipts_sha256'].items():
        check(hashlib.sha256((here/'evidence'/name).read_bytes()).hexdigest() == sha, 'receipt hash')
    def load(root, receipt_name, prefixes):
        receipt = json.loads((here/'evidence'/receipt_name).read_text()); rows = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefixes):continue
            raw = (root/name).read_bytes()
            check(hashlib.sha256(raw).hexdigest() == sha, 'source hash')
            rows.append(json.loads(raw))
        return rows
    scored = load(original, 'broad-screen-038.json', ('electricity:', 'pedestrian:'))
    history = load(warm, 'broad-warm-screen-043.json', ('warmup-electricity:', 'warmup-pedestrian:'))+scored
    episodes = load(warm, 'broad-warm-screen-043.json', ('episodes.json',))[0]
    controls = load(control, 'broad-ensemble-045.json', ('electricity:', 'pedestrian:'))
    check((len(scored), len(history), len(episodes), len(controls)) == (416, 541, 541, 416), 'complete sources')
    raw_by_id = {(r['series_id'], r['origin']): r for r in history}
    meta = {(r['series_id'], r['origin']): r for r in episodes}
    old = {(r['series_id'], r['origin']): r for r in controls}
    check(len(raw_by_id) == len(meta) == 541 and len(old) == 416, 'source uniqueness')
    # Verify metadata joins back to archived executed cohorts. Feature derivation
    # was checked against original observed histories in the frozen 043 audit.
    for r in episodes:
        source = raw_by_id[(r['series_id'], r['origin'])]
        check(r['cv'] == source['cv'] and r['last_target'] == source['last_target'], 'episode execution join')
        check(r['domain'] == r['series_id'].split(':')[0], 'domain identity')
        check(r['outcome_recorded_at'] == source['last_target'], 'disclosed recording assumption')

    def fit_audit(fitted, pairs, masses):
        w = fitted['weights']
        check(len(w) == 6 and all(math.isfinite(v) and v >= 0 for v in w), 'simplex')
        near(math.fsum(w), 1., 1e-10)
        check(fitted['success'], 'successful optimization')
        expected_hash = hashlib.sha256(json.dumps({'pairs': pairs, 'masses': masses}, separators=(',', ':')).encode()).hexdigest()
        check(fitted['input_sha256'] == expected_hash, 'exact training pairs')
        gradient = [2e-6*(v-1/6) for v in w]
        loss = 1e-6*math.fsum((v-1/6)**2 for v in w); initial = 0.; omitted = 0.
        for pair, mass in zip(pairs, masses, strict=True):
            check(len(pair['actual']) == 24 and all(len(pair['point'][m]) == 24 for m in ORDER), 'training horizon')
            logs = [[math.log1p(pair['point'][m][i]) for m in ORDER] for i in range(24)]
            y = [math.log1p(v) for v in pair['actual']]
            error = [math.fsum(w[j]*logs[i][j] for j in range(6))-y[i] for i in range(24)]
            norm = math.sqrt(avg([e*e for e in error])); loss += mass*norm
            initial += mass*math.sqrt(avg([(avg(logs[i])-y[i])**2 for i in range(24)]))
            if norm <= 1e-8:omitted += mass*norm
            else:
                for j in range(6):gradient[j] += mass*math.fsum(logs[i][j]*error[i] for i in range(24))/(24*norm)
        gap = math.fsum(w[j]*gradient[j] for j in range(6))-min(gradient)+omitted
        near(fitted['objective'], loss); near(fitted['initial_objective'], initial)
        near(fitted['convex_gap_bound'], gap, 1e-8); near(fitted['near_zero_loss_bound'], omitted)
        check(gap <= 1e-5+1e-10 and loss <= initial+1e-8, 'frozen acceptance thresholds')

    rows = []; iterations = 0; fits = 0
    for source in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
        key = source['series_id'], source['origin']; current = meta[key]
        row = json.loads((directory/f'{source["series_id"]}-{source["round"]:02d}.json').read_text())
        check((row['series_id'], row['origin'], row['round']) == (*key, source['round']), 'identity')
        check(row['actual'] == source['actual'] and row['current_features'] == current['features'], 'unchanged task and observed context')
        now = datetime.fromisoformat(source['origin'])
        candidates = [r for r in episodes if r['domain'] == current['domain']
            and datetime.fromisoformat(r['origin']) < now
            and datetime.fromisoformat(r['last_target']) <= now
            and datetime.fromisoformat(r['outcome_recorded_at']) <= now]
        origins = sorted({datetime.fromisoformat(r['origin']) for r in candidates})[-8:]
        pool = sorted([r for r in candidates if datetime.fromisoformat(r['origin']) in origins], key=lambda r: (r['origin'], r['series_id']))
        diagnostic = row['retrieval']; ready = len(pool) >= 16 and len(origins) >= 3
        check(diagnostic['ready'] == ready and diagnostic['distinct_origins'] == len(origins), 'sufficiency')
        check(len(pool) == len(diagnostic['candidates']), 'candidate count')
        means = [avg([r['features'][j] for r in pool]) for j in range(12)]
        scales = [max(.1, math.sqrt(avg([(r['features'][j]-means[j])**2 for r in pool]))) for j in range(12)]
        for j in range(12):
            near(diagnostic['location'][j], means[j]); near(diagnostic['scale'][j], scales[j])
        distances = []
        for r, observed in zip(pool, diagnostic['candidates'], strict=True):
            distance = math.fsum(((r['features'][j]-current['features'][j])/scales[j])**2 for j in range(12))
            check((r['series_id'], r['origin']) == (observed['series_id'], observed['origin']), 'visible candidate')
            near(observed['distance'], distance)
            distances.append((distance, r['origin'], r['series_id']))
        expected = sorted(distances)[:16] if ready else []
        check([(r['origin'], r['series_id']) for r in diagnostic['selected']] == [(at, sid) for _, at, sid in expected], 'independently ranked neighbors')
        for observed, (distance, _, _) in zip(diagnostic['selected'], expected, strict=True):near(observed['distance'], distance)
        if ready:
            past = [raw_by_id[(sid, at)] for _, at, sid in expected]
            pairs = [{'point': {m: source['folds'][m][i]['point'] for m in ORDER},
                      'actual': source['folds'][ORDER[0]][i]['actual']} for i in range(3)]
            pairs += [{'point': r['point'], 'actual': r['actual']} for r in past]
            fit_audit(row['fit'], pairs, [1/6]*3+[.5/len(past)]*len(past))
            fits += 1; iterations += row['fit']['iterations']
        else:check(row['fit'] == old[key]['control_fit'], 'control fallback')
        check(row['point']['control'] == old[key]['point']['cv_ensemble'], 'byte-identical strong control')
        check(row['derived_forecasts'] and len(row['point']['ledger']) == 24, 'derived completion')
        for i, p in enumerate(row['point']['ledger']):
            near(p, math.expm1(math.fsum(row['fit']['weights'][j]*math.log1p(source['point'][m][i]) for j, m in enumerate(ORDER))))
        for arm in ('control', 'ledger'):near(row['scores'][arm], metric(row['point'][arm], source['actual']))
        rows.append({k: row[k] for k in ('series_id', 'round', 'scores')})
    report = json.loads((directory/'report.json').read_text())
    check(report['rows'] == rows and len(rows) == 416, 'full report')
    def summary(observed, subset):
        check(observed['cases'] == len(subset), 'aggregate count')
        scores = {arm: avg([r['scores'][arm] for r in subset]) for arm in ('control', 'ledger')}
        for arm, value in scores.items():near(observed['mean_rmsle'][arm], value)
        near(observed['primary_ledger_reduction'], 1-scores['ledger']/scores['control'])
    summary(report['overall'], rows)
    for domain in ('electricity', 'pedestrian'):
        summary(report['domains'][domain], [r for r in rows if r['series_id'].startswith(domain+':')])
    check(report['weight_fits'] == fits and report['optimizer_iterations'] == iterations, 'costs')
    check(report['api_calls'] == report['additional_provider_fits'] == 0, 'no new inference')
    check(report['inherited_forecast_computations'] == 12984, 'common inherited cost')
    check(report['development_gate_passed'] == (report['overall']['primary_ledger_reduction'] >= .2
        and all(v['primary_ledger_reduction'] > 0 for v in report['domains'].values())), 'gate')
    result = {'checks': checks, 'failures': 0, 'cases': len(rows), 'verified_weight_fits': fits,
              'report_sha256': hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
              'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Independent scalar time filtering, feature scaling/distances/ranking, pair hashes, objective/certificate, derived forecasts and aggregate scores. Frozen 043 features reused; original providers not refit. No reserved-data access.'}
    destination.write_text(json.dumps(result, indent=2)+'\n'); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'warm', 'control', 'directory'):p.add_argument(name)
    a = p.parse_args(); print(json.dumps(audit(a.original, a.warm, a.control, a.directory), indent=2))
