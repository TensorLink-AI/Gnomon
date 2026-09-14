"""Independent scalar verification of 050 block mixtures and convex bounds."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

MODELS = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')


def audit(original, warm, control, contexts, directory):
    original, warm, control, contexts, directory = map(Path, (original, warm, control, contexts, directory))
    output = directory/'verification.json'
    if output.exists():raise FileExistsError(output)
    here = Path(__file__).parent; checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:raise AssertionError(message)
    def near(a, b, tolerance=1e-9):
        check(math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), f'{a} != {b}')
    def avg(xs):return math.fsum(xs)/len(xs)
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, sha in manifest['code_sha256'].items():
        check(hashlib.sha256((here/name).read_bytes()).hexdigest() == sha, 'code changed')
    for name, sha in manifest['source_receipts_sha256'].items():
        check(hashlib.sha256((here/'evidence'/name).read_bytes()).hexdigest() == sha, 'receipt changed')
    check(manifest['regularizer'] == .01 and manifest['smoothing_epsilon'] == 1e-6 and manifest['ftol'] == 1e-14, 'frozen numerical parameters')
    def load(root, receipt_name, prefixes):
        receipt = json.loads((here/'evidence'/receipt_name).read_text()); rows = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefixes):continue
            raw = (root/name).read_bytes()
            check(hashlib.sha256(raw).hexdigest() == sha, 'source artifact changed')
            rows.append(json.loads(raw))
        return rows
    scored = load(original, 'broad-screen-038.json', ('electricity:', 'pedestrian:'))
    earlier = load(warm, 'broad-warm-screen-043.json', ('warmup-electricity:', 'warmup-pedestrian:'))
    episodes = load(warm, 'broad-warm-screen-043.json', ('episodes.json',))[0]
    controls = load(control, 'broad-ensemble-045.json', ('electricity:', 'pedestrian:'))
    contexts = load(contexts, 'broad-context-ensemble-047.json', ('electricity:', 'pedestrian:'))
    check(tuple(map(len, (scored, earlier, episodes, controls, contexts))) == (416, 125, 541, 416, 416), 'full source cohorts')
    def index(rows):
        values = {(r['series_id'], r['origin']): r for r in rows}
        check(len(values) == len(rows), 'identity uniqueness')
        return values
    history = index(scored+earlier); metadata = index(episodes); controls = index(controls); contexts = index(contexts)

    def objective(weights, pairs, masses, anchor):
        loss = .01/4*math.fsum((weights[b][j]-anchor[j])**2 for b in range(4) for j in range(6))
        gradient = [[.02/4*(weights[b][j]-anchor[j]) for j in range(6)] for b in range(4)]
        exact_loss = loss
        for pair, mass in zip(pairs, masses, strict=True):
            logs = [[math.log1p(pair['point'][m][h]) for m in MODELS] for h in range(24)]
            error = [math.fsum(weights[h//6][j]*logs[h][j] for j in range(6))-math.log1p(pair['actual'][h]) for h in range(24)]
            squared = avg([e*e for e in error]); norm = math.sqrt(squared+1e-12)
            loss += mass*norm; exact_loss += mass*math.sqrt(squared)
            for b in range(4):
                for j in range(6):gradient[b][j] += mass*math.fsum(logs[h][j]*error[h] for h in range(b*6, (b+1)*6))/(24*norm)
        gap = math.fsum(math.fsum(weights[b][j]*gradient[b][j] for j in range(6))-min(gradient[b]) for b in range(4))
        return loss, gradient, gap, exact_loss

    def certificate(observed, pairs, masses, anchor):
        w = observed['weights']
        check(len(w) == 4 and all(len(block) == 6 and all(math.isfinite(v) and v >= 0 for v in block) for block in w), 'feasible dimensions')
        for block in w:near(math.fsum(block), 1., 1e-10)
        check(observed['anchor'] == anchor and observed['success'] is True, 'anchor and solver status')
        expected = hashlib.sha256(json.dumps({'pairs': pairs, 'masses': masses, 'anchor': anchor}, separators=(',', ':')).encode()).hexdigest()
        check(observed['input_sha256'] == expected, 'exact paired training inputs')
        value, _, gap, exact = objective(w, pairs, masses, anchor)
        initial = objective([anchor]*4, pairs, masses, anchor)[0]
        near(observed['objective'], value); near(observed['initial_objective'], initial)
        near(observed['convex_gap_bound'], gap, 1e-8)
        check(gap <= 1e-5+1e-10 and value <= initial+1e-8, 'frozen convex acceptance')
        check(-1e-12 <= value-exact <= 1e-6+1e-12, 'smoothing perturbation bound')
        check(observed['smoothing_epsilon'] == observed['maximum_objective_smoothing_error'] == 1e-6 and observed['near_zero_loss_bound'] == 0., 'smoothing disclosure')
        # All real-data fits passed without the synthetic-case refinement. Keep
        # this explicit instead of claiming that unexercised path was audited.
        check(observed['certificate_refinements'] == [], 'no refinement on completed cohort')
        check(observed['refinement_start_weights'] == w, 'unchanged post-SLSQP weights')

    checked = []; iterations = 0
    for source in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
        key = source['series_id'], source['origin']; now = datetime.fromisoformat(key[1])
        name = f'{key[0]}-{source["round"]:02d}.json'; row = json.loads((directory/name).read_text())
        check((row['series_id'], row['origin'], row['round']) == (*key, source['round']), 'task identity')
        check(row['source_execution_case'] == name and row['derived_forecasts'] is True, 'derived provenance')
        check(row['actual'] == source['actual'], 'actuals unchanged')
        refs = [{'series_id': r['series_id'], 'origin': r['origin']} for r in contexts[key]['retrieval']['selected']]
        check(row['neighbors'] == refs and len(refs) == 16, 'frozen context retrieval')
        past = []
        for ref in refs:
            identity = ref['series_id'], ref['origin']; meta = metadata[identity]
            at, close, recorded = (datetime.fromisoformat(meta[k]) for k in ('origin', 'last_target', 'outcome_recorded_at'))
            check(all(t.tzinfo is not None for t in (now, at, close, recorded)), 'explicit clocks')
            check(at < now and close <= now and recorded <= now, 'visible historical outcomes')
            check(meta['domain'] == key[0].split(':')[0] and at.timetz() == now.timetz(), 'domain and hourly alignment')
            past.append(history[identity])
        pairs = []
        for i in range(3):
            ends = [source['folds'][m][i]['end'] for m in MODELS]
            check(len(set(ends)) == 1 and (730-ends[0]) % 24 == 0, 'current CV hourly phase')
            check(all(source['folds'][m][i]['actual'] == source['folds'][MODELS[0]][i]['actual'] for m in MODELS), 'matched CV targets')
            pairs.append({'point': {m: source['folds'][m][i]['point'] for m in MODELS}, 'actual': source['folds'][MODELS[0]][i]['actual']})
        anchor = controls[key]['control_fit']['weights']
        certificate(row['fits']['block_cv'], pairs, [1/3]*3, anchor)
        certificate(row['fits']['block_ledger'], pairs+[{'point': r['point'], 'actual': r['actual']} for r in past], [1/6]*3+[.5/16]*16, anchor)
        check(row['point']['global_cv'] == controls[key]['point']['cv_ensemble'], 'fixed strong comparator')
        for arm in ('block_cv', 'block_ledger'):
            w = row['fits'][arm]['weights']; iterations += row['fits'][arm]['iterations']
            check(len(row['point'][arm]) == 24, 'full derived horizon')
            for h, p in enumerate(row['point'][arm]):
                expected = math.expm1(math.fsum(w[h//6][j]*math.log1p(source['point'][m][h]) for j, m in enumerate(MODELS)))
                near(p, expected)
        for arm, point in row['point'].items():
            near(row['scores'][arm], math.sqrt(avg([(math.log1p(p)-math.log1p(y))**2 for p, y in zip(point, source['actual'], strict=True)])))
        checked.append({k: row[k] for k in ('series_id', 'round', 'scores')})
    report = json.loads((directory/'report.json').read_text())
    check(report['rows'] == checked and len(checked) == 416, 'complete report')
    def summary(observed, rows):
        check(observed['cases'] == len(rows), 'aggregate count')
        means = {arm: avg([r['scores'][arm] for r in rows]) for arm in ('global_cv', 'block_cv', 'block_ledger')}
        for arm, value in means.items():near(observed['mean_rmsle'][arm], value)
        near(observed['primary_reduction'], 1-means['block_ledger']/means['block_cv'])
        near(observed['reduction_vs_global_guard'], 1-means['block_ledger']/means['global_cv'])
        near(observed['control_reduction_vs_global'], 1-means['block_cv']/means['global_cv'])
    summary(report['overall'], checked)
    for domain in ('electricity', 'pedestrian'):summary(report['domains'][domain], [r for r in checked if r['series_id'].startswith(domain+':')])
    check(report['weight_fits'] == 832 and report['optimizer_iterations'] == iterations and report['certificate_refinements'] == 0, 'optimization cost')
    check(report['provider_fits'] == report['api_calls'] == 0 and report['inherited_forecast_computations'] == 12984, 'inference cost')
    gate = report['overall']['primary_reduction'] >= .2 and report['overall']['reduction_vs_global_guard'] >= .2 and all(v['primary_reduction'] > 0 and v['reduction_vs_global_guard'] > 0 for v in report['domains'].values())
    check(report['development_gate_passed'] == gate, 'both comparator gates')
    result = {'checks': checks, 'failures': 0, 'cases': 416, 'verified_weight_fits': 832,
        'report_sha256': hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Independent scalar objectives, gradients, gap and smoothing bounds; all weights, derived forecasts, scores, time/phase constraints, source hashes and comparator gates. Real-data fits needed no certificate refinements; that path has synthetic coverage only.'}
    output.write_text(json.dumps(result, indent=2)+'\n'); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'warm', 'control', 'contexts', 'directory'):p.add_argument(name)
    print(json.dumps(audit(**vars(p.parse_args())), indent=2))
