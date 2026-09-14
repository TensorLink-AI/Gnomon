"""Independent tree traversal, evidence weights, quadratic fits and scores."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np

MODELS = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')
SETTINGS = dict(n_estimators=64, max_depth=4, min_samples_leaf=8, max_features=1.,
                bootstrap=False, random_state=17, n_jobs=1, criterion='squared_error')


def audit(original, warm, control, intraday, directory):
    original, warm, control, intraday, directory = map(Path, (original, warm, control, intraday, directory))
    output = directory/'verification.json'
    if output.exists():raise FileExistsError(output)
    here = Path(__file__).parent; checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:raise AssertionError(message)
    def near(a, b, tolerance=1e-9):check(bool(np.allclose(a, b, atol=tolerance, rtol=tolerance)), 'numerical disagreement')
    def avg(values):return math.fsum(values)/len(values)
    def matrix(pair):
        errors = [[math.log1p(pair['point'][m][h])-math.log1p(pair['actual'][h]) for m in MODELS] for h in range(24)]
        return np.array([[avg([e[i]*e[j] for e in errors]) for j in range(6)] for i in range(6)])
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, sha in manifest['code_sha256'].items():check(hashlib.sha256((here/name).read_bytes()).hexdigest() == sha, 'code hash')
    for name, sha in manifest['source_receipts_sha256'].items():check(hashlib.sha256((here/'evidence'/name).read_bytes()).hexdigest() == sha, 'receipt hash')
    def load(root, receipt_name, prefixes):
        receipt = json.loads((here/'evidence'/receipt_name).read_text()); rows = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefixes):continue
            raw = (root/name).read_bytes(); check(hashlib.sha256(raw).hexdigest() == sha, 'source artifact hash')
            rows.append(json.loads(raw))
        return rows
    scored = load(original, 'broad-screen-038.json', ('electricity:', 'pedestrian:'))
    earlier = load(warm, 'broad-warm-screen-043.json', ('warmup-electricity:', 'warmup-pedestrian:'))
    episodes = load(warm, 'broad-warm-screen-043.json', ('episodes.json',))[0]
    controls = load(control, 'broad-ensemble-045.json', ('electricity:', 'pedestrian:'))
    intradays = load(intraday, 'broad-intraday-050.json', ('electricity:', 'pedestrian:'))
    check(tuple(map(len, (scored, earlier, episodes, controls, intradays))) == (416, 125, 541, 416, 416), 'all source cases')
    def index(rows):
        values = {(r['series_id'], r['origin']): r for r in rows}; check(len(values) == len(rows), 'unique identities'); return values
    history = index(scored+earlier); meta = index(episodes); controls = index(controls); intradays = index(intradays)
    def leaf(tree, features):
        node = 0; depth = 0
        while tree['children_left'][node] != -1:
            check(depth < 4, 'tree depth and acyclic traversal')
            j = tree['feature'][node]
            check(0 <= j < 12 and math.isfinite(tree['threshold'][node]), 'valid split')
            node = tree['children_left'][node] if features[j] <= tree['threshold'][node] else tree['children_right'][node]
            check(0 <= node < len(tree['feature']), 'tree child index'); depth += 1
        check(tree['children_right'][node] == -1, 'leaf structure')
        return node
    forests = {}; gram_cache = {}
    def historical_gram(key):
        if key not in gram_cache:gram_cache[key] = matrix(history[key])
        return gram_cache[key]
    def forest(name, domain, origin):
        if name in forests:return forests[name]
        record = json.loads((directory/name).read_text()); now = datetime.fromisoformat(origin)
        eligible = sorted([r for r in episodes if r['domain'] == domain and datetime.fromisoformat(r['origin']) < now
            and datetime.fromisoformat(r['last_target']) <= now and datetime.fromisoformat(r['outcome_recorded_at']) <= now], key=lambda r: (r['origin'], r['series_id']))
        refs = [{'series_id': r['series_id'], 'origin': r['origin']} for r in eligible]
        check(record['eligible'] == refs, 'all and only eligible historical records')
        check(record['origin'] == record['effective_source_as_of'] == record['effective_recorded_as_of'] == origin and record['domain'] == domain, 'effective query')
        ready = len(eligible) >= 32 and len({r['origin'] for r in eligible}) >= 3
        check(record['ready'] == ready and ready, 'supported full cohort')
        check(record['settings'] == SETTINGS and len(record['trees']) == 64, 'frozen estimator settings')
        check(record['features'] == [r['features'] for r in eligible], 'predecision features')
        expected_matrices = np.array([historical_gram((r['series_id'], r['origin'])) for r in eligible])
        near(record['matrices'], expected_matrices)
        check(min(np.linalg.eigvalsh(g).min() for g in expected_matrices) >= -1e-9, 'PSD historical labels')
        expected_hash = hashlib.sha256(json.dumps({'features': record['features'], 'matrices': record['matrices']}, separators=(',', ':')).encode()).hexdigest()
        check(record['input_sha256'] == expected_hash, 'forest training hash')
        # sklearn applies tree split thresholds to float32 feature values.
        features = np.asarray(record['features'], dtype=np.float32).tolist(); memberships = []
        for tree in record['trees']:
            sizes = [len(tree[k]) for k in ('children_left', 'children_right', 'feature', 'threshold')]
            check(len(set(sizes)) == 1 and sizes[0] >= 1, 'tree array shapes')
            groups = {}
            for i, x in enumerate(features):groups.setdefault(leaf(tree, x), []).append(i)
            check(all(len(group) >= 8 for group in groups.values()), 'minimum historical leaf support')
            memberships.append(groups)
        forests[name] = record, memberships, expected_matrices
        return forests[name]
    def fit_check(observed, expected, anchor):
        g = np.array(observed['matrix']); w = np.array(observed['weights']); a = np.array(anchor)
        near(g, expected); check(observed['anchor'] == anchor and observed['success'] is True, 'shared anchor and successful fit')
        check(w.shape == (6,) and np.isfinite(w).all() and (w >= 0).all(), 'feasible weights'); near(w.sum(), 1., 1e-10)
        eigenvalue = np.linalg.eigvalsh(g).min(); near(observed['minimum_eigenvalue'], eigenvalue)
        check(eigenvalue >= -1e-9, 'PSD weight-fit matrix')
        value = float(w@g@w+.001*np.sum((w-a)**2)); initial = float(a@g@a)
        gradient = 2*g@w+.002*(w-a); gap = float(w@gradient-gradient.min())
        near(observed['objective'], value); near(observed['initial_objective'], initial); near(observed['convex_gap_bound'], gap)
        check(gap <= 1e-5+1e-10 and value <= initial+1e-8, 'convex acceptance threshold')
    checked = []; iterations = 0
    for source in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
        key = source['series_id'], source['origin']; domain = key[0].split(':')[0]
        row = json.loads((directory/f'{key[0]}-{source["round"]:02d}.json').read_text())
        check((row['series_id'], row['origin'], row['round']) == (*key, source['round']), 'task identity')
        check(row['actual'] == source['actual'] and row['derived_forecasts'] is True, 'unchanged evidence and derived label')
        check(row['forest_ref'] == f'forest-{domain}-{source["round"]:02d}.json', 'one forest per domain and origin')
        record, groups, matrices = forest(row['forest_ref'], domain, key[1])
        prediction = row['historical_prediction']; current = np.asarray(meta[key]['features'], dtype=np.float32).tolist()
        check(prediction['current_features'] == meta[key]['features'], 'query uses current observed context')
        weights = np.zeros(len(record['eligible'])); leaves = []
        for tree, group in zip(record['trees'], groups, strict=True):
            node = leaf(tree, current); leaves.append(node); support = group[node]
            for i in support:weights[i] += 1/(64*len(support))
        check(prediction['leaves'] == leaves, 'independent query traversal')
        near(prediction['record_weights'], weights); near(weights.sum(), 1., 1e-10)
        check((weights >= 0).all(), 'nonnegative historical relevance weights')
        expected_history = np.tensordot(weights, matrices, axes=(0, 0)); near(prediction['matrix'], expected_history)
        pairs = [{'point': {m: source['folds'][m][i]['point'] for m in MODELS}, 'actual': source['folds'][MODELS[0]][i]['actual']} for i in range(3)]
        current_g = np.mean([matrix(pair) for pair in pairs], axis=0); near(row['cv_matrix'], current_g)
        anchor = controls[key]['control_fit']['weights']
        fit_check(row['fits']['gram_cv'], current_g, anchor)
        fit_check(row['fits']['gram_ledger'], .5*(current_g+expected_history), anchor)
        check(row['point']['global_cv'] == controls[key]['point']['cv_ensemble'] and row['point']['intraday_cv'] == intradays[key]['point']['block_cv'], 'both unchanged strong controls')
        for arm in ('gram_cv', 'gram_ledger'):
            w = row['fits'][arm]['weights']; iterations += row['fits'][arm]['iterations']
            expected = [math.expm1(math.fsum(w[j]*math.log1p(source['point'][m][h]) for j, m in enumerate(MODELS))) for h in range(24)]
            near(row['point'][arm], expected)
        for arm, point in row['point'].items():
            check(len(point) == 24, 'full forecast horizon')
            near(row['scores'][arm], math.sqrt(avg([(math.log1p(p)-math.log1p(y))**2 for p, y in zip(point, source['actual'], strict=True)])))
        checked.append({k: row[k] for k in ('series_id', 'round', 'scores')})
    report = json.loads((directory/'report.json').read_text()); check(report['rows'] == checked and len(checked) == 416, 'complete report')
    def summary(observed, rows):
        check(observed['cases'] == len(rows), 'aggregate count')
        means = {arm: avg([r['scores'][arm] for r in rows]) for arm in ('global_cv', 'intraday_cv', 'gram_cv', 'gram_ledger')}
        for arm, value in means.items():near(observed['mean_rmsle'][arm], value)
        for arm in ('gram_cv', 'global_cv', 'intraday_cv'):near(observed['reductions'][arm], 1-means['gram_ledger']/means[arm])
    summary(report['overall'], checked)
    for domain in ('electricity', 'pedestrian'):summary(report['domains'][domain], [r for r in checked if r['series_id'].startswith(domain+':')])
    check(report['evidence_forest_fits'] == len(forests) == 52 and report['evidence_trees'] == 3328 and report['weight_fits'] == 832, 'all model costs')
    check(report['optimizer_iterations'] == iterations and report['forecast_provider_fits'] == report['api_calls'] == 0, 'iteration and execution costs')
    gate = all(v >= .2 for v in report['overall']['reductions'].values()) and all(v > 0 for d in report['domains'].values() for v in d['reductions'].values())
    check(report['development_gate_passed'] == gate, 'all comparator gates')
    result = {'checks': checks, 'failures': 0, 'cases': 416, 'verified_evidence_forests': 52, 'verified_weight_fits': 832,
        'report_sha256': hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Independent scalar Gram entries, saved-tree traversal, leaf support and record-weight predictions; temporal eligibility, quadratic certificates, forecasts and scores. Does not independently implement or prove optimal tree split training. No new provider fits or reserved access.'}
    output.write_text(json.dumps(result, indent=2)+'\n'); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'warm', 'control', 'intraday', 'directory'):p.add_argument(name)
    print(json.dumps(audit(**vars(p.parse_args())), indent=2))
