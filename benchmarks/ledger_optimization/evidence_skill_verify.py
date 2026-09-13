"""Independent array-based reproduction of evidence-risk diagnostic 048."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ORDER = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')
METHODS = ('cv', 'historical', 'blend')
KINDS = ('correct', 'opposite', 'predicted_tied', 'actual_tied')


def audit(original, warm, control, selected, directory):
    original, warm, control, selected, directory = map(Path, (original, warm, control, selected, directory))
    output = directory/'verification.json'
    if output.exists():raise FileExistsError(output)
    here = Path(__file__).parent; checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:raise AssertionError(message)
    def near(observed, expected):
        check(bool(np.allclose(observed, expected, atol=1e-10, rtol=1e-9)), 'numeric reproduction')
    def kind(p, a):
        if abs(a) <= 1e-12:return 'actual_tied'
        if abs(p) <= 1e-12:return 'predicted_tied'
        return 'correct' if (p > 0) == (a > 0) else 'opposite'
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, sha in manifest['code_sha256'].items():
        check(hashlib.sha256((here/name).read_bytes()).hexdigest() == sha, 'frozen code')
    for name, sha in manifest['source_receipts_sha256'].items():
        check(hashlib.sha256((here/'evidence'/name).read_bytes()).hexdigest() == sha, 'frozen receipt')
    def load(root, receipt_name, prefixes):
        receipt = json.loads((here/'evidence'/receipt_name).read_text()); result = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefixes):continue
            raw = (root/name).read_bytes()
            check(hashlib.sha256(raw).hexdigest() == sha, 'frozen artifact')
            result.append(json.loads(raw))
        return result
    scored = load(original, 'broad-screen-038.json', ('electricity:', 'pedestrian:'))
    previous = load(warm, 'broad-warm-screen-043.json', ('warmup-electricity:', 'warmup-pedestrian:'))
    metadata = load(warm, 'broad-warm-screen-043.json', ('episodes.json',))[0]
    controls = load(control, 'broad-ensemble-045.json', ('electricity:', 'pedestrian:'))
    choices = load(selected, 'broad-context-ensemble-047.json', ('electricity:', 'pedestrian:'))
    def index(rows):
        result = {(r['series_id'], r['origin']): r for r in rows}
        check(len(result) == len(rows), 'unique identity')
        return result
    sources = index(scored); history = index(scored+previous); episodes = index(metadata)
    controls = index(controls); choices = index(choices)
    rows = json.loads((directory/'cases.json').read_text())
    check(len(rows) == len(sources) == len(controls) == len(choices) == 416, 'all scored cases')
    check(len(history) == len(episodes) == 541, 'warm historical cohorts')
    check([(r['series_id'], r['origin']) for r in rows] == sorted(sources, key=lambda key: (key[1], key[0])), 'full case identity order')
    derived = []; triangles = np.triu_indices(6, 1)
    for row in rows:
        key = row['series_id'], row['origin']; source = sources[key]; choice = choices[key]
        check(row['round'] == source['round'], 'round')
        refs = [{'series_id': r['series_id'], 'origin': r['origin']} for r in choice['retrieval']['selected']]
        check(row['neighbors'] == refs and len(refs) == 16, 'frozen neighborhood')
        now = datetime.fromisoformat(key[1]); past = []
        for r in refs:
            historical_key = r['series_id'], r['origin']; meta = episodes[historical_key]
            check(datetime.fromisoformat(meta['origin']) < now
                  and datetime.fromisoformat(meta['last_target']) <= now
                  and datetime.fromisoformat(meta['outcome_recorded_at']) <= now, 'visible evidence')
            check(meta['domain'] == key[0].split(':')[0], 'domain')
            past.append(history[historical_key])
        cv = np.array([source['cv'][m] for m in ORDER])
        historical = np.array([[r['scores'][m] for m in ORDER] for r in past]).mean(axis=0)
        estimates = {'cv': cv, 'historical': historical, 'blend': (cv+historical)/2}
        actual = np.array([source['scores'][m] for m in ORDER])
        near([row['actual_risks'][m] for m in ORDER], actual)
        actual_differences = (actual[:, None]-actual[None, :])[triangles]
        risks = {}
        for method, estimate in estimates.items():
            near([row['risk_estimates'][method][m] for m in ORDER], estimate)
            difference = (estimate[:, None]-estimate[None, :])[triangles]
            squared = (difference-actual_differences)**2
            risks[method] = {'risk_mse': float(np.mean((estimate-actual)**2)),
                'risk_mae': float(np.mean(np.abs(estimate-actual))), 'contrast_mse': float(squared.mean()),
                'kinds': [kind(p, a) for p, a in zip(difference, actual_differences)]}
            observed = row['risk_assessment'][method]
            for metric in ('risk_mse', 'risk_mae', 'contrast_mse'):near(observed[metric], risks[method][metric])
            check(len(observed['contrasts']) == 15, 'all contrasts')
            for j, pair in enumerate(observed['contrasts']):
                check((pair['left'], pair['right']) == (ORDER[triangles[0][j]], ORDER[triangles[1][j]]), 'pair identity')
                near(pair['predicted_difference'], difference[j]); near(pair['actual_difference'], actual_differences[j])
                near(pair['squared_error'], squared[j]); check(pair['ordering'] == risks[method]['kinds'][j], 'pair ordering')
        w = np.array([controls[key]['control_fit']['weights'], choice['fit']['weights']]).T
        def losses(pair):
            logs = np.log1p(np.array([pair['point'][m] for m in ORDER]).T)
            residuals = logs@w-np.log1p(pair['actual'])[:, None]
            return np.sqrt(np.mean(residuals**2, axis=0))
        cv_losses = np.array([losses({'point': {m: source['folds'][m][i]['point'] for m in ORDER},
                                     'actual': source['folds'][ORDER[0]][i]['actual']}) for i in range(3)]).mean(axis=0)
        past_losses = np.array([losses(r) for r in past]).mean(axis=0)
        proposed = {'cv': cv_losses, 'historical': past_losses, 'blend': (cv_losses+past_losses)/2}
        gains = {method: float(value[0]-value[1]) for method, value in proposed.items()}
        for method in METHODS:
            near(row['gain_estimates'][method], gains[method])
            for j, arm in enumerate(('control', 'ledger')):near(row['estimated_ensemble_losses'][arm][method], proposed[method][j])
        realized = losses(source); gain = float(realized[0]-realized[1])
        near(row['actual_gain'], gain)
        near(gain, choice['scores']['control']-choice['scores']['ledger'])
        derived.append({'series_id': key[0], 'risks': risks, 'gains': gains, 'actual': gain})
    report = json.loads((directory/'report.json').read_text())
    def summary(observed, subset):
        check(observed['cases'] == len(subset), 'case count')
        mse_cv = np.mean([r['risks']['cv']['contrast_mse'] for r in subset])
        for method in METHODS:
            target = observed['risk_prediction'][method]
            for metric in ('risk_mse', 'risk_mae', 'contrast_mse'):near(target[metric], np.mean([r['risks'][method][metric] for r in subset]))
            if mse_cv:
                near(target['contrast_skill_vs_cv'], 1-np.mean([r['risks'][method]['contrast_mse'] for r in subset])/mse_cv)
            else:check(target['contrast_skill_vs_cv'] is None, 'undefined skill')
            counts = {k: sum(r['risks'][method]['kinds'].count(k) for r in subset) for k in KINDS}
            check(target['pair_ordering_counts'] == counts, 'pair count aggregate')
            prediction = observed['ensemble_gain_prediction'][method]
            near(prediction['prediction_mse'], np.mean([(r['gains'][method]-r['actual'])**2 for r in subset]))
            near(prediction['mean_predicted_gain'], np.mean([r['gains'][method] for r in subset]))
            check(prediction['ordering_counts'] == {k: sum(kind(r['gains'][method], r['actual']) == k for r in subset) for k in KINDS}, 'gain sign counts')
        actuals = np.array([r['actual'] for r in subset]); realized = observed['actual_gain']
        near(realized['mean'], actuals.mean()); near(realized['sum_gain'], actuals[actuals > 1e-12].sum())
        near(realized['sum_loss'], -actuals[actuals < -1e-12].sum())
        check(realized['helped'] == int((actuals > 1e-12).sum()) and realized['hurt'] == int((actuals < -1e-12).sum())
              and realized['tied'] == int((np.abs(actuals) <= 1e-12).sum()), 'help/hurt/tied')
    summary(report['overall'], derived)
    for domain in ('electricity', 'pedestrian'):summary(report['domains'][domain], [r for r in derived if r['series_id'].startswith(domain+':')])
    check(report['policy_changed'] is False and report['diagnostic_only'] is True, 'diagnostic scope')
    check(report['provider_calls'] == report['api_calls'] == report['weight_fits'] == 0, 'zero new execution')
    result = {'checks': checks, 'failures': 0, 'cases': len(rows),
        'report_sha256': hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
        'cases_sha256': hashlib.sha256((directory/'cases.json').read_bytes()).hexdigest(),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Independent array-based estimates, all 15 model contrasts, sign counts, ensemble losses/gains and aggregates; frozen source hashes and time visibility. No new fits or reserved access.'}
    output.write_text(json.dumps(result, indent=2)+'\n'); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'warm', 'control', 'selected', 'directory'):p.add_argument(name)
    print(json.dumps(audit(**vars(p.parse_args())), indent=2))
