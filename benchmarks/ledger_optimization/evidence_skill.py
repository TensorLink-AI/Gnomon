"""Audit predictive evidence without changing the completed 047 forecasts."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time

MODELS = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')
ESTIMATES = ('cv', 'historical', 'blend')
TOLERANCE = 1e-12


def ordering(predicted, actual):
    if abs(actual) <= TOLERANCE:return 'actual_tied'
    if abs(predicted) <= TOLERANCE:return 'predicted_tied'
    return 'correct' if predicted*actual > 0 else 'opposite'


def assess_risks(estimates, actual):
    """Each case contributes equally; contrasts are not independent samples."""
    if set(actual) != set(MODELS) or set(estimates) != set(ESTIMATES):
        raise ValueError('Complete six-model risks and three estimates required')
    if any(set(v) != set(MODELS) for v in estimates.values()):
        raise ValueError('Incomplete model estimate')
    if any(not math.isfinite(x) or x < 0 for values in [actual, *estimates.values()] for x in values.values()):
        raise ValueError('Finite nonnegative RMSLE required')
    result = {}
    for name, prediction in estimates.items():
        contrasts = []
        for i, left in enumerate(MODELS):
            for right in MODELS[i+1:]:
                expected = prediction[left]-prediction[right]
                realized = actual[left]-actual[right]
                contrasts.append({'left': left, 'right': right, 'predicted_difference': expected,
                    'actual_difference': realized, 'squared_error': (expected-realized)**2,
                    'ordering': ordering(expected, realized)})
        result[name] = {'risk_mse': mean((prediction[m]-actual[m])**2 for m in MODELS),
            'risk_mae': mean(abs(prediction[m]-actual[m]) for m in MODELS),
            'contrast_mse': mean(r['squared_error'] for r in contrasts), 'contrasts': contrasts}
    return result


def ensemble_loss(points, actual, weights):
    """Compute log-space loss directly, without rounding derived forecasts."""
    if len(weights) != 6 or any(not math.isfinite(w) or w < 0 for w in weights) or abs(sum(weights)-1) > 1e-8:
        raise ValueError('Simplex weights required')
    if len(actual) != 24 or any(len(points[m]) != 24 for m in MODELS):
        raise ValueError('Complete 24-step forecast required')
    residual = [math.fsum(weights[j]*math.log1p(points[m][i]) for j, m in enumerate(MODELS))
                - math.log1p(actual[i]) for i in range(24)]
    return math.sqrt(mean(e*e for e in residual))


def summarize(rows):
    risk = {}; gains = {}
    for method in ESTIMATES:
        counts = {k: 0 for k in ('correct', 'opposite', 'predicted_tied', 'actual_tied')}
        for row in rows:
            for pair in row['risk_assessment'][method]['contrasts']:counts[pair['ordering']] += 1
        risk[method] = {key: mean(r['risk_assessment'][method][key] for r in rows)
                        for key in ('risk_mse', 'risk_mae', 'contrast_mse')}
        risk[method]['pair_ordering_counts'] = counts
        gains[method] = {'prediction_mse': mean((r['gain_estimates'][method]-r['actual_gain'])**2 for r in rows),
            'mean_predicted_gain': mean(r['gain_estimates'][method] for r in rows),
            'ordering_counts': {k: sum(ordering(r['gain_estimates'][method], r['actual_gain']) == k for r in rows)
                                for k in counts}}
    denominator = risk['cv']['contrast_mse']
    for method in ESTIMATES:
        risk[method]['contrast_skill_vs_cv'] = 1-risk[method]['contrast_mse']/denominator if denominator else None
    actuals = [r['actual_gain'] for r in rows]
    return {'cases': len(rows), 'risk_prediction': risk, 'ensemble_gain_prediction': gains,
            'actual_gain': {'mean': mean(actuals), 'helped': sum(x > TOLERANCE for x in actuals),
                'hurt': sum(x < -TOLERANCE for x in actuals), 'tied': sum(abs(x) <= TOLERANCE for x in actuals),
                'sum_gain': math.fsum(x for x in actuals if x > TOLERANCE),
                'sum_loss': -math.fsum(x for x in actuals if x < -TOLERANCE)}}


def run(original, warm, control, selected, output):
    original, warm, control, selected, output = map(Path, (original, warm, control, selected, output))
    here = Path(__file__).parent; hashes = {}
    def load(root, receipt_name, prefixes):
        path = here/'evidence'/receipt_name
        hashes[receipt_name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = json.loads(path.read_text()); rows = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefixes):continue
            raw = (root/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:raise ValueError('Source changed: '+name)
            rows.append(json.loads(raw))
        return rows
    scored = load(original, 'broad-screen-038.json', ('electricity:', 'pedestrian:'))
    earlier = load(warm, 'broad-warm-screen-043.json', ('warmup-electricity:', 'warmup-pedestrian:'))
    episodes = load(warm, 'broad-warm-screen-043.json', ('episodes.json',))[0]
    controls = load(control, 'broad-ensemble-045.json', ('electricity:', 'pedestrian:'))
    choices = load(selected, 'broad-context-ensemble-047.json', ('electricity:', 'pedestrian:'))
    if tuple(map(len, (scored, earlier, episodes, controls, choices))) != (416, 125, 541, 416, 416):
        raise ValueError('Incomplete frozen sources')
    by_key = {(r['series_id'], r['origin']): r for r in earlier+scored}
    metadata = {(r['series_id'], r['origin']): r for r in episodes}
    control_map = {(r['series_id'], r['origin']): r for r in controls}
    choice_map = {(r['series_id'], r['origin']): r for r in choices}
    if tuple(map(len, (by_key, metadata, control_map, choice_map))) != (541, 541, 416, 416):
        raise ValueError('Duplicate source identities')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):(output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'code_sha256': {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
        for name in ('evidence_skill.py', 'BROAD_EVIDENCE_SKILL_048.md')},
        'source_receipts_sha256': hashes, 'provider_calls': 0, 'api_calls': 0,
        'diagnostic_only': True, 'policies_changed': False, 'inherited_forecast_computations': 12984})
    started = time.monotonic(); cpu = time.process_time(); results = []
    for source in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
        key = source['series_id'], source['origin']; choice = choice_map[key]
        refs = choice['retrieval']['selected']; now = datetime.fromisoformat(key[1]); past = []
        if len(refs) != 16 or len({(r['series_id'], r['origin']) for r in refs}) != 16:
            raise ValueError('Expected sixteen unique frozen neighbors')
        for ref in refs:
            neighbor = metadata[(ref['series_id'], ref['origin'])]
            at, close, recorded = (datetime.fromisoformat(neighbor[field]) for field in ('origin', 'last_target', 'outcome_recorded_at'))
            if any(t.tzinfo is None for t in (now, at, close, recorded)) or not (at < now and close <= now and recorded <= now):
                raise ValueError('Nonvisible historical evidence')
            if neighbor['domain'] != key[0].split(':')[0]:raise ValueError('Cross-domain neighbor')
            past.append(by_key[(ref['series_id'], ref['origin'])])
        estimates = {'cv': dict(source['cv']),
                     'historical': {m: mean(r['scores'][m] for r in past) for m in MODELS}}
        estimates['blend'] = {m: .5*(estimates['cv'][m]+estimates['historical'][m]) for m in MODELS}
        weights = {'control': control_map[key]['control_fit']['weights'], 'ledger': choice['fit']['weights']}
        losses = {}
        for arm, w in weights.items():
            cv_loss = mean(ensemble_loss({m: source['folds'][m][i]['point'] for m in MODELS},
                                        source['folds'][MODELS[0]][i]['actual'], w) for i in range(3))
            past_loss = mean(ensemble_loss(r['point'], r['actual'], w) for r in past)
            losses[arm] = {'cv': cv_loss, 'historical': past_loss, 'blend': .5*(cv_loss+past_loss)}
        gain_estimates = {m: losses['control'][m]-losses['ledger'][m] for m in ESTIMATES}
        # Prospective estimates above are now fixed. Current outcomes below
        # assess them; nothing produced here feeds a choice or a training set.
        result = {'series_id': key[0], 'origin': key[1], 'round': source['round'],
            'neighbors': [{'series_id': r['series_id'], 'origin': r['origin']} for r in refs],
            'risk_estimates': estimates, 'actual_risks': source['scores'],
            'risk_assessment': assess_risks(estimates, source['scores']),
            'estimated_ensemble_losses': losses, 'gain_estimates': gain_estimates,
            'actual_gain': choice['scores']['control']-choice['scores']['ledger']}
        results.append(result)
    report = {'overall': summarize(results),
        'domains': {domain: summarize([r for r in results if r['series_id'].startswith(domain+':')])
                    for domain in ('electricity', 'pedestrian')},
        'seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
        'provider_calls': 0, 'api_calls': 0, 'weight_fits': 0, 'policy_changed': False,
        'diagnostic_only': True, 'confidence_interval': None,
        'limitations': ['Repeated development cases; not held-out or causal agent evidence.',
            'Pairwise contrasts and successive origins are dependent.',
            'Ensemble gain estimates are losses on fitting records, not out-of-sample validation.',
            'Historical availability assumes period-end recording; original forecast computations reused.']}
    save('cases.json', results); save('report.json', report); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'warm', 'control', 'selected', 'output'):parser.add_argument(name)
    args = parser.parse_args(); print(json.dumps(run(**vars(args)), indent=2))
