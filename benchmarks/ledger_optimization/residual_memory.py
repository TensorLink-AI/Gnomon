"""Shared signed-log-error correction from explicitly supplied training pairs."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time

MODELS = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')


def log_forecast(point, weights):
    if len(weights) != 6 or any(not math.isfinite(w) or w < 0 for w in weights) or abs(sum(weights)-1) > 1e-8:
        raise ValueError('Six simplex weights required')
    if set(point) != set(MODELS) or any(len(point[m]) != 24 for m in MODELS):
        raise ValueError('Complete six-model 24-step forecasts required')
    if any(not math.isfinite(p) or p < 0 for values in point.values() for p in values):
        raise ValueError('Finite nonnegative predictions required')
    return [math.fsum(weights[j]*math.log1p(point[m][h]) for j, m in enumerate(MODELS)) for h in range(24)]


def fit(pairs, masses, weights):
    if not pairs or len(pairs) != len(masses) or any(not math.isfinite(x) or x < 0 for x in masses) or abs(sum(masses)-1) > 1e-10:
        raise ValueError('Normalized nonnegative masses required')
    residuals = []
    for pair in pairs:
        forecast = log_forecast(pair['point'], weights); actual = pair['actual']
        if len(actual) != 24 or any(not math.isfinite(y) or y < 0 for y in actual):
            raise ValueError('Finite nonnegative 24-step actuals required')
        residuals.append([math.log1p(y)-p for y, p in zip(actual, forecast, strict=True)])
    bias = [.5*math.fsum(mass*r[h] for mass, r in zip(masses, residuals, strict=True)) for h in range(24)]
    empirical = math.fsum(mass*mean((r[h]-bias[h])**2 for h in range(24)) for mass, r in zip(masses, residuals, strict=True))
    penalty = mean(b*b for b in bias)
    gradient = [(4*bias[h]-2*math.fsum(mass*r[h] for mass, r in zip(masses, residuals, strict=True)))/24 for h in range(24)]
    return {'correction': bias, 'residuals': residuals, 'masses': masses, 'weights': weights,
        'objective': empirical+penalty, 'empirical_loss': empirical, 'penalty': penalty,
        'max_absolute_gradient': max(abs(v) for v in gradient),
        'input_sha256': hashlib.sha256(json.dumps({'pairs': pairs, 'masses': masses, 'weights': weights}, separators=(',', ':')).encode()).hexdigest()}


def apply(point, weights, correction):
    if len(correction) != 24 or any(not math.isfinite(b) for b in correction):
        raise ValueError('Finite 24-step correction required')
    logs = [p+b for p, b in zip(log_forecast(point, weights), correction, strict=True)]
    return {'point': [math.expm1(max(0., p)) for p in logs],
            'clipped_leads': [i for i, p in enumerate(logs) if p < 0.]}


def metric(point, actual):
    return math.sqrt(mean((math.log1p(p)-math.log1p(y))**2 for p, y in zip(point, actual, strict=True)))


def summarize(rows):
    scores = {arm: mean(r['scores'][arm] for r in rows) for arm in ('uncorrected', 'cv_correction', 'ledger_correction')}
    return {'cases': len(rows), 'mean_rmsle': scores,
        'primary_reduction': 1-scores['ledger_correction']/scores['cv_correction'],
        'reduction_vs_uncorrected_guard': 1-scores['ledger_correction']/scores['uncorrected'],
        'control_correction_reduction_vs_uncorrected': 1-scores['cv_correction']/scores['uncorrected']}


def run(original, warm, control, contexts, output):
    original, warm, control, contexts, output = map(Path, (original, warm, control, contexts, output))
    here = Path(__file__).parent; hashes = {}
    def load(root, receipt_name, prefixes):
        path = here/'evidence'/receipt_name; hashes[receipt_name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = json.loads(path.read_text()); result = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefixes):continue
            raw = (root/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:raise ValueError('Frozen source changed: '+name)
            result.append(json.loads(raw))
        return result
    scored = load(original, 'broad-screen-038.json', ('electricity:', 'pedestrian:'))
    earlier = load(warm, 'broad-warm-screen-043.json', ('warmup-electricity:', 'warmup-pedestrian:'))
    episodes = load(warm, 'broad-warm-screen-043.json', ('episodes.json',))[0]
    controls = load(control, 'broad-ensemble-045.json', ('electricity:', 'pedestrian:'))
    contexts = load(contexts, 'broad-context-ensemble-047.json', ('electricity:', 'pedestrian:'))
    def index(rows):
        result = {(r['series_id'], r['origin']): r for r in rows}
        if len(result) != len(rows):raise ValueError('Duplicate cohort')
        return result
    if tuple(map(len, (scored, earlier, episodes, controls, contexts))) != (416, 125, 541, 416, 416):raise ValueError('Incomplete sources')
    by_key = index(scored+earlier); metadata = index(episodes); control_map = index(controls); context_map = index(contexts)
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):(output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'code_sha256': {n: hashlib.sha256((here/n).read_bytes()).hexdigest()
        for n in ('residual_memory.py', 'BROAD_RESIDUAL_MEMORY_049.md')}, 'source_receipts_sha256': hashes,
        'api_calls': 0, 'provider_fits': 0, 'inherited_forecast_computations': 12984,
        'inherited_047_weight_fits': 416, 'prior_047_weight_fits_used_for_prediction': False,
        'training_surrogate': 'weighted squared log residual plus unit ridge penalty',
        'primary_metric': 'mean case RMSLE', 'recording_assumption': 'period-end from 042/043'})
    started = time.monotonic(); cpu = time.process_time(); rows = []; fits = 0; clipped = {'cv_correction': 0, 'ledger_correction': 0}; pending = None
    try:
        for source in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
            key = source['series_id'], source['origin']; pending = key; now = datetime.fromisoformat(key[1])
            weights = control_map[key]['control_fit']['weights']
            refs = context_map[key]['retrieval']['selected']; past = []
            if len(refs) != 16 or len({(r['series_id'], r['origin']) for r in refs}) != 16:raise ValueError('Sixteen unique contexts required')
            for r in refs:
                identity = r['series_id'], r['origin']; meta = metadata[identity]
                at, close, recorded = (datetime.fromisoformat(meta[k]) for k in ('origin', 'last_target', 'outcome_recorded_at'))
                if any(t.tzinfo is None for t in (now, at, close, recorded)) or not (at < now and close <= now and recorded <= now):raise ValueError('Nonvisible outcome')
                if meta['domain'] != key[0].split(':')[0] or at.timetz() != now.timetz():raise ValueError('Domain or lead phase mismatch')
                past.append(by_key[identity])
            cv_pairs = []
            for i in range(3):
                ends = {source['folds'][m][i]['end'] for m in MODELS}
                if len(ends) != 1 or (730-next(iter(ends))) % 24:raise ValueError('CV phase mismatch')
                cv_pairs.append({'point': {m: source['folds'][m][i]['point'] for m in MODELS}, 'actual': source['folds'][MODELS[0]][i]['actual']})
            fitted = {'cv_correction': fit(cv_pairs, [1/3]*3, weights)}; fits += 1
            fitted['ledger_correction'] = fit(cv_pairs+[{'point': r['point'], 'actual': r['actual']} for r in past],
                                               [1/6]*3+[.5/len(past)]*len(past), weights); fits += 1
            forecasts = {arm: apply(source['point'], weights, result['correction']) for arm, result in fitted.items()}
            points = {'uncorrected': control_map[key]['point']['cv_ensemble'], **{arm: result['point'] for arm, result in forecasts.items()}}
            # Both fits and derived forecasts exist before scoring current actuals.
            row = {'series_id': key[0], 'origin': key[1], 'round': source['round'],
                'source_execution_case': f'{key[0]}-{source["round"]:02d}.json', 'derived_forecasts': True,
                'neighbors': [{'series_id': r['series_id'], 'origin': r['origin']} for r in refs],
                'weights': weights, 'fits': fitted, 'point': points, 'actual': source['actual'],
                'clipped_leads': {arm: result['clipped_leads'] for arm, result in forecasts.items()},
                'scores': {arm: metric(p, source['actual']) for arm, p in points.items()}}
            save(f'{key[0]}-{source["round"]:02d}.json', row)
            for arm in clipped:clipped[arm] += len(row['clipped_leads'][arm])
            rows.append({k: row[k] for k in ('series_id', 'round', 'scores')})
        overall = summarize(rows); domains = {d: summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity', 'pedestrian')}
        report = {'overall': overall, 'domains': domains, 'rows': rows, 'vector_fits': fits, 'clipped_leads': clipped,
            'seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu, 'api_calls': 0, 'provider_fits': 0,
            'inherited_forecast_computations': 12984, 'development_gate_passed':
            overall['primary_reduction'] >= .2 and overall['reduction_vs_uncorrected_guard'] >= .2
            and all(v['primary_reduction'] > 0 and v['reduction_vs_uncorrected_guard'] > 0 for v in domains.values())}
        save('report.json', report); return report
    except BaseException as error:
        save('FAILED.json', {'task': pending, 'error': type(error).__name__, 'message': str(error),
            'completed_cases': len(rows), 'completed_vector_fits': fits, 'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'warm', 'control', 'contexts', 'output'):parser.add_argument(name)
    r = run(**vars(parser.parse_args())); print(json.dumps({k: v for k, v in r.items() if k != 'rows'}, indent=2))
