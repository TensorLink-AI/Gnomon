"""Frozen prospective learned-risk development comparison with strong controls."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import time
import numpy as np

from .conditional_risk import MODELS, fit_weights, gram, predict, train, visible
from .evidence_ensemble import combine
from .residual_memory import metric


def summarize(rows):
    scores = {arm: mean(r['scores'][arm] for r in rows) for arm in ('global_cv', 'intraday_cv', 'gram_cv', 'gram_ledger')}
    return {'cases': len(rows), 'mean_rmsle': scores,
            'reductions': {arm: 1-scores['gram_ledger']/scores[arm] for arm in ('gram_cv', 'global_cv', 'intraday_cv')}}


def run(original, warm, control, intraday, output):
    original, warm, control, intraday, output = map(Path, (original, warm, control, intraday, output))
    here = Path(__file__).parent; hashes = {}
    def load(root, receipt_name, prefixes):
        path = here/'evidence'/receipt_name; hashes[receipt_name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = json.loads(path.read_text()); rows = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefixes):continue
            raw = (root/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:raise ValueError('Frozen source changed: '+name)
            rows.append(json.loads(raw))
        return rows
    scored = load(original, 'broad-screen-038.json', ('electricity:', 'pedestrian:'))
    earlier = load(warm, 'broad-warm-screen-043.json', ('warmup-electricity:', 'warmup-pedestrian:'))
    episodes = load(warm, 'broad-warm-screen-043.json', ('episodes.json',))[0]
    controls = load(control, 'broad-ensemble-045.json', ('electricity:', 'pedestrian:'))
    intradays = load(intraday, 'broad-intraday-050.json', ('electricity:', 'pedestrian:'))
    if tuple(map(len, (scored, earlier, episodes, controls, intradays))) != (416, 125, 541, 416, 416):raise ValueError('Incomplete cohort')
    def index(rows):
        result = {(r['series_id'], r['origin']): r for r in rows}
        if len(result) != len(rows):raise ValueError('Duplicate task identity')
        return result
    history = index(scored+earlier); meta = index(episodes); controls = index(controls); intradays = index(intradays)
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):(output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'code_sha256': {n: hashlib.sha256((here/n).read_bytes()).hexdigest()
        for n in ('broad_conditional_risk.py', 'conditional_risk.py', 'evidence_ensemble.py', 'residual_memory.py', 'BROAD_CONDITIONAL_RISK_051.md')},
        'source_receipts_sha256': hashes, 'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'scikit-learn')},
        'api_calls': 0, 'forecast_provider_fits': 0, 'inherited_forecast_computations': 12984,
        'recording_assumption': 'period-end from 042/043', 'training_objective': 'conditional expected MSLE plus anchor penalty'})
    started = time.monotonic(); cpu = time.process_time(); forests = {}; rows = []; fits = 0; iterations = 0; forest_count = 0; pending = None
    try:
        for source in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
            key = source['series_id'], source['origin']; domain = key[0].split(':')[0]; batch = domain, key[1]; pending = {'task': key, 'phase': 'evidence'}
            if batch not in forests:
                available = visible(episodes, domain, key[1])
                ready = len(available) >= 32 and len({r['origin'] for r in available}) >= 3
                training = {'eligible': [{'series_id': r['series_id'], 'origin': r['origin']} for r in available],
                    'domain': domain, 'origin': key[1], 'ready': ready, 'effective_source_as_of': key[1], 'effective_recorded_as_of': key[1]}
                model = None
                if ready:
                    # Compute labels only after temporal eligibility. No current
                    # or future case outcomes are presented to this model.
                    matrices = [gram(history[(r['series_id'], r['origin'])]).tolist() for r in available]
                    model, details = train([r['features'] for r in available], matrices)
                    training.update(details); forest_count += 1
                name = f'forest-{domain}-{source["round"]:02d}.json'; save(name, training)
                forests[batch] = model, training, name
            model, training, model_name = forests[batch]
            pairs = [{'point': {m: source['folds'][m][i]['point'] for m in MODELS},
                      'actual': source['folds'][MODELS[0]][i]['actual']} for i in range(3)]
            cv_matrix = np.mean([gram(p) for p in pairs], axis=0)
            prediction = predict(model, training, meta[key]['features']) if model is not None else None
            historical = np.asarray(prediction['matrix']) if prediction is not None else cv_matrix
            blended = .5*(cv_matrix+historical)
            anchor = controls[key]['control_fit']['weights']
            pending['phase'] = 'cv_weights'; fits += 1; cv_fit = fit_weights(cv_matrix, anchor); iterations += cv_fit['iterations']
            pending['phase'] = 'ledger_weights'; fits += 1; ledger_fit = fit_weights(blended, anchor); iterations += ledger_fit['iterations']
            points = {'global_cv': controls[key]['point']['cv_ensemble'], 'intraday_cv': intradays[key]['point']['block_cv'],
                      'gram_cv': combine(source['point'], cv_fit['weights']), 'gram_ledger': combine(source['point'], ledger_fit['weights'])}
            # Score only after all current proposals are fixed.
            row = {'series_id': key[0], 'origin': key[1], 'round': source['round'], 'forest_ref': model_name,
                'historical_prediction': prediction, 'cv_matrix': cv_matrix.tolist(), 'derived_forecasts': True,
                'fits': {'gram_cv': cv_fit, 'gram_ledger': ledger_fit}, 'point': points, 'actual': source['actual'],
                'scores': {arm: metric(p, source['actual']) for arm, p in points.items()}}
            save(f'{key[0]}-{source["round"]:02d}.json', row); rows.append({k: row[k] for k in ('series_id', 'round', 'scores')})
            save('status.json', {'completed_cases': len(rows), 'evidence_forest_fits': forest_count, 'weight_fits': fits, 'iterations': iterations, 'seconds': time.monotonic()-started})
        overall = summarize(rows); domains = {d: summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity', 'pedestrian')}
        report = {'overall': overall, 'domains': domains, 'rows': rows, 'evidence_forest_fits': forest_count,
            'evidence_trees': forest_count*64, 'weight_fits': fits, 'optimizer_iterations': iterations,
            'seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
            'api_calls': 0, 'forecast_provider_fits': 0, 'inherited_forecast_computations': 12984,
            'development_gate_passed': all(v >= .2 for v in overall['reductions'].values()) and all(v > 0 for d in domains.values() for v in d['reductions'].values())}
        save('report.json', report); return report
    except BaseException as error:
        save('FAILED.json', {'task': pending, 'error': type(error).__name__, 'message': str(error),
            'certificate': getattr(error, 'certificate', None), 'completed_cases': len(rows),
            'evidence_forest_fits': forest_count, 'weight_fits_started': fits, 'completed_fit_iterations': iterations,
            'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'warm', 'control', 'intraday', 'output'):p.add_argument(name)
    r = run(**vars(p.parse_args())); print(json.dumps({k: v for k, v in r.items() if k != 'rows'}, indent=2))
