"""Shared ensemble development comparison with explicit fit evidence."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import time

from .evidence_ensemble import MODELS, combine, fit
from .broad_screen import rmsle


def visible(history, series, origin):
    now = datetime.fromisoformat(origin)
    if now.tzinfo is None:raise ValueError('Explicit cutoff required')
    valid = []
    for r in history:
        if r['series_id'] != series:continue
        at, closed = datetime.fromisoformat(r['origin']), datetime.fromisoformat(r['last_target'])
        if at < now and closed <= now:valid.append(r)
    valid.sort(key=lambda r: datetime.fromisoformat(r['origin']))
    if len({r['origin'] for r in valid}) != len(valid):raise ValueError('Duplicate matched origin')
    return valid[-4:]


def cv_pairs(row):
    return [{'point': {m: row['folds'][m][i]['point'] for m in MODELS},
             'actual': row['folds'][MODELS[0]][i]['actual']} for i in range(3)]


def summarize(rows):
    means = {p: mean(r['scores'][p] for r in rows) for p in ('cv_ensemble', 'ledger_ensemble', 'hard_cv', 'uniform')}
    return {'cases': len(rows), 'mean_rmsle': means,
        'primary_ledger_reduction': 1-means['ledger_ensemble']/means['cv_ensemble'],
        'control_ensemble_reduction_vs_hard_cv': 1-means['cv_ensemble']/means['hard_cv'],
        'ledger_reduction_vs_hard_cv': 1-means['ledger_ensemble']/means['hard_cv']}


def run(original, warm, output):
    original, warm, output = map(Path, (original, warm, output)); here = Path(__file__).parent
    def load(root, receipt, prefix):
        result = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefix):continue
            raw = (root/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:raise ValueError('Frozen evidence changed: '+name)
            result.append(json.loads(raw))
        return result
    receipt38 = json.loads((here/'evidence/broad-screen-038.json').read_text())
    receipt43 = json.loads((here/'evidence/broad-warm-screen-043.json').read_text())
    scored = load(original, receipt38, ('electricity:', 'pedestrian:'))
    earlier = load(warm, receipt43, ('warmup-electricity:', 'warmup-pedestrian:'))
    if len(scored) != 416 or len(earlier) != 125:raise ValueError('Incomplete frozen cohorts')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'code_sha256': {n: hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('broad_ensemble.py', 'evidence_ensemble.py', 'BROAD_ENSEMBLE_044.md', 'broad_screen.py')},
        'source_receipts_sha256': {n: hashlib.sha256((here/'evidence'/n).read_bytes()).hexdigest() for n in ('broad-screen-038.json', 'broad-warm-screen-043.json')},
        'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')},
        'inherited_forecast_computations': 12984, 'additional_provider_fits': 0, 'api_calls': 0,
        'recording_assumption': 'period-end outcome recording from 042/043'})
    started = time.monotonic(); cpu = time.process_time(); rows = []; fits = 0; iterations = 0
    pending = None
    try:
        for row in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
            pending = {'series_id': row['series_id'], 'round': row['round'], 'phase': 'control'}
            cv = cv_pairs(row); cv_masses = [1/3]*3
            fits += 1; control = fit(cv, cv_masses); iterations += control['iterations']
            past = visible(earlier+scored, row['series_id'], row['origin'])
            if len(past) >= 3:
                pending['phase'] = 'ledger'
                pairs = cv+[{'point': r['point'], 'actual': r['actual']} for r in past]
                masses = [1/6]*3+[.5/len(past)]*len(past)
                fits += 1; ledger = fit(pairs, masses); iterations += ledger['iterations']
            else:ledger = control
            # Current actuals enter reporting only after both fits are complete.
            points = {'cv_ensemble': combine(row['point'], control['weights']),
                      'ledger_ensemble': combine(row['point'], ledger['weights']),
                      'uniform': combine(row['point'], [1/6]*6),
                      'hard_cv': row['point'][row['selection']['cv']]}
            result = {'series_id': row['series_id'], 'round': row['round'], 'origin': row['origin'],
                'source_execution_case': f'{row["series_id"]}-{row["round"]:02d}.json',
                'derived_forecasts': True, 'control_fit': control, 'ledger_fit': ledger,
                'retrieved': [{'origin': r['origin'], 'round': r['round']} for r in past],
                'point': points, 'actual': row['actual'], 'scores': {p: rmsle(point, row['actual']) for p, point in points.items()}}
            save(f'{row["series_id"]}-{row["round"]:02d}.json', result)
            rows.append({k: result[k] for k in ('series_id', 'round', 'scores')})
            save('status.json', {'completed_cases': len(rows), 'weight_fits': fits, 'iterations': iterations, 'seconds': time.monotonic()-started})
        overall = summarize(rows)
        domains = {d: summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity', 'pedestrian')}
        report = {'overall': overall, 'domains': domains,
            'mature_round_ge_10': summarize([r for r in rows if r['round'] >= 10]),
            'later_development_round_18_25': summarize([r for r in rows if r['round'] >= 18]),
            'development_gate_passed': overall['primary_ledger_reduction'] >= .2 and all(v['primary_ledger_reduction'] > 0 for v in domains.values()),
            'weight_fits': fits, 'optimizer_iterations': iterations, 'seconds': time.monotonic()-started,
            'cpu_seconds': time.process_time()-cpu, 'api_calls': 0, 'additional_provider_fits': 0,
            'inherited_forecast_computations': 12984, 'rows': rows}
        save('report.json', report); return report
    except BaseException as error:
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error), 'task': pending,
            'certificate': getattr(error, 'certificate', None), 'completed_cases': len(rows),
            'weight_fits_started': fits, 'completed_fit_iterations': iterations, 'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('original'); p.add_argument('warm'); p.add_argument('output'); a = p.parse_args()
    r = run(a.original, a.warm, a.output)
    print(json.dumps({k: v for k, v in r.items() if k != 'rows'}, indent=2))
