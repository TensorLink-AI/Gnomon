"""Frozen intraday-mixture comparison using archived forecasts and evidence."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import time

from .intraday_ensemble import MODELS, combine, fit
from .residual_memory import metric


def summarize(rows):
    scores = {arm: mean(r['scores'][arm] for r in rows) for arm in ('global_cv', 'block_cv', 'block_ledger')}
    return {'cases': len(rows), 'mean_rmsle': scores,
        'primary_reduction': 1-scores['block_ledger']/scores['block_cv'],
        'reduction_vs_global_guard': 1-scores['block_ledger']/scores['global_cv'],
        'control_reduction_vs_global': 1-scores['block_cv']/scores['global_cv']}


def run(original, warm, control, contexts, output):
    original, warm, control, contexts, output = map(Path, (original, warm, control, contexts, output))
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
    contexts = load(contexts, 'broad-context-ensemble-047.json', ('electricity:', 'pedestrian:'))
    if tuple(map(len, (scored, earlier, episodes, controls, contexts))) != (416, 125, 541, 416, 416):raise ValueError('Incomplete frozen evidence')
    def index(rows):
        values = {(r['series_id'], r['origin']): r for r in rows}
        if len(values) != len(rows):raise ValueError('Duplicate identity')
        return values
    history = index(scored+earlier); metadata = index(episodes); controls = index(controls); contexts = index(contexts)
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):(output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'code_sha256': {n: hashlib.sha256((here/n).read_bytes()).hexdigest()
        for n in ('broad_intraday.py', 'intraday_ensemble.py', 'residual_memory.py', 'BROAD_INTRADAY_050.md')},
        'source_receipts_sha256': hashes, 'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')},
        'api_calls': 0, 'provider_fits': 0, 'inherited_forecast_computations': 12984,
        'recording_assumption': 'period-end from 042/043', 'regularizer': .01,
        'lead_blocks': [[0, 5], [6, 11], [12, 17], [18, 23]], 'prior_047_weights_used': False,
        'smoothing_epsilon': 1e-6, 'ftol': 1e-14, 'maximum_certificate_refinements': 32})
    started = time.monotonic(); cpu = time.process_time(); rows = []; fits = 0; iterations = 0; refinements = 0; pending = None
    try:
        for source in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
            key = source['series_id'], source['origin']; origin = datetime.fromisoformat(key[1])
            pending = {'series_id': key[0], 'round': source['round'], 'arm': 'block_cv'}
            anchor = controls[key]['control_fit']['weights']; refs = contexts[key]['retrieval']['selected']; past = []
            if len(refs) != 16 or len({(r['series_id'], r['origin']) for r in refs}) != 16:raise ValueError('Sixteen unique contexts required')
            for ref in refs:
                identity = ref['series_id'], ref['origin']; meta = metadata[identity]
                at, close, recorded = (datetime.fromisoformat(meta[k]) for k in ('origin', 'last_target', 'outcome_recorded_at'))
                if any(t.tzinfo is None for t in (origin, at, close, recorded)) or not(at < origin and close <= origin and recorded <= origin):raise ValueError('Nonvisible evidence')
                if meta['domain'] != key[0].split(':')[0] or at.timetz() != origin.timetz():raise ValueError('Domain or lead-hour mismatch')
                past.append(history[identity])
            cv = []
            for i in range(3):
                ends = {source['folds'][m][i]['end'] for m in MODELS}
                if len(ends) != 1 or (730-next(iter(ends))) % 24:raise ValueError('CV lead-hour mismatch')
                cv.append({'point': {m: source['folds'][m][i]['point'] for m in MODELS}, 'actual': source['folds'][MODELS[0]][i]['actual']})
            fits += 1; cv_fit = fit(cv, [1/3]*3, anchor); iterations += cv_fit['iterations']
            pending['arm'] = 'block_ledger'; fits += 1
            ledger_fit = fit(cv+[{'point': r['point'], 'actual': r['actual']} for r in past], [1/6]*3+[.5/16]*16, anchor)
            iterations += ledger_fit['iterations']
            refinements += len(cv_fit['certificate_refinements'])+len(ledger_fit['certificate_refinements'])
            points = {'global_cv': controls[key]['point']['cv_ensemble'],
                      'block_cv': combine(source['point'], cv_fit['weights']), 'block_ledger': combine(source['point'], ledger_fit['weights'])}
            # Actual current outcomes are used only after both proposals exist.
            row = {'series_id': key[0], 'origin': key[1], 'round': source['round'],
                'source_execution_case': f'{key[0]}-{source["round"]:02d}.json', 'derived_forecasts': True,
                'neighbors': [{'series_id': r['series_id'], 'origin': r['origin']} for r in refs],
                'fits': {'block_cv': cv_fit, 'block_ledger': ledger_fit}, 'point': points, 'actual': source['actual'],
                'scores': {arm: metric(p, source['actual']) for arm, p in points.items()}}
            save(f'{key[0]}-{source["round"]:02d}.json', row)
            rows.append({k: row[k] for k in ('series_id', 'round', 'scores')})
            save('status.json', {'completed_cases': len(rows), 'weight_fits': fits, 'iterations': iterations, 'seconds': time.monotonic()-started})
        overall = summarize(rows); domains = {d: summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity', 'pedestrian')}
        report = {'overall': overall, 'domains': domains, 'rows': rows, 'weight_fits': fits,
            'optimizer_iterations': iterations, 'seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
            'certificate_refinements': refinements,
            'api_calls': 0, 'provider_fits': 0, 'inherited_forecast_computations': 12984,
            'development_gate_passed': overall['primary_reduction'] >= .2 and overall['reduction_vs_global_guard'] >= .2
                and all(v['primary_reduction'] > 0 and v['reduction_vs_global_guard'] > 0 for v in domains.values())}
        save('report.json', report); return report
    except BaseException as error:
        save('FAILED.json', {'task': pending, 'error': type(error).__name__, 'message': str(error),
            'certificate': getattr(error, 'certificate', None), 'completed_cases': len(rows),
            'weight_fits_started': fits, 'completed_fit_iterations': iterations, 'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'warm', 'control', 'contexts', 'output'):p.add_argument(name)
    r = run(**vars(p.parse_args())); print(json.dumps({k: v for k, v in r.items() if k != 'rows'}, indent=2))
