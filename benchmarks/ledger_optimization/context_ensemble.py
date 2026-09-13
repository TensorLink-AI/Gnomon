"""Retrieve temporally visible matched contexts; fit the frozen shared ensemble."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import time

import numpy as np
from scipy.optimize import minimize

from .ensemble_refinement import isolated
from .evidence_ensemble import combine
from .broad_ensemble import cv_pairs
from .broad_screen import rmsle


def retrieve(current, episodes):
    """No actuals or realized scores are needed to choose the neighborhood."""
    now = datetime.fromisoformat(current['origin'])
    if now.tzinfo is None:
        raise ValueError('Explicit current origin required')
    candidates = []; seen = set()
    for row in episodes:
        if row['domain'] != current['domain']:
            continue
        at, close, recorded = (datetime.fromisoformat(row[k]) for k in
                               ('origin', 'last_target', 'outcome_recorded_at'))
        if any(t.tzinfo is None for t in (at, close, recorded)):
            raise ValueError('Explicit evidence timestamps required')
        if at >= now or close > now or recorded > now:
            continue
        key = row['series_id'], at
        if key in seen or close <= at or recorded < close:
            raise ValueError('Duplicate or inconsistent visible evidence')
        seen.add(key); candidates.append(row)
    origins = sorted({datetime.fromisoformat(r['origin']) for r in candidates})[-8:]
    candidates = sorted([r for r in candidates if datetime.fromisoformat(r['origin']) in origins],
                        key=lambda r: (r['origin'], r['series_id']))
    current_x = np.asarray(current['features'], dtype=float)
    if current_x.shape != (12,) or not np.isfinite(current_x).all():
        raise ValueError('Twelve finite current features required')
    diagnostic = {'candidates': [], 'selected': [], 'distinct_origins': len(origins),
                  'ready': len(candidates) >= 16 and len(origins) >= 3,
                  'location': None, 'scale': None}
    if not candidates:
        return diagnostic
    x = np.asarray([r['features'] for r in candidates], dtype=float)
    if x.shape != (len(candidates), 12) or not np.isfinite(x).all():
        raise ValueError('Twelve finite historical features required')
    location = x.mean(axis=0); scale = np.maximum(x.std(axis=0), .1)
    distances = np.square((x-current_x)/scale).sum(axis=1)
    diagnostic.update(location=location.tolist(), scale=scale.tolist())
    diagnostic['candidates'] = [{'series_id': r['series_id'], 'origin': r['origin'],
                                 'distance': float(distance)}
                                for r, distance in zip(candidates, distances, strict=True)]
    if diagnostic['ready']:
        diagnostic['selected'] = sorted(diagnostic['candidates'],
            key=lambda r: (r['distance'], r['origin'], r['series_id']))[:16]
    return diagnostic


def summarize(rows):
    scores = {p: mean(r['scores'][p] for r in rows) for p in ('control', 'ledger')}
    return {'cases': len(rows), 'mean_rmsle': scores,
            'primary_ledger_reduction': 1-scores['ledger']/scores['control']}


def run(original, warm, control, output):
    original, warm, control, output = map(Path, (original, warm, control, output))
    here = Path(__file__).parent
    sources = {}
    def load(root, receipt_name, prefixes):
        receipt_path = here/'evidence'/receipt_name
        sources[receipt_name] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        receipt = json.loads(receipt_path.read_text()); rows = []
        for name, sha in receipt['files'].items():
            if not name.startswith(prefixes):continue
            raw = (root/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('Frozen source changed: '+name)
            rows.append(json.loads(raw))
        return rows
    scored = load(original, 'broad-screen-038.json', ('electricity:', 'pedestrian:'))
    earlier = load(warm, 'broad-warm-screen-043.json', ('warmup-electricity:', 'warmup-pedestrian:'))
    episodes = load(warm, 'broad-warm-screen-043.json', ('episodes.json',))[0]
    controls = load(control, 'broad-ensemble-045.json', ('electricity:', 'pedestrian:'))
    if (len(scored), len(earlier), len(episodes), len(controls)) != (416, 125, 541, 416):
        raise ValueError('Incomplete sources')
    by_key = {(r['series_id'], r['origin']): r for r in earlier+scored}
    metadata = {(r['series_id'], r['origin']): r for r in episodes}
    control_map = {(r['series_id'], r['origin']): r for r in controls}
    if len(by_key) != 541 or len(metadata) != 541 or len(control_map) != 416:
        raise ValueError('Duplicate source identity')
    # Retrieval sees only timestamps and predecision context, never scores.
    context = [{k: r[k] for k in ('series_id', 'origin', 'last_target',
                'outcome_recorded_at', 'domain', 'features')} for r in episodes]
    numerical = isolated('benchmarks.ledger_optimization._ensemble_047', here/'evidence_ensemble.py')
    def refined(*args, **kwargs):
        kwargs['options'] = {**kwargs['options'], 'ftol': 1e-12}
        return minimize(*args, **kwargs)
    numerical.minimize = refined
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'code_sha256': {n: hashlib.sha256((here/n).read_bytes()).hexdigest()
        for n in ('context_ensemble.py', 'evidence_ensemble.py', 'ensemble_refinement.py',
                  'broad_ensemble.py', 'broad_screen.py', 'BROAD_CONTEXT_ENSEMBLE_047.md')},
        'source_receipts_sha256': sources, 'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')},
        'api_calls': 0, 'additional_provider_fits': 0, 'inherited_forecast_computations': 12984,
        'control': 'byte-identical 045 CV ensemble', 'recording_assumption': 'period-end from 042/043'})
    started = time.monotonic(); cpu = time.process_time(); rows = []; fits = 0; iterations = 0; pending = None
    try:
        for source in sorted(scored, key=lambda r: (r['origin'], r['series_id'])):
            key = source['series_id'], source['origin']; pending = key
            retrieval = retrieve(metadata[key], context)
            prior = [by_key[(r['series_id'], r['origin'])] for r in retrieval['selected']]
            if retrieval['ready']:
                pairs = cv_pairs(source)+[{'point': r['point'], 'actual': r['actual']} for r in prior]
                fits += 1
                fitted = numerical.fit(pairs, [1/6]*3+[.5/len(prior)]*len(prior))
                iterations += fitted['iterations']
            else:
                fitted = control_map[key]['control_fit']
            # Future task actuals enter the metric only after the weights exist.
            points = {'control': control_map[key]['point']['cv_ensemble'],
                      'ledger': combine(source['point'], fitted['weights'])}
            row = {'series_id': key[0], 'origin': key[1], 'round': source['round'],
                   'current_features': metadata[key]['features'], 'retrieval': retrieval,
                   'fit': fitted, 'derived_forecasts': True, 'point': points, 'actual': source['actual'],
                   'scores': {p: rmsle(v, source['actual']) for p, v in points.items()}}
            save(f'{key[0]}-{source["round"]:02d}.json', row)
            rows.append({k: row[k] for k in ('series_id', 'round', 'scores')})
            save('status.json', {'completed_cases': len(rows), 'weight_fits': fits,
                                 'iterations': iterations, 'seconds': time.monotonic()-started})
        overall = summarize(rows)
        domains = {d: summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity', 'pedestrian')}
        report = {'overall': overall, 'domains': domains, 'rows': rows, 'weight_fits': fits,
                  'optimizer_iterations': iterations, 'seconds': time.monotonic()-started,
                  'cpu_seconds': time.process_time()-cpu, 'api_calls': 0, 'additional_provider_fits': 0,
                  'inherited_forecast_computations': 12984,
                  'development_gate_passed': overall['primary_ledger_reduction'] >= .2
                    and all(v['primary_ledger_reduction'] > 0 for v in domains.values())}
        save('report.json', report)
        return report
    except BaseException as error:
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error), 'task': pending,
            'certificate': getattr(error, 'certificate', None), 'completed_cases': len(rows),
            'weight_fits_started': fits, 'completed_fit_iterations': iterations,
            'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'warm', 'control', 'output'):p.add_argument(name)
    a = p.parse_args(); report = run(a.original, a.warm, a.control, a.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
