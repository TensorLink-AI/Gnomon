"""Nondeployable certified error interval for fixed six-forecast combinations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from scipy.optimize import minimize

from .ensemble_refinement import isolated


def interval(certificate, rmsle):
    low = max(0., certificate['objective']-certificate['convex_gap_bound']-(5/6)*1e-6)
    if low > rmsle+1e-10:raise ValueError('Contradictory optimization interval')
    return low, rmsle


def summary(rows):
    avg = lambda key: math.fsum(r[key] for r in rows)/len(rows)
    control = avg('control_rmsle'); low = avg('lower_bound'); high = avg('upper_bound')
    return {'cases': len(rows), 'control_rmsle': control,
            'hindsight_minimum_mean_rmsle_bounds': [low, high],
            'hindsight_improvement_bounds': [1-high/control, 1-low/control],
            'target_rmsle': .8*control, 'twenty_percent_ruled_out': low > .8*control}


def run(source, comparison, output):
    source, comparison, output = map(Path, (source, comparison, output)); here = Path(__file__).parent
    receipt38 = json.loads((here/'evidence/broad-screen-038.json').read_text())
    receipt45 = json.loads((here/'evidence/broad-ensemble-045.json').read_text())
    if hashlib.sha256((comparison/'report.json').read_bytes()).hexdigest() != receipt45['files']['report.json']:
        raise ValueError('Stronger control evidence changed')
    controls = {(r['series_id'], r['round']): r['scores']['cv_ensemble'] for r in json.loads((comparison/'report.json').read_text())['rows']}
    records = []
    for name, sha in receipt38['files'].items():
        if not name.startswith(('electricity:', 'pedestrian:')):continue
        raw = (source/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:raise ValueError('Original forecast evidence changed')
        records.append(json.loads(raw))
    if len(records) != len(controls) or len(records) != 416:raise ValueError('All fixed cases required')
    numerical = isolated('benchmarks.ledger_optimization._oracle_046', here/'evidence_ensemble.py')
    def refined(*args, **kwargs):
        kwargs['options'] = {**kwargs['options'], 'ftol': 1e-12}
        return minimize(*args, **kwargs)
    numerical.minimize = refined
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):(output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'code_sha256': {n: hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('ensemble_opportunity.py', 'BROAD_ENSEMBLE_BOUND_046.md', 'evidence_ensemble.py', 'ensemble_refinement.py')},
        'receipts_sha256': {n: hashlib.sha256((here/'evidence'/n).read_bytes()).hexdigest() for n in ('broad-screen-038.json', 'broad-ensemble-045.json')},
        'ftol': 1e-12, 'hindsight_only': True, 'api_calls': 0, 'new_provider_fits': 0})
    start = time.monotonic(); rows = []; fits = 0; iterations = 0; pending = None
    try:
        for source_row in sorted(records, key=lambda r: (r['series_id'], r['round'])):
            key = source_row['series_id'], source_row['round']; pending = key
            pair = {'point': source_row['point'], 'actual': source_row['actual']}
            fits += 1; certificate = numerical.fit([pair], [1.]); iterations += certificate['iterations']
            point = numerical.combine(source_row['point'], certificate['weights'])
            score = math.sqrt(math.fsum((math.log1p(p)-math.log1p(a))**2 for p, a in zip(point, source_row['actual'], strict=True))/24)
            low, high = interval(certificate, score)
            rows.append({'series_id': key[0], 'round': key[1], 'control_rmsle': controls[key],
                         'lower_bound': low, 'upper_bound': high, 'fit': certificate, 'point': point})
        report = {'overall': summary(rows), 'domains': {d: summary([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity', 'pedestrian')},
            'weight_fits': fits, 'iterations': iterations, 'seconds': time.monotonic()-start,
            'new_provider_fits': 0, 'api_calls': 0, 'reserved_data_accessed': False, 'hindsight_only': True, 'rows': rows}
        save('report.json', report); return report
    except BaseException as error:
        save('partial.json', rows)
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error), 'task': pending,
            'certificate': getattr(error, 'certificate', None), 'fits_started': fits,
            'completed_fits': len(rows), 'completed_iterations': iterations, 'seconds': time.monotonic()-start})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source'); p.add_argument('comparison'); p.add_argument('output'); a = p.parse_args()
    r = run(a.source, a.comparison, a.output)
    print(json.dumps({k: v for k, v in r.items() if k != 'rows'}, indent=2))
