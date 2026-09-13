"""Fixed warm-up forecasts and contextual selector over unchanged scored cases."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import sys
import time

from .broad_screen import compute_case, select as recent_select
from .broad_calibration import choose as calibrated_select
from .warm_context import features, select


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(rows):
    means = {k: mean(r['scores'][k] for r in rows) for k in ('cv', 'context', 'past', 'blended', 'calibrated')}
    return {'cases': len(rows), 'mean_rmsle': means,
            'relative_reduction': {k: 1-v/means['cv'] for k, v in means.items()},
            'context_overrides': sum(r['context']['provider'] != r['context']['control_provider'] for r in rows)}


def run(warmup, panel, original, output):
    warmup, panel, original, output = map(Path, (warmup, panel, original, output))
    here = Path(__file__).parent
    receipt42 = json.loads((here/'evidence/broad-warmup-042.json').read_text())
    receipt38 = json.loads((here/'evidence/broad-screen-038.json').read_text())
    receipt37 = json.loads((here/'evidence/broad-panel-037.json').read_text())
    for name in ('warmup-spans.json', 'boundaries.json'):
        if digest(warmup/name) != receipt42['files'][name]:raise ValueError('Warmup evidence changed')
    if digest(panel/'development-spans.json') != receipt37['files']['development-spans.json']:
        raise ValueError('Development spans changed')
    manifest38 = json.loads((original/'manifest.json').read_text())
    if digest(original/'manifest.json') != receipt38['files']['manifest.json']:raise ValueError('Original manifest changed')
    for name in ('hourly_numerical.py', 'broad_screen.py'):
        if digest(here/name) != manifest38['code_sha256'][name]:raise ValueError('Frozen numerical helper changed')
    original_rows = []
    for name, sha in receipt38['files'].items():
        if name.startswith(('electricity:', 'pedestrian:')):
            if digest(original/name) != sha:raise ValueError('Scored evidence changed')
            original_rows.append(json.loads((original/name).read_text()))
    if len(original_rows) != 416:raise ValueError('All original tasks required')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'python': sys.version, 'executable': sys.executable,
        'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn')},
        'code_sha256': {n: digest(here/n) for n in ('broad_warm_screen.py', 'warm_context.py', 'BROAD_WARM_SCREEN_043.md', 'broad_screen.py', 'hourly_numerical.py', 'broad_calibration.py')},
        'source_receipts_sha256': {n: digest(here/'evidence'/n) for n in ('broad-warmup-042.json', 'broad-screen-038.json', 'broad-panel-037.json')},
        'api_calls': 0, 'reserved_counts_accessed': 0, 'recording_assumption': 'outcomes recorded at nominal horizon close',
        'planned_additional_forecast_computations': 3000, 'inherited_forecast_computations': 9984})
    spans = json.loads((warmup/'warmup-spans.json').read_text())
    originals = json.loads((panel/'development-spans.json').read_text())
    tasks = json.loads((warmup/'boundaries.json').read_text())
    save('excluded_warmup.json', [t for t in tasks if not t['ready']])
    computations = 0; estimator_fits = 0; contexts = 0; episodes = []; results = []
    start = time.monotonic(); cpu = time.process_time()
    def charge(model):
        nonlocal computations, estimator_fits
        computations += 1; estimator_fits += model in ('ridge_short', 'ridge_long', 'forest')
    def episode(row, observed):
        return {**{k: row[k] for k in ('series_id', 'round', 'origin', 'last_target', 'cv', 'scores')},
            'domain': row['series_id'].split(':')[0], 'outcome_recorded_at': row['last_target'],
            'features': features(row['cv'], observed)}
    try:
        completed = 0
        for task in tasks:
            if not task['ready']:continue
            series = task['series_id']; index = task['round']+8
            row = compute_case(spans[series], index, [], charge)
            row['series_id'] = series; row['round'] = task['round']
            if row['origin'].replace('+00:00', '') != task['origin']:raise ValueError('Warm-up origin shifted')
            save(f'warmup-{series}-{index:02d}.json', row)
            low, high = row['history_indices']
            episodes.append(episode(row, spans[series]['values'][low:high]))
            completed += 1
            save('status.json', {'phase': 'warmup_forecasts', 'completed_warmup_cases': completed,
                'forecast_computations': computations, 'estimator_fits': estimator_fits, 'seconds': time.monotonic()-start})
        if completed != 125 or computations != 3000:raise ValueError('Warmup cohort/cost mismatch')
        for row in original_rows:
            low, high = row['history_indices']
            episodes.append(episode(row, originals[row['series_id']]['values'][low:high]))
        save('episodes.json', episodes)
        for row in sorted(original_rows, key=lambda r: (r['origin'], r['series_id'])):
            current = next(e for e in episodes if e['series_id'] == row['series_id'] and e['round'] == row['round'])
            choice = select(row['cv'], current['features'], current['domain'], row['origin'], episodes)
            contexts += choice['contextual_fits']
            own = [e for e in episodes if e['series_id'] == row['series_id']]
            recent = recent_select(row['cv'], own, row['origin'])
            calibrated = calibrated_select(row['cv'], own, row['series_id'], row['origin'])
            picked = {'cv': row['selection']['cv'], 'context': choice['provider'],
                      'past': recent['past'], 'blended': recent['blended'], 'calibrated': calibrated['provider']}
            if picked['cv'] != choice['control_provider']:raise ValueError('Control changed')
            result = {'series_id': row['series_id'], 'round': row['round'], 'origin': row['origin'],
                'context': choice, 'recent': recent, 'calibrated': calibrated,
                'scores': {k: row['scores'][m] for k, m in picked.items()}}
            results.append(result)
            save('status.json', {'phase': 'selection', 'completed_scored_cases': len(results),
                'contextual_fits': contexts, 'forecast_computations': computations, 'seconds': time.monotonic()-start})
        overall = summary(results)
        domains = {s: summary([r for r in results if r['series_id'].startswith(s+':')]) for s in ('electricity', 'pedestrian')}
        report = {'overall': overall, 'domains': domains,
            'mature_round_ge_10': summary([r for r in results if r['round'] >= 10]),
            'early_development_round_0_17': summary([r for r in results if r['round'] < 18]),
            'later_development_round_18_25': summary([r for r in results if r['round'] >= 18]),
            'development_gate_passed': overall['relative_reduction']['context'] >= .2 and all(s['relative_reduction']['context'] > 0 for s in domains.values()),
            'completed_warmup_cases': completed, 'excluded_warmup_cases': 3,
            'additional_forecast_computations': computations, 'additional_estimator_fits': estimator_fits,
            'contextual_estimator_fits': contexts, 'total_common_forecast_computations': 9984+computations,
            'api_calls': 0, 'reserved_counts_accessed': 0, 'seconds': time.monotonic()-start,
            'cpu_seconds': time.process_time()-cpu, 'rows': results}
        save('report.json', report)
        return report
    except BaseException as error:
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error),
             'forecast_computations_started': computations, 'estimator_fits_started': estimator_fits,
             'contextual_fits': contexts, 'completed_scored_cases': len(results), 'seconds': time.monotonic()-start})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('warmup', 'panel', 'original', 'output'):
        parser.add_argument(name)
    args = parser.parse_args()
    result = run(args.warmup, args.panel, args.original, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
