"""Frozen online CV-error calibration using only matured matched development evidence."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time

MODELS = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')


def choose(cv, history, series_id, origin):
    now = datetime.fromisoformat(origin)
    if now.tzinfo is None:
        raise ValueError('Timezone required')

    def validate(scores):
        if set(scores) != set(MODELS) or any(not math.isfinite(v) or v < 0 for v in scores.values()):
            raise ValueError('Complete finite nonnegative cohort required')

    validate(cv)
    eligible = []; seen = set()
    for row in history:
        if row['series_id'] != series_id:
            raise ValueError('Foreign series in historical cohort')
        at, close, recorded = (datetime.fromisoformat(row[k]) for k in
                               ('origin', 'last_target', 'outcome_recorded_at'))
        if any(t.tzinfo is None for t in (at, close, recorded)):
            raise ValueError('Evidence timezone required')
        if at >= now or close > now or recorded > now:
            continue
        if at in seen or close <= at or recorded < close:
            raise ValueError('Invalid historical origin or availability')
        seen.add(at); validate(row['cv']); validate(row['scores'])
        eligible.append(row)
    recent = sorted(eligible, key=lambda r: datetime.fromisoformat(r['origin']))[-8:]
    control = min(MODELS, key=lambda m: (cv[m], MODELS.index(m)))
    bias = {m: 0. for m in MODELS}; weight = 0.
    if len(recent) >= 4:
        bias = {m: mean(r['scores'][m]-r['cv'][m] for r in recent) for m in MODELS}
        weight = len(recent)/(len(recent)+4)
    estimates = {m: max(0., cv[m]+weight*bias[m]) for m in MODELS}
    selected = min(MODELS, key=lambda m: (estimates[m], cv[m], MODELS.index(m)))
    return {'provider': selected, 'cv_provider': control, 'matched_origins': len(recent),
            'retrieved_origins': [r['origin'] for r in recent], 'bias': bias,
            'shrinkage_weight': weight, 'estimated_rmsle': estimates,
            'basis': 'past_cv_error_correction' if weight else 'insufficient_history_use_current_cv'}


def summary(rows):
    baseline = mean(r['cv_rmsle'] for r in rows)
    selected = mean(r['calibrated_rmsle'] for r in rows)
    deltas = [r['cv_rmsle']-r['historical_rmsle'] for r in rows if r['historical_changed']]
    return {'cases': len(rows), 'cv_rmsle': baseline, 'calibrated_rmsle': selected,
            'relative_reduction': 1-selected/baseline,
            'calibrated_overrides': sum(r['selection']['provider'] != r['selection']['cv_provider'] for r in rows),
            'original_historical_overrides': {'count': len(deltas),
                'helped': sum(d > 0 for d in deltas), 'hurt': sum(d < 0 for d in deltas),
                'tied': sum(d == 0 for d in deltas),
                'total_rmsle_saved': sum(max(0., d) for d in deltas),
                'total_rmsle_added': sum(max(0., -d) for d in deltas)}}


def run(source, output):
    source, output = Path(source), Path(output)
    receipt_path = Path(__file__).with_name('evidence')/'broad-screen-038.json'
    receipt = json.loads(receipt_path.read_text())
    data = []
    for name, sha in receipt['files'].items():
        if not name.startswith(('electricity:', 'pedestrian:')):
            continue
        raw = (source/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError('Screen evidence changed: '+name)
        data.append(json.loads(raw))
    if len(data) != 416:
        raise ValueError('All 416 frozen cases required')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'source_receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256(Path(__file__).with_name('BROAD_CALIBRATION_039.md').read_bytes()).hexdigest(),
        'additional_forecast_computations': 0, 'api_calls': 0,
        'inherited_forecast_computations': 9984, 'inherited_estimator_fits': 4992})
    started = time.monotonic(); cpu = time.process_time(); rows = []
    try:
        series_ids = sorted({r['series_id'] for r in data})
        for series in series_ids:
            history = []
            for row in sorted((r for r in data if r['series_id'] == series), key=lambda r: r['origin']):
                selection = choose(row['cv'], history, series, row['origin'])
                control = selection['cv_provider']
                if control != row['selection']['cv']:
                    raise ValueError('Original CV choice changed')
                rows.append({'series_id': series, 'round': row['round'], 'origin': row['origin'],
                    'selection': selection, 'cv_rmsle': row['scores'][control],
                    'calibrated_rmsle': row['scores'][selection['provider']],
                    'historical_rmsle': row['scores'][row['selection']['past']],
                    'historical_changed': row['selection']['past'] != control})
                history.append({**{k: row[k] for k in ('series_id', 'origin', 'last_target', 'scores', 'cv')},
                                'outcome_recorded_at': row['last_target']})
        overall = summary(rows)
        domains = {s: summary([r for r in rows if r['series_id'].startswith(s+':')]) for s in ('electricity', 'pedestrian')}
        estimates = {s: {m: {'mean_cv_rmsle': mean(r['cv'][m] for r in data if r['series_id'].startswith(s+':')),
                           'mean_actual_rmsle': mean(r['scores'][m] for r in data if r['series_id'].startswith(s+':'))}
                         for m in MODELS} for s in domains}
        result = {'overall': overall, 'domains': domains,
            'early_development_round_0_17': summary([r for r in rows if r['round'] < 18]),
            'later_development_round_18_25': summary([r for r in rows if r['round'] >= 18]),
            'mature_round_ge_10': summary([r for r in rows if r['round'] >= 10]),
            'original_recipe_cv_vs_actual': estimates,
            'development_gate_passed': overall['relative_reduction'] >= .2 and all(r['relative_reduction'] > 0 for r in domains.values()),
            'seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
            'additional_forecasts': 0, 'api_calls': 0, 'reserved_future_reads': 0, 'rows': rows}
        save('report.json', result)
        return result
    except BaseException as error:
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error), 'completed_cases': len(rows)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source'); parser.add_argument('output')
    args = parser.parse_args()
    result = run(args.source, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'original_recipe_cv_vs_actual')}, indent=2))
