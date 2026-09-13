"""Frozen 038 development screen; no raw-source, final-outcome or API access."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import sys
import time

from . import hourly_numerical as numerical

SOURCE_SHA = '60c8db1de333290d7cc04ca3a52dc4c4d46d8b93867105c5cb9be03adb8de15f'
MODELS = tuple(numerical.RECIPES)


def rmsle(point, actual):
    import math
    return math.sqrt(mean((math.log1p(p)-math.log1p(a))**2 for p, a in zip(point, actual, strict=True)))


def select(cv, history, origin):
    now = datetime.fromisoformat(origin)
    if now.tzinfo is None or set(cv) != set(MODELS):
        raise ValueError('Explicit origin and complete CV cohort required')
    eligible = []
    seen = set()
    for row in history:
        at, close, recorded = (datetime.fromisoformat(row[k]) for k in
                               ('origin', 'last_target', 'outcome_recorded_at'))
        if any(t.tzinfo is None for t in (at, close, recorded)):
            raise ValueError('Explicit evidence timezone required')
        if at >= now or close > now or recorded > now:
            continue
        if close <= at or recorded < close or at in seen or set(row['scores']) != set(MODELS):
            raise ValueError('Invalid or incomplete matched historical evidence')
        seen.add(at)
        eligible.append(row)
    recent = sorted(eligible, key=lambda r: datetime.fromisoformat(r['origin']))[-4:]
    control = min(MODELS, key=lambda m: (cv[m], MODELS.index(m)))
    past = blended = control
    if len(recent) >= 3:
        averages = {m: mean(r['scores'][m] for r in recent) for m in MODELS}
        past = min(MODELS, key=lambda m: (averages[m], cv[m], MODELS.index(m)))
        blended = min(MODELS, key=lambda m: ((averages[m]+cv[m])/2, cv[m], MODELS.index(m)))
    return {'cv': control, 'past': past, 'blended': blended,
            'history_origins': [r['origin'] for r in recent], 'matched_origins': len(recent)}


def compute_case(span, round_number, history, before_predict=None):
    start = datetime.fromisoformat(span['start_label']).replace(tzinfo=timezone.utc)
    stop = 730 + 168*round_number
    low = stop-730
    values = span['values'][low:stop]  # No current production targets given to selection.
    labels = [start+timedelta(hours=i) for i in range(low, stop+24)]
    origin = (start+timedelta(hours=stop)).isoformat()
    points, folds, cv = {}, {}, {}
    for model in MODELS:
        folds[model] = []
        for end in (658, 682, 706):
            if before_predict:
                before_predict(model)
            point = numerical.predict(values[:end], labels[:end], labels[end:end+24], model)
            actual = values[end:end+24]
            folds[model].append({'end': end, 'point': point, 'actual': actual, 'rmsle': rmsle(point, actual)})
        cv[model] = mean(f['rmsle'] for f in folds[model])
        if before_predict:
            before_predict(model)
        points[model] = numerical.predict(values, labels[:730], labels[730:], model)
    choice = select(cv, history, origin)
    actual = span['values'][stop:stop+24]
    scores = {m: rmsle(p, actual) for m, p in points.items()}
    return {'round': round_number, 'origin': origin,
            'last_target': (start+timedelta(hours=stop+24)).isoformat(),
            'history_indices': [low, stop], 'target_indices': [stop, stop+24],
            'history_sha256': hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest(),
            'cv': cv, 'selection': choice, 'point': points, 'actual': actual, 'scores': scores, 'folds': folds}


def summarize(rows):
    means = {k: mean(r[k+'_rmsle'] for r in rows) for k in ('cv', 'past', 'blended', 'hindsight')}
    return {'cases': len(rows), 'means': means,
            'past_relative_reduction': 1-means['past']/means['cv'],
            'blended_relative_reduction': 1-means['blended']/means['cv'],
            'hindsight_relative_reduction': 1-means['hindsight']/means['cv'],
            'past_selection_changes': sum(r['selection']['cv'] != r['selection']['past'] for r in rows)}


def run(source, output):
    source, output = Path(source), Path(output)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Only frozen development spans permitted')
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    save('manifest.json', {'source_sha256': SOURCE_SHA, 'recipes': numerical.RECIPES,
        'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('broad_screen.py', 'hourly_numerical.py', 'BROAD_SCREEN_038.md')},
        'python': sys.version, 'executable': sys.executable,
        'packages': {k: importlib.metadata.version(k) for k in ('numpy', 'scikit-learn')},
        'api_calls': 0, 'llm_tokens': 0, 'planned_forecast_computations': 9984,
        'planned_estimator_fits': 4992, 'planned_cases': 416})
    spans = json.loads(source.read_text())
    started, cpu = time.monotonic(), time.process_time()
    rows = []
    attempts = {'forecast_computations_started': 0, 'estimator_fits_started': 0}

    def before_predict(model):
        attempts['forecast_computations_started'] += 1
        attempts['estimator_fits_started'] += numerical.RECIPES[model]['kind'] in ('ridge', 'forest')
    try:
        for series, span in sorted(spans.items()):
            history = []
            for round_number in range(26):
                row = compute_case(span, round_number, history, before_predict)
                row['series_id'] = series
                save(f'{series}-{round_number:02d}.json', row)
                rows.append({'series_id': series, 'round': round_number, 'selection': row['selection'],
                    **{k+'_rmsle': row['scores'][row['selection'][k]] for k in ('cv', 'past', 'blended')},
                    'hindsight_rmsle': min(row['scores'].values())})
                history.append({'origin': row['origin'], 'last_target': row['last_target'],
                                'outcome_recorded_at': row['last_target'], 'scores': row['scores']})
                save('status.json', {'completed_cases': len(rows), 'forecast_computations': len(rows)*24,
                     'estimator_fits': len(rows)*12, 'seconds': time.monotonic()-started})
        assert len(rows) == 416
        domains = {s: summarize([r for r in rows if r['series_id'].startswith(s+':')])
                   for s in ('electricity', 'pedestrian')}
        overall = summarize(rows)
        result = {'overall': overall, 'domains': domains,
            'mature_round_ge_10': summarize([r for r in rows if r['round'] >= 10]),
            'development_gate_passed': overall['past_relative_reduction'] >= .2 and
                                      all(r['past_relative_reduction'] > 0 for r in domains.values()),
            'api_calls': 0, 'llm_tokens': 0, 'forecast_computations': len(rows)*24,
            'estimator_fits': len(rows)*12, 'seconds': time.monotonic()-started,
            'cpu_seconds': time.process_time()-cpu, 'rows': rows,
            'limitation': 'Development mechanism screen; not agent treatment, held-out evidence or independent-case significance.'}
        save('report.json', result)
        return result
    except BaseException as error:
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error),
             'completed_cases': len(rows), **attempts, 'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source'); parser.add_argument('output')
    args = parser.parse_args()
    result = run(args.source, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
