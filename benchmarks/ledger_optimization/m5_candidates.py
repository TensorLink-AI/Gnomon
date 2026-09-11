"""Frozen development-only M5 candidate computation and past-only policy screen."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import importlib.util
import json
from math import cos, isfinite, log1p, pi, sin, sqrt
import os
from pathlib import Path
import platform
from statistics import mean
import time

from gnomon import ForecastRequest

from .cv_context_screen import leader, select, visible_history

MANIFEST_SHA = '6343970b713d8ccc49c5f3d6a633e3c876511ef4fd16c19f832d34a792243dd9'
RECIPES_SHA = '74f987cd9b260900795c3c8e1f6697a76ea87415f597a0d825b7c8bbc20d49eb'
PROVIDERS = ('sf_seasonal_naive_7', 'sf_window_average_28', 'sf_auto_ets_7',
             'sf_auto_arima_7', 'sf_auto_arima_promo_7', 'sf_auto_theta_7',
             'sf_croston_optimized', 'sf_mstl_7')
ORIGINS = tuple(365 + 14 * i for i in range(26))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, separators=(',', ':'))
        stream.write('\n')


def request(rows, origin):
    history = rows[max(0, origin - 729):origin + 1]
    if len(history) < 56 or origin < 0 or origin >= len(rows):
        raise ValueError('Insufficient origin-bounded history')
    dates = [datetime.fromisoformat(r['ds']) for r in history]
    future = [dates[-1] + timedelta(days=i) for i in range(1, 15)]

    def covariates(times):
        return tuple((0.0, sin(2 * pi * (t - timedelta(days=1)).weekday() / 7),
                      cos(2 * pi * (t - timedelta(days=1)).weekday() / 7)) for t in times)

    return ForecastRequest(history=tuple(float(r['y']) for r in history), horizon=14,
                           season=7, frequency='D', cutoff=dates[-1].isoformat(),
                           known_time_cutoff=dates[-1].isoformat(), recorded_time_cutoff=dates[-1].isoformat(),
                           timestamps=tuple(t.isoformat() for t in dates),
                           future_timestamps=tuple(t.isoformat() for t in future),
                           past_covariates=covariates(dates), future_covariates=covariates(future),
                           past_covariate_names=('onpromotion', 'dow_sin', 'dow_cos'),
                           future_covariate_names=('onpromotion', 'dow_sin', 'dow_cos'),
                           series_id=history[-1]['unique_id'], unit='unit_sales')


def metric(point, actual):
    if (len(point) != len(actual) or not actual or
            any(not isfinite(float(x)) for x in [*point, *actual]) or any(x < 0 for x in actual)):
        raise ValueError('Require complete finite forecasts and nonnegative actuals')
    return {'rmsle': sqrt(mean((log1p(max(0, p)) - log1p(a)) ** 2 for p, a in zip(point, actual, strict=True))),
            'mae': mean(abs(p - a) for p, a in zip(point, actual, strict=True))}


def compute_case(rows, round_index, predict):
    origin = ORIGINS[round_index]
    req = request(rows, origin)
    actual = [float(r['y']) for r in rows[origin + 1:origin + 15]]
    cv = []
    for index in (origin - 28, origin - 14):
        cv_req = request(rows, index)
        if cv_req.future_timestamps[-1] > req.cutoff:
            raise ValueError('CV actuals extend beyond forecast origin')
        cv.append((cv_req, [float(r['y']) for r in rows[index + 1:index + 15]]))
    case = {'series_id': req.series_id, 'round': round_index, 'origin': req.cutoff,
            'future_timestamps': list(req.future_timestamps), 'outcome_recorded_at': req.future_timestamps[-1],
            'request': asdict(req), 'actual': actual, 'predictions': {}, 'scores': {}, 'mae': {},
            'current_card': {}, 'candidate_metadata': {}, 'cv_evidence': {}}
    for provider in PROVIDERS:
        started = time.monotonic()
        point, metadata = predict(provider, req)
        scores = metric(point, actual)
        folds = []
        for cv_req, truth in cv:
            cv_point, cv_metadata = predict(provider, cv_req)
            cv_scores = metric(cv_point, truth)
            folds.append({'request': asdict(cv_req), 'point': list(cv_point), 'actual': truth,
                          'metrics': cv_scores, 'metadata': cv_metadata})
        case['predictions'][provider] = list(point)
        case['scores'][provider], case['mae'][provider] = scores['rmsle'], scores['mae']
        case['candidate_metadata'][provider] = dict(metadata, elapsed_seconds=time.monotonic() - started)
        case['current_card'][provider] = {'cv_rmsle': mean(f['metrics']['rmsle'] for f in folds),
                                          'cv_mae': mean(f['metrics']['mae'] for f in folds),
                                          'forecast_total': sum(point), 'status': 'ok'}
        case['cv_evidence'][provider] = folds
    return case


def load_panel(manifest_path):
    if digest(manifest_path) != MANIFEST_SHA:
        raise ValueError('Only registered M5 development manifest is allowed')
    manifest = json.loads(manifest_path.read_text())
    path = Path(manifest['development']['path'])
    if digest(path) != manifest['development']['sha256']:
        raise ValueError('Development input changed')
    groups = {}
    with path.open(newline='') as stream:
        for r in csv.DictReader(stream):
            if float(r['onpromotion']) != 0 or not isfinite(float(r['y'])) or float(r['y']) < 0:
                raise ValueError('Invalid registered observation/covariate')
            groups.setdefault(r['unique_id'], []).append(r)
    expected = {c['series_id'] for c in manifest['splits']['development']}
    if set(groups) != expected or len(groups) != 8:
        raise ValueError('Only the eight development identities are permitted')
    for group in groups.values():
        group.sort(key=lambda r: r['ds'])
        times = [datetime.fromisoformat(r['ds']) for r in group]
        if len(group) != 730 or any(b - a != timedelta(days=1) for a, b in zip(times, times[1:])):
            raise ValueError('Incomplete development time grid')
    return manifest, groups


def load_recipes(path):
    if digest(path) != RECIPES_SHA:
        raise ValueError('Pinned forecasting recipe changed')
    spec = importlib.util.spec_from_file_location('m5_pinned_recipes', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.CANDIDATES) != PROVIDERS:
        raise ValueError('Recipe portfolio changed')
    return module


def series_job(rows, recipe_path, output):
    module = load_recipes(Path(recipe_path))
    directory = Path(output) / rows[0]['unique_id']
    directory.mkdir()
    for index in range(26):
        case = compute_case(rows, index, module.forecast_candidate_result)
        save_new(directory / f'{index:02d}.json', case)
        print(json.dumps({'series': case['series_id'], 'completed_origin': index}), flush=True)
    info = module._forecast_cached.cache_info()._asdict()
    save_new(directory / 'cache.json', info)
    return rows[0]['unique_id'], info


def screen(cases):
    if len(cases) != 208 or len({(c['series_id'], c['round']) for c in cases}) != 208:
        raise ValueError('Require all 208 unique development cases')
    rows = []
    for c in cases:
        for p in PROVIDERS:
            computed = metric(c['predictions'][p], c['actual'])
            if abs(computed['rmsle'] - c['scores'][p]) > 1e-12 or abs(computed['mae'] - c['mae'][p]) > 1e-12:
                raise ValueError('Score arithmetic mismatch')
        history = visible_history(c, cases, conditioned=False)
        cv = leader(c)
        mae_provider = min(PROVIDERS, key=lambda p: (mean(h['mae'][p] for h in history), p)) if len(history) >= 4 else cv
        choices = {'current_cv': cv, 'frozen_support': select(c, history, 4, .5),
                   'lifetime_mae_diagnostic': mae_provider}
        rows.append({'series_id': c['series_id'], 'round': c['round'], 'origin': c['origin'],
                     'visible_origins': [h['origin'] for h in history], 'choices': choices,
                     'rmsle': {name: c['scores'][p] for name, p in choices.items()},
                     'future_aware_oracle_rmsle': min(c['scores'].values()),
                     'production_fallbacks': {p: c['candidate_metadata'][p] for p in PROVIDERS if c['candidate_metadata'][p]['fallback_used']}})
    summaries = {}
    predicates = {'all': lambda c: True, 'cold': lambda c: c['round'] < 4,
                  'mature': lambda c: c['round'] >= 4, 'training': lambda c: c['round'] < 18,
                  'later_development': lambda c: c['round'] >= 18}
    for name, predicate in predicates.items():
        group = [r for r in rows if predicate(r)]
        values = {p: mean(r['rmsle'][p] for r in group) for p in rows[0]['choices']}
        oracle = mean(r['future_aware_oracle_rmsle'] for r in group)
        cv = values['current_cv']
        summaries[name] = {'case_count': len(group), 'mean_case_rmsle': values,
                           'future_aware_oracle_rmsle': oracle,
                           'oracle_gain_vs_cv': 1 - oracle / cv if cv else None,
                           'support_gain_vs_cv': 1 - values['frozen_support'] / cv if cv else None}
    return {'scope': 'M5 development only; no live agent or confirmation comparison',
            'target_established': False, 'api_calls': 0, 'reserved_accessed': False,
            'summaries': summaries, 'cases': rows,
            'production_forecasts': len(cases) * len(PROVIDERS),
            'production_fallbacks': sum(len(r['production_fallbacks']) for r in rows),
            'cv_forecasts': len(cases) * len(PROVIDERS) * 2,
            'cv_fallbacks': sum(f['metadata']['fallback_used'] for c in cases for folds in c['cv_evidence'].values() for f in folds),
            'limitations': ['Lifetime MAE is not a 1.1.9 agent comparison.', 'Oracle uses future outcomes; not deployable.',
                            'Availability assumed at period end; promotion feature unavailable/zero.',
                            'Development performance cannot establish the final target.']}


def runtime():
    requirements = Path(__file__).with_name('m5-numerical-requirements.txt')
    for line in requirements.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, version = line.split('==')
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Numerical package pin mismatch: {name}')
    from gnomon import forecast_adapter
    return {'python': platform.python_version(),
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'request_module': forecast_adapter.__file__, 'request_module_sha256': digest(Path(forecast_adapter.__file__)),
            'thread_limits': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-manifest', type=Path, required=True)
    parser.add_argument('--recipes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refuse to overwrite any candidate run')
    manifest, groups = load_panel(args.panel_manifest)
    if digest(args.recipes) != RECIPES_SHA:
        raise ValueError('Pinned forecasting source mismatch')
    env = runtime()
    if any(v != '1' for v in env['thread_limits'].values()):
        raise ValueError('All numerical thread limits must be one')
    source_paths = [Path(__file__), Path(__file__).with_name('M5_CANDIDATES_015.md'),
                    Path(__file__).with_name('m5-numerical-requirements.txt'), Path(__file__).with_name('cv_context_screen.py')]
    frozen = {p.name: digest(p) for p in source_paths}
    args.output.mkdir(parents=True)
    jobs = args.output / 'series'
    jobs.mkdir()
    receipt = {'status': 'running', 'panel_manifest_sha256': digest(args.panel_manifest),
               'development_sha256': manifest['development']['sha256'], 'recipe_sha256': RECIPES_SHA,
               'recipe_path': str(args.recipes), 'source': frozen, 'runtime': env,
               'workers': 2, 'api_calls': 0, 'reserved_accessed': False, 'target_established': False}
    save_new(args.output / 'freeze.json', receipt)
    failures, caches = {}, {}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(series_job, rows, str(args.recipes), str(jobs)): sid for sid, rows in groups.items()}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                _, caches[sid] = future.result()
            except Exception as error:
                failures[sid] = f'{type(error).__name__}: {error}'
                print(json.dumps({'series': sid, 'failure': failures[sid]}), flush=True)
    changed = [p.name for p in source_paths if digest(p) != frozen[p.name]]
    if digest(Path(env['request_module'])) != env['request_module_sha256']:
        changed.append('typed_request_module')
    if digest(args.recipes) != RECIPES_SHA or digest(Path(manifest['development']['path'])) != manifest['development']['sha256']:
        changed.append('recipe_or_input')
    status = {'status': 'failed' if failures or changed else 'complete', 'failed_series': failures,
              'source_changes': changed, 'elapsed_seconds': time.monotonic() - started, 'recipe_cache_by_series': caches}
    if failures or changed:
        save_new(args.output / 'completion.json', status)
        raise ValueError('Preparation failed; retain partial artifacts, no complete score')
    cases = [json.loads((jobs / sid / f'{i:02d}.json').read_text()) for sid in sorted(groups) for i in range(26)]
    cases.sort(key=lambda c: (c['origin'], c['series_id']))
    save_new(args.output / 'cases.json', cases)
    result = screen(cases)
    save_new(args.output / 'completion.json', status)
    result['inputs'] = {'cases_path': str(args.output / 'cases.json'), 'cases_sha256': digest(args.output / 'cases.json'),
                        'freeze_sha256': digest(args.output / 'freeze.json'), 'completion_sha256': digest(args.output / 'completion.json')}
    save_new(args.output / 'screen.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
