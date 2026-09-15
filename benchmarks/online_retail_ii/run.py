"""Host-only baseline scoring and causally ordered case exports."""
from collections import defaultdict
from datetime import timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import dump, load_panel, origin, PHASES, sha
from .models import CANDIDATES, forecast, metrics


def fingerprint(series, history, dates, future):
    task = {'series_id': series, 'history': list(map(float, history)), 'timestamps': list(dates),
            'future_timestamps': list(future), 'unit': 'units', 'horizon': len(future)}
    return hashlib.sha256(json.dumps(task, sort_keys=True, allow_nan=False).encode()).hexdigest()


def summarize(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[r['method']].append(r)
    result = {}
    for method, rows in grouped.items():
        total_actual = sum(r['metrics']['actual_sum'] for r in rows)
        defined = [r['metrics']['mase'] for r in rows if r['metrics']['mase'] is not None]
        result[method] = {'cases': len(rows), 'mean_rmsle': float(np.mean([r['metrics']['rmsle'] for r in rows])),
            'mean_mae': float(np.mean([r['metrics']['mae'] for r in rows])),
            'mean_mase': float(np.mean(defined)) if defined else None, 'undefined_mase_cases': len(rows)-len(defined),
            'aggregate_wape': sum(r['metrics']['absolute_error_sum'] for r in rows)/total_actual if total_actual else None,
            'fallback_cases': sum(r['fallback_used'] for r in rows), 'clipped_points': sum(r['clipped_points'] for r in rows)}
    return result


def run_baselines(panel, output, smoke=False):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh run directory')
    manifest, frame = load_panel(panel)
    if manifest['phase'] != 'development':
        raise ValueError('This development runner cannot unlock validation/final; freeze agent and runtime first')
    selected = manifest['selected_products']
    if smoke:
        selected = {s: selected[s] for group in ('sparse', 'intermittent', 'frequent')
                    for s in sorted(k for k, v in selected.items() if v['stratum'] == group)[:2]}
    indices = list(PHASES[manifest['phase']])[:2] if smoke else list(PHASES[manifest['phase']])
    output.mkdir(parents=True)
    runtime = {n: importlib.metadata.version(n) for n in ('gnomon-forecast','numpy','pandas','statsforecast','statsmodels','scikit-learn','openpyxl')}
    plan = {'source_manifest_sha256': sha(Path(panel)/'manifest.json'), 'protocol_sha256': manifest['protocol_sha256'],
            'panel_sha256': manifest['panel_sha256'], 'phase': manifest['phase'], 'smoke': smoke,
            'series': sorted(selected), 'origins': indices, 'planned_cases': len(selected)*len(indices),
            'models': list(CANDIDATES), 'runtime': runtime, 'seeds': [7],
            'gnomon_agent_arm_executed': False, 'engy_calls': 0,
            'code_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}}
    dump(output/'plan.json', plan)
    rows, cache, history_evidence = [], {}, defaultdict(list)
    attempts = 0; fits = 0; cache_hits = 0; started = time.monotonic()
    panels = {s: f.sort_values('date').reset_index(drop=True) for s, f in frame.groupby('series_id') if s in selected}

    def numerical(series, model, y, day):
        nonlocal attempts, fits, cache_hits
        attempts += 1
        key = (series, model, day.isoformat(), hashlib.sha256(np.asarray(y, dtype=np.float64).tobytes()).hexdigest())
        if key not in cache:
            before = time.monotonic(); value = forecast(model, y, 14, day)
            value['elapsed_seconds'] = time.monotonic()-before
            cache[key] = value; fits += 1
        else:
            cache_hits += 1
        return cache[key]

    for index in indices:
        day = origin(index)
        for series in sorted(selected):
            f = panels[series]; before = f[f.date.dt.date <= day]; after = f[(f.date.dt.date > day) & (f.date.dt.date <= day+timedelta(days=14))]
            y = before.value.to_numpy(dtype=float); actual = after.value.to_numpy(dtype=float)
            if len(actual) != 14 or len(y) < 126:
                raise ValueError('Incomplete visible history or scoring horizon')
            current, cv = {}, {}
            for model in CANDIDATES:
                folds = []
                for offset in (42, 28, 14):
                    train, target = y[:-offset], y[-offset:][:14]
                    prediction = numerical(series, model, train, day-timedelta(days=offset))
                    folds.append({'origin': (day-timedelta(days=offset)).isoformat(),
                                  'rmsle': metrics(prediction['point'], target, train)['rmsle'],
                                  'fallback_used': prediction['fallback_used'], 'error': prediction['error']})
                cv[model] = {'folds': folds, 'mean_rmsle': float(np.mean([r['rmsle'] for r in folds])),
                             'eligible': not any(r['fallback_used'] for r in folds)}
                current[model] = numerical(series, model, y, day)
            ranked = sorted((m for m in CANDIDATES if cv[m]['eligible']), key=lambda m: (cv[m]['mean_rmsle'], CANDIDATES.index(m)))
            if not ranked:
                raise ValueError('No valid current-CV candidate')
            dates = before.date.dt.strftime('%Y-%m-%d').tolist(); future = after.date.dt.strftime('%Y-%m-%d').tolist()
            request_id = fingerprint(series, y, dates, future)
            case_id = f'{series}-{day.isoformat()}'
            case = output/'agent-cases'/case_id; case.mkdir(parents=True)
            before.rename(columns={'date':'timestamp','value':'value'})[['timestamp','value']].to_csv(case/'history.csv', index=False)
            dump(case/'task.json', {'case_id': case_id, 'series_id': series, 'unit': 'units', 'origin': day.isoformat(),
                'horizon': 14, 'future_timestamps': future, 'request_fingerprint': request_id, 'candidates': list(CANDIDATES),
                'target': manifest['target'], 'availability': manifest['source_availability'],
                'instruction': 'Select a typed executed forecast by execution_id. Current CV optimizes RMSLE. Do not infer stockouts or causal explanations from sales alone.'})
            dump(case/'current-cv.json', {'metric': 'mean_fold_rmsle', 'candidates': cv, 'ranking': ranked})
            matured = [r for r in history_evidence[series] if r['target_end'] <= day.isoformat()]
            dump(case/'matured-outcomes.json', {'as_of': day.isoformat(), 'records': matured,
                                              'availability_is_assumed': True})
            methods = dict(current)
            methods['rolling_cv_select'] = {**current[ranked[0]], 'selected_provider': ranked[0]}
            top = ranked[:3]
            methods['cv_top3_ensemble'] = {'provider': 'cv_top3_ensemble', 'members': top,
                'point': np.expm1(np.mean([np.log1p(current[m]['point']) for m in top], axis=0)).tolist(),
                'fallback_used': any(current[m]['fallback_used'] for m in top),
                'clipped_points': sum(current[m]['clipped_points'] for m in top)}
            chosen = ranked[0]
            if len(matured) >= 4:
                chosen = min(CANDIDATES, key=lambda m: (np.mean([r['scores'][m] for r in matured[-4:]]), CANDIDATES.index(m)))
            methods['historical_recent4_diagnostic'] = {**current[chosen], 'selected_provider': chosen,
                                                       'matured_origins': len(matured), 'is_gnomon_agent_result': False}
            scores = {}
            for method, prediction in methods.items():
                m = metrics(prediction['point'], actual, y)
                row = {'case_id': case_id, 'series_id': series, 'origin': day.isoformat(), 'origin_index': index,
                    'stratum': selected[series]['stratum'], 'method': method, 'request_fingerprint': request_id,
                    **prediction, 'metrics': m}
                rows.append(row)
                if method in CANDIDATES:
                    scores[method] = m['rmsle']
                with (output/'host-scores.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            history_evidence[series].append({'origin': day.isoformat(), 'target_end': future[-1],
                'future_timestamps': future, 'actual': actual.tolist(), 'scores': scores,
                'predictions': {m: current[m]['point'] for m in CANDIDATES},
                'fallbacks': {m: current[m]['fallback_used'] for m in CANDIDATES},
                'models_sha256': sha(Path(__file__).with_name('models.py'))})
            print(json.dumps({'case_complete': case_id, 'completed_cases': len(rows)//len(methods), 'planned_cases': plan['planned_cases']}), flush=True)
    report = {'status': 'complete', 'scope': 'development_smoke' if smoke else 'development',
        'cases': plan['planned_cases'], 'methods': summarize(rows), 'numerical_requests': attempts,
        'numerical_computations': fits, 'cache_hits': cache_hits, 'cached_computations_are_not_new_fits': True,
        'elapsed_seconds': time.monotonic()-started, 'gnomon_agent_arm_executed': False, 'engy_calls': 0,
        'final_evidence_opened': False, 'objective_established': False, 'plan_sha256': sha(output/'plan.json')}
    if {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')} != plan['code_sha256']:
        dump(output/'incomplete.json', {'cause': 'benchmark_source_changed_during_run', 'retain_all_outputs': True})
        raise ValueError('Benchmark source changed during execution; preserve this attempt')
    dump(output/'report.json', report)
    return report
