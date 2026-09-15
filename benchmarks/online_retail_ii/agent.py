"""A common execution-bound tool for matched Hermes integrations.

Only an exported case directory is required. The tool does not receive the ZIP,
host panel, test actuals, or host baseline scores.
"""
from datetime import date, datetime, time, timedelta
import importlib.metadata
import json
from pathlib import Path
import uuid
import math
from zoneinfo import ZoneInfo

import pandas as pd

from .data import sha
from .models import CANDIDATES, forecast
from .run import fingerprint


def load_case(case):
    case = Path(case); task = json.loads((case/'task.json').read_text())
    if task['unit'] != 'units' or task['horizon'] != 14:
        raise ValueError('Changed task unit or horizon')
    if task['future_timestamps'] != [(date.fromisoformat(task['origin'])+timedelta(days=i)).isoformat() for i in range(1,15)]:
        raise ValueError('Forecast dates do not match the original task')
    history = pd.read_csv(case/'history.csv', parse_dates=['timestamp'])
    dates = history.timestamp.dt.strftime('%Y-%m-%d').tolist()
    actual_id = fingerprint(task['series_id'], history.value, dates, task['future_timestamps'])
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError('Duplicate or unsorted observed dates')
    if actual_id != task['request_fingerprint'] or dates[-1] != task['origin']:
        raise ValueError('Case history, task identity or timestamps changed')
    return task, history


def execute(case, output, provider, backend='gnomon'):
    if provider not in CANDIDATES or backend not in ('direct', 'gnomon'):
        raise ValueError('Select a registered provider and direct/gnomon backend')
    task, history = load_case(case)
    result = None; metadata = {}
    if backend == 'direct':
        result = forecast(provider, history.value, task['horizon'], date.fromisoformat(task['origin']))
        execution_id = str(uuid.uuid4())
    else:
        if importlib.metadata.version('gnomon-forecast') != '1.2.0':
            raise ValueError('This experiment requires the pinned Gnomon 1.2.0 distribution')
        from gnomon import ForecastRequest, ForecastResult, InferenceEngine
        engine = InferenceEngine(cache_size=0)
        def adapter(request):
            nonlocal result
            result = forecast(provider, request.history, request.horizon, date.fromisoformat(task['origin']))
            return ForecastResult(point=tuple(result['point']), timestamps=request.future_timestamps,
                                  series_id=request.series_id, unit=request.unit,
                                  metadata={'numerical_provider': provider, 'fallback_used': result['fallback_used']})
        engine.register(provider, adapter, deterministic=True, revision='online-retail-ii-v1/'+sha(Path(__file__).with_name('models.py')))
        zone = ZoneInfo('Europe/London')
        def stamp(day):
            return datetime.combine(date.fromisoformat(day), time(), zone).isoformat()
        request = ForecastRequest(history=tuple(map(float, history.value)), horizon=task['horizon'],
            series_id=task['series_id'], unit='units', frequency='D', season=7,
            timestamps=tuple(stamp(d.strftime('%Y-%m-%d')) for d in history.timestamp),
            future_timestamps=tuple(stamp(d) for d in task['future_timestamps']), cutoff=stamp(task['origin']))
        execution = engine.forecast(provider, request)
        returned = execution.to_dict(); execution_id = returned['execution_id']
        metadata = {'gnomon_execution': returned, 'gnomon_distribution': '1.2.0'}
        if list(execution.result.point) != result['point']:
            raise ValueError('Gnomon changed the registered numerical forecast')
    completion = {'operation': 'forecast', 'operation_succeeded': True,
        'case_id': task['case_id'], 'series_id': task['series_id'], 'unit': 'units',
        'provider': provider, 'execution_id': execution_id, 'request_fingerprint': task['request_fingerprint'],
        'future_timestamps': task['future_timestamps'], 'point': result['point'],
        'fallback_used': result['fallback_used'], 'executed_provider': result['executed_provider'],
        'clipped_points': result['clipped_points'],
        'numerical_error': result['error'], 'backend': backend,
        'recording_semantics': 'benchmark replay executed now; not an ex-ante historical ledger record',
        'models_sha256': sha(Path(__file__).with_name('models.py')), **metadata}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    destination = output/(execution_id+'.json')
    with destination.open('x') as stream:
        stream.write(json.dumps(completion, indent=2, allow_nan=False)+'\n')
    return completion


def resolve(case, executions, selection=None):
    """No prose scraping; conflicting explicit references never trigger recovery."""
    task, _ = load_case(case)
    matched = []
    for p in sorted(Path(executions).glob('*.json')):
        item = json.loads(p.read_text())
        points = item.get('point')
        valid_points = (isinstance(points, list) and len(points) == task['horizon']
                        and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in points))
        if (item.get('operation_succeeded') is True and item.get('case_id') == task['case_id']
                and item.get('request_fingerprint') == task['request_fingerprint']
                and item.get('series_id') == task['series_id']
                and item.get('unit') == task['unit'] and item.get('provider') in CANDIDATES and valid_points
                and item.get('future_timestamps') == task['future_timestamps']):
            matched.append(item)
    if selection is not None:
        chosen = [x for x in matched if x['execution_id'] == selection]
        if len(chosen) != 1:
            return {'resolved': False, 'status': 'conflicting_final_selection'}
        return {'resolved': True, 'status': 'explicit_selection', 'execution': chosen[0]}
    if len(matched) == 1:
        return {'resolved': True, 'status': 'recovered_single_execution', 'execution': matched[0]}
    return {'resolved': False, 'status': 'ambiguous_multiple_executions' if matched else 'no_successful_execution'}
