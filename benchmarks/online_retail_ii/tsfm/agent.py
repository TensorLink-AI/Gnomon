"""Execution-bound replay of one retained remote forecast through Gnomon 1.2.0."""
from pathlib import Path
from datetime import date,datetime,time
from zoneinfo import ZoneInfo
import importlib.metadata,json,math
from benchmarks.online_retail_ii.full_span.agent import load_case,execute as builtin_execute
from benchmarks.online_retail_ii.full_span.models import CANDIDATES as BUILTINS
from .client import PROVIDERS,digest

CANDIDATES=(*BUILTINS,*PROVIDERS)

def execute(case,output,provider,backend='gnomon'):
    if provider not in PROVIDERS:return builtin_execute(case,output,provider,backend)
    if backend!='gnomon' or importlib.metadata.version('gnomon-forecast')!='1.2.0':raise ValueError('Pinned Gnomon 1.2.0 required')
    from gnomon import ForecastRequest,ForecastResult,InferenceEngine
    task,history=load_case(case);frozen=json.loads((Path(case)/(provider+'-forecast.json')).read_text())
    if frozen['request_fingerprint']!=task['request_fingerprint'] or not frozen['available']:raise ValueError('TSFM forecast unavailable or belongs to another task')
    if digest(frozen['point'])!=frozen['point_sha256']:raise ValueError('Frozen TSFM forecast changed')
    zone=ZoneInfo('Europe/London')
    stamp=lambda d:datetime.combine(date.fromisoformat(d),time(),zone).isoformat()
    req=ForecastRequest(history=tuple(map(float,history.value)),horizon=14,season=7,frequency='D',
        series_id=task['series_id'],unit='units',timestamps=tuple(stamp(d.strftime('%Y-%m-%d')) for d in history.timestamp),
        future_timestamps=tuple(stamp(d) for d in task['future_timestamps']),cutoff=stamp(task['origin']))
    engine=InferenceEngine(cache_size=0)
    def predict(r):return ForecastResult(point=tuple(frozen['point']),timestamps=r.future_timestamps,series_id=r.series_id,unit=r.unit,
        metadata={'source':'retained_chutes_response','api_meta':frozen['api_meta'],'training_cutoff':'unknown',
            'forecast_generated_now_for_retrospective_replay':True})
    engine.register(provider,predict,deterministic=True,revision=frozen['revision'])
    execution=engine.forecast(provider,req);result=execution.to_dict()
    completion={'operation':'forecast','operation_succeeded':True,'case_id':task['case_id'],'series_id':task['series_id'],
        'unit':'units','provider':provider,'execution_id':result['execution_id'],'request_fingerprint':task['request_fingerprint'],
        'future_timestamps':task['future_timestamps'],'point':list(execution.result.point),'fallback_used':False,
        'executed_provider':provider,'clipped_points':frozen['clipped_points'],'numerical_error':None,'backend':'gnomon',
        'recording_semantics':'retrospective replay of retained live API forecast; not historically available model',
        'gnomon_execution':result,'gnomon_distribution':'1.2.0','tsfm_api_meta':frozen['api_meta'],
        'api_response_sha256':frozen['api_response_sha256'],'remote_call_reused_across_seeds':True}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with (out/(result['execution_id']+'.json')).open('x') as f:json.dump(completion,f,allow_nan=False)
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
