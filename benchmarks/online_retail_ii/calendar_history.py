"""Public-API daily replay comparison across UK clock transitions.

The pinned 1.2.0 ledger compares elapsed origin lags. The benchmark declares
local calendar days; offset changes must not change that task. Never suppress
an identity rejection without checking the recorded identities and local grid.
"""
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo


def compare_calendar_history(ledger, **query):
    original=ledger.compare_history(**query)
    if original['status']!='incompatible_evidence':return original
    if original.get('reason')!='task_or_provider_identity_changed' or original.get('excluded'):
        return original
    origins=original.get('origins',[])
    if not origins or len(origins)!=original.get('observed_origins'):return original
    zone=ZoneInfo('Europe/London');shapes=set();identities={};elapsed=set()
    def local(value):return datetime.fromisoformat(value).astimezone(zone)
    for origin in origins:
        for model in origin['models']:
            run=ledger.execution(model['execution_id']);req=run['request']
            if req['frequency']!='D' or len(req['future_timestamps'])!=query['horizon'] or not req['timestamps']:
                return original
            end=local(req['timestamps'][-1]);at=local(req['cutoff'])
            future=[local(t) for t in req['future_timestamps']]
            if any(t!=end+timedelta(days=i) for i,t in enumerate(future,1)):return original
            # Compare the local calendar lag and every other scored task dimension.
            shape={k:req.get(k) for k in ('frequency','season','horizon','series_id','unit',
                'past_covariate_names','future_covariate_names','quantiles','samples')}
            shape.update(related_series_count=len(req['related_series']),
                local_origin_lag_seconds=(at.replace(tzinfo=None)-end.replace(tzinfo=None)).total_seconds(),
                local_target_clock=end.time().isoformat())
            shapes.add(json.dumps(shape,sort_keys=True))
            identities.setdefault(run['provider'],set()).add(json.dumps(run['provider_identity'],sort_keys=True))
            elapsed.add((datetime.fromisoformat(req['cutoff'])-datetime.fromisoformat(req['timestamps'][-1])).total_seconds())
    if len(shapes)!=1 or any(len(v)!=1 for v in identities.values()) or len(elapsed)<2:return original
    # Each origin is still admitted independently by the pinned public API,
    # including ex-ante recording, actual visibility, identity and horizon checks.
    verified=[]
    for origin in origins:
        part=ledger.compare_history(**{**query,'start':origin['origin'],'end':origin['origin']})
        if part['status']!='ok' or part['matched_origins']!=1 or part.get('excluded'):return original
        if part['origins'][0]!=origin:return original
        verified.extend(part['origins'])
    result=dict(original)
    result.update(status='ok',reason=None,next_step='consider_evidence_and_sample_count_before_model_selection',
        origins=verified,models=[{'provider':p,'revision':revision,
            'mae':sum(next(m['mae'] for m in o['models'] if m['provider']==p) for o in verified)/len(verified)}
            for p,revision in query['providers'].items()],
        calendar_comparison={'timezone':'Europe/London','basis':'identical_local_daily_grid_and_provider_identity',
            'original_status':original['status'],'original_reason':original['reason'],
            'elapsed_origin_lags_seconds':sorted(elapsed),'independently_admitted_origins':len(verified),
            'predictions_or_actuals_changed':False})
    return result
