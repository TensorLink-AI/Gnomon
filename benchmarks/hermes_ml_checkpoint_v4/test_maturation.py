"""Synthetic maturation boundary checks, with no model/network execution."""
from copy import deepcopy
import json
from pathlib import Path
from .maturation import plan


def fixture():
    origin='2026-01-01T00:00:00Z'
    times=['2026-01-02T00:00:00Z','2026-01-03T00:00:00Z']
    task={'origin':times[-1],'series_id':'sales','unit':'widgets','horizon':2}
    prior=[{'origin':origin,'outcome_recorded_at':times[-1],'future_timestamps':times,
            'series_id':'sales','unit':'widgets','actual':[2,4],'execution_id':'chosen'}]
    result={'event':'result','kind':'forecast','task_origin':origin,
            'execution':{'execution_id':'chosen'},'point':[2,3],
            'request':{'cutoff':origin,'series_id':'sales','unit':'widgets',
                       'horizon':2,'future_timestamps':times}}
    other=deepcopy(result);other['execution']['execution_id']='alternative';other['point']=[1,1]
    return [result,other],prior,task


def run_checks(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);checks=[]
    events,prior,task=fixture()
    before=deepcopy((events,prior,task))
    groups,excluded=plan(events,prior,task)
    assert len(groups)==1 and len(groups[0]['results'])==2 and excluded==[]
    assert (events,prior,task)==before
    checks.append('chosen and unchosen executions mature against the same visible outcome without mutation')
    events.append({'event':'matured','execution_id':'chosen'})
    assert [r['execution']['execution_id'] for r in plan(events,prior,task)[0][0]['results']]==['alternative']
    events.append({'event':'matured','execution_id':'alternative'})
    assert plan(events,prior,task)==([],[])
    checks.append('partial sync recovery and exact retry score each execution once')
    for name,change,reason in [
        ('recorded after query', lambda e,p,t:p[0].update(outcome_recorded_at='2026-01-04T00:00:00Z'), 'outcome_not_recorded_by_query'),
        ('target after query', lambda e,p,t:t.update(origin='2026-01-02T00:00:00Z'), 'outcome_not_recorded_by_query'),
        ('recording before maturity', lambda e,p,t:p[0].update(outcome_recorded_at='2026-01-02T00:00:00Z'), 'invalid_outcome_recording_time'),
        ('wrong series', lambda e,p,t:e[0]['request'].update(series_id='other'), 'series_or_unit_mismatch'),
        ('wrong unit', lambda e,p,t:e[0]['request'].update(unit=None), 'series_or_unit_mismatch'),
        ('wrong origin', lambda e,p,t:e[0]['request'].update(cutoff='2025-12-31T00:00:00Z'), 'forecast_origin_mismatch'),
        ('wrong horizon', lambda e,p,t:e[0]['request'].update(horizon=3), 'target_identity_mismatch'),
        ('wrong targets', lambda e,p,t:e[0]['request'].update(future_timestamps=['2026-01-02T12:00:00Z','2026-01-03T00:00:00Z']), 'target_identity_mismatch'),
        ('point length', lambda e,p,t:e[0].update(point=[1]), 'target_identity_mismatch'),
        ('actual length', lambda e,p,t:p[0].update(actual=[1]), 'target_identity_mismatch'),
        ('nonfinite actual', lambda e,p,t:p[0].update(actual=[1,float('nan')]), 'invalid_outcome_values'),
        ('outcome wrong unit', lambda e,p,t:p[0].update(unit='kg'), 'series_or_unit_mismatch'),
    ]:
        events,prior,task=fixture();events=events[:1];change(events,prior,task)
        groups,excluded=plan(events,prior,task)
        assert groups==[] and excluded==[{'execution_id':'chosen','reason':reason}], name
        checks.append(name+' excluded')
    events,prior,task=fixture();events[0]['kind']='backtest'
    assert len(plan(events,prior,task)[0][0]['results'])==1
    checks.append('retrospective forecasts never reclassified as production')
    for name,change in [('duplicate outcome',lambda e,p,t:p.append(deepcopy(p[0]))),
                        ('duplicate execution',lambda e,p,t:e.append(deepcopy(e[0]))),
                        ('naive query time',lambda e,p,t:t.update(origin='2026-01-03'))]:
        events,prior,task=fixture();change(events,prior,task)
        try:plan(events,prior,task)
        except ValueError:pass
        else:raise AssertionError(name+' should reject before any writes')
        checks.append(name+' rejects')
    (root/'passed.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2)+'\n')
    return checks
