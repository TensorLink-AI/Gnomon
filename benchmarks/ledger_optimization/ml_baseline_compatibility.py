"""Synthetic public-API compatibility check for the frozen 1.1.9 comparator.

Run unchanged in isolated published 1.1.9 and pinned 1.2.0 interpreters. This
tests the execution/visibility contract, not forecasting accuracy or agent value.
"""
import argparse
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import sys


def check(root):
    from gnomon import ForecastResult, GnomonSession, InferenceEngine, TemporalLedger
    from gnomon.forecast_adapter import AdapterCapabilities
    from gnomon.ids import FixedClock
    from gnomon.build_info import build_info
    root.mkdir(parents=True,exist_ok=False)
    origin='2026-01-03T00:00:00+00:00'
    observed='2026-01-06T00:00:00+00:00'
    request={'history':[1,2,3], 'timestamps':[f'2026-01-0{i}T00:00:00+00:00' for i in (1,2,3)],
             'future_timestamps':[f'2026-01-0{i}T00:00:00+00:00' for i in (4,5)],
             'horizon':2,'series_id':'synthetic_sales','unit':'widgets','cutoff':origin,
             'known_time_cutoff':origin,'frequency':'D','season':1,
             'past_covariates':[[0],[1],[0]],'future_covariates':[[1],[0]],
             'past_covariate_names':['known_feature'],'future_covariate_names':['known_feature']}
    db=TemporalLedger(root/'ledger.db',clock=FixedClock(datetime.fromisoformat(origin)))
    engine=InferenceEngine(ledger=db);calls=[]
    for name in ('synthetic_last','synthetic_mean'):
        def provider(r,name=name):
            calls.append(name)
            assert list(r.future_covariates)==[(1.0,),(0.0,)]
            point=r.history[-1] if name=='synthetic_last' else sum(r.history)/len(r.history)
            return ForecastResult(point=(point,)*r.horizon,series_id=r.series_id,
                                  unit=r.unit,timestamps=r.future_timestamps)
        engine.register(name,provider,revision='synthetic-contract-v1',deterministic=True,
                        lifecycle='stateless',capabilities=AdapterCapabilities(past_covariates=True,future_covariates=True))
    session=GnomonSession(engine=engine)
    executions={name:session.forecast(name,request) for name in ('synthetic_last','synthetic_mean')}
    assert all(v['status']=='ok' for v in executions.values()),executions
    ids=[v['execution_id'] for v in executions.values()]
    assert list(executions['synthetic_last']['result']['point'])==[3,3]
    assert list(executions['synthetic_mean']['result']['point'])==[2,2]
    pending=db.evaluate(ids[0],source_as_of=origin,recorded_as_of=origin)
    assert pending['n']==0 and pending['status']=='pending' and pending['mae'] is None
    db.clock=FixedClock(datetime.fromisoformat(observed))
    db.append_actual(actuals=[{'series_id':request['series_id'],'unit':'widgets','valid_time':t,
                              'value':a,'source_available_at':t}
                             for t,a in zip(request['future_timestamps'],[4,5],strict=True)])
    assert db.actuals_as_of(request['series_id'],unit='widgets',source_as_of=observed,
                           recorded_as_of=request['future_timestamps'][-1])==[]
    partial=db.evaluate(ids[0],source_as_of=request['future_timestamps'][0],recorded_as_of=observed)
    assert partial['n']==1 and partial['mae']==1 and partial['status']=='partial'
    try:db.evaluate(ids[0],source_as_of=request['future_timestamps'][0],recorded_as_of=observed,allow_partial=False)
    except Exception as exc:
        assert type(exc).__name__=='ForecastAdapterError' and 'complete actual horizon' in str(exc)
        strict_rejection={'type':type(exc).__name__,'message':str(exc)}
    else:raise AssertionError('Strict partial scoring should reject')
    complete=db.evaluate(execution_ids=ids,source_as_of=observed,recorded_as_of=observed,allow_partial=False)
    again=db.evaluate(execution_ids=ids,source_as_of=observed,recorded_as_of=observed,allow_partial=False)
    assert [v['evaluation_id'] for v in complete]==[v['evaluation_id'] for v in again]
    for score,expected in zip(complete,[(1.5,math.sqrt(2.5),-1.5),(2.5,math.sqrt(6.5),-2.5)],strict=True):
        assert score['n']==2 and score['status']=='complete'
        for field,value in zip(('mae','rmse','bias'),expected,strict=True):assert abs(score[field]-value)<1e-12
    comparison=db.compare_history(series_id=request['series_id'],horizon=2,unit='widgets',
        providers={n:'synthetic-contract-v1' for n in executions},start=origin,end=origin,
        source_as_of=observed,recorded_as_of=observed)
    assert comparison['matched_origins']==1,comparison
    assert len(comparison['models'])==2
    assert len(calls)==2, 'Scoring/comparison must not execute a provider'
    saved=db.evaluations(ids[0],recorded_as_of=observed)
    assert any(e['evaluation_id']==complete[0]['evaluation_id'] for e in saved)
    result={'passed':True,'distribution_version':importlib.metadata.version('gnomon-forecast'),
            'build':build_info(),'python':sys.executable,'request':request,
            'executions':executions,'pending':pending,'partial':partial,
            'strict_rejection':strict_rejection,'complete':complete,'comparison':comparison,
            'provider_calls':calls,'provider_calls_during_scoring_or_comparison':0,
            'original_submission_changed':False,
            'limitations':['Synthetic callable; no ML fit, Engy call, production data or reserved outcome.',
                           'A compatible 1.1.9 API does not establish performance of a 1.1.9 agent.',
                           'Incumbent evidence formatting for dynamic configurations still needs a prospective specification.']}
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    r=check(p.parse_args().output)
    print(json.dumps({k:r[k] for k in ('passed','distribution_version','build','provider_calls_during_scoring_or_comparison')}))
