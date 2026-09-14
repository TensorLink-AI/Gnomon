"""Exercise087 against the pinned published1.2.0 ledger using synthetic data."""
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from .agent_review import review
from benchmarks.hermes_ml_checkpoint_v4 import dev_evidence_summary


def run(output):
    import gnomon
    from gnomon import TemporalLedger,InferenceEngine,ForecastResult
    from gnomon.ids import FixedClock
    from gnomon.build_info import build_info
    if importlib.metadata.version('gnomon-forecast')!='1.2.0':raise ValueError('Pinned1.2.0 runtime required')
    build=build_info()
    if build['commit']!='a38cd0cad35383e5f10021abf3aa20d4c16923be':raise ValueError('Wrong runtime build')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);first=datetime(2026,1,3,tzinfo=timezone.utc)
    db=TemporalLedger(root/'ledger.db',clock=FixedClock(first));engine=InferenceEngine(ledger=db);calls=[];records=[]
    for name,value in [('a',2),('b',4),('c',3)]:
        def provider(r,name=name,value=value):
            calls.append(name)
            return ForecastResult(point=(value,)*r.horizon,series_id=r.series_id,unit=r.unit,timestamps=r.future_timestamps)
        engine.register(name,provider,revision='synthetic-v1:'+name,deterministic=True,lifecycle='stateless')
    for number in range(3):
        origin=first+timedelta(days=2*number);db.clock=FixedClock(origin)
        times=[(origin+timedelta(days=i)).isoformat() for i in (1,2)]
        request={'history':[1,1,1],'timestamps':[(origin-timedelta(days=i)).isoformat() for i in (2,1,0)],
                 'future_timestamps':times,'horizon':2,'series_id':'synthetic_sales','unit':'widgets',
                 'cutoff':origin.isoformat(),'known_time_cutoff':origin.isoformat(),'frequency':'D'}
        ids=[]
        for name in ('a','b','c'):
            execution=engine.forecast(name,request).to_dict();ids.append(execution['execution_id'])
            records.append({'event':'result','kind':'forecast','request':request,'config_id':name,
                            'config':{'synthetic_provider':name},'execution':execution})
        db.clock=FixedClock(origin+timedelta(days=2))
        db.append_actual(actuals=[{'series_id':'synthetic_sales','unit':'widgets','value':1,
                                  'valid_time':t,'source_available_at':t} for t in times])
        db.evaluate(execution_ids=ids,source_as_of=times[-1],recorded_as_of=times[-1],allow_partial=False)
    task={'series_id':'synthetic_sales','unit':'widgets','horizon':2,'origin':(first+timedelta(days=7)).isoformat()}
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before=sha(root/'ledger.db');call_count=len(calls)
    result=review(db,records,task,dev_evidence_summary,root/'full-before.json',limit=1)
    assert sha(root/'ledger.db')==before and len(calls)==call_count
    assert result['full_evidence']['sha256']==sha(root/'full-before.json')
    window=result['cards'][0]['windows']['last_4_origins']
    assert window['matched_origins']==3 and window['n']==6 and window['lowest_error_config_ids']==['a']
    assert result['cards'][0]['windows']['lifetime']['same_as']=='last_4_origins'
    next_call=result['pagination']['next_call']
    assert next_call['arguments']=={'offset':1,'limit':1}
    following=review(db,records,task,dev_evidence_summary,root/'full-next.json',**next_call['arguments'])
    assert following['cards'][0]['config_ids']!=result['cards'][0]['config_ids']
    assert sha(root/'ledger.db')==before and len(calls)==call_count
    db.clock=FixedClock(first+timedelta(days=20))
    db.append_actual(series_id='synthetic_sales',unit='widgets',value=999,
                     valid_time=(first+timedelta(days=1)).isoformat(),source_available_at=(first+timedelta(days=20)).isoformat())
    after_revision=sha(root/'ledger.db')
    revised=review(db,records,task,dev_evidence_summary,root/'full-after.json',limit=1)
    assert sha(root/'ledger.db')==after_revision and len(calls)==call_count
    assert (root/'full-before.json').read_bytes()==(root/'full-after.json').read_bytes()
    assert result['cards']==revised['cards'] and result['query']==revised['query']
    for n,v in (('brief-before',result),('brief-after',revised),('brief-next',following),('records',records),('task',task)):
        (root/(n+'.json')).write_text(json.dumps(v,indent=2)+'\n')
    report={'passed':True,'distribution_version':'1.2.0','build':build,'package_path':gnomon.__file__,
            'python':sys.executable,'synthetic_forecasts':call_count,'synthetic_actual_rows':7,
            'provider_calls_during_review':len(calls)-call_count,'ledger_unchanged_during_review':True,
            'hidden_revision_preserved_same_evidence':True,'full_evidence_hash_verified':True,
            'custom_page_limit_preserved':True,'next_call_executed':True,
            'api_calls':0,'protected_access':False}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
