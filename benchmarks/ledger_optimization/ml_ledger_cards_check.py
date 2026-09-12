"""Executable synthetic adapter checks using either published ledger runtime."""
import argparse
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from ml_ledger_cards import incumbent_cards,compact_cards


def run(root):
    from gnomon import ForecastResult, InferenceEngine, TemporalLedger
    from gnomon.ids import FixedClock
    from gnomon.build_info import build_info
    root.mkdir(parents=True,exist_ok=False)
    first=datetime(2026,1,3,tzinfo=timezone.utc);db=TemporalLedger(root/'ledger.db',clock=FixedClock(first))
    engine=InferenceEngine(ledger=db);calls=[];records=[];queries=[]
    for name,value in [('a',2),('b',4),('c',3)]:
        def provider(r,name=name,value=value):
            calls.append(name)
            return ForecastResult(point=(value,)*r.horizon,series_id=r.series_id,unit=r.unit,timestamps=r.future_timestamps)
        engine.register(name,provider,revision='synthetic-v1:'+name,deterministic=True,lifecycle='stateless')
    for number in range(15):
        origin=first+timedelta(days=2*number);db.clock=FixedClock(origin)
        times=[(origin+timedelta(days=i)).isoformat() for i in (1,2)]
        request={'history':[1,1,1],'timestamps':[(origin-timedelta(days=i)).isoformat() for i in (2,1,0)],
                 'future_timestamps':times,'horizon':2,'series_id':'synthetic_sales','unit':'widgets',
                 'cutoff':origin.isoformat(),'known_time_cutoff':origin.isoformat(),'frequency':'D'}
        executions=[]
        for name in (['a','b'] if number<11 else ['b','c']):
            execution=engine.forecast(name,request).to_dict();executions.append(execution['execution_id'])
            records.append({'event':'result','kind':'forecast','request':request,'config_id':name,
                            'config':{'synthetic_provider':name},'execution':execution})
        db.clock=FixedClock(origin+timedelta(days=2))
        db.append_actual(actuals=[{'series_id':'synthetic_sales','unit':'widgets','value':1,
                                  'valid_time':t,'source_available_at':t} for t in times])
        db.evaluate(execution_ids=executions,source_as_of=times[-1],recorded_as_of=times[-1],allow_partial=False)
    now=(first+timedelta(days=31)).isoformat()
    task={'series_id':'synthetic_sales','unit':'widgets','horizon':2,'origin':now}
    provider_count=len(calls);before=hashlib.sha256((root/'ledger.db').read_bytes()).hexdigest()
    card=incumbent_cards(db,records,task);assert card['total_pairs']==3
    indexed={tuple(sorted((c['left_config_id'],c['right_config_id']))):c for c in card['cards']}
    assert indexed['a','b']['windows']['lifetime']['matched_origins']==11
    assert indexed['a','b']['windows']['last_12_origins']['matched_origins']==8
    assert indexed['a','b']['windows']['last_4_origins']['matched_origins']==0
    assert indexed['b','c']['windows']['last_4_origins']['matched_origins']==4
    assert indexed['a','c']['windows']['lifetime']['matched_origins']==0
    assert card['cards'][0]['left_config_id']=='b' and card['cards'][0]['right_config_id']=='c'
    checks=['pair cohorts survive empty three-model intersection','global recent windows, not last successful matches',
            'empty pair disclosed','score-independent recency ordering']
    # Independently ask the public ledger for each exact window and compare.
    for entry in card['cards']:
        for label,count in [('last_4_origins',4),('last_12_origins',12),('lifetime',15)]:
            query={'series_id':'synthetic_sales','horizon':2,'unit':'widgets','providers':entry['providers'],
                   'start':(first+timedelta(days=2*(15-count))).isoformat(),
                   'end':(first+timedelta(days=28)).isoformat(),'source_as_of':now,'recorded_as_of':now}
            q=db.compare_history(**query);queries.append({'arguments':query,'result':q})
            window=entry['windows'][label]
            assert window['matched_origins']==q['matched_origins'] and window['n']==q['n']
            assert window['models']==q['models'],(label,window,q)
    checks.append('all nine pair/window summaries equal independent public queries')
    pages=[incumbent_cards(db,records,task,offset=i,limit=1) for i in range(3)]
    assert [p['cards'][0] for p in pages]==card['cards']
    assert [p['next_offset'] for p in pages]==[1,2,None]
    assert incumbent_cards(db,records,task,pair=['a','b'])['cards']==[indexed['a','b']]
    checks.append('exact pagination and explicit pair retrieval')
    compact=compact_cards(card,'evidence.json')
    assert all('origins' not in w for c in compact['cards'] for w in c['windows'].values())
    for index,entry in enumerate(compact['cards']):
        for name,w in entry['windows'].items():
            assert w['models']==card['cards'][index]['windows'][name]['models']
            target=card
            for key in w['origins_reference'].split('#/')[1].split('/'):
                target=target[int(key)] if isinstance(target,list) else target[key]
            assert target==card['cards'][index]['windows'][name]['origins']
    checks.append('compact summaries retain exact metrics and resolvable evidence references')
    assert len(calls)==provider_count
    assert hashlib.sha256((root/'ledger.db').read_bytes()).hexdigest()==before
    checks.append('zero provider calls and byte-identical ledger during all reviews')
    # Keep future actual revisions invisible despite the physically present row.
    db.clock=FixedClock(first+timedelta(days=40))
    db.append_actual(series_id='synthetic_sales',unit='widgets',value=999,
                     valid_time=(first+timedelta(days=1)).isoformat(),
                     source_available_at=(first+timedelta(days=40)).isoformat())
    assert incumbent_cards(db,records,task)==card
    checks.append('future-recorded/source-available revision invisible at query cutoff')
    wrong=deepcopy(records);wrong[1]['execution']['provider']='a'
    try:incumbent_cards(db,wrong,task)
    except ValueError:pass
    else:raise AssertionError('Conflicting identity accepted')
    checks.append('conflicting provider identity rejected')
    for mode in ('backtest','future'):
        extra=deepcopy(records[0])
        if mode=='backtest':extra['kind']='backtest'
        else:
            extra['request']['cutoff']=now
            extra['request']['future_timestamps']=[(datetime.fromisoformat(now)+timedelta(days=i)).isoformat() for i in (1,2)]
        extended=incumbent_cards(db,records+[extra],task)
        assert extended['cards']==card['cards']
    checks.append('retrospective and unclosed forecasts cannot alter production cards')
    result={'passed':True,'checks':checks,'distribution_version':importlib.metadata.version('gnomon-forecast'),
            'build':build_info(),'python':sys.executable,'card':card,'direct_queries':queries,
            'synthetic_provider_calls':provider_count,'provider_calls_during_review':len(calls)-provider_count,
            'api_calls':0,'final_data_opened':False}
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    result=run(p.parse_args().output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('card','direct_queries')},indent=2))
