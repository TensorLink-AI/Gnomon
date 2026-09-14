"""Synthetic two-origin, real-backend probe through guarded pinned Hermes dispatch.

No dataset/holdout or API access. A host-installed observer counts entry into the
actual numerical.predict implementation independently of core.execute's counter.
Observer source is in the protected manifest; it is test instrumentation only.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from .execution_boundary_093 import LabBoundary
from .hermes_boundary_093 import HermesBoundary, bounded_agent_class, native_callbacks


PROJECT_FILES=('lab.py','core.py','numerical.py','check_forecast.py','policy.py','maturation.py',
               'ledger_cards.py','dev_evidence_summary.py','agent_review.py','visible_agent_review.py',
               'ml_ledger_cards.py','history_091.py','review_bridge.py')
OBSERVER='''import json,hashlib,numerical
from pathlib import Path
def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def install():
    original=numerical.predict
    def measured(request,config):
        row={'origin':json.loads(Path('task.json').read_text())['origin'],
             'request_sha256':fingerprint(request),'config':config}
        with Path('observed-fits.jsonl').open('a') as f:f.write(json.dumps({'event':'begin',**row})+'\\n')
        result=original(request,config)
        with Path('observed-fits.jsonl').open('a') as f:f.write(json.dumps({'event':'end',**row})+'\\n')
        return result
    numerical.predict=measured
'''


class ObservedBoundary(LabBoundary):
    BOOTSTRAP=("import runpy,sys; p=sys.argv.pop(1); sys.path.insert(0,p); "
               "import fit_observer; fit_observer.install(); sys.argv[0]=p+'/lab.py'; "
               "runpy.run_path(sys.argv[0],run_name='__main__')")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def rows(path):return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['plain','gnomon','ledger'],required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--lab-source',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    work=root/'work';work.mkdir();home=root/'home';home.mkdir()
    os.environ['HERMES_HOME']=str(home)
    dump(home/'config.yaml',{'memory':{'memory_enabled':True,'user_profile_enabled':True},'skills':{'external_dirs':[]}})
    for name in PROJECT_FILES:shutil.copyfile(args.lab_source/name,work/name)
    (work/'fit_observer.py').write_text(OBSERVER)
    available=importlib.util.find_spec('gnomon') is not None
    assert available == (args.arm!='plain')
    if available:assert importlib.metadata.version('gnomon-forecast')=='1.2.0'
    dates=[(datetime(2020,1,1,tzinfo=timezone.utc)+timedelta(days=i)).isoformat() for i in range(758)]
    values=[float(10+i%7+3*(i%11==0)) for i in range(758)]
    events=[];checks=[];blocked=[]
    def deny(*a,**kw):blocked.append('connection');raise RuntimeError('No network in synthetic backend probe')
    def check(label,condition):
        checks.append({'assertion':label,'passed':bool(condition)})
        if not condition:raise AssertionError(label)
    def prepare(offset,prior):
        with (work/'history.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['timestamp','value','promo'])
            w.writerows((dates[i],values[i],int(i%11==0)) for i in range(offset,offset+730))
        with (work/'future.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['timestamp','promo'])
            w.writerows((dates[i],int(i%11==0)) for i in range(offset+730,offset+744))
        dump(work/'task.json',{'series_id':'synthetic-retail','unit':'widgets','horizon':14,'origin':dates[offset+729],
                              'round':offset//14,'future_timestamps':dates[offset+730:offset+744]})
        dump(work/'backend.json',{'arm':args.arm});dump(work/'previous_runs.json',prior)
        dump(work/'agent-budget.json',{'forwarded_requests':1,'remaining_requests':15,'deadline_epoch':time.time()+1200})
        for name in ('forecast.json','execution.json','decision.json','checkpoint.json'):(work/name).unlink(missing_ok=True)
        protected=(*PROJECT_FILES,'fit_observer.py','history.csv','future.csv','task.json','backend.json','previous_runs.json')
        return {n:sha(work/n) for n in protected}
    try:
        with patch.object(socket.socket,'connect',deny),patch.object(socket,'create_connection',deny):
            from run_agent import AIAgent
            from tools.memory_tool import MemoryStore
            cls=bounded_agent_class(AIAgent)
            def make_agent(manifest):
                agent=cls.__new__(cls);agent._flush_messages_to_session_db=None
                agent._memory_store=MemoryStore();agent._memory_store.load_from_disk()
                agent.execution_boundary=HermesBoundary(ObservedBoundary(work,sys.executable,manifest),
                    native=native_callbacks(agent),record=events.append,deadline=time.time()+1200)
                return agent
            agent=make_agent(prepare(0,[]));counter=0
            def invoke(tool,arguments):
                nonlocal counter
                counter+=1;ident=f'{args.arm}-{counter}'
                call=SimpleNamespace(id=ident,function=SimpleNamespace(name=tool,arguments=json.dumps(arguments)))
                messages=[];agent._uniquify_tool_call_ids([call])
                agent._execute_tool_calls(SimpleNamespace(tool_calls=[call]),messages,'synthetic')
                assert messages[0]['tool_call_id']==ident
                return json.loads(messages[0]['content'])
            def lab(operation,expect=0,**kwargs):
                outer=invoke('lab',{'operation':operation,**kwargs});assert outer['status']=='ok',outer
                r=outer['result'];assert r['exit_code']==expect,r
                assert not r['stderr'],r
                return json.loads(r['stdout'])
            config={'model':'ridge','window':90,'lags':7,'alpha':10}
            check('start performs four actual fits',lab('start')['numerical_calls_started']==4)
            baseline=(work/'checkpoint.json').read_bytes()
            check('start retry performs no fits',lab('start')['numerical_calls_started']==0)
            check('backtest performs three fits',lab('backtest',config=config)['numerical_calls_started']==3)
            check('backtest keeps checkpoint',baseline==(work/'checkpoint.json').read_bytes())
            check('commit performs one fit',lab('commit',config=config)['numerical_calls_started']==1)
            check('explicit workflow completed',lab('status')['result']['checkpoint']['selection_after_comparison'])
            check('repeat commit reuses execution',lab('commit',config=config)['numerical_calls_started']==0)
            rejected=invoke('terminal',{'command':'python -c "from numerical import predict; predict({}, {})"'})
            check('direct numerical path blocked',rejected['status']=='error' and not rejected['execution_started'])
            for i in range(17):
                config={**config,'alpha':100+i}
                assert lab('backtest',config=config)['numerical_calls_started']==3
            check('59 attempts before final reserve',lab('status')['budget']['numerical_attempts']==59)
            check('excess batch rejected before work',lab('backtest',expect=2,config={**config,'alpha':500})['numerical_calls_started']==0)
            check('final reserve usable',lab('commit',config=config)['numerical_calls_started']==1)
            check('no work beyond 60',lab('backtest',expect=2,config={**config,'alpha':501})['numerical_calls_started']==0)
            recorded=rows(work/'experiments.jsonl');observed=rows(work/'observed-fits.jsonl')
            attempts=[r for r in recorded if r['event']=='attempt'];results=[r for r in recorded if r['event']=='result']
            begins=[r for r in observed if r['event']=='begin'];ends=[r for r in observed if r['event']=='end']
            check('60 counted attempts equal 60 real entries and returns',len(attempts)==len(results)==len(begins)==len(ends)==60)
            check('all actual fit identities match counted attempts',all(
                o['request_sha256']==fingerprint(a['request']) and o['config']==a['config'] and o['origin']==a['task_origin']
                for o,a in zip(begins,attempts,strict=True)))
            check('all actual fits returned',all({k:v for k,v in a.items() if k!='event'}=={k:v for k,v in b.items() if k!='event'}
                                               for a,b in zip(begins,ends,strict=True)))
            remembered=invoke('memory',{'target':'memory','action':'add','content':'Synthetic origin one: a ridge comparison was completed.'})
            check('native memory write',json.loads(remembered['result'])['success'])
            checkpoint=json.loads((work/'checkpoint.json').read_text())
            prior=[{'origin':dates[729],'outcome_recorded_at':dates[743],'actual':values[730:744],
                    'future_timestamps':dates[730:744],'series_id':'synthetic-retail','unit':'widgets',
                    'execution_id':checkpoint['execution_id']}]
            evidence_before=(work/'experiments.jsonl').read_bytes()
            agent=make_agent(prepare(14,prior))
            check('native memory survives next origin','Synthetic origin one' in agent._memory_store.format_for_system_prompt('memory'))
            synced=lab('sync')
            check('three production alternatives mature',len(synced['result']['matured_execution_ids'])==3)
            check('sync does not fit',synced['numerical_calls_started']==0 and synced['result']['provider_calls']==0)
            check('historical evidence immutable',(work/'experiments.jsonl').read_bytes().startswith(evidence_before))
            check('sync retry adds nothing',lab('sync')['result']['matured_execution_ids']==[])
            review=lab('review')['result']
            if args.arm=='ledger':
                check('ledger review has matched evidence',review['record_count']==3 and bool(review['cards']))
                card=review['cards'][0]
                dump(root/'first-card.json',card)
            check('review and maturation added no numerical invocations',len(rows(work/'observed-fits.jsonl'))==120)
            check('new origin has no selected forecast',lab('status')['result']['checkpoint'] is None)
            check('no network attempts',not blocked)
            dump(root/'passed.json',{'arm':args.arm,'passed':True,'checks':checks,'engy_calls':0,
                 'actual_model_invocations':60,'runtime_python':sys.executable,'runtime_prefix':sys.prefix,
                 'gnomon_version':importlib.metadata.version('gnomon-forecast') if available else None,
                 'first_origin_results':results,'scope':'Synthetic guarded dispatch with real models and ledger; no accuracy claim.'})
    finally:
        dump(root/'probe-evidence.json',{'checks':checks,'events':events,'blocked_network':blocked,
                                      'source_hashes':{n:sha(work/n) for n in (*PROJECT_FILES,'fit_observer.py')}})


if __name__=='__main__':main()
