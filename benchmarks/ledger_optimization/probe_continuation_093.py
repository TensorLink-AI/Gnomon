"""Synthetic four-origin, three-arm continuation on the frozen pod runtime.

Uses actual workers/models/ledger/native memory with scripted upstream replies.
Never reads Engy credentials, development targets, or the reserved final set.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import urllib.request
from unittest.mock import patch

from benchmarks.ledger_optimization import continue_guarded_093 as c
from benchmarks.hermes_ml_checkpoint_v6 import service_admission


def synthetic_jobs():
    times=[(datetime(2020,1,1,tzinfo=timezone.utc)+timedelta(days=i)).isoformat() for i in range(800)]
    values=[float(10+i%7+3*(i%11==0)) for i in range(800)]
    result=[]
    for n in range(4):
        begin=n*14;end=begin+730;future=times[end:end+14]
        result.append({'series_id':'synthetic-continuation','round':n,'origin':times[end-1],
            'future_timestamps':future,'outcome_recorded_at':future[-1],'actual':values[end:end+14],
            'request':{'history':values[begin:end],'timestamps':times[begin:end],
                'future_timestamps':future,'past_covariate_names':['promo'],'future_covariate_names':['promo'],
                'past_covariates':[[float(i%11==0)] for i in range(begin,end)],
                'future_covariates':[[float(i%11==0)] for i in range(end,end+14)],'unit':'widgets'}})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    source={p.name:c.digest(p) for p in c.run.HERE.iterdir() if p.is_file()}
    tested=c.digest(Path(c.__file__));runtime=c.run.runtime_inventory();jobs=synthetic_jobs()
    series='synthetic-continuation';pilot=root/'pilot';pilot.mkdir()
    c.run.dump(pilot/'manifest.json',{'planned':9});c.run.dump(pilot/'host-jobs.json',{series:jobs[:3]})
    runtimes={a:c.run.OTHER/('plain-venv' if a=='plain' else 'gnomon-venv')/'bin/python' for a in c.run.ARMS}
    config={'model':'ridge','window':90,'lags':7,'alpha':10};current={};wire=[];checks=[]
    def check(name,value):
        checks.append({'assertion':name,'passed':bool(value)})
        if not value:raise AssertionError(name)
    def call(name,args):
        return {'id':f'{name}-{current["n"]}-{len(wire)}','type':'function',
            'function':{'name':name,'arguments':json.dumps(args)}}
    class Response:
        status=200
        def __init__(self,body):self.body=body
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(self.body).encode()
    def upstream(request,**kwargs):
        if request.full_url!='https://api.engy.ai/v1/chat/completions':
            raise AssertionError('Unexpected URL, no real network access permitted')
        payload=json.loads(request.data);current['n']+=1;n=current['n']
        wire.append({'arm':current['arm'],'round':current['round'],'request':payload,'synthetic':True})
        if n>4:raise AssertionError('Unexpected extra model request')
        operations={1:[('lab',{'operation':'review'}),('lab',{'operation':'start'})],
            2:[('lab',{'operation':'backtest','config':config}),('memory',{'target':'memory','action':'add',
                'content':f'Synthetic continuation origin {current["round"]}: compared seasonal and ridge.'})],
            3:[('lab',{'operation':'commit','config':config}),('notes_write',{'path':'decision.json',
                'text':'{"rationale":"synthetic continuation check"}'})],4:[]}[n]
        # Every call in a batch needs a unique ID, even two uses of lab.
        tools=[]
        for index,(name,arguments) in enumerate(operations):
            item=call(name,arguments);item['id']+=f'-{index}';tools.append(item)
        message={'role':'assistant','content':None if tools else 'Synthetic workflow complete.'}
        if tools:message['tool_calls']=tools
        return Response({'id':f'synthetic-{len(wire)}','object':'chat.completion','created':0,
            'model':'deepseek-v4.1-flash','choices':[{'index':0,'message':message,
                'finish_reason':'tool_calls' if tools else 'stop'}],
            'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}})
    def ready(_):
        return 200,b'{"choices":[{"message":{"content":"READY"}}],"usage":{"total_tokens":0},"synthetic":true}',0.0
    original_execute=c.run.execute
    def execute(*args):
        current.update(arm=args[1],round=args[3]['round'],n=0)
        result=original_execute(*args)
        check(f'{args[1]} origin {args[3]["round"]}: full workflow',result['valid'] and result['workflow_complete'])
        check(f'{args[1]} origin {args[3]["round"]}: bounded calls',result['api_calls']==4 and result['numerical_attempts']==8)
        return result
    try:
        with patch.object(urllib.request,'urlopen',upstream),patch.object(service_admission,'probe',ready),\
             patch.object(c.run,'key',side_effect=AssertionError('No credentials in synthetic preflight')),\
             patch.object(c.run,'execute',execute):
            prior={a:[] for a in c.run.ARMS}
            for job in jobs[:3]:
                offset=job['round']%3
                for arm in c.run.ARMS[offset:]+c.run.ARMS[:offset]:
                    c.run.execute(pilot,arm,series,job,prior[arm],runtimes[arm],'synthetic-only')
            first=c.analyze(pilot)
            check('prefix independent audit',first['complete'] and not first['audit_failures'])
            c.run.dump(pilot/'complete.json',{'completed':9})
            before=c.inventory(pilot);resumed=root/'resumed';c.copy_prefix(pilot,resumed,before)
            c.run.dump(resumed/'manifest.json',{'planned':12});c.run.dump(resumed/'host-jobs.json',{series:jobs})
            result=c.continue_chain(0,series,jobs,resumed,runtimes,'synthetic-only')
            check('only three additional sessions',len(result)==3 and all(r['round']==3 for r in result))
            report=c.analyze(resumed)
            check('resumed independent audit',report['complete'] and not report['audit_failures'])
            check('original prefix unchanged',c.inventory(pilot)==before)
            for arm in c.run.ARMS:
                check(arm+': prefix raw records unchanged',all(
                    c.digest(p)==c.digest(resumed/p.relative_to(pilot))
                    for p in (pilot/arm/series).glob('round-*/*') if p.is_file()))
                check(arm+': memory restored into resumed model request',any(
                    'Synthetic continuation origin 2' in json.dumps(w['request']['messages'])
                    for w in wire if w['arm']==arm and w['round']==3))
                check(arm+': native memory remains saved',bool(c.read(resumed/arm/series/'round-3/memory.json')))
            check('48 scripted responses',len(wire)==48)
            check('runtime inventory unchanged',c.run.runtime_inventory()==runtime)
            check('frozen worker sources unchanged',{p.name:c.digest(p) for p in c.run.HERE.iterdir() if p.is_file()}==source)
            check('continuation source unchanged',c.digest(Path(c.__file__))==tested)
            c.run.dump(root/'passed.json',{'passed':True,'engy_calls':0,'checks':checks,
                'continuation_source_sha256':tested,'probe_source_sha256':c.digest(Path(__file__)),
                'frozen_sources':source,'resumed_arms':list(c.run.ARMS),'synthetic_responses':len(wire),
                'numerical_attempts':96,'independent_audit_checks':report['audit_checks'],
                'scope':'Synthetic full workers, state-copy, native memory, ledger, temporal maturation and guarded execution. Not live availability or accuracy evidence.'})
    finally:
        c.run.dump(root/'probe-evidence.json',{'checks':checks,'wire':wire,'engy_calls':0})


if __name__=='__main__':main()
