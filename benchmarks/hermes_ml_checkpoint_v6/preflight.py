"""Synthetic host/transport/worker/analyzer preflight; never opens Engy credentials."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request
from unittest.mock import patch

from . import run, service_admission
from .analyze import analyze
from .transport import dump, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    sources={p.name:sha(p) for p in run.HERE.iterdir() if p.is_file()}
    with (args.fixture/'history.csv').open() as f:history=list(csv.DictReader(f))
    with (args.fixture/'future.csv').open() as f:future=list(csv.DictReader(f))
    task=json.loads((args.fixture/'task.json').read_text())
    times=[r['timestamp'] for r in future]
    req={'history':[float(r['value']) for r in history],'timestamps':[r['timestamp'] for r in history],
         'future_timestamps':times,'past_covariate_names':['promo'],'future_covariate_names':['promo'],
         'past_covariates':[[float(r['promo'])] for r in history],
         'future_covariates':[[float(r['promo'])] for r in future],'unit':task['unit']}
    # Synthetic-only generator from the preceding backend probe; never real holdout data.
    indices=[(datetime.fromisoformat(t)-datetime(2020,1,1,tzinfo=timezone.utc)).days for t in times]
    job={'request':req,'origin':task['origin'],'series_id':'synthetic-host','round':0,
         'future_timestamps':times,'outcome_recorded_at':times[-1],
         'actual':[float(10+i%7+3*(i%11==0)) for i in indices]}
    cases=root/'cases';cases.mkdir();dump(cases/'manifest.json',{'planned':3})
    dump(cases/'host-jobs.json',{'synthetic-host':[job]})
    inventory=run.runtime_inventory();dump(root/'runtime-inventory.json',inventory)
    runtimes={a:run.OTHER/('plain-venv' if a=='plain' else 'gnomon-venv')/'bin/python' for a in run.ARMS}
    config={'model':'ridge','window':90,'lags':7,'alpha':10}
    def tool(i,name,args):return {'id':i,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}
    scripts=[[
        tool('review','lab',{'operation':'review'}),tool('start','lab',{'operation':'start'})],[
        tool('backtest','lab',{'operation':'backtest','config':config}),
        tool('memory','memory',{'target':'memory','action':'add','content':'Synthetic host trial: baseline and ridge compared.'})],[
        tool('commit','lab',{'operation':'commit','config':config}),
        tool('note','notes_write',{'path':'decision.json','text':'{"rationale":"scripted host test"}'})],None]
    wire=[];checks=[];current=[None,0]
    class Response:
        status=200
        def __init__(self,body):self.body=body
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(self.body).encode()
    def upstream(request,**kwargs):
        assert request.full_url=='https://api.engy.ai/v1/chat/completions'
        payload=json.loads(request.data);current[1]+=1;n=current[1]
        wire.append({'arm':current[0],'request':payload,'synthetic':True})
        assert n<=4,'Unexpected extra model request'
        message={'role':'assistant','content':'Synthetic workflow complete.' if scripts[n-1] is None else None}
        if scripts[n-1]:message['tool_calls']=scripts[n-1]
        return Response({'id':f'synthetic-{n}','object':'chat.completion','created':0,'model':'deepseek-v4.1-flash',
            'choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if scripts[n-1] else 'stop'}],
            'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}})
    def ready(_):
        return 200,json.dumps({'choices':[{'message':{'content':'READY'}}],
                               'usage':{'total_tokens':0},'synthetic':True}).encode(),0.0
    def check(label,value):
        checks.append({'assertion':label,'passed':bool(value)})
        if not value:raise AssertionError(label)
    try:
        with patch.object(urllib.request,'urlopen',upstream),patch.object(service_admission,'probe',ready):
            for arm in run.ARMS:
                current[:]=[arm,0]
                value=run.execute(cases,arm,'synthetic-host',job,[],runtimes[arm],'synthetic-preflight-only')
                check(arm+': host grades full workflow',value['valid'] and value['workflow_complete'] and not value['fallback_used'])
                check(arm+': eight numerical attempts, four API calls',value['numerical_attempts']==8 and value['api_calls']==4)
                check(arm+': one real worker, no correction',value['exit_code']==0 and value['corrections']==0)
                check(arm+': complete synthetic usage records',value['usage_complete'] and value['responses']==4)
        analyze(cases)
        report=json.loads((cases/'report.json').read_text())
        check('independent analyzer passes all arms',report['complete'] and not report['audit_failures'])
        check('runtime package parity unchanged',run.runtime_inventory()==inventory)
        check('all frozen sources unchanged',{p.name:sha(p) for p in run.HERE.iterdir() if p.is_file()}==sources)
        dump(root/'passed.json',{'passed':True,'checks':checks,'tested_sources':sources,
            'engy_calls':0,'synthetic_upstream_responses':len(wire),'numerical_attempts':24,
            'scope':'Actual host/worker/transport/analyzer with synthetic model and service responses; not live provider availability or efficacy.'})
    finally:
        dump(root/'preflight-evidence.json',{'checks':checks,'synthetic_upstream_requests':wire,'engy_calls':0})


if __name__=='__main__':main()
