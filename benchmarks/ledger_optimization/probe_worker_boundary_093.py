"""Run the prospective worker against scripted local responses and real models."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import socket
import threading
import time
from unittest.mock import patch

from .probe_backends_boundary_093 import PROJECT_FILES, dump, rows, sha
from .worker_093 import run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    work=root/'work';work.mkdir();out=root/'output';out.mkdir();home=root/'home';home.mkdir()
    os.environ['HERMES_HOME']=str(home)
    dump(home/'config.yaml',{'compression':{'enabled':False},'memory':{'memory_enabled':True,'user_profile_enabled':True},
                           'skills':{'external_dirs':[]},'mcp_servers':{}})
    files=(*PROJECT_FILES,'task.json','history.csv','future.csv','backend.json')
    for name in files:shutil.copyfile(args.fixture/name,work/name)
    dump(work/'previous_runs.json',[])
    shutil.copyfile(Path(__file__).with_name('TASK_093.md'),work/'TASK.md')
    dump(root/'manifest.json',{name:sha(work/name) for name in (*files,'previous_runs.json','TASK.md')})
    deadline=time.time()+480
    dump(work/'agent-budget.json',{'forwarded_requests':0,'remaining_requests':16,'deadline_epoch':deadline})
    config={'model':'ridge','window':90,'lags':7,'alpha':10}
    def tool(ident,name,args):return {'id':ident,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}
    script=[[
        tool('review','lab',{'operation':'review'}),tool('start','lab',{'operation':'start'})],[
        tool('backtest','lab',{'operation':'backtest','config':config}),
        tool('memory','memory',{'target':'memory','action':'add','content':'Synthetic worker: compared a ridge and a baseline.'})],[
        tool('commit','lab',{'operation':'commit','config':config}),
        tool('note','notes_write',{'path':'decision.json','text':json.dumps({'rationale':'Scripted integration test; no performance claim.'})})],None]
    requests=[];discovery=[];blocked=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if 'messages' not in request:
                discovery.append({'path':self.path,'body':request});self.send_response(404);self.end_headers();return
            requests.append(request);i=len(requests)
            dump(work/'agent-budget.json',{'forwarded_requests':i,'remaining_requests':16-i,'deadline_epoch':deadline})
            if i>len(script):self.send_response(500);self.end_headers();return
            message={'role':'assistant','content':'Synthetic workflow completed.' if script[i-1] is None else None}
            if script[i-1]:message['tool_calls']=script[i-1]
            body=json.dumps({'id':f'synthetic-{i}','object':'chat.completion','created':0,'model':'deepseek-v4.1-flash',
                'choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if script[i-1] else 'stop'}],
                'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    original=socket.socket.connect
    def connect(sock,address):
        if isinstance(address,tuple) and address[0]=='127.0.0.1' and address[1]==server.server_port:return original(sock,address)
        blocked.append(str(address));raise RuntimeError('Only scripted loopback provider is available')
    checks=[]
    def check(label,value):
        checks.append({'assertion':label,'passed':bool(value)})
        if not value:raise AssertionError(label)
    try:
        with patch.object(socket.socket,'connect',connect):
            result=run(work,out,f'http://127.0.0.1:{server.server_port}/v1',root/'manifest.json')
        check('full worker completed',result['stop_reason']=='checkpoint_workflow_complete')
        check('one conversation, no corrections',result['attempts']==1 and result['corrections']==0)
        check('four scripted chat responses',len(requests)==4)
        check('budget retained exact request count',result['final_budget']['forwarded_requests']==4 and result['final_budget']['remaining_requests']==12)
        records=rows(work/'experiments.jsonl')
        check('eight numerical attempts',sum(r['event']=='attempt' for r in records)==8)
        check('eight numerical results',sum(r['event']=='result' for r in records)==8)
        checkpoint=json.loads((work/'checkpoint.json').read_text())
        check('checkpoint selected after comparison',checkpoint['selection_after_comparison'])
        check('decision summary saved',(work/'decision.json').exists())
        check('native memory saved',any('Synthetic worker' in p.read_text() for p in (home/'memories').glob('*.md')))
        check('no nonlocal network attempts',not blocked)
        events=rows(out/'boundary-events.jsonl')
        check('all six model tool calls retained',sum(e['stage']=='returned' for e in events)==6)
        check('host completion check audited',sum(e['stage']=='host_completion_check' for e in events)==1)
        dump(root/'passed.json',{'passed':True,'checks':checks,'engy_calls':0,'numerical_attempts':8,
                               'scripted_chat_responses':4,'scope':'Full guarded worker with scripted choices, real models; no efficacy result.'})
    finally:
        server.shutdown();server.server_close()
        dump(root/'transcript.json',{'requests':requests,'discovery':discovery,'checks':checks,'blocked_network':blocked})


if __name__=='__main__':main()
