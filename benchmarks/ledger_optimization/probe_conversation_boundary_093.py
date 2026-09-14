"""Actual pinned Hermes conversation against scripted loopback responses, no Engy.

This verifies schema advertisement and dispatch/persistence through run_conversation.
It does not measure agent performance or exercise the real forecasting providers.
"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from unittest.mock import patch

from benchmarks.ledger_optimization.execution_boundary_093 import LabBoundary
from benchmarks.ledger_optimization.hermes_boundary_093 import HermesBoundary, bounded_agent_class, native_callbacks
from benchmarks.ledger_optimization.boundary_schemas_093 import attach


def main():
    home=Path(os.environ['HERMES_HOME'])
    if home.exists():raise RuntimeError('Use a fresh nonexistent HERMES_HOME.')
    home.mkdir(parents=True)
    (home/'config.yaml').write_text('memory:\n  memory_enabled: true\n  user_profile_enabled: true\nskills:\n  external_dirs: []\n')
    requests=[];discovery_requests=[];blocked=[];events=[]
    def tc(ident,name,raw):return {'id':ident,'type':'function','function':{'name':name,'arguments':raw}}
    script=[{'role':'assistant','content':None,'tool_calls':[
        tc('status','lab','{"operation":"status"}'),
        tc('bypass','terminal','{"command":"python local_probe.py"}'),
        tc('duplicate','lab','{"operation":"start","operation":"status"}')
    ]}, {'role':'assistant','content':None,'tool_calls':[
        tc('remember','memory','{"target":"memory","action":"add","content":"Synthetic conversation lesson: compare recorded origins."}'),
        tc('list','project_list','{}')
    ]}, {'role':'assistant','content':'Synthetic probe complete. No forecast was selected.'}]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if 'messages' not in request:
                discovery_requests.append({'path':self.path,'body':request})
                self.send_response(404);self.send_header('Content-Length','2');self.end_headers();self.wfile.write(b'{}')
                return
            requests.append(request)
            if len(requests)>len(script):
                self.send_response(500);self.end_headers();return
            message=script[len(requests)-1]
            body=json.dumps({'id':f'local-{len(requests)}','object':'chat.completion','created':0,
                             'model':'deepseek-v4.1-flash','choices':[{'index':0,'message':message,
                             'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop'}],
                             'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    original=socket.socket.connect
    def connect(sock,address):
        if isinstance(address,tuple) and address[0]=='127.0.0.1' and address[1]==server.server_port:
            return original(sock,address)
        blocked.append(str(address));raise RuntimeError('Only the scripted loopback provider is available.')
    checks=[]
    def check(label,value):
        checks.append({'assertion':label,'passed':bool(value)})
        if not value:raise AssertionError(label)
    try:
        with patch.object(socket.socket,'connect',connect), tempfile.TemporaryDirectory() as directory:
            from run_agent import AIAgent
            from tools.memory_tool import MemoryStore
            work=Path(directory)
            (work/'lab.py').write_text('print(\'{"status":"ok","provider_calls":0}\')\n')
            lab=LabBoundary(work,sys.executable,{'lab.py':hashlib.sha256((work/'lab.py').read_bytes()).hexdigest()})
            agent=bounded_agent_class(AIAgent)(
                model='deepseek-v4.1-flash',provider='custom',api_mode='chat_completions',
                base_url=f'http://127.0.0.1:{server.server_port}/v1',api_key='synthetic-only',
                max_iterations=4,max_tokens=3072,request_overrides={'temperature':0.2,'seed':7},
                enabled_toolsets=['memory','skills'],quiet_mode=True,skip_context_files=True,
                load_soul_identity=False,skip_background_review=True,skip_memory=False,run_budget_seconds=60)
            agent._disable_streaming=True
            boundary=HermesBoundary(lab,native=native_callbacks(agent),record=events.append,deadline=time.time()+60)
            attach(agent,boundary)
            result=agent.run_conversation(user_message='Run the synthetic tool integration probe. This is not a forecasting task.')
            check('three scripted responses consumed',len(requests)==3)
            advertised={t['function']['name'] for t in requests[0]['tools']}
            check('exactly nine permitted tools',advertised=={'lab','evidence_read','project_list','notes_write','data_summary',
                                                           'memory','skills_list','skill_view','skill_manage'})
            messages=requests[-1]['messages']
            returned={m.get('tool_call_id'):m['content'] for m in messages if m.get('role')=='tool'}
            check('all five calls paired with results',set(returned)=={'status','bypass','duplicate','remember','list'})
            check('duplicate fields rejected',json.loads(returned['duplicate'])['status']=='error')
            check('unknown terminal rejected','does not exist' in returned['bypass'].lower())
            check('two admitted boundary tools',sum(e.get('admitted',False) for e in lab.events)==2)
            # Two admitted lab-boundary tools: lab/status and pure project_list; only lab runs Python.
            check('one lab operation',sum(e['tool']=='lab' and e['admitted'] for e in lab.events)==1)
            fresh=MemoryStore();fresh.load_from_disk()
            check('conversation memory survived','Synthetic conversation lesson' in fresh.format_for_system_prompt('memory'))
            check('no persistence failure',not getattr(agent,'_incremental_persistence_failed',False))
            check('no nonlocal network attempt',not blocked)
            receipt={'checks':checks,'requests':requests,'discovery_requests':discovery_requests,'events':events,'result':result,'engy_calls':0,
                     'scripted_responses':len(requests),'blocked_network':blocked,
                     'scope':'Pinned Hermes full conversation; synthetic tools/provider, no performance result.'}
            (home/'probe-result.json').write_text(json.dumps(receipt,default=str,indent=2))
            print(json.dumps({'passed':True,'checks':len(checks),'receipt':str(home/'probe-result.json'),'engy_calls':0}))
    finally:
        server.shutdown();server.server_close()
        (home/'probe-transcript.json').write_text(json.dumps({'requests':requests,'discovery_requests':discovery_requests,'events':events,'checks':checks,
                                                           'blocked_network':blocked},default=str,indent=2))


if __name__=='__main__':main()
