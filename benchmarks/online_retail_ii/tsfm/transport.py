"""Local metering proxy; real credentials remain outside the Hermes process."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from benchmarks.online_retail_ii.data import dump

MODEL='deepseek-v4.1-flash'
REQUEST_LIMIT=12
MAX_TOKENS=3072
SECONDS=480


class Proxy(ThreadingHTTPServer):
    daemon_threads=False
    def __init__(self,output,key,seed,deadline,stub=False):
        super().__init__(('127.0.0.1',0),Handler)
        self.output,self.key,self.seed,self.deadline=Path(output),key,seed,deadline
        self.calls=0;self.lock=threading.Lock();self.stub=stub;self.budget()
    def budget(self):
        temporary=self.output/'agent-budget.tmp'
        dump(temporary,{'remaining_requests':max(0,REQUEST_LIMIT-self.calls),
            'forwarded_requests':self.calls,'limit_requests':REQUEST_LIMIT,'deadline_epoch':self.deadline})
        temporary.replace(self.output/'agent-budget.json')


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_):pass
    def send(self,status,raw):
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw)));self.end_headers()
        try:self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        self.send(200,json.dumps({'object':'list','data':[{'id':MODEL,'object':'model'}]}).encode())
    def do_POST(self):
        size=int(self.headers.get('Content-Length',0))
        if not 0<size<=4_000_000:return self.send(413,b'{"error":{"message":"request too large"}}')
        try:payload=json.loads(self.rfile.read(size))
        except ValueError:return self.send(400,b'{"error":{"message":"invalid JSON"}}')
        if 'messages' not in payload:return self.send(404,b'{"error":{"message":"metadata not available"}}')
        server=self.server
        with server.lock:
            if server.calls>=REQUEST_LIMIT or time.time()>=server.deadline:
                dump(server.output/f'blocked-{time.time_ns()}.json',{'cause':'budget_exhausted','forwarded':False})
                return self.send(400,b'{"error":{"message":"budget exhausted; use saved typed execution"}}')
            server.calls+=1;number=server.calls;server.budget()
        prefix=server.output/f'api-{number:02d}'
        dump(str(prefix)+'-request.json',payload)
        if number>=10:
            payload['messages']=[*payload['messages'],{'role':'system','content':
                'Selection phase: select an existing execution_id with retail. If no forecast exists, execute one now. Do not spend remaining calls on further inspection.'}]
        payload.update(model=MODEL,temperature=.2,seed=server.seed,max_tokens=MAX_TOKENS,stream=False)
        for key in ('stream_options','max_completion_tokens'):payload.pop(key,None)
        dump(str(prefix)+'-forwarded.json',payload)
        started=time.monotonic()
        if server.stub:
            # Exercise real Hermes dispatch without network or credentials.
            if number==1:
                message={'role':'assistant','content':None,'tool_calls':[{'id':'memory-1','type':'function',
                    'function':{'name':'memory','arguments':json.dumps({'target':'memory','action':'add',
                    'content':'Retail preflight: select only a current task execution; historical observations are not current targets.'})}}]}
            elif number==2:
                message={'role':'assistant','content':None,'tool_calls':[{'id':'cv-1','type':'function',
                    'function':{'name':'retail','arguments':json.dumps({'operation':'cv'})}}]}
            elif number==3:
                message={'role':'assistant','content':None,'tool_calls':[{'id':'forecast-1','type':'function',
                    'function':{'name':'retail','arguments':json.dumps({'operation':'forecast','provider':'paracast_route' if server.seed==7 else 'paracast_ensemble2'})}}]}
            elif number==4:
                tool=next(m for m in reversed(payload['messages']) if m['role']=='tool')
                eid=json.loads(tool['content'])['result']['execution_id']
                message={'role':'assistant','content':None,'tool_calls':[{'id':'select-1','type':'function',
                    'function':{'name':'retail','arguments':json.dumps({'operation':'select','execution_id':eid})}}]}
            else:message={'role':'assistant','content':'Selected the saved executed forecast.'}
            raw=json.dumps({'id':'stub','object':'chat.completion','model':MODEL,
                'choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if number<5 else 'stop'}],
                'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}).encode();status=200
        else:
            request=urllib.request.Request('https://api.engy.ai/v1/chat/completions',
                data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+server.key,'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(request,timeout=min(120,max(.01,server.deadline-time.time()))) as response:
                    raw,status=response.read(),response.status
            except urllib.error.HTTPError as exc:raw,status=exc.read(),exc.code
            except Exception as exc:raw,status=json.dumps({'error':{'message':type(exc).__name__}}).encode(),502
            raw=raw.replace(server.key.encode(),b'[REDACTED]')
        Path(str(prefix)+'-response.json').write_bytes(raw)
        dump(str(prefix)+'-receipt.json',{'status':status,'seconds':time.monotonic()-started,
            'forwarded':not server.stub,'seed_requested':server.seed})
        self.send(status,raw)


@contextmanager
def proxy(output,key,seed,deadline,stub=False):
    server=Proxy(output,key,seed,deadline,stub)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/v1'
    finally:server.shutdown();server.server_close();thread.join()
