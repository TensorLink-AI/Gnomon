"""Meter requests, publish budgets, and disclose the protected selection phase."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
from .policy import REQUEST_LIMIT, EXPLORATION_REQUESTS, phase, SELECTION_NOTICE
MODEL = 'deepseek-v4-flash-0731'

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Proxy(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, output, key, work=None, deadline=None):
        super().__init__(('127.0.0.1', 0), Handler)
        self.output, self.key = output, key
        self.calls, self.lock = 0, threading.Lock()
        self.work,self.deadline=work,deadline
        self.update_budget()

    def update_budget(self):
        if self.work is not None:
            path=Path(self.work)/'agent-budget.json'
            temporary=path.with_suffix('.tmp')
            seconds = self.deadline-time.time() if self.deadline else None
            dump(temporary,{'limit_requests':REQUEST_LIMIT,'forwarded_requests':min(REQUEST_LIMIT,self.calls),
                            'remaining_requests':max(0,REQUEST_LIMIT-self.calls),'deadline_epoch':self.deadline,
                            'phase':phase(self.calls, seconds),
                            'next_request_phase':phase(self.calls+1, seconds),
                            'exploration_requests_remaining':max(0,EXPLORATION_REQUESTS-self.calls)})
            os.replace(temporary,path)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self.send(200, json.dumps({'object': 'list', 'data': [{'id': MODEL, 'object': 'model'}]}).encode())

    def do_POST(self):
        size = int(self.headers.get('Content-Length', '0'))
        if size > 4_000_000:
            return self.send(413, b'{"error":{"message":"request too large"}}')
        payload = json.loads(self.rfile.read(size))
        if 'messages' not in payload:
            # Hermes may probe a local-looking endpoint for model metadata.
            # Discovery is not an Engy inference call and must not spend budget.
            dump(self.server.output / 'model-metadata-probe.json', {'path': self.path, 'body': payload})
            return self.send(404, b'{"error":{"message":"metadata endpoint not implemented"}}')
        with self.server.lock:
            self.server.calls += 1
            number = self.server.calls
            self.server.update_budget()
        if number > REQUEST_LIMIT:
            dump(self.server.output / f'blocked-request-{number:02d}.json',
                 {'cause':'request_budget_exhausted','forwarded':False,'request':payload})
            return self.send(400, b'{"error":{"message":"evaluation model-call budget exhausted"}}')
        prefix = self.server.output / f'api-{number:02d}'
        dump(str(prefix) + '-request.json', payload)
        seconds = self.server.deadline-time.time() if self.server.deadline else None
        current_phase = phase(number, seconds)
        if current_phase == 'selection':
            # Explicit intervention, recorded beside original and forwarded requests.
            notice = SELECTION_NOTICE + f' Requests including this one remaining: {REQUEST_LIMIT-number+1}.'
            payload['messages'] = [*payload['messages'], {'role':'system','content':notice}]
            dump(str(prefix) + '-intervention.json', {'phase':current_phase,'notice':notice})
        # Metering transport enforces the declared settings equally for both arms.
        payload.update(model=MODEL, temperature=0.2, seed=7, max_tokens=3072, stream=False)
        for key in ('stream_options', 'max_completion_tokens'):
            payload.pop(key, None)
        dump(str(prefix) + '-forwarded.json', payload)
        req = urllib.request.Request('https://api.engy.ai/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={
                'Authorization': 'Bearer ' + self.server.key, 'Content-Type': 'application/json'})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                raw, status = response.read(), response.status
        except urllib.error.HTTPError as exc:
            raw, status = exc.read(), exc.code
        except Exception as exc:
            raw = json.dumps({'error': {'message': type(exc).__name__}}).encode()
            status = 502
        # Never retain authorization headers or the real key, including echoed errors.
        raw = raw.replace(self.server.key.encode(), b'[REDACTED]')
        Path(str(prefix) + '-response.json').write_bytes(raw)
        dump(str(prefix) + '-receipt.json', {'status': status, 'seconds': time.monotonic() - started})
        self.send(status, raw)

@contextmanager
def proxy(output, key, work=None, deadline=None):
    server = Proxy(output, key, work, deadline)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
