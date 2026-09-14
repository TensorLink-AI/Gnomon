"""Probe a seed capsule's actual worker construction and metered proxy locally.

Fresh process required. Model responses are scripted; no real Hermes/provider
execution or Engy API request. All evidence stays in the supplied output folder.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def probe(capsule, output):
    capsule, output = Path(capsule).resolve(), Path(output).resolve()
    if any(k.startswith('benchmarks.hermes_ml_checkpoint_v6') for k in sys.modules):
        raise ValueError('Fresh process required; reject cached imports from another capsule')
    manifest = json.loads((capsule / 'capsule.json').read_text())
    package = capsule / 'benchmarks/hermes_ml_checkpoint_v6'
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
    sys.path.insert(0, str(capsule))
    from benchmarks.hermes_ml_checkpoint_v6 import worker, transport
    assert Path(worker.__file__).resolve() == package / 'worker.py'
    assert Path(transport.__file__).resolve() == package / 'transport.py'
    output.mkdir(parents=True, exist_ok=False)
    work = output / 'work'; work.mkdir()
    audit = output / 'worker-output'; audit.mkdir()
    (work / 'agent-budget.json').write_text(json.dumps({'deadline_epoch': time.time()+480}))
    required = ('lab.py', 'core.py', 'numerical.py', 'task.json', 'history.csv',
                'future.csv', 'backend.json', 'previous_runs.json')
    protected = output / 'protected.json'
    protected.write_text(json.dumps({name: 'synthetic' for name in required}))
    constructions = []
    class Agent:
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.tools = []
    fake_agent = ModuleType('run_agent'); fake_agent.AIAgent = Agent
    lifecycle = ModuleType('tools.mcp_tool_lifecycle')
    lifecycle.shutdown_mcp_servers = lambda: None
    def drive(factory, *args, **kwargs):
        for remaining in (16, 8, 1):
            factory(remaining, 120)
        return {'synthetic': True}
    previous_cwd = Path.cwd()
    try:
        with patch.dict(sys.modules, {'run_agent': fake_agent, 'tools.mcp_tool_lifecycle': lifecycle}), \
             patch.object(worker, 'bounded_agent_class', lambda cls: cls), \
             patch.object(worker, 'LabBoundary', lambda *args: SimpleNamespace()), \
             patch.object(worker, 'HermesBoundary', lambda *args, **kwargs: SimpleNamespace()), \
             patch.object(worker, 'native_callbacks', lambda agent: {}), \
             patch.object(worker, 'attach', lambda *args: None), patch.object(worker, 'drive', drive):
            worker.run(work, audit, 'http://127.0.0.1:1/v1', protected)
    finally:
        os.chdir(previous_cwd)
    for arguments in constructions:
        assert arguments['request_overrides'] == {'temperature': .2, 'seed': manifest['requested_seed']}
        assert arguments['model'] == 'deepseek-v4.1-flash' and arguments['max_tokens'] == 3072
    proxy_output = output / 'proxy-output'; proxy_output.mkdir()
    wire = []
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"choices":[{"message":{"content":"scripted"}}],"usage":{"total_tokens":1}}'
    def upstream(request, **kwargs):
        wire.append(json.loads(request.data))
        return Response()
    with patch.object(transport.urllib.request, 'urlopen', upstream), \
         transport.proxy(proxy_output, 'synthetic-only', work, time.time()+480) as url:
        port = int(url.split(':')[-1].split('/')[0])
        for number in range(17):
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
            connection.request('POST', '/v1/chat/completions', json.dumps({
                'model': 'wrong', 'seed': -123, 'messages': [{'role': 'user', 'content': 'synthetic'}]}))
            response = connection.getresponse(); response.read(); connection.close()
            assert response.status == (200 if number < 16 else 400)
    assert len(wire) == 16
    for number, payload in enumerate(wire):
        assert {k: payload[k] for k in manifest['parameters']} == manifest['parameters']
        assert payload['stream'] is False
        assert any(m['role'] == 'system' for m in payload['messages']) == (number >= 12)
        assert json.loads((proxy_output / f'api-{number+1:02d}-forwarded.json').read_text()) == payload
    assert json.loads((work / 'agent-budget.json').read_text())['remaining_requests'] == 0
    result = {'status': 'scripted_seed_probe_passed', 'requested_seed': manifest['requested_seed'],
              'worker_constructions': len(constructions), 'scripted_forwarded_requests': len(wire),
              'seventeenth_request_rejected': True, 'selection_phase_preserved': True,
              'provider_calls': 0, 'engy_calls': 0, 'final_gate_opened': False,
              'capsule_manifest_sha256': hashlib.sha256((capsule / 'capsule.json').read_bytes()).hexdigest(),
              'limitations': 'Mock agent construction and scripted proxy upstream; actual model seed support and full runner/memory integration untested.'}
    (output / 'passed.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.capsule, args.output), indent=2))
