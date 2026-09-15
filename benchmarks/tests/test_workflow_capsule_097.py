"""Verify immutable capsule creation and reminders through the real HTTP proxy."""
import csv
from datetime import datetime, timedelta
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization.workflow_capsule_097 import build, digest


BASE = Path('results/collection-096-dispatch-bundle-001/payload/capsule')


class WorkflowCapsuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_only_declared_files_change_and_original_is_immutable(self):
        before = {str(p.relative_to(BASE)): digest(p.read_bytes()) for p in BASE.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
        result = build(BASE, self.root/'capsule')
        self.assertEqual(result['changed_from_096'], ['PROTOCOL.md', 'TASK.md', 'analyze.py', 'transport.py'])
        self.assertEqual(before, {str(p.relative_to(BASE)): digest(p.read_bytes()) for p in BASE.rglob('*')
                                 if p.is_file() and '__pycache__' not in p.parts})
        self.assertFalse(result['final_gate_opened'])
        with self.assertRaisesRegex(ValueError, 'fresh'):
            build(BASE, self.root/'capsule')

    def test_unrecognized_parent_rejected(self):
        source = self.root/'wrong'; source.mkdir(); (source/'capsule.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'exact frozen'):
            build(source, self.root/'capsule')
        self.assertFalse((self.root/'capsule').exists())

    def test_http_transport_records_progress_and_keeps_original_limits(self):
        capsule = self.root/'capsule'; build(BASE, capsule)
        package = capsule/'benchmarks/hermes_ml_checkpoint_v6'
        name = '_synthetic_workflow_097'
        parent = types.ModuleType(name); parent.__path__ = [str(package)]
        sys.modules[name] = parent
        def cleanup():
            for key in list(sys.modules):
                if key == name or key.startswith(name+'.'):
                    del sys.modules[key]
        self.addCleanup(cleanup)
        spec = importlib.util.spec_from_file_location(name+'.transport', package/'transport.py')
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        work, output = self.root/'work', self.root/'output'; work.mkdir(); output.mkdir()
        times = [(datetime(2020, 1, 1)+timedelta(days=i)).isoformat() for i in range(730)]
        (work/'task.json').write_text(json.dumps({'origin': times[-1]}))
        with (work/'history.csv').open('w') as stream:
            writer = csv.writer(stream); writer.writerow(['timestamp', 'value'])
            writer.writerows((t, 1) for t in times)
        captured = []
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"choices":[{"message":{"content":"synthetic"}}],"usage":{"total_tokens":0}}'
        def upstream(request, **kwargs):
            self.assertEqual(request.full_url, 'https://api.engy.ai/v1/chat/completions')
            captured.append(json.loads(request.data))
            return Response()
        with patch.object(module.urllib.request, 'urlopen', upstream):
            with module.proxy(output, 'SYNTHETIC_NO_CREDENTIAL', work, time.time()+400) as url:
                connection = http.client.HTTPConnection(url.split('/')[2])
                for number in range(1, 18):
                    payload = {'messages': [{'role': 'user', 'content': 'same task'}], 'model': 'ignored'}
                    connection.request('POST', '/v1/chat/completions', json.dumps(payload),
                                       {'Content-Type': 'application/json'})
                    response = connection.getresponse(); response.read()
                    self.assertEqual(response.status, 400 if number == 17 else 200)
                connection.close()
        self.assertEqual(len(captured), 16)
        self.assertEqual(len(list(output.glob('*-workflow-progress.json'))), 3)
        self.assertEqual(len(list(output.glob('*-intervention.json'))), 4)
        for number, payload in enumerate(captured, 1):
            self.assertEqual(payload['model'], 'deepseek-v4.1-flash')
            self.assertEqual(payload['seed'], 7)
            self.assertEqual(payload['max_tokens'], 3072)
            original = json.loads((output/f'api-{number:02d}-request.json').read_text())
            self.assertEqual(original['messages'], [{'role': 'user', 'content': 'same task'}])
            if number in (4, 8, 11):
                receipt = json.loads((output/f'api-{number:02d}-workflow-progress.json').read_text())
                self.assertEqual(payload['messages'][-1]['content'], receipt['notice'])
                self.assertEqual(receipt['progress']['exploration_requests_including_this_one'], 13-number)
            elif number < 13:
                self.assertEqual(payload['messages'], original['messages'])
            else:
                receipt = json.loads((output/f'api-{number:02d}-intervention.json').read_text())
                self.assertEqual(receipt['phase'], 'selection')
                self.assertEqual(payload['messages'][-1]['content'], receipt['notice'])
        self.assertFalse((work/'experiments.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
