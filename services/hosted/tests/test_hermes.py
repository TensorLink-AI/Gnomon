"""Opt-in pinned Hermes client integration; CI supplies a real checkout."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from gnomon_hosted.storage import Store
from hosted_probe_support import serving


def test_fresh_hermes_processes_share_evidence(tmp_path):
    hermes = os.environ.get('GNOMON_HERMES_ROOT')
    if not hermes:
        pytest.skip('Set GNOMON_HERMES_ROOT to the pinned Hermes checkout')
    python = os.environ.get('GNOMON_HERMES_PYTHON', sys.executable)
    script = Path(__file__).resolve().parents[1] / 'scripts/hermes_client.py'
    store = Store.initialize(tmp_path / 'service')
    project = store.create_project('hermes')['project_id']
    a = store.issue_token(project, 'hermes-a', ['forecast.create', 'evidence.read'])['token']
    b = store.issue_token(project, 'hermes-b', ['evidence.read'])['token']
    def invoke(url, token, tool, arguments):
        result = subprocess.run([python, str(script), '--hermes-root', hermes, '--url', url + '/mcp'],
            input=json.dumps({'tool': tool, 'arguments': arguments}), text=True, capture_output=True, timeout=90,
            env={**os.environ, 'GNOMON_SERVICE_TOKEN': token})
        assert result.returncode == 0, result.stderr
        reply = json.loads(result.stdout)
        assert reply['status'] == 'ok', reply
        return reply
    at = datetime.now(timezone.utc)
    request = {'history': [1, 2], 'horizon': 1, 'series_id': 'hermes-test',
               'timestamps': [(at-timedelta(days=d)).isoformat() for d in (2, 1)],
               'future_timestamps': [(at+timedelta(days=1)).isoformat()]}
    with serving(store.root) as url:
        result = invoke(url, a, 'gnomon_forecast', {'provider': 'last_value', 'request': request, 'idempotency_key': 'one'})
    with serving(store.root) as url:
        resolved = invoke(url, b, 'gnomon_hosted', {'action': 'resolve', 'reference': result['result']['reference']})
        assert resolved['result']['execution_id'] == result['result']['execution_id']
