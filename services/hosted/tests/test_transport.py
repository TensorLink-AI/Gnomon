import asyncio
from urllib.parse import urljoin

import httpx
from gnomon_hosted.storage import Store
from hosted_probe_support import serving, call


def test_sse_owner_binding_and_input_boundaries(tmp_path):
    store = Store.initialize(tmp_path / 'service')
    project = store.create_project('one')['project_id']
    a = store.issue_token(project, 'a', ['evidence.read'])['token']
    b = store.issue_token(project, 'b', ['evidence.read'])['token']
    with serving(store.root) as url:
        async def probe():
            async with httpx.AsyncClient(timeout=10) as client:
                async with client.stream('GET', url + '/sse', headers={'Authorization': 'Bearer ' + a}) as stream:
                    assert stream.status_code == 200
                    lines = stream.aiter_lines()
                    async for line in lines:
                        if line.startswith('data: '):
                            endpoint = urljoin(url, line[6:])
                            break
                    payload = {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
                        'protocolVersion': '2025-03-26', 'capabilities': {}, 'clientInfo': {'name': 'test', 'version': '1'}}}
                    wrong = await client.post(endpoint, json=payload, headers={'Authorization': 'Bearer ' + b})
                    assert wrong.status_code in (403, 404)
                    right = await client.post(endpoint, json=payload, headers={'Authorization': 'Bearer ' + a})
                    assert right.status_code == 202
        asyncio.run(probe())
        too_big = httpx.post(url + '/mcp', headers={'Authorization': 'Bearer ' + a}, content=b'x' * (1024 * 1024 + 1))
        assert too_big.status_code == 413
        for headers in ({'Host': 'evil.example'}, {'Origin': 'https://evil.example'}):
            assert httpx.get(url + '/health', headers=headers).status_code == 403
        forbidden = call(url, a, 'gnomon_hosted', {'action': 'dataset.put', 'request': {'history': [1], 'horizon': 1}, 'idempotency_key': 'no'})
        assert forbidden['error']['code'] == 'FORBIDDEN'


def test_action_specific_validation_reports_contract_and_recovers(tmp_path):
    store = Store.initialize(tmp_path / 'service')
    project = store.create_project('validation')['project_id']
    token = store.issue_token(project, 'pilot', ['evidence.read', 'decision.create'])['token']
    with serving(store.root) as url:
        snapshot = call(url, token, 'gnomon_hosted', {
            'action': 'snapshot.save', 'decision_id': 'example', 'idempotency_key': 'snapshot'})
        assert snapshot['error']['code'] == 'INVALID_ARGUMENTS'
        assert 'recorded_as_of' in snapshot['error']['message']
        assert 'source_as_of' in snapshot['error']['message']
        analysis = call(url, token, 'gnomon_hosted', {
            'action': 'analysis.submit', 'decision_id': 'private-value-not-for-errors',
            'idempotency_key': 'analysis'})
        message = analysis['error']['message']
        assert 'snapshot_id' in message and 'method_version' in message
        assert 'Allowed fields for this operation:' in message
        assert 'private-value-not-for-errors' not in message
        assert call(url, token, 'gnomon_hosted', {'action': 'info'})['status'] == 'ok'
