from datetime import datetime, timedelta, timezone
import json
import sqlite3

from gnomon_hosted.storage import Store, now
from hosted_probe_support import serving, call, ditto_peer
from test_end_to_end import ok


def seed(url, token):
    # Requests have historical targets here, explicitly a retrospective fixture.
    at = datetime.now(timezone.utc) - timedelta(days=1)
    request = {'history': [10, 11, 12], 'horizon': 1, 'series_id': 'test-sales', 'unit': 'widgets',
        'timestamps': [(at - timedelta(days=i)).isoformat() for i in (3, 2, 1)],
        'future_timestamps': [at.isoformat()]}
    forecast = ok(call(url, token, 'gnomon_forecast', {'provider': 'last_value', 'request': request, 'idempotency_key': 'forecast'}))
    decision = ok(call(url, token, 'gnomon_ledger', {'operation': 'record_decision_summary',
        'execution_id': forecast['result']['execution_id'], 'rationale': 'Test hypothesis', 'assumptions': [], 'invalidation_conditions': [], 'context': [], 'idempotency_key': 'decision'}))
    did = decision['result']['result']['decision_id']
    ok(call(url, token, 'gnomon_ledger', {'operation': 'append_actual', 'series_id': 'test-sales', 'unit': 'widgets',
        'valid_time': at.isoformat(), 'value': 13, 'source_available_at': now(), 'idempotency_key': 'actual'}))
    cuts = {'source_as_of': now(), 'recorded_as_of': now()}
    lesson = ok(call(url, token, 'gnomon_ledger', {'operation': 'record_lesson', 'decision_id': did,
        'lesson': 'Synthetic test: baseline underpredicted by one widget; cause unknown.', **cuts, 'idempotency_key': 'lesson'}))
    queued = ok(call(url, token, 'gnomon_hosted', {'action': 'export.enqueue',
        'lesson_id': lesson['result']['result']['lesson_id'], 'recorded_as_of': now(), 'idempotency_key': 'enqueue'}))
    return queued['result']['export_id']


def test_durable_delivery_recall_and_tampering(tmp_path):
    root, peer = tmp_path / 'service', tmp_path / 'peer'
    store = Store.initialize(root)
    project = store.create_project('test')['project_id']
    token = store.issue_token(project, 'agent', ['forecast.create', 'decision.create', 'actual.create', 'memory.export', 'evidence.read'])['token']
    store.configure_ditto(project, 'https://api.heyditto.ai/mcp', 'TEST_DITTO_KEY', 'test-graph')
    with ditto_peer(peer) as endpoint:
        # Test fixture bypasses the operator's HTTPS-only URL configuration for loopback.
        with store.connect() as conn:
            conn.execute('UPDATE projects SET ditto_url=? WHERE id=?', (endpoint, project))
        with serving(root, env={'TEST_DITTO_KEY': 'fixture-not-a-secret'}) as url:
            eid = seed(url, token)
        # Queue survives a server restart before the network write.
        with serving(root, env={'TEST_DITTO_KEY': 'fixture-not-a-secret'}) as url:
            sent = ok(call(url, token, 'gnomon_hosted', {'action': 'export.deliver', 'export_id': eid}))
            assert sent['state'] == 'acknowledged', sent
            again = ok(call(url, token, 'gnomon_hosted', {'action': 'export.deliver', 'export_id': eid}))
            assert again['memory_id'] == sent['memory_id']
        with serving(root, env={'TEST_DITTO_KEY': 'fixture-not-a-secret'}) as url:
            args = {'action': 'memory.recall', 'query': 'test sales', 'source_as_of': now(), 'recorded_as_of': now()}
            recalled = ok(call(url, token, 'gnomon_hosted', args))
            assert recalled['returned'] == 1, json.dumps(recalled)
            assert recalled['lessons'][0]['current_review']['metrics']['mae'] == 1
            assert recalled['lessons'][0]['narrative_verified'] is False
            with sqlite3.connect(peer / 'memories.db') as conn:
                assert conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0] == 1
                conn.execute('UPDATE memories SET content=?', (json.dumps({'fake': 'evidence'}),))
            tampered = ok(call(url, token, 'gnomon_hosted', args))
            assert tampered['returned'] == 0
            assert tampered['excluded'][0]['reason'] == 'unverified_or_cutoff_ineligible'


def test_ambiguous_save_never_retries_and_can_reconcile(tmp_path):
    root, peer = tmp_path / 'service', tmp_path / 'peer'
    store = Store.initialize(root)
    project = store.create_project('test')['project_id']
    token = store.issue_token(project, 'agent', ['forecast.create', 'decision.create', 'actual.create', 'memory.export', 'evidence.read'])['token']
    store.configure_ditto(project, 'https://api.heyditto.ai/mcp', 'TEST_DITTO_KEY', 'test-graph')
    with ditto_peer(peer) as endpoint:
        with store.connect() as conn:
            conn.execute('UPDATE projects SET ditto_url=? WHERE id=?', (endpoint, project))
        (peer / 'fail-after-save').touch()
        with serving(root, env={'TEST_DITTO_KEY': 'fixture-not-a-secret'}) as url:
            eid = seed(url, token)
            sent = ok(call(url, token, 'gnomon_hosted', {'action': 'export.deliver', 'export_id': eid}))
            assert sent['state'] == 'uncertain'
        (peer / 'fail-after-save').unlink()
        with serving(root, env={'TEST_DITTO_KEY': 'fixture-not-a-secret'}) as url:
            again = ok(call(url, token, 'gnomon_hosted', {'action': 'export.deliver', 'export_id': eid}))
            assert again['state'] == 'uncertain'
            with sqlite3.connect(peer / 'memories.db') as conn:
                rows = conn.execute('SELECT id FROM memories').fetchall()
            assert len(rows) == 1
            bad = call(url, token, 'gnomon_hosted', {'action': 'export.reconcile', 'export_id': eid, 'memory_id': 'unknown'})
            assert bad['error']['code'] == 'UNVERIFIED'
            good = ok(call(url, token, 'gnomon_hosted', {'action': 'export.reconcile', 'export_id': eid, 'memory_id': rows[0][0]}))
            assert good['state'] == 'acknowledged'


def test_wrong_graph_and_retired_project_never_export(tmp_path):
    import httpx
    root, peer = tmp_path / 'service', tmp_path / 'peer'
    store = Store.initialize(root)
    project = store.create_project('test')['project_id']
    token = store.issue_token(project, 'agent', ['forecast.create', 'decision.create', 'actual.create', 'memory.export', 'evidence.read'])['token']
    store.configure_ditto(project, 'https://api.heyditto.ai/mcp', 'TEST_DITTO_KEY', 'wrong-graph')
    with ditto_peer(peer) as endpoint:
        with store.connect() as conn:
            conn.execute('UPDATE projects SET ditto_url=? WHERE id=?', (endpoint, project))
        with serving(root, env={'TEST_DITTO_KEY': 'fixture-not-a-secret'}) as url:
            eid = seed(url, token)
            sent = ok(call(url, token, 'gnomon_hosted', {'action': 'export.deliver', 'export_id': eid}))
            assert sent['state'] == 'pending'
            ledger = store.ledger(project)
            store.disable_project(project)
            assert httpx.post(url + '/mcp', headers={'Authorization': 'Bearer ' + token}, json={}).status_code == 401
            with ledger.transaction() as conn:
                assert conn.execute('SELECT state FROM hosted_exports WHERE id=?', (eid,)).fetchone()[0] == 'cancelled'
            with sqlite3.connect(peer / 'memories.db') as conn:
                assert conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0] == 0
