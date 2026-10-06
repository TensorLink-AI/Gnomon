from datetime import datetime, timedelta, timezone
import json
import time

import httpx
import pytest

from gnomon_hosted.storage import Store, now
from gnomon_hosted.cli import backup, restore
from hosted_probe_support import serving, call


def request():
    at = datetime.now(timezone.utc)
    return {'history': [10., 11., 12.], 'horizon': 2, 'series_id': 'sales', 'unit': 'widgets',
            'timestamps': [(at - timedelta(days=d)).isoformat() for d in (3, 2, 1)],
            'future_timestamps': [(at + timedelta(seconds=d)).isoformat() for d in (1, 2)]}


def ok(reply):
    assert reply['status'] == 'ok', reply
    return reply


def test_real_mcp_handoff_restart_revisions_and_restore(tmp_path):
    root = tmp_path / 'service'
    store = Store.initialize(root)
    project = store.create_project('sales')['project_id']
    a = store.issue_token(project, 'hermes-a', ['forecast.create', 'decision.create', 'evidence.read'])
    b = store.issue_token(project, 'actual-importer', ['actual.create'])
    c = store.issue_token(project, 'hermes-b', ['evidence.read', 'decision.create'])
    other = store.create_project('other')['project_id']
    outsider = store.issue_token(other, 'outsider', ['evidence.read'])
    req = request()
    with serving(root) as url:
        submitted = ok(call(url, a['token'], 'gnomon_forecast', {'provider': 'last_value', 'request': req, 'idempotency_key': 'forecast-1'}))
        execution = submitted['result']['execution_id']
        ref = submitted['result']['reference']
        duplicate = ok(call(url, a['token'], 'gnomon_forecast', {'provider': 'last_value', 'request': req, 'idempotency_key': 'forecast-1'}))
        assert duplicate['request_id'] == submitted['request_id']
        assert duplicate['result']['execution_id'] == execution
        conflict = call(url, a['token'], 'gnomon_forecast', {'provider': 'historical_mean', 'request': req, 'idempotency_key': 'forecast-1'})
        assert conflict['error']['code'] == 'IDEMPOTENCY_CONFLICT'
        decision = ok(call(url, a['token'], 'gnomon_ledger', {'operation': 'record_decision_summary',
            'execution_id': execution, 'rationale': 'Level persists.', 'assumptions': ['No shift.'],
            'invalidation_conditions': ['Level changes.'], 'context': [], 'idempotency_key': 'decision-1'}))
        did = decision['result']['result']['decision_id']
        pending = ok(call(url, c['token'], 'gnomon_ledger', {'operation': 'review_decision',
                     'decision_id': did, 'source_as_of': now(), 'recorded_as_of': now()}))
        assert pending['scoring_status'] == 'pending'
        # No test clock is exposed to remote clients. Wait for actual valid times.
        while datetime.now(timezone.utc) <= datetime.fromisoformat(req['future_timestamps'][-1]):
            time.sleep(.01)
        rows = [{'series_id': 'sales', 'unit': 'widgets', 'valid_time': t, 'value': value,
                 'source_available_at': now(), 'source_ref': 'integration-test'}
                for t, value in zip(req['future_timestamps'], [13, 14])]
        denied = call(url, a['token'], 'gnomon_ledger', {'operation': 'append_actual', 'actuals': rows, 'idempotency_key': 'actuals'})
        assert denied['error']['code'] == 'FORBIDDEN'
        ok(call(url, b['token'], 'gnomon_ledger', {'operation': 'append_actual', 'actuals': rows, 'idempotency_key': 'actuals'}))
        cutoff = now()
        saved = ok(call(url, c['token'], 'gnomon_hosted', {'action': 'review.save', 'decision_id': did,
                    'source_as_of': cutoff, 'recorded_as_of': cutoff, 'idempotency_key': 'review-1'}))
        assert saved['result']['review']['metrics']['mae'] == 1.5
        review_ref = saved['result']['reference']
        lesson = ok(call(url, c['token'], 'gnomon_ledger', {'operation': 'record_lesson', 'decision_id': did,
                    'lesson': 'Baseline underpredicted; cause unverified.', 'source_as_of': cutoff,
                    'recorded_as_of': cutoff, 'idempotency_key': 'lesson-1'}))
        lid = lesson['result']['result']['lesson_id']
        assert call(url, outsider['token'], 'gnomon_hosted', {'action': 'resolve', 'reference': ref})['error']['code'] == 'UNAVAILABLE'
        assert httpx.get(url + '/sse').status_code == 401
        assert httpx.get(url + '/health', headers={'Origin': 'https://evil.example'}).status_code == 403
    # A fresh OS process owns the actual server; clients also reconnect from scratch.
    with serving(root) as url:
        resolved = ok(call(url, c['token'], 'gnomon_hosted', {'action': 'resolve', 'reference': ref}, sse=True))
        assert resolved['result']['execution_id'] == execution
        old_review = ok(call(url, c['token'], 'gnomon_hosted', {'action': 'resolve', 'reference': review_ref}))
        assert old_review['result']['metrics']['mae'] == 1.5
        revised = {**rows[1], 'value': 20, 'source_available_at': now()}
        ok(call(url, b['token'], 'gnomon_ledger', {'operation': 'append_actual', **revised, 'idempotency_key': 'revision-1'}))
        current = ok(call(url, c['token'], 'gnomon_ledger', {'operation': 'review_decision', 'decision_id': did,
                     'source_as_of': now(), 'recorded_as_of': now()}))
        assert current['result']['metrics']['mae'] == 4.5
        exported = ok(call(url, c['token'], 'gnomon_ledger', {'operation': 'export_lesson', 'lesson_id': lid, 'recorded_as_of': now()}))
        assert exported['result']['metrics']['mae'] == 1.5
        store.revoke(c['token_id'])
        assert httpx.post(url + '/mcp', headers={'Authorization': 'Bearer ' + c['token']}, json={}).status_code == 401
        with pytest.raises(Exception, match='Stop the service'):
            backup(root, tmp_path / 'blocked-backup')
    bundle = tmp_path / 'backup'
    backup(root, bundle)
    restored = tmp_path / 'restored'
    restore(bundle, restored)
    restored_store = Store(restored)
    fresh = restored_store.issue_token(project, 'fresh-after-restore', ['evidence.read'])
    with serving(restored) as url:
        result = ok(call(url, fresh['token'], 'gnomon_hosted', {'action': 'resolve', 'reference': review_ref}))
        assert result['result']['metrics']['mae'] == 1.5
        assert httpx.post(url + '/mcp', headers={'Authorization': 'Bearer ' + a['token']}, json={}).status_code == 401
    assert json.loads((bundle / 'manifest.json').read_text())['service_id'] == store.service_id
