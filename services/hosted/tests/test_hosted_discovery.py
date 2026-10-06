import json
from datetime import datetime, timedelta, timezone

import jsonschema
import pytest

from gnomon_hosted.app import tools
from gnomon_hosted.contracts import CONTRACTS, action_schema, validate_action
from gnomon_hosted.runtime import Runtime
from gnomon_hosted.storage import Store, ServiceError, now
from hosted_probe_support import serving, call
from test_submissions import fixture, snapshot


def test_all_action_contracts_validate_and_reject_unrelated_fields():
    declared = next(t.inputSchema for t in tools() if t.name == 'gnomon_hosted')
    assert set(declared['properties']['action']['enum']) == set(CONTRACTS)
    for action in CONTRACTS:
        schema = action_schema(action)
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema['additionalProperties'] is False
        assert set(schema['properties']) <= set(declared['properties'])
        assert set(schema['required']) <= set(schema['properties'])
        with pytest.raises(ServiceError, match='Allowed fields'):
            validate_action({'action': action, 'secret-field-value': 'must-not-echo'})
    # Independent copies: discovery callers cannot mutate runtime requirements.
    action_schema('snapshot.save')['required'].clear()
    with pytest.raises(ServiceError, match='decision_id'):
        validate_action({'action': 'snapshot.save'})
    with pytest.raises(ServiceError, match='exactly one'):
        validate_action({'action': 'export.enqueue', 'recorded_as_of': now(),
                         'idempotency_key': 'x', 'lesson_id': 'a', 'analysis_id': 'b'})


def test_schema_discovery_over_mcp_and_wrong_action_fields(tmp_path):
    store, identity, runtime, token, args = fixture(tmp_path)
    with serving(store.root) as url:
        for action in ('snapshot.save', 'analysis.submit', 'forecast.submit'):
            reply = call(url, token, 'gnomon_hosted', {'action': 'schema.get', 'target_action': action})
            assert reply['schema'] == action_schema(action)
        bad = call(url, token, 'gnomon_hosted', {'action': 'analysis.submit', 'decision_id': 'x'})
        assert bad['error']['code'] == 'INVALID_ARGUMENTS'
        assert 'snapshot_id' in bad['error']['message']
        assert 'Allowed fields' in bad['error']['message']
        assert call(url, token, 'gnomon_hosted', {'action': 'info'})['status'] == 'ok'


def test_info_is_configuration_discovery_not_provider_execution(tmp_path, monkeypatch):
    config = tmp_path / 'providers.toml'
    config.write_text('''[providers.private-plugin]
kind = "custom"
entrypoint = "missing_module:must_not_load"
[providers.ephemeris]
kind = "ephemeris"
base_url = "https://secret-endpoint.invalid"
api_key_env = "PRIVATE_PROVIDER_KEY"
''')
    store = Store.initialize(tmp_path / 'service')
    project = store.create_project('info', str(config))['project_id']
    token = store.issue_token(project, 'reader', ['evidence.read'])['token']
    # Configuration exists but no credentials have been provided.
    with store.connect() as conn:
        conn.execute('UPDATE projects SET ditto_url=?,ditto_graph=?,ditto_token_env=? WHERE id=?',
                     ('https://private-ditto.invalid', 'project-graph', 'PRIVATE_DITTO_KEY', project))
    monkeypatch.setenv('PRIVATE_DITTO_KEY', 'do-not-echo-secret')
    runtime = Runtime(store)
    r = runtime.call(store.authenticate(token), 'gnomon_hosted', {'action': 'info'})
    assert r['versions']['core'] and r['versions']['hosted']
    assert r['providers']['last_value']['availability'] == 'built_in'
    assert r['providers']['ephemeris']['availability'] == 'configured_not_probed'
    assert r['ditto'] == {'configured': True, 'credential_present': True, 'graph': 'project-graph', 'health_checked': False}
    assert r['external_calls'] == r['ledger_writes'] == 0
    text = json.dumps(r)
    for secret in ('do-not-echo-secret', 'private-ditto.invalid', 'secret-endpoint.invalid', 'missing_module', 'PRIVATE_DITTO_KEY', str(config)):
        assert secret not in text


def test_lifecycle_tracks_pending_complete_stale_and_exports_without_writes(tmp_path):
    store, identity, runtime, token, args = fixture(tmp_path)
    f = runtime.call(identity, 'gnomon_hosted', args)['result']
    d = runtime.call(identity, 'gnomon_ledger', {'operation': 'record_decision_summary',
        'execution_id': f['execution_id'], 'rationale': 'Synthetic status test',
        'assumptions': [], 'invalidation_conditions': [], 'context': [], 'idempotency_key': 'd'})['result']['result']['decision_id']
    ledger = store.ledger(identity.project)
    def status():
        with ledger.transaction() as conn:
            before = conn.execute('SELECT COUNT(*) FROM hosted_requests').fetchone()[0]
        reply = runtime.call(identity, 'gnomon_hosted', {'action': 'decision.status', 'decision_id': d})
        with ledger.transaction() as conn:
            assert conn.execute('SELECT COUNT(*) FROM hosted_requests').fetchone()[0] == before
        assert reply['external_calls'] == reply['ledger_writes'] == 0
        return reply
    assert status()['actuals']['status'] == 'pending'
    runtime.call(identity, 'gnomon_ledger', {'operation': 'append_actual', 'series_id': 's', 'unit': 'widgets',
        'valid_time': args['request']['future_timestamps'][0], 'source_available_at': args['request']['future_timestamps'][0],
        'value': 5, 'idempotency_key': 'a'})
    assert status()['actuals']['status'] == 'complete'
    snap = snapshot(runtime, identity, d)
    a = runtime.call(identity, 'gnomon_hosted', {'action': 'analysis.submit', 'snapshot_id': snap['snapshot_id'],
        'method': 'test', 'method_version': '1', 'metrics': {'mae': 2}, 'lesson': 'Synthetic', 'idempotency_key': 'analysis'})['result']
    with store.connect() as conn:
        conn.execute("UPDATE projects SET ditto_url='https://example.invalid',ditto_connection='test' WHERE id=?", (identity.project,))
    e = runtime.call(identity, 'gnomon_hosted', {'action': 'export.enqueue', 'analysis_id': a['analysis_id'],
        'recorded_as_of': now(), 'idempotency_key': 'export'})['result']
    r = status()
    assert r['analyses'][0]['numerically_verified'] is False
    assert r['exports'][0]['state'] == 'pending'
    assert any(s['action'] == 'export.deliver' for s in r['next_steps'])
    with ledger.transaction() as conn:
        conn.execute("UPDATE hosted_exports SET state='uncertain' WHERE id=?", (e['export_id'],))
    assert any(s['action'] == 'export.reconcile' for s in status()['next_steps'])
    with ledger.transaction() as conn:
        conn.execute("UPDATE hosted_exports SET state='acknowledged',memory_id='ditto-test' WHERE id=?", (e['export_id'],))
    assert status()['exports'][0]['memory_id'] == 'ditto-test'
    # Revision is available before the next snapshot cutoff (avoid same-clock precision races).
    at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    runtime.call(identity, 'gnomon_ledger', {'operation': 'append_actual', 'series_id': 's', 'unit': 'widgets',
        'valid_time': args['request']['future_timestamps'][0], 'source_available_at': at,
        'value': 9, 'idempotency_key': 'revised'})
    assert status()['analyses'][0]['evidence_changed'] is True
    foreign = store.create_project('other')['project_id']
    other = store.authenticate(store.issue_token(foreign, 'reader', ['evidence.read'])['token'])
    with pytest.raises(ServiceError, match='unavailable'):
        runtime.call(other, 'gnomon_hosted', {'action': 'decision.status', 'decision_id': d})
    writer = store.authenticate(store.issue_token(identity.project, 'outcome', ['actual.create'])['token'])
    with pytest.raises(ServiceError) as denied:
        runtime.call(writer, 'gnomon_hosted', {'action': 'decision.status', 'decision_id': d})
    assert denied.value.code == 'FORBIDDEN'


def test_lifecycle_includes_core_lessons_reviews_and_evaluations(tmp_path):
    from test_submissions import prepare
    store, identity, runtime, token, args = fixture(tmp_path)
    f, d = prepare(runtime, identity, args)
    snap = snapshot(runtime, identity, d)
    cutoffs = {k: snap['snapshot'][k] for k in ('source_as_of', 'recorded_as_of')}
    runtime.call(identity, 'gnomon_ledger', {'operation': 'evaluate', 'execution_id': f['execution_id'],
                                          **cutoffs, 'idempotency_key': 'eval'})
    review = runtime.call(identity, 'gnomon_hosted', {'action': 'review.save', 'decision_id': d,
                                                    **cutoffs, 'idempotency_key': 'review'})
    lesson = runtime.call(identity, 'gnomon_ledger', {'operation': 'record_lesson', 'decision_id': d,
        'lesson': 'One synthetic observation, no causal claim.', **cutoffs, 'idempotency_key': 'lesson'})
    status = runtime.call(identity, 'gnomon_hosted', {'action': 'decision.status', 'decision_id': d})
    assert status['evaluations']
    assert status['reviews'][0]['review_id'] == review['result']['review_id']
    assert status['lessons'][0]['lesson_id'] == lesson['result']['result']['lesson_id']
    assert status['lessons'][0]['evidence_changed'] is False
    assert status['blockers']
    assert not any(s['action'] == 'export.enqueue' for s in status['next_steps'])
