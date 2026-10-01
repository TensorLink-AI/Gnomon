from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys

import pytest
from gnomon_hosted.runtime import Runtime
from gnomon_hosted.storage import Store, ServiceError, now
from gnomon.evidence import score_snapshot
from hosted_probe_support import serving, call


def fixture(tmp_path):
    store = Store.initialize(tmp_path / 'service')
    project = store.create_project('shared')['project_id']
    token = store.issue_token(project, 'local-agent', ['forecast.create', 'decision.create', 'evidence.read', 'actual.create', 'memory.export'])['token']
    identity = store.authenticate(token)
    at = datetime.now(timezone.utc) - timedelta(days=2)
    req = {'series_id': 's', 'unit': 'widgets', 'history': [1, 2], 'horizon': 1,
           'timestamps': [(at - timedelta(days=i)).isoformat() for i in (2, 1)],
           'future_timestamps': [at.isoformat()]}
    args = {'action': 'forecast.submit', 'provider': 'local-model', 'revision': 'v1',
            'request': req, 'result': {'point': [3]}, 'computed_at': at.isoformat(), 'idempotency_key': 'f'}
    return store, identity, Runtime(store), token, args


def prepare(runtime, identity, args):
    f = runtime.call(identity, 'gnomon_hosted', args)['result']
    d = runtime.call(identity, 'gnomon_ledger', {'operation': 'record_decision_summary',
        'execution_id': f['execution_id'], 'rationale': 'Local example, retrospective submission.',
        'assumptions': [], 'invalidation_conditions': [], 'context': [], 'idempotency_key': 'd'})['result']['result']['decision_id']
    runtime.call(identity, 'gnomon_ledger', {'operation': 'append_actual', 'series_id': 's', 'unit': 'widgets',
        'valid_time': args['request']['future_timestamps'][0], 'source_available_at': now(), 'value': 5, 'idempotency_key': 'a'})
    return f, d


def snapshot(runtime, identity, decision, key='snapshot'):
    # This host can step its wall clock backwards. Wait for real time to catch
    # recorded evidence; never alter timestamps or weaken cutoff validation.
    import time
    ledger = runtime.store.ledger(identity.project)
    floor = max([ledger.decision(decision)['recorded_at'],
                 *[a['recorded_at'] for a in ledger.actuals_as_of('s', unit='widgets')],
                 *[a['source_available_at'] for a in ledger.actuals_as_of('s', unit='widgets')]])
    deadline = time.monotonic() + 10
    while now() < floor and time.monotonic() < deadline:
        time.sleep(.05)
    cutoff = now()
    return runtime.call(identity, 'gnomon_hosted', {'action': 'snapshot.save', 'decision_id': decision,
        'source_as_of': cutoff, 'recorded_as_of': cutoff, 'idempotency_key': key})['result']


def test_local_compute_restart_and_revision_over_real_mcp(tmp_path):
    store, identity, runtime, token, args = fixture(tmp_path)
    # Forecast is actually computed by local Gnomon in an independent process.
    args['result'] = json.loads(subprocess.check_output([sys.executable, '-c',
        'import json,sys; from dataclasses import asdict; from gnomon import GnomonSession; '
        's=GnomonSession.from_config(); print(json.dumps(asdict(s.engine.forecast("last_value",json.load(sys.stdin)).result)))'],
        input=json.dumps(args['request']).encode()))
    args['provider'] = 'last_value'
    with serving(store.root) as url:
        f = call(url, token, 'gnomon_hosted', args)
        assert f['status'] == 'ok', f
        duplicate = call(url, token, 'gnomon_hosted', args)
        assert f == duplicate
    # New server and client connection; core execution still resolves.
    with serving(store.root) as url:
        r = call(url, token, 'gnomon_hosted', {'action': 'resolve', 'reference': f['result']['reference']})['result']
        assert r['evidence'] == 'client_submitted'
        assert r['provider_identity']['execution_verified'] is False
        assert r['recorded_at'] > args['computed_at']
        assert r['result']['point'] == [2]
    _, d = prepare(runtime, identity, args)
    snap = snapshot(runtime, identity, d)
    scored = json.loads(subprocess.check_output([sys.executable, '-c',
        'import json,sys; from gnomon.evidence import score_snapshot; print(json.dumps(score_snapshot(json.load(sys.stdin))))'],
        input=json.dumps(snap['snapshot']).encode()))
    assert scored['metrics']['mae'] == 3
    analysis = {'action': 'analysis.submit', 'snapshot_id': snap['snapshot_id'],
        'method': scored['method'], 'method_version': scored['method_version'], 'metrics': scored['metrics'],
        'lesson': 'Underprediction on one synthetic observation; no causal claim.', 'idempotency_key': 'analysis'}
    with serving(store.root) as url:
        saved = call(url, token, 'gnomon_hosted', analysis)
        assert saved['result']['analysis']['numerically_verified'] is False
    runtime.call(identity, 'gnomon_ledger', {'operation': 'append_actual', 'series_id': 's', 'unit': 'widgets',
        'valid_time': args['request']['future_timestamps'][0], 'source_available_at': now(), 'value': 9, 'idempotency_key': 'revision'})
    fresh = snapshot(runtime, identity, d, 'new')
    assert score_snapshot(fresh['snapshot'])['metrics']['mae'] == 7
    with serving(store.root) as url:
        old = call(url, token, 'gnomon_hosted', {'action': 'resolve', 'reference': snap['reference']})['result']
        assert old == snap['snapshot']
        assert score_snapshot(old)['metrics']['mae'] == 3
        restored = call(url, token, 'gnomon_hosted', {'action': 'resolve', 'reference': saved['result']['reference']})['result']
        assert restored['metrics']['mae'] == 3 and restored['numerically_verified'] is False


@pytest.mark.parametrize('change', [
    {'recorded_at': '2000-01-01T00:00:00Z'}, {'result': {'point': [True]}},
    {'result': {'point': [float('nan')]}}, {'result': {'point': []}},
    {'result': {'point': [1], 'series_id': 'other'}},
    {'result': {'point': [1], 'quantiles': [{'0.1': 3, '0.9': 2}]}},
    {'computed_at': '2999-01-01T00:00:00Z'},
])
def test_invalid_submissions_leave_no_execution(tmp_path, change):
    store, identity, runtime, _, args = fixture(tmp_path)
    with pytest.raises((ServiceError, ValueError)):
        runtime.call(identity, 'gnomon_hosted', args | change)
    with store.ledger(identity.project).transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM executions').fetchone()[0] == 0


def test_permissions_provenance_and_conflicts(tmp_path):
    store, identity, runtime, _, args = fixture(tmp_path)
    f, d = prepare(runtime, identity, args)
    snap = snapshot(runtime, identity, d)
    reader = store.authenticate(store.issue_token(identity.project, 'reader', ['evidence.read'])['token'])
    with pytest.raises(ServiceError, match='Permission'):
        runtime.call(reader, 'gnomon_hosted', args)
    other = store.create_project('other')['project_id']
    outsider = store.authenticate(store.issue_token(other, 'outsider', ['evidence.read', 'decision.create'])['token'])
    with pytest.raises(ServiceError):
        runtime.call(outsider, 'gnomon_hosted', {'action': 'resolve', 'reference': snap['reference']})
    with pytest.raises(ServiceError) as error:
        runtime.call(identity, 'gnomon_hosted', args | {'result': {'point': [4]}})
    assert error.value.code == 'IDEMPOTENCY_CONFLICT'
    # Forged numerical claims can be stored, but never receive server verification.
    analysis = {'action': 'analysis.submit', 'snapshot_id': snap['snapshot_id'], 'method': 'custom',
        'method_version': '1', 'metrics': {'mae': 999}, 'lesson': 'Client claim.', 'idempotency_key': 'claim'}
    with pytest.raises(ServiceError):
        runtime.call(outsider, 'gnomon_hosted', analysis)
    saved = runtime.call(identity, 'gnomon_hosted', analysis)['result']['analysis']
    assert saved['metrics']['mae'] == 999 and saved['numerically_verified'] is False
    with pytest.raises(ServiceError):
        runtime.call(identity, 'gnomon_hosted', analysis | {'numerically_verified': True})


def test_client_lesson_ditto_recall_never_claims_numeric_verification(tmp_path):
    from hosted_probe_support import ditto_peer
    store, identity, runtime, token, args = fixture(tmp_path)
    _, d = prepare(runtime, identity, args)
    snap = snapshot(runtime, identity, d)
    analysis = runtime.call(identity, 'gnomon_hosted', {'action': 'analysis.submit',
        'snapshot_id': snap['snapshot_id'], 'method': 'custom', 'method_version': 'v1',
        'metrics': {'mae': 999}, 'lesson': 'Deliberately incorrect client arithmetic to test trust labels.',
        'idempotency_key': 'analysis'})['result']
    store.configure_ditto(identity.project, 'https://api.heyditto.ai/mcp', 'TEST_DITTO_KEY', 'test-graph')
    with ditto_peer(tmp_path / 'peer') as endpoint:
        with store.connect() as conn:
            conn.execute('UPDATE projects SET ditto_url=? WHERE id=?', (endpoint, identity.project))
        queued = runtime.call(identity, 'gnomon_hosted', {'action': 'export.enqueue',
            'analysis_id': analysis['analysis_id'], 'recorded_as_of': now(), 'idempotency_key': 'export'})['result']
        with serving(store.root, env={'TEST_DITTO_KEY': 'fixture'}) as url:
            sent = call(url, token, 'gnomon_hosted', {'action': 'export.deliver', 'export_id': queued['export_id']})
            assert sent['state'] == 'acknowledged', sent
        runtime.call(identity, 'gnomon_ledger', {'operation': 'append_actual', 'series_id': 's', 'unit': 'widgets',
            'valid_time': args['request']['future_timestamps'][0], 'source_available_at': now(), 'value': 9,
            'idempotency_key': 'revision'})
        # Restart and fresh reader; resolve evidence and recompute locally.
        reader = store.issue_token(identity.project, 'fresh-reader', ['evidence.read'])['token']
        with serving(store.root, env={'TEST_DITTO_KEY': 'fixture'}) as url:
            recalled = call(url, reader, 'gnomon_hosted', {'action': 'memory.recall', 'query': 'client arithmetic',
                'source_as_of': now(), 'recorded_as_of': now()})
            assert recalled['returned'] == 1, recalled
            lesson = recalled['lessons'][0]
            assert lesson['numerically_verified'] is False and lesson['narrative_verified'] is False
            assert lesson['evidence_changed'] is True
            assert lesson['lesson']['metrics']['mae'] == 999
            assert score_snapshot(lesson['current_evidence'])['metrics']['mae'] == 6
            assert 'current_review' not in lesson  # No server numerical review in store-only mode.


def test_cli_local_forecast_exact_retry_and_local_analysis(tmp_path):
    import os
    store, identity, runtime, token, args = fixture(tmp_path)
    request_file, submission = tmp_path / 'request.json', tmp_path / 'submission.json'
    request_file.write_text(json.dumps(args['request']))
    def cli(url, *arguments, stdin=None):
        return subprocess.run([sys.executable, '-m', 'gnomon_hosted.client', '--url', url + '/mcp', *arguments],
            input=stdin, capture_output=True, text=True, timeout=30,
            env={**os.environ, 'GNOMON_SERVICE_TOKEN': token})
    with serving(store.root) as url:
        result = cli(url, 'forecast', '--request', str(request_file), '--provider', 'last_value',
            '--submission-file', str(submission), '--idempotency-key', 'cli-forecast')
        assert result.returncode == 0, result.stderr
        forecast = json.loads(result.stdout)
        assert submission.stat().st_mode & 0o777 == 0o600
        retry = cli(url, 'call', stdin=submission.read_text())
        assert json.loads(retry.stdout) == forecast
        again = cli(url, 'forecast', '--request', str(request_file), '--provider', 'last_value',
            '--submission-file', str(submission), '--idempotency-key', 'cli-forecast')
        assert again.returncode != 0
        with store.ledger(identity.project).transaction() as conn:
            assert conn.execute('SELECT COUNT(*) FROM executions').fetchone()[0] == 1
    _, d = prepare(runtime, identity, args)
    snap = snapshot(runtime, identity, d)
    reference_file = tmp_path / 'reference.json'
    reference_file.write_text(json.dumps(snap['reference']))
    with serving(store.root) as url:
        analysis = cli(url, 'analyze', '--snapshot-reference', str(reference_file), '--lesson', 'Test only.',
                       '--idempotency-key', 'cli-analysis')
        assert analysis.returncode == 0, analysis.stderr
        saved = json.loads(analysis.stdout)
        assert saved['local_computation']['metrics']['mae'] == 2
        assert saved['result']['analysis']['numerically_verified'] is False


def test_submitted_forecast_and_receipt_rollback_together(tmp_path):
    import sqlite3
    store, identity, runtime, _, args = fixture(tmp_path)
    with store.ledger(identity.project).transaction() as conn:
        conn.execute("CREATE TRIGGER fault BEFORE INSERT ON hosted_audit BEGIN SELECT RAISE(ABORT, 'fault'); END")
    with pytest.raises(sqlite3.IntegrityError):
        runtime.call(identity, 'gnomon_hosted', args)
    with store.ledger(identity.project).transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM executions').fetchone()[0] == 0
        assert conn.execute('SELECT result FROM hosted_requests').fetchone()[0] is None


def test_snapshot_missing_and_ineligible_evidence_and_no_scoring(tmp_path, monkeypatch):
    import gnomon.decision_memory
    store, identity, runtime, _, args = fixture(tmp_path)
    _, d = prepare(runtime, identity, args)
    def forbidden(*args, **kwargs):
        raise AssertionError('Snapshot must not invoke numerical scoring')
    monkeypatch.setattr(gnomon.decision_memory, 'point_error_metrics', forbidden)
    at = args['request']['future_timestamps'][0]
    packet = runtime.call(identity, 'gnomon_hosted', {'action': 'snapshot.save', 'decision_id': d,
        'source_as_of': at, 'recorded_as_of': now(), 'idempotency_key': 'missing'})['result']['snapshot']
    assert packet['actuals'] == []
    local = score_snapshot(packet)
    assert local['scoring_status'] == 'pending' and local['metrics']['mae'] is None
    with pytest.raises(ValueError):
        runtime.call(identity, 'gnomon_hosted', {'action': 'snapshot.save', 'decision_id': d,
            'source_as_of': at, 'recorded_as_of': at, 'idempotency_key': 'predates-decision'})


def test_atomic_analysis_export_and_failed_configuration(tmp_path):
    store, identity, runtime, _, args = fixture(tmp_path)
    _, d = prepare(runtime, identity, args)
    snap = snapshot(runtime, identity, d)
    analysis = {'action': 'analysis.submit', 'snapshot_id': snap['snapshot_id'], 'method': 'custom',
        'method_version': '1', 'metrics': {'mae': 2}, 'lesson': 'Synthetic example.', 'export_to_ditto': True,
        'idempotency_key': 'atomic-fails'}
    with pytest.raises(ServiceError) as error:
        runtime.call(identity, 'gnomon_hosted', analysis)
    assert error.value.code == 'DITTO_NOT_CONFIGURED'
    with store.ledger(identity.project).transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_records WHERE kind='client_analysis'").fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM hosted_exports').fetchone()[0] == 0
    store.configure_ditto(identity.project, 'https://api.heyditto.ai/mcp', 'TEST_KEY', 'test')
    args = analysis | {'idempotency_key': 'atomic-success'}
    saved = runtime.call(identity, 'gnomon_hosted', args)
    assert saved['result']['export']['state'] == 'pending'
    assert runtime.call(identity, 'gnomon_hosted', args) == saved
    with store.ledger(identity.project).transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_records WHERE kind='client_analysis'").fetchone()[0] == 1
        assert conn.execute('SELECT COUNT(*) FROM hosted_exports').fetchone()[0] == 1
