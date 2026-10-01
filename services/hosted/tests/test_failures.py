from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest
from gnomon_hosted.runtime import Runtime
from gnomon_hosted.storage import Store, ServiceError, now
from gnomon_hosted import worker


def setup(tmp_path):
    store = Store.initialize(tmp_path / 'service')
    project = store.create_project('test')['project_id']
    token = store.issue_token(project, 'agent', ['actual.create', 'evidence.read', 'forecast.create'])['token']
    return store, store.authenticate(token), Runtime(store)


def actual():
    at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    return {'operation': 'append_actual', 'series_id': 's', 'valid_time': at,
            'source_available_at': at, 'value': 1, 'idempotency_key': 'one'}


def test_concurrent_duplicate_is_one_effect_and_one_receipt(tmp_path):
    store, identity, runtime = setup(tmp_path)
    args = actual()
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda _: runtime.call(identity, 'gnomon_ledger', args), range(12)))
    assert len({r['request_id'] for r in replies}) == 1
    ledger = store.ledger(identity.project)
    assert len(ledger.actuals_as_of('s')) == 1
    with ledger.transaction() as conn:
        assert conn.execute('SELECT COUNT(*) FROM hosted_requests').fetchone()[0] == 1
    # Simulate lost response: fetch the same stable receipt in a fresh runtime.
    result = Runtime(Store(store.root)).call(identity, 'gnomon_ledger', args)
    assert result['state'] == 'completed'
    rid = ledger.actuals_as_of('s')[0]['actual_id']
    assert ledger.actual(rid)['revision'] == 0


def test_receipt_failure_rolls_back_core_actual(tmp_path):
    store, identity, runtime = setup(tmp_path)
    ledger = store.ledger(identity.project)
    with ledger.transaction() as conn:
        conn.execute("CREATE TRIGGER inject_failure BEFORE INSERT ON hosted_audit BEGIN SELECT RAISE(ABORT, 'fault'); END")
    with pytest.raises(sqlite3.IntegrityError):
        runtime.call(identity, 'gnomon_ledger', actual())
    assert ledger.actuals_as_of('s') == []
    with ledger.transaction() as conn:
        row = conn.execute('SELECT state,result FROM hosted_requests').fetchone()
        assert row['state'] == 'outcome_unknown' and row['result'] is None


def test_provider_unknown_is_durable_and_not_retried(tmp_path, monkeypatch):
    store, identity, runtime = setup(tmp_path)
    calls = []
    def fail(*args):
        calls.append(1)
        raise ServiceError('OUTCOME_UNKNOWN', 'Injected provider timeout')
    monkeypatch.setattr(worker, 'forecast', fail)
    at = datetime.now(timezone.utc)
    args = {'provider': 'last_value', 'request': {'series_id': 's', 'history': [1, 2], 'horizon': 1,
        'timestamps': [(at - timedelta(days=d)).isoformat() for d in (2, 1)],
        'future_timestamps': [(at + timedelta(days=1)).isoformat()]}, 'idempotency_key': 'forecast'}
    with pytest.raises(ServiceError, match='timeout'):
        runtime.call(identity, 'gnomon_forecast', args)
    store.recover()
    retry = Runtime(Store(store.root)).call(identity, 'gnomon_forecast', args)
    assert retry['state'] == 'outcome_unknown' and len(calls) == 1


def test_crash_recovery_preserves_completed_records(tmp_path):
    store, identity, runtime = setup(tmp_path)
    first = runtime.call(identity, 'gnomon_ledger', actual())
    ledger = store.ledger(identity.project)
    with ledger.transaction() as conn:
        conn.execute('INSERT INTO hosted_requests VALUES(?,?,?,?,?,?,?,?,?)', ('unfinished', identity.principal,
            'unfinished', 'digest', 'forecast', 'running', None, now(), now()))
    Store(store.root).recover()
    with ledger.transaction() as conn:
        assert conn.execute('SELECT state FROM hosted_requests WHERE id=?', (first['request_id'],)).fetchone()[0] == 'completed'
        assert conn.execute("SELECT state FROM hosted_requests WHERE id='unfinished'").fetchone()[0] == 'outcome_unknown'
    assert len(ledger.actuals_as_of('s')) == 1


def test_remote_fields_and_privileges_fail_closed(tmp_path):
    store, identity, runtime = setup(tmp_path)
    bad = actual() | {'recorded_at': '2020-01-01T00:00:00Z'}
    with pytest.raises(Exception):
        runtime.call(identity, 'gnomon_ledger', bad)
    assert store.ledger(identity.project).actuals_as_of('s') == []
    for name, args in [('gnomon_python', {}), ('gnomon_ledger', {'operation': 'sql', 'query': 'SELECT 1'}),
                       ('gnomon_forecast', {'provider': 'last_value', 'path': '/etc/passwd', 'idempotency_key': 'p'})]:
        with pytest.raises(ServiceError):
            runtime.call(identity, name, args)


def test_real_provider_process_timeout_is_bounded():
    import multiprocessing
    import time
    before = {p.pid for p in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(ServiceError, match='timed out'):
        worker.forecast(None, 'last_value', {'history': [1, 2], 'horizon': 1}, timeout=.0001)
    assert time.monotonic() - started < 5
    assert {p.pid for p in multiprocessing.active_children()} <= before


def test_backup_tampering_is_rejected_before_restore(tmp_path):
    from gnomon_hosted.cli import backup, restore
    store, identity, runtime = setup(tmp_path)
    runtime.call(identity, 'gnomon_ledger', actual())
    bundle, target = tmp_path / 'backup', tmp_path / 'restored'
    backup(store.root, bundle)
    with open(bundle / 'control.db', 'ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ServiceError, match='manifest'):
        restore(bundle, target)
    assert not target.exists()


def test_receipt_reports_core_writes_after_enclosing_commit(tmp_path):
    store, identity, runtime = setup(tmp_path)
    args = actual()
    first = runtime.call(identity, 'gnomon_ledger', args)
    assert first['result']['execution_diagnostics']['ledger_writes'] == 1
    duplicate = runtime.call(identity, 'gnomon_ledger', args)
    assert duplicate == first  # The original committed receipt, not a new write.
    second = runtime.call(identity, 'gnomon_ledger', args | {'idempotency_key': 'new-request-same-actual'})
    assert second['result']['execution_diagnostics']['ledger_writes'] == 0
    assert len(store.ledger(identity.project).actuals_as_of('s')) == 1
