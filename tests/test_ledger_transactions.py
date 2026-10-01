"""Application receipts must commit or roll back with their core ledger effects."""
import sqlite3
import pytest
from gnomon import TemporalLedger


def actual(value=1):
    return dict(series_id='s', valid_time='2026-01-01T00:00:00Z', value=value,
                source_available_at='2026-01-02T00:00:00Z')


def test_extension_transaction_rolls_back_receipt_and_actual(tmp_path):
    ledger = TemporalLedger(tmp_path / 'ledger.db')
    with ledger.transaction() as conn:
        conn.execute('CREATE TABLE receipts(id TEXT PRIMARY KEY)')
    with pytest.raises(RuntimeError):
        with ledger.transaction() as conn:
            ledger.append_actual(**actual())
            conn.execute("INSERT INTO receipts VALUES('one')")
            raise RuntimeError('simulated response preparation failure')
    assert ledger.actuals_as_of('s') == []
    with ledger.transaction() as conn:
        assert conn.execute('SELECT count(*) FROM receipts').fetchone()[0] == 0
        aid = ledger.append_actual(**actual())
        conn.execute('INSERT INTO receipts VALUES(?)', (aid,))
    reopened = TemporalLedger(ledger.path, create=False)
    assert reopened.actuals_as_of('s')[0]['actual_id'] == aid


def test_caught_nested_failure_does_not_leak_partial_mutation(tmp_path):
    ledger = TemporalLedger(tmp_path / 'ledger.db')
    with ledger.transaction():
        try:
            with ledger.transaction() as conn:
                ledger.append_actual(**actual())
                conn.execute('INSERT INTO missing_table VALUES(1)')
        except sqlite3.OperationalError:
            pass
        ledger.append_actual(**actual(2))
    assert [r['value'] for r in ledger.actuals_as_of('s')] == [2]
    assert ledger.actuals_as_of('s')[0]['revision'] == 0
