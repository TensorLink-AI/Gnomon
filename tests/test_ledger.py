from dataclasses import replace
from datetime import datetime
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import multiprocessing

import pytest

from gnomon import ForecastRequest, ForecastResult, InferenceEngine, TemporalLedger
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock


def clock(day):
    return FixedClock(datetime.fromisoformat(f"2025-01-{day:02d}T00:00:00+00:00"))


@pytest.fixture
def setup(tmp_path):
    ledger = TemporalLedger(tmp_path / "ledger.db", clock=clock(3))
    engine = InferenceEngine(ledger=ledger, cache_size=2)
    def model(r):
        return ForecastResult((2,) * r.horizon, timestamps=r.future_timestamps, series_id=r.series_id, unit=r.unit)
    engine.register("user", model, revision="r1", deterministic=True)
    request = ForecastRequest((1, 2), 2, series_id="shop", unit="USD", snapshot_id="frozen-input",
                              timestamps=("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
                              future_timestamps=("2025-01-03T00:00:00Z", "2025-01-04T00:00:00Z"))
    return ledger, engine, request


def actual(ledger, value=3, day=3, available_day=3):
    return ledger.append_actual(series_id="shop", valid_time=f"2025-01-{day:02d}T00:00:00Z",
                                value=value, source_available_at=f"2025-01-{available_day:02d}T00:00:00Z", unit="USD")


def test_execution_ids_are_distinct_and_payloads_are_durable_and_deduplicated(setup):
    ledger, engine, req = setup
    a, b = engine.forecast("user", req), engine.forecast("user", req)
    assert a.execution_id != b.execution_id and b.cache_hit
    first = ledger.execution(a.execution_id)
    second = TemporalLedger(ledger.path).execution(b.execution_id)
    assert first["payload_id"] == second["payload_id"]
    assert first["request"]["history"] == [1, 2]
    assert first["result"]["point"] == [2, 2]
    assert first["revision"] == "r1" and first["request"]["snapshot_id"] == "frozen-input"
    with sqlite3.connect(ledger.path) as conn:
        assert conn.execute("SELECT count(*) FROM payloads").fetchone()[0] == 1
        for command in ("DELETE FROM executions", "UPDATE payloads SET payload_json='{}'"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(command)


def test_actual_source_and_recorded_replay_differ_and_exact_reingest_is_idempotent(setup):
    ledger, _, _ = setup
    first = actual(ledger)
    ledger.clock = clock(10)
    correction = actual(ledger, value=30, available_day=4)
    assert actual(ledger, value=30, available_day=4) == correction
    assert ledger.actuals_as_of("shop", unit="USD", source_as_of="2025-01-05T00:00:00Z")[0]["actual_id"] == correction
    historical = ledger.actuals_as_of("shop", unit="USD", source_as_of="2025-01-05T00:00:00Z", recorded_as_of="2025-01-05T00:00:00Z")
    assert historical[0]["actual_id"] == first
    assert ledger.actuals_as_of("shop", unit="EUR") == []
    assert ledger.actuals_as_of("shop", unit="USD")[0]["revision"] == 1


def test_partial_complete_and_revised_scores_append_without_losing_old_results(setup):
    ledger, engine, req = setup
    run = engine.forecast("user", req)
    pending = ledger.evaluate(run.execution_id)
    assert pending["status"] == "pending" and pending["mae"] is None
    first = actual(ledger, value=3)
    partial = ledger.evaluate(run.execution_id)
    assert partial["n"] == 1 and partial["status"] == "partial" and partial["mae"] == 1
    assert partial["actual_ids"] == [first]
    assert ledger.pending()[0]["missing_steps"] == [1]
    with pytest.raises(ForecastAdapterError, match="complete actual"):
        ledger.evaluate(run.execution_id, allow_partial=False)
    ledger.clock = clock(4)
    actual(ledger, value=4, day=4, available_day=4)
    complete = ledger.evaluate(run.execution_id, allow_partial=False)
    assert complete["status"] == "complete" and complete["mae"] == 1.5
    assert ledger.pending() == []
    ledger.clock = clock(10)
    actual(ledger, value=30, available_day=5)
    revised = ledger.evaluate(run.execution_id)
    assert revised["mae"] == 15
    scores = ledger.evaluations(run.execution_id)
    assert [s["mae"] for s in scores] == [None, 1, 1.5, 15]
    assert len(ledger.evaluations(run.execution_id, recorded_as_of="2025-01-05T00:00:00Z")) == 3
    assert ledger.execution(run.execution_id)["result"]["point"] == [2, 2]


@pytest.mark.parametrize("value", ["false", "true", None, 0, 1, [], {}])
def test_invalid_partial_scoring_flag_is_rejected_without_appending(setup, value):
    from gnomon import GnomonSession
    from gnomon.contracts import GnomonError
    ledger, engine, req = setup
    run = engine.forecast("user", req)
    actual(ledger)
    with pytest.raises(ForecastAdapterError, match="allow_partial must be"):
        ledger.evaluate(run.execution_id, allow_partial=value)
    with GnomonSession(engine, ledger=ledger) as session, pytest.raises(GnomonError) as caught:
        session.call("gnomon_ledger", {"operation": "evaluate", "execution_id": run.execution_id,
                                      "allow_partial": value})
    assert caught.value.code == "INVALID_ARGUMENTS"
    assert ledger.evaluations(run.execution_id) == []


@pytest.mark.parametrize("magnitude", [1e200, 1e308])
def test_large_finite_errors_score_without_intermediate_overflow(setup, magnitude):
    ledger, engine, req = setup
    a, b = engine.forecast("user", req), engine.forecast("user", req)
    actual(ledger, value=magnitude)
    actual(ledger, value=magnitude, day=4, available_day=4)
    score = ledger.evaluate(a.execution_id, allow_partial=False)
    assert score["mae"] == pytest.approx(magnitude)
    assert score["rmse"] == pytest.approx(magnitude)
    assert score["bias"] == pytest.approx(-magnitude)
    assert all(row["mae"] == pytest.approx(magnitude)
               for row in ledger.compare([a.execution_id, b.execution_id])["models"])
    assert ledger.evaluations(a.execution_id)[0] == score


def test_unrepresentable_error_is_refused_without_appending(setup):
    ledger, engine, req = setup
    engine.register("large", lambda r: ForecastResult((1e308,) * r.horizon,
                    timestamps=r.future_timestamps, series_id=r.series_id, unit=r.unit))
    run = engine.forecast("large", req)
    actual(ledger, value=-1e308)
    with pytest.raises(ForecastAdapterError, match="finite numeric range"):
        ledger.evaluate(run.execution_id)
    assert ledger.evaluations(run.execution_id) == []


def test_matched_comparison_freezes_vintages_and_rejects_different_inputs(setup):
    ledger, engine, req = setup
    a, b = engine.forecast("user", req), engine.forecast("user", req)
    actual(ledger)
    comparison = ledger.compare([a.execution_id, b.execution_id])
    assert comparison["n"] == 1 and comparison["matched_steps"] == [0]
    assert [m["mae"] for m in comparison["models"]] == [1, 1]
    changed = engine.forecast("user", replace(req, history=(9, 2)))
    with pytest.raises(ForecastAdapterError, match="matched inputs"):
        ledger.compare([a.execution_id, changed.execution_id])


def test_decisions_and_outcome_corrections_do_not_execute_actions(setup):
    ledger, engine, req = setup
    run = engine.forecast("user", req)
    decision = ledger.record_decision(execution_ids=[run.execution_id], policy={"version": "1"},
                                      inputs={"stock": 1}, action={"order": 3})
    ledger.append_decision_outcome(decision, outcome={"sales": 2}, source_available_at="2025-01-04T00:00:00Z")
    ledger.append_decision_outcome(decision, outcome={"sales": 3}, source_available_at="2025-01-05T00:00:00Z")
    record = ledger.decision(decision)
    assert record["authorization_ref"] is None
    assert [o["outcome"]["sales"] for o in record["outcomes"]] == [2, 3]
    with pytest.raises(ForecastAdapterError):
        ledger.record_decision(execution_ids=["unknown"], inputs={}, policy={}, action={})


def test_concurrent_append_preserves_revisions(setup):
    ledger, _, _ = setup
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda n: actual(ledger, value=n), range(8)))
    assert len(set(ids)) == 8
    with sqlite3.connect(ledger.path) as conn:
        assert sorted(row[0] for row in conn.execute("SELECT revision FROM actuals")) == list(range(8))


def _process_append(path, value):
    # Each spawned process creates its own ledger/SQLite connections. The shared
    # correction must deduplicate even while distinct corrections are appended.
    ledger = TemporalLedger(path, clock=clock(3))
    return actual(ledger, value=value), actual(ledger, value=1000)


def test_separate_processes_initialize_and_append_without_lost_revisions(tmp_path):
    path = tmp_path / "concurrent.db"
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_process_append, [path] * 8, range(8)))
    assert len({distinct for distinct, _ in results}) == 8
    assert len({shared for _, shared in results}) == 1
    with sqlite3.connect(path) as conn:
        assert sorted(row[0] for row in conn.execute("SELECT revision FROM actuals")) == list(range(9))
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_failed_study_insert_rolls_back_payload_and_metadata(setup):
    ledger, engine, req = setup
    original = engine.forecast("user", req)
    report = {"study_id": "failed-study", "evidence": "rolling_origin_backtest",
              "action_authorized": False, "folds": [{"runs": {
                  "missing": {"execution_id": "not-recorded"}}}]}
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        ledger.record_study(report)
    with sqlite3.connect(ledger.path) as conn:
        assert conn.execute("SELECT count(*) FROM payloads").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM studies").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM study_executions").fetchone()[0] == 0
    assert ledger.execution(original.execution_id)["result"]["point"] == [2, 2]


def test_failed_schema_upgrade_is_atomic_and_can_be_retried(setup, monkeypatch):
    ledger, engine, req = setup
    original = engine.forecast("user", req)
    with sqlite3.connect(ledger.path) as conn:
        conn.execute("DROP TABLE study_executions")
        conn.execute("DROP TABLE studies")
        conn.execute("PRAGMA user_version=2")
        before = conn.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall()
    connect = TemporalLedger._connect

    @contextmanager
    def fail_new_trigger(self):
        with connect(self) as conn:
            conn.set_authorizer(lambda action, name, *_: sqlite3.SQLITE_DENY
                                if action == sqlite3.SQLITE_CREATE_TRIGGER
                                and name == "immutable_studies_UPDATE" else sqlite3.SQLITE_OK)
            yield conn

    with monkeypatch.context() as patch:
        patch.setattr(TemporalLedger, "_connect", fail_new_trigger)
        with pytest.raises(sqlite3.DatabaseError, match="authorized"):
            TemporalLedger(ledger.path)
    with sqlite3.connect(ledger.path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall() == before
    upgraded = TemporalLedger(ledger.path)
    assert upgraded.execution(original.execution_id)["result"]["point"] == [2, 2]
    with sqlite3.connect(ledger.path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3


def test_ledger_rejects_naive_time_and_other_databases(setup, tmp_path):
    ledger, _, _ = setup
    with pytest.raises(ForecastAdapterError, match="explicit timezone"):
        ledger.append_actual(series_id="shop", valid_time="2025-01-03", value=2, source_available_at="2025-01-03")
    other = tmp_path / "other.db"
    with sqlite3.connect(other) as conn:
        conn.execute("CREATE TABLE existing (id TEXT)")
    with pytest.raises(ForecastAdapterError, match="separate ledger"):
        TemporalLedger(other)
    with sqlite3.connect(ledger.path) as conn:
        conn.execute("PRAGMA user_version=999")
    with pytest.raises(ForecastAdapterError, match="schema version"):
        TemporalLedger(ledger.path)
