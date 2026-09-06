from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
import json
import sqlite3

import pytest

from gnomon import ForecastRequest, ForecastResult, GnomonSession, InferenceEngine, TemporalLedger
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock


def timestamp(day, hour=0):
    return f"2025-01-{day:02d}T{hour:02d}:00:00+00:00"


def set_clock(ledger, day, hour=0):
    ledger.clock = FixedClock(datetime.fromisoformat(timestamp(day, hour)))


def observation(day, value=10, available=None):
    return dict(series_id="sales", unit="USD", valid_time=timestamp(day), value=value,
                source_available_at=timestamp(available or day))


def request(origin=2, **changes):
    return replace(ForecastRequest(tuple(range(1, origin + 1)), 2, series_id="sales", unit="USD", frequency="D",
                   timestamps=tuple(timestamp(d) for d in range(1, origin + 1)),
                   future_timestamps=(timestamp(origin + 1), timestamp(origin + 2)),
                   cutoff=timestamp(origin), known_time_cutoff=timestamp(origin),
                   recorded_time_cutoff=timestamp(origin), snapshot_id=f"snapshot-{origin}"), **changes)


@pytest.fixture
def setup(tmp_path):
    ledger = TemporalLedger(tmp_path / "ledger.db")
    set_clock(ledger, 2, 12)
    engine = InferenceEngine(ledger=ledger, cache_size=10)
    for name, value in (("a", 8), ("b", 4)):
        engine.register(name, lambda r, v=value: ForecastResult((v,) * r.horizon, timestamps=r.future_timestamps,
                        series_id=r.series_id, unit=r.unit), revision="v1", deterministic=True)
    return ledger, engine


def history(ledger, **changes):
    args = dict(series_id="sales", unit="USD", horizon=2, providers={"a": "v1", "b": "v1"},
                start=timestamp(2), end=timestamp(4), source_as_of=timestamp(9), recorded_as_of=timestamp(9))
    return ledger.compare_history(**{**args, **changes})


def populate(ledger, engine):
    ids = []
    for origin in (2, 3, 4):
        set_clock(ledger, origin, 12)
        ids.extend(engine.forecast(p, request(origin)).execution_id for p in ("a", "b"))
    set_clock(ledger, 8)
    ledger.append_actual(actuals=[observation(d) for d in range(3, 7)])
    return ids


def test_feedback_survives_sessions_and_tracks_revisions(setup):
    ledger, engine = setup
    eid = engine.forecast("a", request()).execution_id
    assert ledger.search(series_id="sales")["items"][0]["status"] == "waiting"
    set_clock(ledger, 5)
    ledger.append_actual(actuals=[observation(3), observation(4)])
    fresh = TemporalLedger(ledger.path, clock=ledger.clock)
    row = fresh.search(status="ready")["items"][0]
    assert row["execution_id"] == eid and row["actuals_available"] == 2 and row["next_step"] == "evaluate"
    assert fresh.pending()[0]["status"] == "ready"  # The original disappearing-work bug.
    score = fresh.evaluate(execution_ids=[eid], allow_partial=False)[0]
    assert fresh.evaluate(eid) == score
    assert fresh.search(status="scored")["items"][0]["evaluation_id"] == score["evaluation_id"]
    set_clock(fresh, 8)
    fresh.append_actual(**observation(3, 30, available=6))
    stale = fresh.search(status="stale")["items"][0]
    assert stale["score_state"] == "stale" and stale["mae"] is None and stale["next_step"] == "rescore"
    assert fresh.search(recorded_as_of=timestamp(5))["items"][0]["score_state"] == "current"
    revised = fresh.evaluate(eid)
    assert revised["mae"] != score["mae"]
    assert len(fresh.evaluations(eid)) == 2
    # An explicit historical rescore must not make the current score look stale.
    fresh.evaluate(eid, recorded_as_of=timestamp(5))
    assert fresh.search()["items"][0]["evaluation_id"] == revised["evaluation_id"]


def test_partial_feedback_and_source_cutoffs(setup):
    ledger, engine = setup
    eid = engine.forecast("a", request()).execution_id
    set_clock(ledger, 5)
    ledger.append_actual(**observation(3))
    ledger.evaluate(eid)
    row = ledger.search()["items"][0]
    assert (row["status"], row["score_state"], row["actuals_available"]) == ("waiting", "current", 1)
    ledger.append_actual(**observation(4, available=8))
    assert ledger.search()["items"][0]["status"] == "waiting"
    row = ledger.search(source_as_of=timestamp(9))["items"][0]
    assert row["status"] == "stale" and row["actuals_available"] == 2


def test_search_filters_pagination_and_high_water_mark(setup):
    ledger, engine = setup
    ids = [engine.forecast("a", request()).execution_id for _ in range(3)]
    engine.forecast("b", request())
    engine.forecast("a", request(series_id="elsewhere"))
    args = dict(series_id="sales", provider="a", horizon=2, unit="USD", start=timestamp(2), end=timestamp(3), limit=1)
    page = ledger.search(**args)
    engine.forecast("a", request())
    found = [page["items"][0]["execution_id"]]
    while page["next_cursor"]:
        page = ledger.search(**args, cursor=page["next_cursor"])
        found.extend(r["execution_id"] for r in page["items"])
    assert found == ids
    cursor = ledger.search(**args)["next_cursor"]
    with pytest.raises(ForecastAdapterError, match="cursor"):
        ledger.search(**{**args, "provider": "b"}, cursor=cursor)
    assert ledger.search(start=timestamp(3))["items"] == []


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": True}, {"limit": 101}, {"horizon": False},
    {"status": "bogus"}, {"cursor": "bad cursor"}, {"cursor": "W10="}, {"start": "2025-01-01"},
    {"start": timestamp(5), "end": timestamp(2)}, {"provider": []}, {"unit": 1}])
def test_search_rejects_invalid_inputs(setup, kwargs):
    with pytest.raises(ForecastAdapterError):
        setup[0].search(**kwargs)


def test_status_filter_has_bounded_work_and_empty_continuation(setup):
    ledger, engine = setup
    for _ in range(201):
        engine.forecast("a", request())
    page = ledger.search(status="ready")
    assert page["items"] == [] and page["scanned"] == 200 and page["next_cursor"]
    page = ledger.search(status="ready", cursor=page["next_cursor"])
    assert page["scanned"] == 1 and page["next_cursor"] is None


def test_batch_actuals_are_atomic_idempotent_and_bounded(setup):
    ledger, _ = setup
    with pytest.raises(ForecastAdapterError):
        ledger.append_actual(actuals=[observation(3), {**observation(4), "value": float("nan")}])
    assert ledger.actuals_as_of("sales", unit="USD") == []
    values = [observation(3), observation(4)]
    assert ledger.append_actual(actuals=values) == ledger.append_actual(actuals=values)
    for values in ([], [observation(3)] * 1001, [{**observation(3), "unexpected": True}], "bad"):
        with pytest.raises(ForecastAdapterError):
            ledger.append_actual(actuals=values)
    with pytest.raises(ForecastAdapterError):
        ledger.append_actual(actuals=[observation(3)], series_id="mixed")


def test_batch_evaluations_are_atomic_idempotent_and_bounded(setup):
    ledger, engine = setup
    eid = engine.forecast("a", request()).execution_id
    with pytest.raises(ForecastAdapterError):
        ledger.evaluate(execution_ids=[eid, "unknown"])
    assert ledger.evaluations(eid) == []
    for ids in ([], [eid, eid], [1], [str(n) for n in range(101)], "bad"):
        with pytest.raises(ForecastAdapterError):
            ledger.evaluate(execution_ids=ids)
    with pytest.raises(ForecastAdapterError):
        ledger.evaluate(eid, execution_ids=[eid])
    with ThreadPoolExecutor(max_workers=6) as pool:
        scores = list(pool.map(lambda _: ledger.evaluate(eid), range(6)))
    assert len({s["evaluation_id"] for s in scores}) == 1
    assert len(ledger.evaluations(eid)) == 1


def test_scores_and_comparisons_reject_future_executions(setup):
    ledger, engine = setup
    ids = [engine.forecast(p, request()).execution_id for p in ("a", "b")]
    with pytest.raises(ForecastAdapterError, match="recorded"):
        ledger.evaluate(ids[0], recorded_as_of=timestamp(1))
    with pytest.raises(ForecastAdapterError, match="recorded"):
        ledger.compare(ids, recorded_as_of=timestamp(1))


def test_matched_history_reuses_predictions_and_does_not_write(setup):
    ledger, engine = setup
    ids = populate(ledger, engine)
    # A cached repeat is not an independent observation.
    set_clock(ledger, 4, 13)
    assert engine.forecast("a", request(4)).cache_hit
    with sqlite3.connect(ledger.path) as conn:
        before = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    result = history(ledger)
    assert result["status"] == "ok" and result["matched_origins"] == 3 and result["n"] == 6
    assert result["models"] == [{"provider": "a", "revision": "v1", "mae": 2},
                                {"provider": "b", "revision": "v1", "mae": 6}]
    assert result["duplicates_ignored"] == 1 and result["provider_calls"] == 0
    assert {m["execution_id"] for o in result["origins"] for m in o["models"]} == set(ids)
    with sqlite3.connect(ledger.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == before
    assert history(TemporalLedger(ledger.path))["models"] == result["models"]


def test_history_actual_revisions_and_recorded_replay(setup):
    ledger, engine = setup
    populate(ledger, engine)
    original = history(ledger)
    set_clock(ledger, 10)
    ledger.append_actual(**observation(3, 100, available=8))
    assert history(ledger)["models"] == original["models"]
    revised = history(ledger, recorded_as_of=timestamp(11))
    assert revised["models"] != original["models"]
    assert revised["origins"][0]["actual_ids"] != original["origins"][0]["actual_ids"]


@pytest.mark.parametrize("change,reason", [
    ({"history": (99, 2)}, "matched inputs"),
    ({"snapshot_id": "other"}, "matched inputs"),
    ({"known_time_cutoff": timestamp(3)}, "source_cutoff_after_origin"),
])
def test_history_refuses_unmatched_or_leaky_forecasts(setup, change, reason):
    ledger, engine = setup
    engine.forecast("a", request())
    engine.forecast("b", request(**change))
    set_clock(ledger, 8)
    ledger.append_actual(actuals=[observation(3), observation(4)])
    result = history(ledger)
    assert result["status"] == "insufficient_evidence"
    assert any(reason in r["reason"] for r in result["excluded"])


def test_history_reports_missing_models_actuals_and_versions(setup):
    ledger, engine = setup
    engine.forecast("a", request())
    result = history(ledger)
    assert result["excluded"][0]["missing_providers"] == ["b"]
    engine.forecast("b", request())
    assert history(ledger)["next_step"] == "supply_missing_actuals"
    result = history(ledger, providers={"a": "v2", "b": "v1"})
    assert any(r["reason"] == "provider_version_mismatch" for r in result["excluded"])


def test_history_accepts_origin_between_latest_observation_and_target(setup):
    ledger, engine = setup
    # The inspected-data path uses the snapshot's as_of, not necessarily the last observation time.
    req = request(cutoff=timestamp(2, 12), known_time_cutoff=timestamp(2, 12))
    for p in ("a", "b"):
        engine.forecast(p, req)
    set_clock(ledger, 8)
    ledger.append_actual(actuals=[observation(3), observation(4)])
    result = history(ledger)
    assert result["matched_origins"] == 1
    assert result["origins"][0]["origin"] == "2025-01-02T12:00:00.000000+00:00"


def test_history_refuses_an_origin_after_the_execution_clock(setup):
    ledger, engine = setup
    for p in ("a", "b"):
        engine.forecast(p, request(cutoff=timestamp(2, 13)))
    assert history(ledger)["excluded"][0]["reason"] == "forecast_origin_after_execution"


def test_history_excludes_post_target_forecasts_and_unknown_identity(setup):
    ledger, engine = setup
    set_clock(ledger, 3)
    engine.forecast("a", request())
    engine.forecast("b", request())
    assert history(ledger)["excluded"][0]["reason"] == "forecast_not_recorded_before_target"
    # Legacy execution payloads are preserved, never assigned inferred capabilities.
    set_clock(ledger, 4, 12)
    old = engine.forecast("a", request(4))
    ledger.record_execution(replace(old, execution_id="old-record", provider_identity=None))
    engine.forecast("b", request(4))
    assert any("provider_identity" in r["reason"] for r in history(ledger)["excluded"])


def test_history_does_not_choose_best_retry_or_mix_task_configurations(setup):
    ledger, engine = setup
    populate(ledger, engine)
    set_clock(ledger, 2, 13)
    engine.forecast("a", request(history=(20, 2)))
    result = history(ledger)
    assert result["matched_origins"] == 2
    assert any(r["reason"] == "ambiguous_inputs_at_origin" for r in result["excluded"])
    set_clock(ledger, 5, 12)
    for p in ("a", "b"):
        engine.forecast(p, request(5, season=7))
    set_clock(ledger, 8)
    ledger.append_actual(**observation(7))
    result = history(ledger, end=timestamp(5))
    assert result["status"] == "incompatible_evidence" and result["models"] == []


@pytest.mark.parametrize("kwargs", [{"providers": {"a": "latest", "b": "v1"}}, {"providers": {"a": "v1"}},
    {"providers": []}, {"providers": {"a": None, "b": "v1"}}, {"series_id": "__default__"},
    {"start": timestamp(5)}, {"end": timestamp(10)}, {"horizon": True}, {"recorded_as_of": "2025-01-09"}])
def test_history_validates_contract(setup, kwargs):
    with pytest.raises(ForecastAdapterError):
        history(setup[0], **kwargs)


def test_batch_and_discovery_tool_permissions_and_schema(setup):
    ledger, engine = setup
    eid = engine.forecast("a", request()).execution_id
    set_clock(ledger, 8)
    reader = GnomonSession(engine)
    args = {"operation": "append_actual", "actuals": [observation(3), observation(4)]}
    with pytest.raises(GnomonError, match="authorization"):
        reader.call("gnomon_ledger", args)
    writer = GnomonSession(engine, allow_outcome_writes=True)
    schema = next(t["inputSchema"] for t in writer.tools() if t["name"] == "gnomon_ledger")
    for call in (args, {"operation": "evaluate", "execution_ids": [eid], "allow_partial": False},
                 {"operation": "search", "series_id": "sales", "status": "scored", "limit": 1}):
        variants = [v for v in schema["oneOf"] if v["properties"]["operation"]["const"] == call["operation"]
                    and set(v["required"]) <= set(call) and set(call) <= set(v["properties"])]
        assert len(variants) == 1
        assert writer.call("gnomon_ledger", call)["status"] == "ok"
    assert reader.call("gnomon_ledger", {"operation": "search"})["result"]["items"][0]["score_state"] == "current"
    for v in schema["oneOf"]:
        if v["properties"]["operation"]["const"] == "evaluate":
            assert not {"execution_id", "execution_ids"} <= set(v["properties"])
    json.dumps(schema, allow_nan=False)


def test_study_discovery_respects_recording_time_and_keeps_truth_separate(setup):
    ledger, engine = setup
    run = engine.forecast("a", request())
    set_clock(ledger, 5)
    ledger.record_study({"study_id": "study-1", "evidence": "rolling_origin_backtest", "action_authorized": False,
                         "folds": [{"runs": {"a": {"execution_id": run.execution_id}}}]})
    assert ledger.search()["items"][0]["study_ids"] == ["study-1"]
    assert ledger.search(recorded_as_of=timestamp(4))["items"][0]["study_ids"] == []
    assert ledger.actuals_as_of("sales", unit="USD") == []
    assert any(e["reason"] == "not_production_evidence" for e in history(ledger)["excluded"])


def test_unknown_grid_is_discoverable_but_unscorable(setup):
    ledger, engine = setup
    engine.forecast("a", ForecastRequest((1, 2), 2))
    row = ledger.search(status="unscorable")["items"][0]
    assert row["missing_steps"] is None and row["next_step"] == "record_an_identified_timezone_aware_forecast"


def test_metric_version_changes_make_scores_stale(setup):
    ledger, engine = setup
    eid = engine.forecast("a", request()).execution_id
    set_clock(ledger, 8)
    ledger.append_actual(actuals=[observation(3), observation(4)])
    score = ledger.evaluate(eid)
    # Simulate a later runtime reading an older metric definition.
    import gnomon.ledger as module
    from unittest.mock import patch
    with patch.object(module, "_METRIC_VERSION", "point-errors/2"):
        assert ledger.search()["items"][0]["score_state"] == "stale"
        assert ledger.evaluate(eid)["evaluation_id"] != score["evaluation_id"]


@pytest.mark.parametrize("training,accepted", [(timestamp(1), True), (timestamp(3), False), (None, False)])
def test_pretrained_history_requires_declared_training_cutoff(setup, training, accepted):
    ledger, engine = setup
    engine.register("pretrained", lambda r: ForecastResult((8,) * r.horizon, timestamps=r.future_timestamps,
        series_id=r.series_id, unit=r.unit, metadata={} if training is None else {"training_cutoff": training}),
        revision="weights-v1", lifecycle="pretrained")
    engine.forecast("a", request())
    engine.forecast("pretrained", request())
    set_clock(ledger, 8)
    ledger.append_actual(actuals=[observation(3), observation(4)])
    result = history(ledger, providers={"a": "v1", "pretrained": "weights-v1"})
    assert (result["matched_origins"] == 1) is accepted
    if not accepted:
        assert "training_cutoff" in result["excluded"][0]["reason"]


def test_provider_capability_change_under_same_version_is_not_pooled(setup):
    from gnomon.forecast_adapter import AdapterCapabilities
    ledger, engine = setup
    populate(ledger, engine)
    engine._providers["a"].capabilities = AdapterCapabilities(max_horizon=20)
    set_clock(ledger, 5, 12)
    for p in ("a", "b"):
        engine.forecast(p, request(5))
    set_clock(ledger, 8)
    ledger.append_actual(**observation(7))
    result = history(ledger, end=timestamp(5))
    assert result["status"] == "incompatible_evidence" and result["models"] == []


def test_history_limit_fails_instead_of_ranking_a_truncated_window(setup):
    ledger, engine = setup
    run = engine.forecast("a", request())
    # Populate distinct executions cheaply in one transaction, just as recorded retries.
    with ledger._connect() as conn:
        for n in range(1000):
            ledger._insert_execution(conn, replace(run, execution_id=f"retry-{n}").to_dict())
    with pytest.raises(ForecastAdapterError, match="1000 executions"):
        history(ledger)


def test_cli_and_mcp_feedback_match_python(setup, tmp_path, capsys):
    from gnomon.cli import main
    from gnomon.mcp_server import _handle
    ledger, engine = setup
    populate(ledger, engine)
    config = tmp_path / "providers.toml"
    config.write_text('schema_version=1\nledger_path="ledger.db"\n')
    args = dict(operation="compare_history", series_id="sales", unit="USD", horizon=2,
                providers={"a": "v1", "b": "v1"}, start=timestamp(2), end=timestamp(4),
                source_as_of=timestamp(9), recorded_as_of=timestamp(9))
    expected = GnomonSession(engine).call("gnomon_ledger", args)
    assert main(["ledger", "--providers-config", str(config), "--arguments", json.dumps(args)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    response = _handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "gnomon_ledger", "arguments": args}}, session=GnomonSession(engine))
    assert not response["isError"]
    assert response["structuredContent"] == expected


@pytest.mark.parametrize("lead,expected", [(1, "ok"), (7, "incompatible_evidence")])
def test_unspecified_frequency_requires_matching_elapsed_leads(setup, lead, expected):
    ledger, engine = setup
    for origin, target in ((2, 3), (4, 4 + lead)):
        set_clock(ledger, origin)
        req = request(origin, horizon=1, frequency=None, future_timestamps=(timestamp(target),))
        for p in ("a", "b"):
            engine.forecast(p, req)
    set_clock(ledger, 15)
    ledger.append_actual(actuals=[observation(d) for d in (3, 4 + lead)])
    result = history(ledger, horizon=1, source_as_of=timestamp(15), recorded_as_of=timestamp(15))
    assert result["status"] == expected
    assert result["matched_origins"] == len(result["origins"]) == 2
    assert result["n"] == result["unique_actuals"] == 2
    assert bool(result["models"]) is (expected == "ok")


@pytest.mark.parametrize("frequency,targets,reason", [
    ("D", (timestamp(9), timestamp(10)), "forecast_grid_does_not_match_frequency"),
    ("D", (timestamp(3), timestamp(5)), "forecast_grid_does_not_match_frequency"),
    ("not-a-frequency", (timestamp(3), timestamp(4)), "forecast_grid_unresolved"),
])
def test_declared_grid_is_checked_before_history_aggregation(setup, frequency, targets, reason):
    ledger, engine = setup
    for p in ("a", "b"):
        engine.forecast(p, request(frequency=frequency, future_timestamps=targets))
    set_clock(ledger, 15)
    ledger.append_actual(actuals=[dict(series_id="sales", unit="USD", valid_time=t, source_available_at=t, value=10)
                                  for t in targets])
    result = history(ledger, source_as_of=timestamp(15), recorded_as_of=timestamp(15))
    assert result["status"] == "insufficient_evidence" and result["models"] == []
    assert result["matched_origins"] == result["n"] == result["unique_actuals"] == 0
    assert result["excluded"][0]["reason"] == reason


@pytest.mark.parametrize("frequency,origins,targets", [
    ("MS", ("2025-01-01T00:00:00+00:00", "2025-02-01T00:00:00+00:00"),
           ("2025-02-01T00:00:00+00:00", "2025-03-01T00:00:00+00:00")),
    ("D", ("2025-03-08T00:00:00-05:00", "2025-03-09T00:00:00-05:00"),
          ("2025-03-09T00:00:00-05:00", "2025-03-10T00:00:00-04:00")),
])
def test_calendar_grids_can_have_different_elapsed_step_lengths(setup, frequency, origins, targets):
    ledger, engine = setup
    for origin, target in zip(origins, targets):
        ledger.clock = FixedClock(datetime.fromisoformat(origin) + timedelta(hours=1))
        req = ForecastRequest((1,), 1, series_id="sales", unit="USD", frequency=frequency,
                              timestamps=(origin,), future_timestamps=(target,), cutoff=origin)
        for p in ("a", "b"):
            engine.forecast(p, req)
    cutoff = "2025-04-01T00:00:00+00:00"
    ledger.clock = FixedClock(datetime.fromisoformat(cutoff))
    ledger.append_actual(actuals=[dict(series_id="sales", unit="USD", valid_time=t, source_available_at=t, value=10)
                                  for t in targets])
    result = history(ledger, start=origins[0], end=origins[-1], horizon=1, source_as_of=cutoff, recorded_as_of=cutoff)
    assert result["status"] == "ok" and result["matched_origins"] == 2


def test_month_start_grid_cannot_change_time_of_day(setup):
    from gnomon.ledger_history import _grid_shape
    with pytest.raises(ForecastAdapterError, match="does_not_match_frequency"):
        _grid_shape({"timestamps": ["2025-01-01T00:00:00Z"], "future_timestamps": ["2025-02-01T12:00:00Z"],
                     "cutoff": None, "frequency": "MS"})


def test_declared_grid_does_not_hide_different_issue_leads(setup):
    ledger, engine = setup
    for origin, hour in ((2, 0), (4, 12)):
        set_clock(ledger, origin, 12)
        for p in ("a", "b"):
            engine.forecast(p, request(origin, cutoff=timestamp(origin, hour)))
    set_clock(ledger, 8)
    ledger.append_actual(actuals=[observation(d) for d in range(3, 7)])
    result = history(ledger, end=timestamp(4, 12))
    assert result["status"] == "incompatible_evidence" and result["models"] == []
    assert result["matched_origins"] == len(result["origins"]) == 2
    assert result["n"] == result["unique_actuals"] == 4


def test_pending_preserves_full_missing_steps_while_search_is_bounded(setup):
    ledger, engine = setup
    future = tuple((datetime.fromisoformat(timestamp(2)) + timedelta(days=d)).isoformat() for d in range(1, 31))
    eid = engine.forecast("a", request(horizon=30, future_timestamps=future)).execution_id
    ledger.clock = FixedClock(datetime.fromisoformat("2025-03-01T00:00:00Z"))
    ledger.append_actual(actuals=[dict(series_id="sales", unit="USD", valid_time=future[i], value=10,
                                      source_available_at=future[i]) for i in (0, 8, 22, 29)])
    missing = [i for i in range(30) if i not in (0, 8, 22, 29)]
    full = ledger.pending()[0]
    preview = ledger.search()["items"][0]
    assert full["execution_id"] == preview["execution_id"] == eid
    assert full["missing_steps"] == missing
    assert preview["missing_steps"] == missing[:20]
    assert full["missing_count"] == preview["missing_count"] == len(missing)


def test_null_units_are_supported_in_scalar_and_batch_tool_schemas(setup):
    ledger, engine = setup
    session = GnomonSession(engine, allow_outcome_writes=True)
    schemas = next(t["inputSchema"]["oneOf"] for t in session.tools() if t["name"] == "gnomon_ledger")
    actual = {**observation(3), "unit": None}
    for variant in schemas:
        properties = variant["properties"]
        if "unit" in properties:
            assert set(properties["unit"]["type"]) == {"string", "null"}
        if "actuals" in properties:
            assert "null" in properties["actuals"]["items"]["properties"]["unit"]["type"]
    scalar = session.call("gnomon_ledger", {"operation": "append_actual", **actual})["result"]
    batch = session.call("gnomon_ledger", {"operation": "append_actual", "actuals": [actual]})["result"]
    assert batch == [scalar]
    rows = session.call("gnomon_ledger", {"operation": "actuals_as_of", "series_id": "sales", "unit": None})["result"]
    assert rows[0]["actual_id"] == scalar


def test_oversized_cursor_is_an_argument_error_not_a_sqlite_failure(setup):
    import base64
    ledger, engine = setup
    for _ in range(2):
        engine.forecast("a", request())
    token = json.loads(base64.urlsafe_b64decode(ledger.search(limit=1)["next_cursor"]))
    token["upper"] = 2**80
    cursor = base64.urlsafe_b64encode(json.dumps(token).encode()).decode()
    with pytest.raises(ForecastAdapterError, match="invalid cursor"):
        ledger.search(cursor=cursor)
    with pytest.raises(GnomonError) as error:
        GnomonSession(engine).call("gnomon_ledger", {"operation": "search", "cursor": cursor})
    assert error.value.code == "INVALID_ARGUMENTS"


def test_unrepresentable_actual_is_rejected_and_the_batch_rolls_back(setup):
    ledger, engine = setup
    with pytest.raises(GnomonError) as error:
        GnomonSession(engine, allow_outcome_writes=True).call("gnomon_ledger", {
            "operation": "append_actual", "actuals": [observation(3), observation(4, value=10**1000)]})
    assert error.value.code == "INVALID_ARGUMENTS"
    assert ledger.actuals_as_of("sales", unit="USD") == []


def test_equivalent_frequency_aliases_share_a_comparison_grid(setup):
    ledger, engine = setup
    for origin, frequency in ((2, "D"), (4, "daily")):
        set_clock(ledger, origin)
        for p in ("a", "b"):
            engine.forecast(p, request(origin, frequency=frequency))
    set_clock(ledger, 8)
    ledger.append_actual(actuals=[observation(d) for d in range(3, 7)])
    result = history(ledger)
    assert result["status"] == "ok" and result["matched_origins"] == 2


def test_history_mean_does_not_overflow_for_representable_extreme_losses(setup):
    import sys
    ledger, engine = setup
    for origin in (2, 3, 4):
        set_clock(ledger, origin)
        for p in ("a", "b"):
            engine.forecast(p, request(origin, horizon=1, future_timestamps=(timestamp(origin + 1),)))
    set_clock(ledger, 8)
    ledger.append_actual(actuals=[observation(d, value=sys.float_info.max) for d in (3, 4, 5)])
    result = history(ledger, horizon=1)
    assert result["status"] == "ok" and result["matched_origins"] == 3
    assert all(m["mae"] == sys.float_info.max for m in result["models"])
