"""Task-preserving recovery and evidence diagnostics from independent 1.1.2 reports."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import shlex

import pytest

from gnomon import ForecastResult, GnomonSession, InferenceEngine
from gnomon.cli import main
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock
from gnomon.mcp_server import _handle
from gnomon.result_refs import ResultLimits, ResultReferences, encode
from gnomon.session import ledger_schema
from test_acceptance_recovery import actual, cli, forecast, stamp
from test_backtesting import at, configured


@pytest.mark.parametrize("amount,expected", [(7, "2026-05-17"), (0, "2026-05-10"), (-7, "2026-05-03")])
def test_shift_recovery_preserves_task_and_exposes_semantic_choice(amount, expected):
    request = dict(operation="shift", value="2026-05-10", amount=amount, unit="days")
    before = deepcopy(request)
    with GnomonSession(enable_temporal=True) as session:
        response = _handle({"method": "tools/call", "params": {
            "name": "gnomon_temporal", "arguments": request}}, session=session)
        details = response["structuredContent"]["error"]["details"]
        assert details["example_arguments"] == dict(request, mode="calendar")
        assert details["changed_fields"] == ["mode"]
        assert details["choices_required"] == {"mode": ["calendar", "elapsed"]}
        assert request == before
    code, result = cli("temporal", "--arguments", json.dumps(details["example_arguments"]))
    assert code == 0 and result["result"]["date"] == expected


def test_temporal_month_end_recovery_preserves_original_request_and_exposes_choice():
    request = dict(operation="shift", value="2026-01-31", amount=1, unit="months", mode="calendar")
    _, response = cli("temporal", "--arguments", json.dumps(request))
    details = response["error"]["details"]
    assert details["example_kind"] == "parameter_preserving_example"
    assert details["supplied_arguments"] == request
    assert details["changed_fields"] == ["invalid_date"]
    assert details["choices_required"] == {"invalid_date": ["reject", "clamp"]}
    assert cli("temporal", "--arguments", json.dumps(details["example_arguments"]))[0] == 0


def test_interval_recovery_preserves_valid_nested_fields():
    request = dict(operation="interval", left={"start": "2026-01-02T00:00:00Z"},
                   right={"start": "2026-01-03T00:00:00Z", "end": "2026-01-06T00:00:00Z"})
    _, response = cli("temporal", "--arguments", json.dumps(request))
    recovery = response["error"]["details"]["example_arguments"]
    assert recovery["left"]["start"] == request["left"]["start"]
    assert recovery["right"] == request["right"]
    assert cli("temporal", "--arguments", json.dumps(recovery))[0] == 0


@pytest.mark.parametrize("batch", [False, True])
def test_live_coverage_refreshes_without_rewriting_score_evidence(tmp_path, batch):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        session.ledger.clock = FixedClock(datetime(2026, 1, 20, tzinfo=timezone.utc))
        eid = forecast(session)
        session.ledger.clock = FixedClock(datetime(2026, 1, 24, tzinfo=timezone.utc))
        actual(session, 21)
        query = dict(operation="evaluate", **({"execution_ids": [eid]} if batch else {"execution_id": eid}))
        def score(arguments):
            result = session.call("gnomon_ledger", arguments, compact=False)["result"]
            return result[0] if batch else result
        first = score(query)
        saved = session.ledger.evaluations(eid)
        session.ledger.clock = FixedClock(datetime(2026, 1, 25, tzinfo=timezone.utc))
        actual(session, 22, None, available=25)
        second = score(query)
        assert second["evaluation_reused"] and not first["evaluation_reused"]
        assert second["evaluation_id"] == first["evaluation_id"]
        assert second["coverage"]["other_units_at_missing_steps"] == []
        assert second["current_coverage"]["other_units_at_missing_steps"] == [None]
        assert second["current_coverage"]["fraction"] == 0.5
        assert second["coverage_basis"] == "saved_evaluation"
        assert second["current_coverage_basis"] == "current_query"
        assert session.ledger.evaluations(eid) == saved
        assert session.ledger.evaluate(eid) == saved[0]
        for field in ("source_as_of", "recorded_as_of"):
            hidden = score(dict(query, **{field: stamp(24)}))
            assert hidden["current_coverage"]["other_units_at_missing_steps"] == []
        with pytest.raises(GnomonError) as exc:
            score(dict(query, allow_partial=False))
        assert exc.value.details["coverage"]["other_units_at_missing_steps"] == [None]
        actual(session, 22)
        complete = score(query)
        assert complete["complete"] and complete["mae"] == 1.5
        assert complete["evaluation_id"] != first["evaluation_id"]
        assert complete["current_coverage"]["other_units_at_missing_steps"] == []


@pytest.mark.parametrize("fields", [(), ("source_as_of",), ("recorded_as_of",), ("source_as_of", "recorded_as_of")])
def test_cutoff_metadata_distinguishes_supplied_and_defaulted(tmp_path, fields):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        eid = forecast(session)
        for operation, identity in (("evaluate", {"execution_id": eid}),
                                    ("actuals_as_of", {"series_id": "sales"}), ("search", {}), ("pending", {})):
            query = session.call("gnomon_ledger", dict(operation=operation, **identity,
                **{field: "2099-01-01T00:00:00Z" for field in fields}), compact=False)["query"]
            for field in ("source_as_of", "recorded_as_of"):
                assert query[field + "_defaulted"] is (field not in fields)
                if field in fields or operation in ("search", "pending"):
                    assert query[field] is not None
            assert query["omitted_cutoffs"] == (None if len(fields) == 2 else query["cutoff_default"])
        query = session.call("gnomon_ledger", dict(operation="actuals_as_of", series_id="sales", unit="widgets"))["query"]
        assert query["unit_defaulted"] is False and query["omitted_unit"] is None


@pytest.mark.parametrize("folds,budget,expected", [(2, 4, 2), (3, 6, 3), (3, 4, 2)])
def test_routing_readiness_uses_actual_matched_folds(tmp_path, folds, budget, expected):
    session, ref = configured(tmp_path, ledger=True)
    with session:
        report = session.evaluate(ref, candidates=["trend"], baseline="last_value", horizon=2,
                                  folds=folds, budget={"max_calls": budget})
        readiness = report["routing_readiness"]
        assert readiness["matched_folds"] == expected and readiness["default_min_folds"] == 3
        assert readiness["ready"] is (expected >= 3)
        assert any(issue["action"] == "evaluate_more_matched_folds" for issue in readiness["issues"]) is (expected < 3)
        routed = session.route(ref, study_id=report["study_id"], candidates=["trend"], baseline="last_value",
                               horizon=2, source_as_of=at(30).isoformat(), recorded_as_of=at(35).isoformat())
        assert (routed.get("reason") == "insufficient_replayable_matched_folds") is (expected < 3)


def test_schema_index_entries_execute_and_cache_example_hits(tmp_path):
    code, index = cli("schemas")
    assert code == 0
    for command in index["schemas"].values():
        code, schema = cli(*shlex.split(command)[1:])
        assert code == 0 and schema["type"] == "object"
    config = tmp_path / "providers.toml"
    config.write_text("cache_size = 8\n")
    with GnomonSession.from_config(config) as session:
        assert "cache_size = 8" in session.capabilities()["cache"]["enable"]
        request = {"history": [1, 2, 3], "horizon": 2}
        assert not session.forecast("last_value", request)["cache_hit"]
        assert session.forecast("last_value", request)["cache_hit"]


def test_series_selector_is_explained_before_execution(capsys):
    with pytest.raises(SystemExit):
        main(["infer", "--help"])
    help_text = capsys.readouterr().out
    assert "Select an existing series" in help_text and "__default__" in help_text
    with GnomonSession.from_config() as session:
        tool = next(t for t in session.tools() if t["name"] == "gnomon_forecast")
        assert "not a new label" in tool["inputSchema"]["oneOf"][1]["properties"]["series_id"]["description"]


def test_custom_provider_return_recovery():
    with InferenceEngine() as engine:
        engine.register("bad", lambda r: {"point": [1]})
        with pytest.raises(ForecastAdapterError, match=r"ForecastResult\(point="):
            engine.forecast("bad", {"history": [1, 2], "horizon": 2})
        engine.register("fixed", lambda request: ForecastResult(point=(request.history[-1],) * request.horizon))
        assert engine.forecast("fixed", {"history": [1, 2], "horizon": 2}).result.point == (2, 2)


@pytest.mark.parametrize("value", [0, 0.0, 2.5, -2.5])
def test_ledger_actual_recovery_preserves_numeric_observation(tmp_path, value):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        session.allow_outcome_writes = True
        request = dict(operation="append_actual", series_id="sales", valid_time=stamp(21),
                       value=value, unit="widgets", source_ref="synthetic-zero-or-fraction")
        with pytest.raises(GnomonError) as exc:
            session.call("gnomon_ledger", request, compact=False)
        details = exc.value.details
        example = details["example_arguments"]
        assert example == dict(request, source_available_at="2026-02-01T00:00:00Z")
        assert type(example["value"]) is type(value)
        assert details["changed_fields"] == ["source_available_at"]
        assert details["example_kind"] == "task_template"
        assert not session.ledger.actuals_as_of("sales", unit="widgets")
        # Supply the fixture's actual availability, not an inferred date.
        example["source_available_at"] = stamp(23)
        session.call("gnomon_ledger", example, compact=False)
        assert session.ledger.actuals_as_of("sales", unit="widgets")[0]["value"] == value


def test_bad_actual_batch_does_not_become_an_unrelated_single_write(tmp_path):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        session.allow_outcome_writes = True
        rows = [dict(series_id="sales", valid_time=stamp(21), value=0, source_available_at=stamp(23)),
                dict(series_id="sales", valid_time=stamp(22), value=2.5)]
        with pytest.raises(GnomonError) as exc:
            session.call("gnomon_ledger", dict(operation="append_actual", actuals=rows))
        details = exc.value.details
        assert details["example_arguments"] == dict(operation="append_actual", actuals=rows)
        assert details["example_kind"] == "task_template"
        assert not session.ledger.actuals_as_of("sales")
        example = details["example_arguments"]
        example["actuals"][1]["source_available_at"] = stamp(23)
        session.call("gnomon_ledger", example)
        assert [r["value"] for r in session.ledger.actuals_as_of("sales")] == [0, 2.5]
        assert "source_available_at" not in rows[1]


def test_nonfinite_batch_error_remains_a_structured_mcp_error(tmp_path):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        session.allow_outcome_writes = True
        rows = [dict(series_id="sales", valid_time=stamp(21), value=0, source_available_at=stamp(23)),
                dict(series_id="sales", valid_time=stamp(22), value=float("inf"), source_available_at=stamp(23))]
        response = _handle({"method": "tools/call", "params": {"name": "gnomon_ledger",
            "arguments": dict(operation="append_actual", actuals=rows)}}, session=session)
        assert response["isError"]
        error = response["structuredContent"]["error"]
        assert error["code"] == "INVALID_ARGUMENTS"
        example = error["details"]["example_arguments"]["actuals"]
        assert example[0]["value"] == 0
        assert example[1]["value"] == "REPLACE_NONFINITE_VALUE"
        assert not session.ledger.actuals_as_of("sales")


def test_forecast_recovery_keeps_observed_history_and_request_identity():
    request = dict(history=[101, 102, 103, 104, 105, 106, 107], horizon=0, season=7, unit="widgets", series_id="sales")
    with GnomonSession.from_config() as session:
        with pytest.raises(GnomonError) as exc:
            session.call("gnomon_forecast", dict(provider="seasonal_naive", request=request, use_cache=False))
        details = exc.value.details
        example = details["example_arguments"]
        assert example == dict(provider="seasonal_naive", request=dict(request, horizon=2), use_cache=False)
        assert details["changed_fields"] == ["request.horizon"]
        assert session.call("gnomon_forecast", example)["result"]["point"] == (101, 102)
    _, response = cli("infer", "--provider", "seasonal_naive", "--request", json.dumps(request))
    corrected = response["error"]["details"]["example_arguments"]
    assert corrected["history"] == request["history"] and corrected["season"] == 7
    assert cli("infer", "--provider", "seasonal_naive", "--request", json.dumps(corrected))[1]["result"]["point"] == [101, 102]


def test_invalid_history_is_not_replaced_by_synthetic_observations():
    with GnomonSession.from_config() as session:
        with pytest.raises(GnomonError) as exc:
            session.call("gnomon_forecast", dict(provider="last_value", request=dict(history=[1, "bad"], horizon=2)))
        details = exc.value.details
        assert "history" not in details["example_arguments"]["request"]
        assert details["example_kind"] == "task_template"
        assert "real history" in details["guidance"]


def test_frozen_reference_recovery_keeps_reference_and_parameters(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("timestamp,value\n2026-01-01,101\n2026-01-02,102\n")
    with GnomonSession.from_config() as session:
        ref = session.data.inspect(str(data), frequency="D")["data_ref"]
        request = dict(provider="last_value", data_ref=ref, horizon=0, series_id="__default__", season=2, use_cache=False)
        with pytest.raises(GnomonError) as exc:
            session.call("gnomon_forecast", request)
        example = exc.value.details["example_arguments"]
        assert example == dict(request, horizon=2)
        assert session.call("gnomon_forecast", example)["result"]["point"] == (102, 102)


def test_comparison_recovery_requires_two_real_matching_ids(tmp_path):
    schema = next(v for v in ledger_schema()["oneOf"] if v["properties"]["operation"]["const"] == "compare")
    assert schema["properties"]["execution_ids"]["minItems"] == 2
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        one, two = forecast(session), forecast(session)
        for ids in ([one], [one, one], []):
            with pytest.raises(GnomonError) as exc:
                session.call("gnomon_ledger", dict(operation="compare", execution_ids=ids))
            details = exc.value.details
            example = details["example_arguments"]
            assert len(example["execution_ids"]) == len(set(example["execution_ids"])) == 2
            assert details["placeholder_execution_ids"]
            if ids:
                assert example["execution_ids"][0] == one
            found = session.call("gnomon_ledger", details["discovery_arguments"], compact=False)["result"]["items"]
            assert {r["execution_id"] for r in found} == {one, two}
            example["execution_ids"] = [one, two]
            assert session.call("gnomon_ledger", example)["status"] == "ok"


def test_comparison_enforces_advertised_upper_bound_before_reading_executions(tmp_path, monkeypatch):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        one, two = forecast(session), forecast(session)
        # Record matched execution retries with distinct IDs in one transaction.
        run = session.ledger.execution(one)
        with session.ledger._connect() as conn:
            for i in range(99):
                conn.execute("INSERT INTO executions VALUES (?,?,?,?,?)", (
                    f"bounded-{i}", run["fingerprint"], run["payload_id"], run["recorded_at"], 0))
        ids = [one, two, *[f"bounded-{i}" for i in range(99)]]
        assert session.call("gnomon_ledger", dict(operation="compare", execution_ids=ids[:100]), compact=False)["status"] == "ok"
        def unexpected_connection():
            pytest.fail("invalid comparison must be rejected before opening the ledger")
        monkeypatch.setattr(session.ledger, "_connect", unexpected_connection)
        for invalid in (ids, "one", [one, {}], [one, ""]):
            with pytest.raises(GnomonError, match="at most 100"):
                session.call("gnomon_ledger", dict(operation="compare", execution_ids=invalid), compact=False)


def test_legacy_score_reconstruction_is_not_labeled_saved_coverage(tmp_path):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        eid = forecast(session)
        original = session.ledger.evaluate(eid)
        legacy = {k: v for k, v in original.items() if k not in ("coverage", "complete")}
        legacy["evaluation_id"] = "synthetic-legacy-evaluation"
        # Append a fixture in the pre-coverage payload shape; no existing evidence is modified.
        with session.ledger._connect() as conn:
            conn.execute("INSERT INTO evaluations VALUES (?,?,?,?)",
                         (legacy["evaluation_id"], eid, legacy["recorded_at"], json.dumps(legacy)))
        actual(session, 21, unit="other")
        for result in (session.ledger.evaluate(eid), session.call("gnomon_ledger", dict(operation="evaluate", execution_id=eid))["result"]):
            assert result["evaluation_id"] == legacy["evaluation_id"]
            assert result["coverage_basis"] == "reconstructed_current_query"
            assert result["coverage"]["other_units_at_missing_steps"] == ["other"]
        assert session.ledger.evaluations(eid) == [original, legacy]


@pytest.mark.parametrize("status,n,complete", [("pending", 0, False), ("partial", 1, False), ("complete", 2, True)])
@pytest.mark.parametrize("limit", [2048, 8192])
def test_large_score_summary_keeps_completion_and_both_coverage_counts(status, n, complete, limit):
    refs = ResultReferences(ResultLimits(max_response_bytes=limit))
    try:
        coverage = dict(required_steps=2, matched_steps=n, fraction=n/2, unit="widgets", missing_steps=list(range(n, 2)),
                        other_units_at_missing_steps=[])
        current = dict(coverage, other_units_at_missing_steps=[None] if not complete else [])
        result = dict(status=status, n=n, horizon=2, complete=complete, evaluation_reused=True,
                      coverage_basis="saved_evaluation", current_coverage_basis="current_query",
                      coverage=coverage, current_coverage=current, details="x" * 12000)
        response = dict(status="ok", operation="evaluate", scoring_status=status, complete=complete, result=result)
        projected = refs.project(response)
        assert len(encode(projected).encode()) <= limit
        assert projected["partial_scope"] == "response_payload"
        assert projected["summary"]["scoring_status"] == status
        assert projected["summary"]["complete"] is complete
        summary = projected["summary"]["result"]
        assert summary["coverage"]["matched_steps"] == n
        assert summary["current_coverage"]["other_units_at_missing_steps_count"] == (0 if complete else 1)
        assert summary["evaluation_reused"] is True
        assert summary["coverage_basis"] == "saved_evaluation"
        assert refs.value(projected["result_ref"]) == response
    finally:
        refs.close()


def test_response_budget_prioritizes_scoring_state_over_optional_identifiers():
    refs = ResultReferences(ResultLimits(max_response_bytes=2048))
    try:
        value = dict(status="ok", scoring_status="partial", complete=False, operation="evaluate",
                     result=dict(status="partial", n=1, complete=False,
                                 coverage=dict(required_steps=2, matched_steps=1, fraction=0.5)),
                     padding="x" * 12000)
        for key in ("execution_id", "study_id", "data_ref", "provider", "revision", "series_id", "unit", "reason",
                    "next_step", "metric_version", "aggregation", "evidence"):
            value[key] = "x" * 150
        projected = refs.project(value)
        assert len(encode(projected).encode()) <= 2048
        assert projected["summary_truncated"] is True
        assert projected["summary"]["complete"] is False
        assert projected["summary"]["result"]["coverage"]["fraction"] == 0.5
        assert refs.value(projected["result_ref"]) == value
    finally:
        refs.close()


def test_generic_cli_study_example_is_not_presented_as_a_task_retry():
    code, response = cli("evaluate", "--arguments", '{}')
    assert code == 2
    details = response["error"]["details"]
    assert details["example_kind"] == "schema_illustration"
    assert "not a retry" in details["example_guidance"]
