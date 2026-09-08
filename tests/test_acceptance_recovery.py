"""Regression journeys from the published 1.1.1 first-use feedback."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json

import pytest

from gnomon import ForecastRequest, GnomonSession
from gnomon.cli import main
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock
from gnomon.mcp_server import _handle
from gnomon.session import ledger_schema


def cli(*args):
    out = StringIO()
    with redirect_stdout(out):
        code = main(list(map(str, args)))
    return code, json.loads(out.getvalue())


def stamp(day):
    return f"2026-01-{day:02d}T00:00:00Z"


def forecast(session):
    return session.forecast("last_value", dict(history=[19, 20], horizon=2, unit="widgets",
        series_id="sales", future_timestamps=[stamp(21), stamp(22)]))["execution_id"]


def actual(session, day, unit="widgets", available=23):
    return session.ledger.append_actual(series_id="sales", valid_time=stamp(day), value=day,
        source_available_at=stamp(available), unit=unit)


def test_scoring_coverage_units_cutoffs_and_exact_retries(tmp_path):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        session.ledger.clock = FixedClock(datetime(2026, 1, 20, tzinfo=timezone.utc))
        eid = forecast(session)
        session.ledger.clock = FixedClock(datetime(2026, 1, 24, tzinfo=timezone.utc))
        actual(session, 21)
        actual(session, 22, None)
        query = dict(operation="evaluate", execution_id=eid)
        score = session.call("gnomon_ledger", query, compact=False)
        assert score["status"] == "ok" and score["scoring_status"] == "partial" and not score["complete"]
        coverage = score["result"]["coverage"]
        assert (coverage["required_steps"], coverage["matched_steps"], coverage["fraction"]) == (2, 1, 0.5)
        assert coverage["other_units_at_missing_steps"] == [None]
        assert score["result"]["mae"] == 1
        assert session.call("gnomon_ledger", query, compact=False) == score
        assert len(session.ledger.evaluations(eid)) == 1
        hidden = session.call("gnomon_ledger", dict(query, source_as_of=stamp(22)), compact=False)
        assert hidden["scoring_status"] == "pending" and hidden["result"]["mae"] is None
        assert hidden["result"]["coverage"]["other_units_at_missing_steps"] == []
        recorded_hidden = session.call("gnomon_ledger", dict(query, recorded_as_of=stamp(23)), compact=False)
        assert recorded_hidden["result"]["coverage"]["other_units_at_missing_steps"] == []
        with pytest.raises(GnomonError) as caught:
            session.call("gnomon_ledger", dict(query, allow_partial=False), compact=False)
        error = caught.value.to_dict()["error"]
        assert error["details"]["coverage"]["missing_steps"] == [1]
        recovery = error["details"]["example_arguments"]
        assert recovery["operation"] == "evaluate" and recovery["execution_id"] == eid
        assert session.call("gnomon_ledger", recovery, compact=False)["scoring_status"] == "partial"
        actual(session, 22)
        complete = session.call("gnomon_ledger", dict(query, allow_partial=False), compact=False)
        assert complete["complete"] and complete["result"]["mae"] == 1.5
        assert complete["result"]["coverage"]["other_units_at_missing_steps"] == []
        # Existing CLI success semantics remain compatible.
        code, response = cli("ledger", "--ledger-path", session.ledger.path, "--arguments", json.dumps(query))
        assert code == 0 and response["scoring_status"] == "complete"


def test_score_metadata_is_immutable_while_search_diagnostics_are_current(tmp_path):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        eid = forecast(session)
        original = session.ledger.evaluate(eid)
        actual(session, 21, unit="other")
        assert session.ledger.evaluate(eid) == original
        assert session.ledger.evaluations(eid) == [original]
        row = session.ledger.search()["items"][0]
        assert row["coverage"]["other_units_at_missing_steps"] == ["other"]


def test_ledger_schema_explains_both_evaluate_forms():
    variants = [v for v in ledger_schema()["oneOf"] if v["properties"]["operation"]["const"] == "evaluate"]
    assert len(variants) == 2
    for variant in variants:
        field = variant["properties"]["allow_partial"]
        assert field["default"] is True and "result.coverage" in field["description"]


def test_ledger_cutoff_recovery_and_naive_forecast_explanation(tmp_path):
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        eid = forecast(session)
        with pytest.raises(GnomonError) as caught:
            session.call("gnomon_ledger", dict(operation="evaluate", execution_id=eid, recorded_as_of=stamp(1)))
        details = caught.value.details
        assert details["example_arguments"]["recorded_as_of"] == details["execution_recorded_at"]
        assert session.call("gnomon_ledger", details["example_arguments"])["scoring_status"] == "pending"
        naive = session.forecast("last_value", dict(history=[1, 2], horizon=1, series_id="sales",
                                                     future_timestamps=["2026-01-21T00:00:00"]))
        with pytest.raises(GnomonError) as caught:
            session.call("gnomon_ledger", dict(operation="evaluate", execution_id=naive["execution_id"]))
        assert caught.value.details["field"] == "request.future_timestamps"
        assert "Changing scoring cutoffs cannot repair" in str(caught.value)


@pytest.mark.parametrize("days, fills, allowed", [([1, 2, 4], 1, False),
    ([i for i in range(1, 21) if i != 10], 1, True),
    ([i for i in range(1, 25) if i not in (10, 11, 12, 13)], 4, False)])
def test_gap_advice_predicts_fraction_and_run_budgets(tmp_path, days, fills, allowed):
    path = tmp_path / "gap.csv"
    path.write_text("timestamp,value\n" + "".join(f"2026-01-{d:02d},{d}\n" for d in days))
    args = ("infer", "--provider", "last_value", "--input", path, "--frequency", "D", "--horizon", "2")
    code, result = cli(*args, "--repair", "safe")
    assert code == 2
    error = result["error"]
    budget = error["details"]["repair_budget"]
    assert budget["filled"] == fills and budget["denominator"] == len(days)
    assert budget["within_budget"] is allowed
    assert any(r["action"] == "allow_interpolation" for r in error["repair_options"]) is allowed
    assert "business_daily" not in error["message"]  # These fixtures contain observed weekends.
    code, result = cli(*args, "--repair", "aggressive")
    assert code == (0 if allowed else 2)
    if not allowed:
        assert result["error"]["code"] == "EXCESSIVE_REPAIR"


def test_duplicate_recovery_and_season_parameters(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("timestamp,value\n2026-01-01,1\n2026-01-02,2\n2026-01-02,2\n2026-01-03,3\n")
    args = ("infer", "--provider", "last_value", "--input", path, "--horizon", "2")
    _, error = cli(*args)
    assert error["error"]["repair_options"][0]["action"] == "deduplicate_identical_rows"
    code, result = cli(*args, "--repair", "safe")
    assert code == 0 and result["result"]["point"] == [3, 3]
    path.write_text("timestamp,value\n2026-01-01,1\n2026-01-02,2\n")
    _, result = cli("infer", "--provider", "seasonal_naive", "--input", path,
                    "--frequency", "D", "--horizon", "2", "--season", "7")
    details = result["error"]["details"]
    assert details["required_history"] == 7 and details["observed_history"] == 2
    assert details["input_options"]["season"] == 7 and "example_arguments" not in details


def test_mcp_forecast_error_has_runnable_example_and_cli_crosswalk():
    with GnomonSession.from_config() as session:
        response = _handle({"method": "tools/call", "params": {"name": "gnomon_forecast",
            "arguments": {"provider": "last_value", "input": "data.csv", "horizon": 2}}}, session=session)
        error = response["structuredContent"]["error"]
        details = error["details"]
        assert details["next_tool"] == "gnomon_inspect"
        assert details["inspect_arguments"] == {"input": "data.csv"}
        assert session.call("gnomon_forecast", details["example_arguments"])["result"]["point"] == (3, 3)


def test_type_temporal_and_configuration_recovery():
    with pytest.raises(ForecastAdapterError, match="finite observations"):
        ForecastRequest(history=["not-a-number"], horizon=2)
    with GnomonSession.from_config() as session:
        with pytest.raises(ForecastAdapterError, match="session.forecast"):
            session.forecast({"history": [1, 2], "horizon": 2})
    _, result = cli("temporal", "--arguments", json.dumps(dict(operation="shift", value="2026-01-01", amount="1", unit="month")))
    message = result["error"]["message"]
    assert "mode" in message and "amount must be an integer" in message and "months" in message
    assert cli("temporal", "--arguments", json.dumps(result["error"]["details"]["example_arguments"]))[0] == 0
    code, schema = cli("capabilities", "--config-schema", "--providers-config", "/does/not/exist")
    assert code == 0 and "allow_outcome_writes" in schema["properties"]


def test_missing_study_has_distinct_fallback_reason(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("timestamp,value\n" + "".join(f"2026-01-{d:02d},{d}\n" for d in range(1, 21)))
    with GnomonSession.from_config(ledger_path=tmp_path / "ledger.db") as session:
        ref = session.data.inspect(str(path), timezone="UTC")["data_ref"]
        response = session.route(ref, study_id="missing", candidates=["historical_mean"], baseline="last_value",
                                 horizon=2, source_as_of=stamp(20), recorded_as_of=stamp(31))
        assert response["reason"] == "study_not_found" and response["provider_calls"] == 0
