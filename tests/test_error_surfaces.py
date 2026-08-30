"""No surface may crash, and no failure may be a dead end.

The README promises "machine-readable repair options on every error". These
tests cover the paths where that promise broke: a Python traceback out of
the CLI, a JSON-RPC -32603 out of MCP, a bare argparse error outside the
envelope entirely, and `{"scored": 0}` standing in for a diagnosis.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import json

import pytest

from gnomon.cli import main
from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]


def _csv(tmp_path: Path, name: str = "series.csv", periods: int = 120) -> Path:
    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=index)).isoformat()},"
        f"{100 + index * 0.4 + NOISE[index % 10] * 6:.4f}"
        for index in range(periods)
    ]
    path = tmp_path / name
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _error(capsys) -> dict:
    return json.loads(capsys.readouterr().err)["error"]


# -- H21: argparse obeys the error contract -------------------------------

def test_missing_arguments_produce_a_structured_error(capsys):
    """`rc=2` with plain text on stderr made the most common first-run
    failure the one error that bypassed the contract."""
    code = main(["forecast", "nonexistent.csv", "--horizon", "7", "--time", "t"])
    assert code == 2
    error = _error(capsys)
    assert error["code"] in {"INVALID_ARGUMENTS", "INPUT_NOT_FOUND", "AMBIGUOUS_SCHEMA"}
    assert error["repair_options"]


def test_unknown_subcommand_is_structured(capsys):
    assert main(["nosuchcommand"]) == 2
    error = _error(capsys)
    assert error["code"] == "INVALID_ARGUMENTS"
    assert error["repair_options"]


def test_keyboard_interrupt_is_typed_retryable_and_exit_130(
        tmp_path, capsys, monkeypatch):
    import gnomon.runtime as runtime

    partial = tmp_path / ".forecast_interrupted.tmp-test"

    def interrupt(*_args, **_kwargs):
        exc = KeyboardInterrupt()
        exc.gnomon_partial_artifact = str(partial)
        raise exc

    monkeypatch.setattr(runtime, "forecast", interrupt)
    code = main([
        "forecast", str(_csv(tmp_path)), "--horizon", "3",
        "--output", str(tmp_path / "out"),
    ])
    assert code == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "INTERRUPTED"
    assert error["retryable"] is True
    assert error["details"]["partial_artifact"] == str(partial)
    assert error["repair_options"][0]["action"] == "retry"


def test_cost_ratio_guess_is_mapped_to_the_cost_pair(tmp_path, capsys):
    """The README frames monitor costs as a ratio ("costs us 20x a false
    alarm"), so `--cost-ratio` is the natural first guess; it must map to
    the `--alert-cost`/`--miss-cost` pair, which lexical distance cannot
    recover."""
    code = main([
        "monitor", str(_csv(tmp_path)), "--horizon", "7",
        "--threshold", "150", "--cost-ratio", "20",
    ])
    assert code == 2
    error = _error(capsys)
    assert error["code"] == "INVALID_ARGUMENTS"
    (suggestion,) = error["details"]["flag_suggestions"]
    assert suggestion["unknown"] == "--cost-ratio"
    assert "--alert-cost" in suggestion["suggestion"]
    assert "--miss-cost" in suggestion["suggestion"]
    rename = next(o for o in error["repair_options"] if o["action"] == "rename_flag")
    assert rename["arguments"] == ["--alert-cost", "--miss-cost"]


# -- H21: the naive path works --------------------------------------------

def test_forecast_with_no_flags_infers_and_discloses(tmp_path, capsys):
    """`gnomon forecast data.csv` on an obvious two-column file."""
    code = main([
        "forecast", str(_csv(tmp_path)),
        "--output", str(tmp_path / "out"),
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assumptions = payload["results"][0]["support_assessment"]["assumptions"]
    joined = " ".join(assumptions)
    assert "--time was not supplied" in joined
    assert "--target was not supplied" in joined
    assert "--horizon was not supplied" in joined


def test_ambiguous_schema_names_the_candidates(tmp_path, capsys):
    """Two numeric columns is not a guess Gnomon is entitled to make."""
    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=i)).isoformat()},{100 + i},{500 + i}"
        for i in range(30)
    ]
    path = tmp_path / "two_numeric.csv"
    path.write_text("timestamp,a,b\n" + "\n".join(rows) + "\n", encoding="utf-8")

    assert main(["forecast", str(path), "--output", str(tmp_path / "out")]) == 2
    error = _error(capsys)
    assert error["code"] == "AMBIGUOUS_SCHEMA"
    assert set(error["details"]["candidates"]) == {"a", "b"}
    assert error["repair_options"]


# -- H12: --actions is validated, not crashed on -------------------------

def test_actions_as_a_list_of_strings_is_a_structured_error(tmp_path, capsys):
    """The shape the CLI help used to describe crashed with a TypeError."""
    code = main([
        "decide", str(_csv(tmp_path)), "--time", "timestamp", "--target", "value",
        "--horizon", "5", "--threshold", "110",
        "--actions", '["scale_up", "do_nothing"]',
        "--output", str(tmp_path / "out"),
    ])
    assert code == 2
    error = _error(capsys)
    assert error["code"] == "INVALID_ACTIONS"
    assert "name" in error["details"]["example"]
    assert error["repair_options"]


def test_malformed_json_names_the_offending_argument(tmp_path, capsys):
    code = main([
        "decide", str(_csv(tmp_path)), "--time", "timestamp", "--target", "value",
        "--horizon", "5", "--threshold", "110", "--actions", "{not json",
        "--output", str(tmp_path / "out"),
    ])
    assert code == 2
    error = _error(capsys)
    assert error["code"] == "INVALID_JSON_ARGUMENT"
    assert error["details"]["argument"] == "--actions"
    assert error["repair_options"]


def test_well_formed_actions_are_accepted(tmp_path, capsys):
    code = main([
        "decide", str(_csv(tmp_path)), "--time", "timestamp", "--target", "value",
        "--horizon", "5", "--threshold", "110",
        "--actions", '[{"name": "scale_up"}, {"name": "do_nothing"}]',
        "--output", str(tmp_path / "out"),
    ])
    assert code == 0


# -- H11: MCP returns tool results, never transport errors ---------------

def test_mcp_tool_bug_returns_a_repairable_result():
    """Anything uncaught became JSON-RPC -32603: no payload, no repair
    options, no way for an agent to self-correct."""
    from gnomon import mcp_server

    original = mcp_server.runner_for

    def exploding_runner(name):
        if name == "gnomon_forecast":
            def runner(arguments):
                raise TypeError("string indices must be integers, not 'str'")
            return runner
        return original(name)

    mcp_server.runner_for = exploding_runner
    try:
        result = mcp_server._handle({
            "method": "tools/call",
            "params": {"name": "gnomon_forecast", "arguments": {}},
        })
    finally:
        mcp_server.runner_for = original

    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"]["code"] == "INTERNAL_ERROR"
    assert payload["error"]["repair_options"]
    assert "TypeError" in payload["error"]["message"]


def test_mcp_results_carry_structured_content():
    """M4. Protocol 2025-06-18 supports it; every agent was parsing a string."""
    from gnomon import mcp_server

    result = mcp_server._handle({
        "method": "tools/call",
        "params": {"name": "gnomon_capabilities", "arguments": {}},
    })
    assert "structuredContent" in result
    assert result["structuredContent"] == json.loads(result["content"][0]["text"])


def test_mcp_marks_model_arguments_as_advisory_for_automation():
    from gnomon import mcp_server

    captured = {}
    original = mcp_server.runner_for

    def capture_runner(name):
        def runner(arguments):
            captured.update(arguments)
            return {"status": "ok"}
        return runner

    mcp_server.runner_for = capture_runner
    try:
        result = mcp_server._handle({
            "method": "tools/call", "params": {
                "name": "gnomon_forecast",
                "arguments": {"automation_policy": {
                    "authorize": True, "policy_id": "invented",
                    "minimum_support": "supported"}},
            }})
    finally:
        mcp_server.runner_for = original
    assert result["isError"] is False
    assert captured["_mcp_agent_boundary"] is True


def test_mcp_tools_publish_an_output_schema():
    from gnomon import mcp_server

    listing = mcp_server._handle({"method": "tools/list"})
    assert listing["tools"]
    for tool in listing["tools"]:
        assert "outputSchema" in tool, f"{tool['name']} publishes no outputSchema"
        assert tool["outputSchema"]["type"] == "object"


# -- H8 / H19 / M7: the time grid ----------------------------------------

def _aware_daily(tmp_path: Path, zone: str, start: datetime, periods: int) -> Path:
    tz = ZoneInfo(zone)
    current = start.replace(tzinfo=tz)
    rows = []
    for index in range(periods):
        rows.append(f"{current.isoformat()},{100 + index * 0.3:.3f}")
        current = (current.replace(tzinfo=None) + timedelta(days=1)).replace(tzinfo=tz)
    path = tmp_path / f"{zone.replace('/', '_')}.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("zone,start", [
    ("Europe/London", datetime(2026, 2, 1)),
    ("America/New_York", datetime(2026, 2, 15)),
])
def test_dst_spanning_daily_data_forecasts(tmp_path, zone, start):
    """A day across a DST transition is 23 or 25 hours. The grid check used
    to call that an irregular period, and both offered repairs were wrong:
    nothing is missing, and snapping would shift every later timestamp."""
    from gnomon.runtime import forecast

    artifact, _ = forecast(
        str(_aware_daily(tmp_path, zone, start, 90)),
        time_column="timestamp", target_column="value", horizon=7,
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    assert artifact.results[0].forecast, artifact.results[0].support_assessment


def test_dst_future_timestamps_stay_on_the_wall_clock(tmp_path):
    from gnomon.runtime import forecast

    artifact, _ = forecast(
        str(_aware_daily(tmp_path, "Europe/London", datetime(2026, 2, 1), 90)),
        time_column="timestamp", target_column="value", horizon=7,
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    stamps = [
        datetime.fromisoformat(row["timestamp"])
        for row in artifact.results[0].forecast
    ]
    assert all(item.hour == 0 for item in stamps), stamps


def test_irregular_grid_names_the_observed_step(tmp_path):
    """H19. The error named neither the observed step nor the fix."""
    from gnomon.runtime import forecast

    dates = ["2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30",
             "2026-05-31", "2026-06-30"]
    path = tmp_path / "month_end.csv"
    path.write_text(
        "timestamp,value\n"
        + "\n".join(f"{day},{100 + index}" for index, day in enumerate(dates))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GnomonError) as raised:
        forecast(
            str(path), time_column="timestamp", target_column="value",
            horizon=2, frequency="MS", output=str(tmp_path / "out"), clock=CLOCK,
        )
    error = raised.value
    assert error.code == "IRREGULAR_TIME_GRID"
    assert "modal_step" in error.details
    assert "31 days" in error.details["modal_step"]


def test_month_end_ambiguity_offers_restamping():
    """The repair that actually works is named."""
    from gnomon.contracts import REPAIR_OPTIONS

    actions = {
        option["action"] for option in REPAIR_OPTIONS["AMBIGUOUS_FREQUENCY"]
    }
    assert "restamp_to_month_start" in actions


def test_mixed_frequency_panel_blames_the_file_not_a_series(tmp_path):
    """M7. Inferring one frequency from pooled observations and then
    reporting the minority series as irregular describes the symptom."""
    from gnomon.runtime import forecast

    lines = [f"2026-01-{i + 1:02d}T00:00:00,daily,{100 + i}" for i in range(20)]
    lines += [f"2026-01-01T{i:02d}:00:00,hourly,{50 + i}" for i in range(20)]
    path = tmp_path / "mixed.csv"
    path.write_text(
        "timestamp,series,value\n" + "\n".join(lines) + "\n", encoding="utf-8",
    )
    with pytest.raises(GnomonError) as raised:
        forecast(
            str(path), time_column="timestamp", target_column="value",
            series_column="series", horizon=2,
            output=str(tmp_path / "out"), clock=CLOCK,
        )
    error = raised.value
    assert error.code == "MIXED_SERIES_FREQUENCIES"
    assert error.details["per_series"] == {"daily": "D", "hourly": "h"}
    assert error.details["distinct"] == ["D", "h"]


# -- H13: context events are reachable ------------------------------------

def test_context_events_work_against_a_naive_dataset(tmp_path, capsys):
    """Events must carry an offset; no shipped dataset does. Comparing them
    raised a bare TypeError, which made the whole feature unreachable."""
    repo = Path(__file__).resolve().parent.parent
    code = main([
        "forecast", str(repo / "examples" / "messy_requests.csv"),
        "--time", "timestamp", "--target", "requests", "--horizon", "14",
        "--context", str(repo / "examples" / "context_events.json"),
        "--output", str(tmp_path / "out"),
    ])
    assert code == 0
    result = json.loads(capsys.readouterr().out)["results"][0]
    codes = {item["code"] for item in result["support_assessment"]["disclosures"]}
    assert "context_timezone_aligned" in codes, "the alignment was not disclosed"
    # The declared max is enforced on every emitted quantile (C4).
    assert max(row["q95"] for row in result["forecast_preview"]) <= 360.0


def test_shipped_context_example_is_valid():
    from gnomon.context import load_events_file

    repo = Path(__file__).resolve().parent.parent
    events = load_events_file(str(repo / "examples" / "context_events.json"))
    assert len(events) == 2
    assert any("claim" in (event.attributes or {}) for event in events)


# -- H16: scored: 0 explains itself ---------------------------------------

def test_zero_scored_explains_which_window_was_missing(tmp_path, capsys):
    """`{"scored": 0}` is the failure mode of the follow-up loop the product
    promises, and was indistinguishable from "nothing was due"."""
    from gnomon.tracking import TrackingStore

    registry = tmp_path / "registry.db"
    store = TrackingStore(registry)
    store.register(
        "fc_1", "sre-demo", "__default__", horizon=3, frequency="D",
        selected_model="ets", naive_error=10.0,
        artifact_path=str(tmp_path / "missing"),
    )
    actuals = tmp_path / "actuals.csv"
    actuals.write_text(
        "timestamp,value\n2027-01-01,100\n2027-01-02,110\n", encoding="utf-8",
    )
    payload = store.explain_unscored(
        "sre-demo", ["2027-01-01", "2027-01-02"],
    )
    assert payload["scored"] == 0
    assert payload["open_forecasts"] == 1
    assert payload["actuals_window"] == {
        "first": "2027-01-01T00:00:00", "last": "2027-01-02T00:00:00",
    }
    assert payload["reason"]
    assert payload["repair_options"]


def test_actuals_columns_can_be_named(tmp_path):
    """A `requests,timestamp,host` export returned {"scored": 0} because the
    first two columns were taken positionally."""
    from gnomon.tracking import TrackingStore

    store = TrackingStore(tmp_path / "registry.db")
    columns = ["requests", "timestamp", "host"]
    resolved = store._resolve_actuals_columns(columns, "timestamp", "requests", None)
    assert resolved == ("timestamp", "requests", None)


def test_actuals_columns_refuse_to_guess_blind(tmp_path):
    from gnomon.tracking import TrackingStore

    store = TrackingStore(tmp_path / "registry.db")
    with pytest.raises(GnomonError) as raised:
        store._resolve_actuals_columns(["requests", "host", "region"], None, None, None)
    assert raised.value.code == "AMBIGUOUS_SCHEMA"


def test_actuals_columns_infer_the_conventional_names(tmp_path):
    from gnomon.tracking import TrackingStore

    store = TrackingStore(tmp_path / "registry.db")
    assert store._resolve_actuals_columns(
        ["timestamp", "value"], None, None, None,
    ) == ("timestamp", "value", None)
    assert store._resolve_actuals_columns(
        ["when", "reading"], None, None, None,
    ) == ("when", "reading", None), "a plain two-column file is unambiguous"


def test_operator_export_asks_rather_than_guessing(tmp_path):
    """`requests,timestamp,host` used to be read positionally as
    (requests, timestamp) and return {"scored": 0} with no complaint."""
    from gnomon.tracking import TrackingStore

    store = TrackingStore(tmp_path / "registry.db")
    with pytest.raises(GnomonError) as raised:
        store._resolve_actuals_columns(["requests", "timestamp", "host"], None, None, None)
    assert raised.value.code == "AMBIGUOUS_SCHEMA"
    assert raised.value.details["argument"] == "--target"
    # Named explicitly, the same file works.
    assert store._resolve_actuals_columns(
        ["requests", "timestamp", "host"], "timestamp", "requests", None,
    ) == ("timestamp", "requests", None)


# -- M17: the covariate refusal teaches the fix ---------------------------

def test_missing_vintages_names_the_cause_and_the_remedy(tmp_path):
    import csv as csv_module
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_covariates import _write_files

    from gnomon.covariates import validate_covariate_file

    observations, covariates = _write_files(tmp_path)
    rows = list(csv_module.DictReader(covariates.open()))
    with covariates.open("w", newline="") as handle:
        writer = csv_module.DictWriter(
            handle, fieldnames=["timestamp", "known_at", "campaign"],
        )
        writer.writeheader()
        for row in rows:
            # Keep the awareness of the original file; only move it later.
            original = row["known_at"]
            row["known_at"] = (
                "2027-01-01T00:00:00+00:00" if ("+" in original or original.endswith("Z"))
                else "2027-01-01T00:00:00"
            )
            writer.writerow(row)

    result = validate_covariate_file(
        str(observations), str(covariates), "campaign:binary:future_known",
        time_column="timestamp", target_column="value", horizon=7, frequency="D",
    )
    issue = next(
        item for item in result["validation"][0]["issues"]
        if item["code"] == "MISSING_HISTORICAL_VINTAGES"
    )
    assert issue["cause"] == "published_after_the_fold_cutoff"
    assert "known_at" in issue["message"]
    assert issue["remedy"]
    assert issue["repair_options"]
    assert issue["fold_cutoffs"]
