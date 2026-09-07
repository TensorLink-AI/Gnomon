"""Independent exact calendar/instant oracles; no forecasts or LLM grading."""

from io import StringIO
import json
import subprocess
import sys

import pytest

from gnomon import GnomonSession, temporal_operation
from gnomon.cli import main
from gnomon.contracts import GnomonError
from gnomon.mcp_server import serve
from gnomon.temporal_ops import TEMPORAL_SCHEMA


def result(operation, **kwargs):
    payload = temporal_operation(operation, **kwargs)
    assert payload["action_authorized"] is False
    assert payload["input_facts"] == "supplied_not_verified"
    return payload["result"]


def test_normalization_preserves_microseconds_and_does_not_guess_a_clock():
    normalized = result("normalize", value="2026-01-02T03:04:05.123456+10:00")
    assert normalized["utc"] == "2026-01-01T17:04:05.123456Z"
    assert normalized["timezone_rules"] == "explicit_fixed_offset"
    assert normalized["utc_offset_seconds"] == 36000
    assert result("normalize", value="2026-01-02T03:04:05+10:00", timezone="UTC")["local"] == "2026-01-01T17:04:05+00:00"


def test_historical_offset_seconds_round_trip_without_truncation():
    normalized = result("normalize", value="1900-01-01T00:00:00+00:09:21")
    assert normalized["utc"] == "1899-12-31T23:50:39Z"
    assert normalized["utc_offset_seconds"] == 561
    assert result("normalize", value=normalized["local"]) == normalized


def test_missing_zone_data_fails_explicitly_but_offsets_still_work(monkeypatch):
    from gnomon import temporal_ops
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    class MissingZone(ZoneInfo):
        def __new__(cls, name):
            raise ZoneInfoNotFoundError(name)

    monkeypatch.setattr(temporal_ops, "ZoneInfo", MissingZone)
    with pytest.raises(ValueError, match="timezone unavailable"):
        result("normalize", value="2024-01-01T00:00:00", timezone="UTC")
    assert result("normalize", value="2024-01-01T00:00:00Z")["utc"] == "2024-01-01T00:00:00Z"


@pytest.mark.parametrize("zone,value,first,second", [
    ("America/New_York", "2024-11-03T01:30:00", "2024-11-03T05:30:00Z", "2024-11-03T06:30:00Z"),
    ("Australia/Lord_Howe", "2024-04-07T01:45:00", "2024-04-06T14:45:00Z", "2024-04-06T15:15:00Z"),
])
def test_ambiguous_folds_require_choice_including_half_hour_transition(zone, value, first, second):
    with pytest.raises(ValueError, match="ambiguous"):
        result("normalize", value=value, timezone=zone)
    for fold, utc in enumerate((first, second)):
        normalized = result("normalize", value=value, timezone=zone, fold=fold)
        assert normalized["utc"] == utc
        assert normalized["ambiguous_local_time"] is True
        assert normalized["fold"] == fold
        assert normalized["timezone_rules"] == "host_zoneinfo_unversioned"
        assert result("normalize", value=utc, timezone=zone) == normalized


@pytest.mark.parametrize("zone,value", [
    ("America/New_York", "2024-03-10T02:30:00"),
    ("Australia/Lord_Howe", "2024-10-06T02:15:00"),
    ("Pacific/Apia", "2011-12-30T12:00:00"),
])
@pytest.mark.parametrize("fold", [None, 0, 1])
def test_nonexistent_wall_times_are_never_silently_shifted(zone, value, fold):
    with pytest.raises(ValueError, match="nonexistent"):
        result("normalize", value=value, timezone=zone, **({"fold": fold} if fold is not None else {}))


def test_elapsed_and_calendar_days_are_distinct_across_dst():
    args = dict(value="2024-03-09T12:00:00", timezone="America/New_York", amount=1, unit="days")
    calendar = result("shift", **args, mode="calendar")
    elapsed = result("shift", **args, mode="elapsed")
    assert calendar["local"] == "2024-03-10T12:00:00-04:00"
    assert elapsed["local"] == "2024-03-10T13:00:00-04:00"
    assert result("duration", start="2024-03-09T12:00:00-05:00", end=calendar["utc"])["duration_seconds"] == "82800"
    assert result("duration", start="2024-03-09T12:00:00-05:00", end=elapsed["utc"])["duration_seconds"] == "86400"
    args["value"] = "2024-11-02T12:00:00"
    assert result("shift", **args, mode="calendar")["local"] == "2024-11-03T12:00:00-05:00"
    assert result("shift", **args, mode="elapsed")["local"] == "2024-11-03T11:00:00-05:00"


def test_calendar_target_fold_and_gap_are_explicit():
    args = dict(value="2024-11-02T01:30:00", timezone="America/New_York", amount=1, unit="days", mode="calendar")
    with pytest.raises(ValueError, match="ambiguous"):
        result("shift", **args)
    assert result("shift", **args, target_fold=1)["utc"] == "2024-11-03T06:30:00Z"
    args["value"] = "2024-03-09T02:30:00"
    with pytest.raises(ValueError, match="nonexistent"):
        result("shift", **args)


@pytest.mark.parametrize("value,amount,unit,expected", [
    ("2024-01-31", 1, "months", "2024-02-29"),
    ("2024-02-29", 1, "years", "2025-02-28"),
    ("2024-03-31", -1, "months", "2024-02-29"),
    ("2000-02-29", 100, "years", "2100-02-28"),
])
def test_gregorian_month_end_requires_explicit_clamping(value, amount, unit, expected):
    args = dict(value=value, amount=amount, unit=unit, mode="calendar")
    with pytest.raises(ValueError, match="no such day"):
        result("shift", **args)
    shifted = result("shift", **args, invalid_date="clamp")
    assert shifted["date"] == expected
    assert shifted["kind"] == "date"
    assert shifted["date_clamped"] is True
    assert "utc" not in shifted


def test_date_boundary_and_fixed_offset_calendar_shift():
    assert result("shift", value="2024-02-28", amount=1, unit="days", mode="calendar")["date"] == "2024-02-29"
    assert result("shift", value="2024-01-07", amount=-1, unit="weeks", mode="calendar")["date"] == "2023-12-31"
    assert result("shift", value="2024-01-31T10:00:00+10:00", amount=1, unit="months", mode="calendar", invalid_date="clamp")["local"] == "2024-02-29T10:00:00+10:00"


@pytest.mark.parametrize("start,end,seconds,micros", [
    ("2024-01-01T00:00:00Z", "2024-01-01T00:00:00.000001Z", "0.000001", "1"),
    ("2024-01-01T00:00:00.000001Z", "2024-01-01T00:00:00Z", "-0.000001", "-1"),
    ("2024-01-01T01:00:00+01:00", "2024-01-01T00:00:00Z", "0", "0"),
    ("0001-01-01T00:00:00Z", "9999-12-31T23:59:59.999999Z", "315537897599.999999", "315537897599999999"),
])
def test_duration_uses_exact_utc_microseconds_not_floats(start, end, seconds, micros):
    duration = result("duration", start=start, end=end)
    assert duration["duration_seconds"] == seconds
    assert duration["duration_microseconds"] == micros


def span(start, end):
    return {"start": f"2024-01-01T{start:02d}:00:00Z", "end": f"2024-01-01T{end:02d}:00:00Z"}


@pytest.mark.parametrize("left,right,relation,inverse,overlap", [
    ((1, 2), (3, 4), "before", "after", False),
    ((1, 2), (2, 3), "meets", "met_by", False),
    ((1, 3), (2, 4), "overlaps", "overlapped_by", True),
    ((1, 2), (1, 3), "starts", "started_by", True),
    ((2, 3), (1, 4), "during", "contains", True),
    ((2, 4), (1, 4), "finishes", "finished_by", True),
    ((1, 2), (1, 2), "equals", "equals", True),
])
def test_all_thirteen_interval_relations_and_inverses(left, right, relation, inverse, overlap):
    relation_result = result("interval", left=span(*left), right=span(*right))
    reversed_result = result("interval", left=span(*right), right=span(*left))
    assert relation_result["relation"] == relation
    assert reversed_result["relation"] == inverse
    assert relation_result["overlap"] == overlap
    assert (relation_result["intersection"] is not None) == overlap
    assert relation_result["left_contains_right"] == (left[0] <= right[0] and left[1] >= right[1])
    assert relation_result["right_contains_left"] == reversed_result["left_contains_right"]


def test_event_order_uses_instants_stable_ties_and_half_open_filter():
    events = [{"event_id": "later", "at": "2024-01-01T01:00:00Z"},
              {"event_id": "first_tie", "at": "2024-01-01T01:00:00+01:00"},
              {"event_id": "second_tie", "at": "2024-01-01T00:00:00Z"}]
    ordered = result("order_events", events=events)
    assert ordered["ordered_event_ids"] == ["first_tie", "second_tie", "later"]
    assert ordered["instant_groups"][0]["event_ids"] == ["first_tie", "second_tie"]
    assert ordered["tie_policy"] == "input_order_not_causality"
    filtered = result("order_events", events=events, **span(0, 1))
    assert filtered["ordered_event_ids"] == ["first_tie", "second_tie"]
    assert filtered["excluded_event_ids"] == ["later"]
    assert filtered["input_count"] == 3 and filtered["included_count"] == 2
    assert result("order_events", events=[])["instant_groups"] == []


@pytest.mark.parametrize("arguments", [
    {}, {"operation": "unknown"}, {"operation": []},
    {"operation": "normalize", "value": "now"},
    *({"operation": "normalize", "value": value} for value in [None, True, "2024-01-01", "2024-01-01T00:00:00", "2024-01-01T00:00:00.0000001Z", "2024-01-01T00:00:00-00:00", "2024-01-01T00:00:00+00:60", "2024-01-01T00:00:60Z", "0" * 1000]),
    *({"operation": "normalize", "value": "2024-01-01T00:00:00", "timezone": zone} for zone in ["../UTC", "/etc/passwd", "Unknown/Nowhere", "x" * 129, True]),
    *({"operation": "normalize", "value": "2024-01-01T00:00:00", "timezone": "UTC", "fold": fold} for fold in [True, 1, 2, "0", None]),
    {"operation": "normalize", "value": "2024-01-01T00:00:00Z", "fold": 0},
    {"operation": "normalize", "value": "2024-01-01T00:00:00Z", "amount": 1},
    {"operation": "duration", "start": "2024-01-01T00:00:00", "end": "2024-01-01T01:00:00Z"},
    {"operation": "interval", "left": span(1, 1), "right": span(1, 2)},
    {"operation": "interval", "left": {**span(0, 1), "closed": True}, "right": span(1, 2)},
    {"operation": "order_events", "events": [], "start": "2024-01-01T00:00:00Z"},
    {"operation": "order_events", "events": [{"event_id": "a", "at": "2024-01-01T00:00:00Z"}] * 2},
    {"operation": "order_events", "events": [{}] * 1001},
    {"operation": "order_events", "events": [{"event_id": "", "at": "2024-01-01T00:00:00Z"}]},
    {"operation": "order_events", "events": {}},
])
def test_invalid_arguments_are_typed_at_session_boundary(arguments):
    with GnomonSession(enable_temporal=True) as session:
        with pytest.raises(GnomonError) as error:
            session.call("gnomon_temporal", arguments)
        assert error.value.code == "INVALID_ARGUMENTS"


@pytest.mark.parametrize("override", [
    {"amount": True}, {"amount": 1.0}, {"amount": 1_000_001}, {"amount": -1_000_001},
    {"unit": "months"}, {"unit": []}, {"mode": "guess"}, {"target_fold": 0},
    {"invalid_date": "clamp"}, {"value": "2024-01-01"},
    {"value": "9999-12-31T23:59:59Z"}, {"value": "0001-01-01T00:00:00Z", "amount": -1},
    {"mode": "calendar", "value": "2024-01-01", "timezone": "UTC"},
    {"mode": "calendar", "unit": "hours"},
])
def test_shift_rejects_invalid_modes_bounds_and_ignored_choices(override):
    arguments = {"value": "2024-01-01T00:00:00Z", "amount": 1, "unit": "days", "mode": "elapsed", **override}
    with pytest.raises(ValueError):
        result("shift", **arguments)


def test_optional_tool_requires_startup_opt_in_and_closed_schema(tmp_path):
    with GnomonSession() as session:
        assert len(session.tools()) == 6
        assert session.capabilities()["temporal"]["enabled"] is False
        with pytest.raises(GnomonError, match="startup"):
            session.call("gnomon_temporal", {"operation": "order_events", "events": [], "enable_temporal": True})
    config = tmp_path / "temporal.toml"
    config.write_text("enable_temporal=true\n")
    with GnomonSession.from_config(config) as session:
        assert len(session.tools()) == 7
        assert session.capabilities()["temporal"]["enabled"] is True
        assert session.tools()[-1]["inputSchema"] == TEMPORAL_SCHEMA
    assert all(schema["additionalProperties"] is False for schema in TEMPORAL_SCHEMA["oneOf"])
    for bad in (1, "true", None):
        with pytest.raises(ValueError, match="boolean"):
            GnomonSession(enable_temporal=bad)
    config.write_text('enable_temporal="true"\n')
    with pytest.raises(ValueError, match="boolean"):
        GnomonSession.from_config(config)


def test_large_event_results_use_shared_exact_retrieval():
    events = [{"event_id": f"event_{index:04d}", "at": "2024-01-01T00:00:00Z"} for index in range(1000)]
    with GnomonSession(enable_temporal=True) as session:
        receipt = session.call("gnomon_temporal", {"operation": "order_events", "events": events})
        assert receipt["partial"] is True
        arguments = {"result_ref": receipt["result_ref"], "pointer": "/result/ordered_event_ids"}
        pieces, offset = [], 0
        while offset is not None:
            page = session.call("gnomon_read", {**arguments, "offset": offset})
            pieces.append(page["text"])
            offset = page["next_offset"]
        assert json.loads("".join(pieces)) == [event["event_id"] for event in events]
        assert session.engine.capabilities() == {}


def test_python_cli_and_real_stdio_temporal_parity_without_providers(tmp_path, capsys):
    arguments = {"operation": "shift", "value": "2024-01-31", "amount": 1, "unit": "months", "mode": "calendar", "invalid_date": "clamp"}
    expected = temporal_operation(**arguments)
    path = tmp_path / "arguments.json"
    path.write_text(json.dumps(arguments))
    assert main(["temporal", "--arguments", "@" + str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert main(["temporal", "--arguments", "[]"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "INVALID_ARGUMENTS"
    config = tmp_path / "temporal.toml"
    config.write_text("enable_temporal=true\n")
    messages = [{"id": 1, "method": "tools/list"}, {"id": 2, "method": "tools/call", "params": {"name": "gnomon_temporal", "arguments": arguments}}]
    completed = subprocess.run([sys.executable, "-m", "gnomon", "mcp", "serve", "--providers-config", str(config)],
        input="".join(json.dumps(message) + "\n" for message in messages), text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    responses = list(map(json.loads, completed.stdout.splitlines()))
    assert "gnomon_temporal" in {tool["name"] for tool in responses[0]["result"]["tools"]}
    assert responses[1]["result"]["structuredContent"] == expected
    # An explicitly empty engine still performs the calculation: no inference dependency.
    with GnomonSession(enable_temporal=True) as session:
        assert session.engine.capabilities() == {}
        output = StringIO()
        assert serve([json.dumps(messages[1])], output, session=session) == 0
        assert json.loads(output.getvalue())["result"]["structuredContent"] == expected
