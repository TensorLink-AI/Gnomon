"""The admission preflight: rejection becomes a repair loop, not a post-mortem.

A proposal used to die silently — the admission verdicts were computed
during the forecast and recorded in evidence the proposer only saw after
the run was spent. The preflight runs the same admission code on the
same data before any forecast, and returns the accepted span grammar
beside every verdict, so an agent can repair and resubmit in one step.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

from gnomon.context import ContextEvent, ContextSource
from gnomon.future_context import (
    parse_bound_span,
    parse_override_scale,
    parse_override_span,
)
from gnomon.preflight import ACCEPTED_SPAN_EXAMPLES, preflight_context_events

START = datetime(2026, 1, 1)


def _write_csv(path: Path, days: int = 120) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for day in range(days):
            writer.writerow([
                (START + timedelta(days=day)).isoformat(),
                200 + 10 * (day % 7) + 0.3 * day,
            ])


def _event(event_id: str, event_type: str, attributes: dict) -> ContextEvent:
    h_start = START + timedelta(days=120)
    return ContextEvent(
        event_id=event_id, event_type=event_type, entity_scope=("*",),
        effective_start=h_start.isoformat() + "+00:00",
        effective_end=(h_start + timedelta(days=6)).isoformat() + "+00:00",
        known_at=START.isoformat() + "+00:00", attributes=attributes,
        source=ContextSource("dataset", "test#preflight"), created_by="llm",
    )


def _events():
    return [
        _event("good-span", "constraint:cap",
               {"source_span": "the value will not exceed 150"}),
        _event("bad-span", "constraint:cap",
               {"source_span": "output may be limited at some point"}),
        _event("claim", "constraint:capacity",
               {"claim": {"kind": "max", "value": 400.0}}),
        _event("fold-gated", "promotion", {"lift": "unknown"}),
    ]


def test_preflight_gives_one_effective_verdict_per_event(tmp_path: Path):
    source = tmp_path / "series.csv"
    _write_csv(source)
    payload = preflight_context_events(
        str(source), time_column="timestamp", target_column="value",
        horizon=7, context_events=_events(),
    )
    assert payload["status"] == "ok"
    assert payload["frequency"] == "D"
    (series,) = payload["series"]
    outcomes = {entry["event_id"]: entry for entry in series["events"]}
    assert outcomes["good-span"]["outcome"] == "would_influence"
    assert outcomes["good-span"]["basis"] == "future_context"
    assert outcomes["claim"]["outcome"] == "would_influence"
    assert outcomes["claim"]["basis"] == "constraint_claim"
    assert outcomes["bad-span"]["outcome"] == "rejected"
    assert any("does not state" in reason
               for reason in outcomes["bad-span"]["reasons"])
    assert outcomes["fold-gated"]["outcome"] == "ablation_gated"
    assert "fold-ablation" in outcomes["fold-gated"]["note"]
    # The repair material rides along with every response.
    assert payload["accepted_span_grammar"] == ACCEPTED_SPAN_EXAMPLES
    assert "same admission code" in payload["basis"]


def test_preflight_matches_the_forecast_verdicts(tmp_path: Path):
    """Deterministic preflight verdicts must be the forecast's verdicts:
    the same events admitted here are the ones the future-context gate
    admits when the forecast actually runs."""
    from gnomon.config import GnomonConfig
    from gnomon.runtime import forecast

    source = tmp_path / "series.csv"
    _write_csv(source)
    events = _events()[:2]
    payload = preflight_context_events(
        str(source), time_column="timestamp", target_column="value",
        horizon=7, context_events=events,
    )
    config = GnomonConfig()
    config.context.future_events = True
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=7, frequency="D", output=str(tmp_path / "out"),
        context_events=events, config=config,
    )
    run_gate = artifact.results[0].future_context
    preflight_gate = payload["series"][0]["future_context"]
    assert [item["event_id"] for item in preflight_gate["admitted"]] == \
        [item["event_id"] for item in run_gate["admitted"]]
    assert [item["code"] for item in preflight_gate["rejected"]] == \
        [item["code"] for item in run_gate["rejected"]]


def test_every_grammar_example_actually_parses():
    """The advertised grammar must not drift from the parser: every
    constraint example yields a bound, every override example a value."""
    for span in ACCEPTED_SPAN_EXAMPLES["constraint"]:
        bound, problem = parse_bound_span(span)
        assert bound is not None, (span, problem)
    for span in ACCEPTED_SPAN_EXAMPLES["override"]:
        value, problem = parse_override_span(span)
        scale, _ = parse_override_scale(span)
        assert value is not None or scale is not None, (span, problem)


def test_census_sensor_shapes_state_their_values():
    """The two sensor-family shapes from the 2026-08 census: a narrated
    level change and a quoted (timestamp, value) tuple. Both state the
    number verbatim; the parser must read exactly it."""
    assert parse_override_span(
        "At 05:27:09, it rapidly and smoothly changes to 1593.0")[0] == 1593.0
    assert parse_override_span("(2022-03-23 00:00:00, 0)")[0] == 0.0
    # a clock time after the movement verb is not a level
    value, _ = parse_override_span("it changes to 5:30 pm operation")
    assert value != 5.0


def test_residue_shapes_from_the_recovery_measurement():
    """The 2026-08 recovery measurement's residue, pinned verbatim: 63 of
    74 remaining rejections were grammar gaps, dominated by bare
    value+window spans. Each shape here was an actual rejected span."""
    assert parse_override_span("0.2 from 05:34:29 until 05:34:46")[0] == 0.2
    assert parse_override_span(
        "the covariate X_0 takes a value of 0.2051 "
        "from 2028-04-23 to 2028-05-05")[0] == 0.2051
    assert parse_override_scale(
        "there is 10.0% of the usual traffic "
        "from 2024-01-15 15:00:00 for 6 hours")[0] == 0.1
    scale, problem = parse_override_scale(
        "Normally 11 times the usual level, but actually 3 times the usual level")
    assert scale == 3.0 and problem is None
    # "no <activity>" states a count of zero — the census caught
    # "no withdrawals" slipping the shorter curated noun list
    assert parse_override_span(
        "resulting in no withdrawals during that period")[0] == 0.0
    bound, _ = parse_bound_span(
        "At full load (=1), the fan turns at a maximum speed of 3000 rpm.")
    assert bound is not None and bound.maximum == 3000.0
    # what must stay out: a delta, an unchanged level, a bare percent of
    # an unstated base, and a date-only event with no stated value
    assert parse_override_span("increased by 5 from 2024-01-01")[0] is None
    assert parse_override_span("no change from 2024-01-01 onwards")[0] is None
    assert parse_override_span(
        "reduced to 50% while the line is partially shut down")[0] is None
    assert parse_override_scale(
        "reduced to 50% while the line is partially shut down")[0] is None
    assert parse_bound_span(
        "At 2022-07-15 06:00:00, we expect that the weather will "
        "become clear.")[0] is None


def test_preflight_is_reachable_from_the_mcp_tool(tmp_path: Path):
    from gnomon.context import event_to_dict
    from gnomon.toolspec import runner_for, visible_tools

    source = tmp_path / "series.csv"
    _write_csv(source)
    events_file = tmp_path / "events.json"
    events_file.write_text(json.dumps({
        "schema_version": "0.1",
        "events": [event_to_dict(event) for event in _events()],
    }))
    runner = runner_for("gnomon_preflight_context")
    assert runner is not None
    payload = runner({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "horizon": 7,
        "context_events_file": str(events_file),
    })
    assert payload["status"] == "ok"
    assert len(payload["series"][0]["events"]) == 4
    assert any(tool["name"] == "gnomon_preflight_context"
               for tool in visible_tools())


def test_preflight_is_reachable_from_the_cli(tmp_path: Path, capsys):
    from gnomon.cli import main
    from gnomon.context import event_to_dict

    source = tmp_path / "series.csv"
    _write_csv(source)
    events_file = tmp_path / "events.json"
    events_file.write_text(json.dumps({
        "schema_version": "0.1",
        "events": [event_to_dict(event) for event in _events()],
    }))
    code = main([
        "context", "preflight", str(source),
        "--time", "timestamp", "--target", "value",
        "--horizon", "7", "--events", str(events_file),
    ])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "ok"
    assert {entry["outcome"] for entry in printed["series"][0]["events"]} == \
        {"would_influence", "rejected", "ablation_gated"}


def test_admission_lanes_are_reachable_from_the_mcp_tool(tmp_path: Path, ):
    """MCP tool calls do not read gnomon.yaml, so the admission lanes must
    be per-call parameters or they are unreachable from the primary agent
    surface — the same gap best_effort had."""
    from gnomon.context import event_to_dict
    from gnomon.toolspec import runner_for, visible_tools

    source = tmp_path / "series.csv"
    _write_csv(source)
    events_file = tmp_path / "events.json"
    events_file.write_text(json.dumps({
        "schema_version": "0.1",
        "events": [event_to_dict(_event(
            "c1", "constraint:cap",
            {"source_span": "the value will not exceed 150"}))],
    }))
    runner = runner_for("gnomon_forecast")
    arguments = {
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "horizon": 7,
        "context_events_file": str(events_file),
    }
    without = runner({**arguments, "output_dir": str(tmp_path / "a")})
    assert without["results"][0]["support"] != "context_trusted"
    with_lane = runner({**arguments, "output_dir": str(tmp_path / "b"),
                        "future_events": True})
    result = with_lane["results"][0]
    assert result["support"] == "context_trusted"
    # the newly admitted bound binds the emitted numbers
    assert all(row["q90"] <= 150.0 + 1e-9
               for row in result["forecast"])
    spec = next(tool for tool in visible_tools()
                if tool["name"] == "gnomon_forecast")
    assert "future_events" in spec["inputSchema"]["properties"]
    assert "structural_events" in spec["inputSchema"]["properties"]


def _inline_events():
    from gnomon.context import event_to_dict
    return [event_to_dict(_event(
        "c1", "constraint:cap",
        {"source_span": "the value will not exceed 150"}))]


def test_inline_context_events_reach_the_forecast_tool(tmp_path: Path):
    """An MCP client holds no filesystem. With a file-only parameter the
    admission lanes were unreachable from the published tool surface —
    a model could be told about future_events and still have no way to
    supply an event. The inline channel closes that."""
    from gnomon.toolspec import runner_for

    source = tmp_path / "series.csv"
    _write_csv(source)
    runner = runner_for("gnomon_forecast")
    payload = runner({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "horizon": 7,
        "output_dir": str(tmp_path / "out"),
        "context_events": _inline_events(),
        "future_events": True,
    })
    result = payload["results"][0]
    assert result["support"] == "context_trusted"
    assert all(row["q90"] <= 150.0 + 1e-9
               for row in result["forecast"])


def test_inline_events_reach_the_preflight_tool(tmp_path: Path):
    from gnomon.toolspec import runner_for

    source = tmp_path / "series.csv"
    _write_csv(source)
    payload = runner_for("gnomon_preflight_context")({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "horizon": 7,
        "context_events": _inline_events(),
    })
    (series,) = payload["series"]
    assert series["events"][0]["outcome"] == "would_influence"


def test_both_channels_concatenate(tmp_path: Path):
    from gnomon.context import event_to_dict
    from gnomon.toolspec import runner_for

    source = tmp_path / "series.csv"
    _write_csv(source)
    events_file = tmp_path / "events.json"
    events_file.write_text(json.dumps({
        "schema_version": "0.1",
        "events": [event_to_dict(_event(
            "from-file", "constraint:cap",
            {"source_span": "values stay between 0 and 340"}))],
    }))
    payload = runner_for("gnomon_preflight_context")({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "horizon": 7,
        "context_events_file": str(events_file),
        "context_events": _inline_events(),
    })
    (series,) = payload["series"]
    assert {entry["event_id"] for entry in series["events"]} == \
        {"from-file", "c1"}


def test_invalid_inline_event_fails_loudly_with_problems(tmp_path: Path):
    import pytest

    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    source = tmp_path / "series.csv"
    _write_csv(source)
    bad = [{"event_id": "e1", "event_type": "constraint:cap",
            "entity_scope": ["*"],
            "effective_start": "not-a-date",
            "effective_end": "also-not", "known_at": "nope"}]
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_preflight_context")({
            "input": str(source), "time_column": "timestamp",
            "target_column": "value", "horizon": 7,
            "context_events": bad,
        })
    assert caught.value.code == "INVALID_CONTEXT_EVENT"
    assert "e1" in caught.value.details["problems"]


def test_preflight_without_any_events_says_what_to_supply(tmp_path: Path):
    import pytest

    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    source = tmp_path / "series.csv"
    _write_csv(source)
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_preflight_context")({
            "input": str(source), "time_column": "timestamp",
            "target_column": "value", "horizon": 7,
        })
    assert "context_events" in caught.value.message
