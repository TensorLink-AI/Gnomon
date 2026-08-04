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
