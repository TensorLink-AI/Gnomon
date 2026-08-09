"""Schema inference on the tool surface.

The CLI infers --time/--target (and defaults --horizon) when the file
leaves no choice, because a bare missing-argument failure on an obvious
two-column file was the most common first-run failure there is. These
tests hold the MCP/agent tool surface to the same standard: the same
inferences, the same strictness, every inference disclosed as an
assumption, and refusals spoken in tool-parameter vocabulary rather than
CLI flags.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.mcp_server import _handle
from gnomon.toolspec import runner_for

REPO = Path(__file__).resolve().parent.parent
DAILY = str(REPO / "examples" / "daily_requests.csv")


def test_inspect_infers_columns_and_discloses() -> None:
    payload = runner_for("gnomon_inspect")({"input": DAILY})
    assert payload["status"] in ("valid", "ok", "complete")
    assumptions = payload["assumptions"]
    assert any("time_column" in item and "'timestamp'" in item
               for item in assumptions)
    assert any("target_column" in item and "'requests'" in item
               for item in assumptions)


def test_forecast_needs_only_the_file(tmp_path) -> None:
    payload = runner_for("gnomon_forecast")({
        "input": DAILY, "output_dir": str(tmp_path),
    })
    assert payload["status"] == "complete"
    result = payload["results"][0]
    assumptions = result["support_assessment"]["assumptions"]
    assert any("time_column" in item for item in assumptions)
    assert any("target_column" in item for item in assumptions)
    assert any("horizon was not supplied" in item for item in assumptions)
    # The default horizon is one seasonal period of the inferred grid.
    assert result["forecast_rows"] >= 1


def test_registry_macro_infers_too(tmp_path) -> None:
    payload = runner_for("gnomon_investigate_change")({
        "input": DAILY, "output_dir": str(tmp_path),
    })
    assert payload["status"] == "complete"
    assumptions = payload["results"][0]["support_assessment"]["assumptions"]
    assert any("time_column" in item for item in assumptions)


def test_explicit_columns_disclose_nothing(tmp_path) -> None:
    payload = runner_for("gnomon_forecast")({
        "input": DAILY, "time_column": "timestamp",
        "target_column": "requests", "horizon": 3,
        "output_dir": str(tmp_path),
    })
    assessment = payload["results"][0]["support_assessment"]
    assert not any("was not supplied" in item
                   for item in assessment.get("assumptions", []))


def test_ambiguous_target_names_candidates_in_tool_vocabulary(tmp_path) -> None:
    source = tmp_path / "vitals.csv"
    source.write_text(
        "timestamp,hr,spo2\n"
        "2026-01-01,60,98\n2026-01-02,62,97\n2026-01-03,61,98\n",
        encoding="utf-8",
    )
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_forecast")({"input": str(source), "horizon": 2})
    error = caught.value
    assert error.code == "AMBIGUOUS_SCHEMA"
    assert error.details["parameter"] == "target_column"
    assert error.details["candidates"] == ["hr", "spo2"]
    # Refusals speak tool parameters, never CLI flags.
    assert "--target" not in error.message
    actions = {option["action"] for option in error.repair_options}
    assert "supply_arguments" in actions
    assert "forecast_all_candidates" in actions


def test_store_inputs_still_require_explicit_columns() -> None:
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_inspect")({"input": "store:whatever"})
    error = caught.value
    assert error.code == "INVALID_ARGUMENTS"
    assert error.details["missing_parameters"] == [
        "time_column", "target_column",
    ]
    assert error.repair_options[0]["arguments"] == [
        "time_column", "target_column",
    ]


def test_ingest_never_infers() -> None:
    # A write to the store under guessed columns would persist the guess:
    # the schema still requires the columns, and the runner raises without
    # them rather than inferring.
    with pytest.raises((GnomonError, KeyError)):
        runner_for("gnomon_ingest")({"input": DAILY, "dataset": "tmp"})


def test_mcp_call_with_only_the_file(tmp_path) -> None:
    result = _handle({"method": "tools/call", "params": {
        "name": "gnomon_forecast",
        "arguments": {"input": DAILY, "output_dir": str(tmp_path)},
    }})
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["status"] == "complete"


def test_schemas_no_longer_require_inferable_parameters() -> None:
    tools = {tool["name"]: tool for tool in _handle({"method": "tools/list"})["tools"]}
    for name in ("gnomon_inspect", "gnomon_forecast", "gnomon_route",
                 "gnomon_investigate_change", "gnomon_detect_anomalies"):
        assert "time_column" not in tools[name]["inputSchema"]["required"]
        assert "target_column" not in tools[name]["inputSchema"]["required"]
    assert "horizon" not in tools["gnomon_forecast"]["inputSchema"]["required"]
    # Verbs whose parameters are not derivable from the data keep them.
    assert "horizon" in tools["gnomon_decide"]["inputSchema"]["required"]
    assert "threshold" in tools["gnomon_monitor"]["inputSchema"]["required"]
    assert "time_column" in tools["gnomon_ingest"]["inputSchema"]["required"]


def test_inline_observations_infer_as_well(tmp_path) -> None:
    rows = [
        {"timestamp": f"2026-01-{day:02d}", "value": 100 + 3 * day}
        for day in range(1, 15)
    ]
    payload = runner_for("gnomon_forecast")({
        "observations": rows, "horizon": 2, "output_dir": str(tmp_path),
    })
    assert payload["status"] == "complete"
    assumptions = payload["results"][0]["support_assessment"]["assumptions"]
    assert any("'value'" in item for item in assumptions)
