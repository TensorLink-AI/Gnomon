"""Phase 4: the agent surface — registry-generated tools, artifact readers,
structured repair options."""

from datetime import datetime, timezone
from pathlib import Path

import json

from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock
from gnomon.macros import monitor
from gnomon.mcp_server import _handle
from gnomon.runtime import forecast
from gnomon.toolspec import TOOLS, runner_for

REPO = Path(__file__).resolve().parent.parent
CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))

NEW_TOOL_NAMES = [
    "gnomon_investigate_change", "gnomon_decide", "gnomon_monitor",
    "gnomon_get_artifact", "gnomon_explain_run",
    "gnomon_status", "gnomon_resolve_outcome",
]


def test_canonical_surface_has_current_tools_only():
    names = [tool["name"] for tool in TOOLS]
    for name in NEW_TOOL_NAMES:
        assert name in names, f"new tool {name} missing"
    retired = {
        "gnomon_covariate_guide", "gnomon_propose_covariates",
        "gnomon_list_open_forecasts", "gnomon_model_performance",
        "gnomon_record_decision", "gnomon_resolve_decision",
        "gnomon_proposer_skill",
    }
    assert retired.isdisjoint(names)


def test_mcp_lists_macro_tools():
    result = _handle({"method": "tools/list"})
    names = {tool["name"] for tool in result["tools"]}
    assert set(NEW_TOOL_NAMES) <= names
    schemas = {tool["name"]: tool["inputSchema"] for tool in result["tools"]}
    assert "actions" in schemas["gnomon_decide"]["properties"]
    assert "threshold" in schemas["gnomon_monitor"]["required"]


def test_macro_schemas_come_from_registry():
    from gnomon.registry import MACROS
    by_name = {tool["name"]: tool for tool in TOOLS}
    for spec in MACROS.values():
        if spec.tool_name == "gnomon_forecast":
            continue  # frozen v0.2 definition
        assert by_name[spec.tool_name]["inputSchema"] == spec.input_schema
        assert by_name[spec.tool_name]["description"] == spec.summary


def test_get_artifact_and_explain_run(tmp_path):
    artifact, directory = forecast(
        str(REPO / "examples" / "daily_requests.csv"),
        time_column="timestamp", target_column="requests", horizon=5,
        output=str(tmp_path), clock=CLOCK,
    )
    fetched = runner_for("gnomon_get_artifact")({
        "artifact_path": str(directory), "include_lineage": True,
    })
    assert fetched["artifact"]["forecast_id"] == artifact.forecast_id
    assert fetched["lineage"]["claims"]

    explained = runner_for("gnomon_explain_run")({"artifact_path": str(directory)})
    assert explained["artifact_id"] == artifact.forecast_id
    assert "__default__" in explained["support_assessments"]
    assert explained["claims"][0]["claim_class"] in ("predictive", "descriptive")
    assert "summary_md" in explained

    selected = runner_for("gnomon_get_artifact")({
        "artifact_path": str(directory),
        "series": "__default__",
        "fields": ["support", "notability"],
        "order_by": "notability",
        "limit": 1,
    })
    [row] = selected["artifact"]["results"]
    assert set(row) == {"series", "support", "notability"}
    assert selected["selection"]["matched"] == 1


def test_explain_run_on_monitor_artifact(tmp_path):
    source = tmp_path / "trend.csv"
    from datetime import date, timedelta
    start = date(2026, 3, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{(start + timedelta(days=i)).isoformat()},{100 + i * 1.5}" for i in range(48)
    ) + "\n")
    payload, directory = monitor(
        str(source), time_column="timestamp", target_column="value",
        horizon=6, threshold=165.0, alert_cost=1.0, miss_cost=9.0,
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    explained = runner_for("gnomon_explain_run")({"artifact_path": str(directory)})
    assert explained["artifact_id"] == payload["monitor_id"]
    assert explained["support_assessments"]


def test_errors_carry_repair_options():
    error = GnomonError("DATASET_NOT_FOUND", "no such dataset", {"available": []})
    envelope = error.to_dict()
    actions = [option["action"] for option in envelope["error"]["repair_options"]]
    assert "list_datasets" in actions and "ingest" in actions
    # Unknown codes degrade to an empty list, never a crash.
    assert GnomonError("BRAND_NEW_CODE", "x").to_dict()["error"]["repair_options"] == []


def test_mcp_tool_error_includes_repair_options(tmp_path):
    result = _handle({"method": "tools/call", "params": {
        "name": "gnomon_decide",
        "arguments": {
            "input": str(REPO / "examples" / "daily_requests.csv"),
            "time_column": "timestamp", "target_column": "nope",
            "horizon": 3, "threshold": 1.0, "actions": [{"name": "a"}],
            "output_dir": str(tmp_path),
        },
    }})
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"]["code"] == "MISSING_COLUMNS"
    assert payload["error"]["repair_options"][0]["action"] == "inspect_dataset"


def test_capabilities_expose_registry():
    from gnomon.runtime import capabilities
    payload = capabilities()
    assert "gnomon_investigate_change" in payload["macros"]
    assert payload["operators"]["evaluate_actions"]["claim_classes"] == ["decision"]
    assert payload["features"]["claim_verifier"] is True
