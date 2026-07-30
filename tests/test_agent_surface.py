"""Phase 4: the agent surface — registry-generated tools, artifact readers,
structured repair options."""

from datetime import datetime, timezone
from pathlib import Path

import json

from aion.contracts import AionError
from aion.ids import FixedClock
from aion.macros import monitor
from aion.mcp_server import _handle
from aion.runtime import forecast
from aion.toolspec import TOOLS, runner_for

REPO = Path(__file__).resolve().parent.parent
CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))

V02_TOOL_NAMES = [
    "aion_capabilities", "aion_inspect", "aion_forecast",
    "aion_covariate_guide", "aion_validate_covariates", "aion_propose_covariates",
    "aion_submit_actuals", "aion_list_open_forecasts", "aion_model_performance",
    "aion_record_decision", "aion_resolve_decision",
]
NEW_TOOL_NAMES = [
    "aion_investigate_change", "aion_decide", "aion_monitor",
    "aion_get_artifact", "aion_explain_run",
]


def test_v02_surface_is_preserved_and_macros_added():
    names = [tool["name"] for tool in TOOLS]
    for name in V02_TOOL_NAMES:
        assert name in names, f"frozen v0.2 tool {name} missing"
    for name in NEW_TOOL_NAMES:
        assert name in names, f"new tool {name} missing"


def test_mcp_lists_macro_tools():
    result = _handle({"method": "tools/list"})
    names = {tool["name"] for tool in result["tools"]}
    assert set(NEW_TOOL_NAMES) <= names
    schemas = {tool["name"]: tool["inputSchema"] for tool in result["tools"]}
    assert "actions" in schemas["aion_decide"]["properties"]
    assert "threshold" in schemas["aion_monitor"]["required"]


def test_macro_schemas_come_from_registry():
    from aion.registry import MACROS
    by_name = {tool["name"]: tool for tool in TOOLS}
    for spec in MACROS.values():
        if spec.tool_name == "aion_forecast":
            continue  # frozen v0.2 definition
        assert by_name[spec.tool_name]["inputSchema"] == spec.input_schema
        assert by_name[spec.tool_name]["description"] == spec.summary


def test_get_artifact_and_explain_run(tmp_path):
    artifact, directory = forecast(
        str(REPO / "examples" / "daily_requests.csv"),
        time_column="timestamp", target_column="requests", horizon=5,
        output=str(tmp_path), clock=CLOCK,
    )
    fetched = runner_for("aion_get_artifact")({
        "artifact_path": str(directory), "include_lineage": True,
    })
    assert fetched["artifact"]["forecast_id"] == artifact.forecast_id
    assert fetched["lineage"]["claims"]

    explained = runner_for("aion_explain_run")({"artifact_path": str(directory)})
    assert explained["artifact_id"] == artifact.forecast_id
    assert "__default__" in explained["support_assessments"]
    assert explained["claims"][0]["claim_class"] in ("predictive", "descriptive")
    assert "summary_md" in explained


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
    explained = runner_for("aion_explain_run")({"artifact_path": str(directory)})
    assert explained["artifact_id"] == payload["monitor_id"]
    assert explained["support_assessments"]


def test_errors_carry_repair_options():
    error = AionError("DATASET_NOT_FOUND", "no such dataset", {"available": []})
    envelope = error.to_dict()
    actions = [option["action"] for option in envelope["error"]["repair_options"]]
    assert "list_datasets" in actions and "ingest" in actions
    # Unknown codes degrade to an empty list, never a crash.
    assert AionError("BRAND_NEW_CODE", "x").to_dict()["error"]["repair_options"] == []


def test_mcp_tool_error_includes_repair_options(tmp_path):
    result = _handle({"method": "tools/call", "params": {
        "name": "aion_decide",
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
    from aion.runtime import capabilities
    payload = capabilities()
    assert "aion_investigate_change" in payload["macros"]
    assert payload["operators"]["evaluate_actions"]["claim_classes"] == ["decision"]
    assert payload["features"]["claim_verifier"] is True
