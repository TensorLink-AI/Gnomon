from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _talk(messages: list[dict]) -> list[dict]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT / "src"), env.get("PYTHONPATH")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-m", "aion", "mcp", "serve"],
        input="".join(json.dumps(message) + "\n" for message in messages),
        capture_output=True, text=True, timeout=60, env=env,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def test_initialize_list_and_call() -> None:
    responses = _talk([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "aion_capabilities", "arguments": {}}},
    ])
    by_id = {response["id"]: response for response in responses}
    assert by_id[1]["result"]["serverInfo"]["name"] == "aion"
    tool_names = [tool["name"] for tool in by_id[2]["result"]["tools"]]
    # The frozen v0.2 tools come first, unchanged; registry-generated macro
    # and artifact tools follow.
    assert tool_names[:11] == [
            "aion_capabilities", "aion_inspect", "aion_forecast",
            "aion_covariate_guide", "aion_validate_covariates",
            "aion_propose_covariates",
        "aion_submit_actuals", "aion_list_open_forecasts",
        "aion_model_performance", "aion_record_decision",
        "aion_resolve_decision",
    ]
    assert set(tool_names[11:]) == {
        "aion_investigate_change", "aion_detect_anomalies", "aion_decide", "aion_monitor",
        "aion_get_artifact", "aion_explain_run",
        "aion_status", "aion_resolve_outcome", "aion_route",
    }
    call = by_id[3]["result"]
    assert call["isError"] is False
    payload = json.loads(call["content"][0]["text"])
    assert payload["interfaces"]["mcp"] is True


def test_tool_error_is_structured_not_fatal() -> None:
    responses = _talk([
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "aion_inspect", "arguments": {
             "input": "/does/not/exist.csv", "time_column": "t", "target_column": "y"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
    ])
    by_id = {response["id"]: response for response in responses}
    call = by_id[1]["result"]
    assert call["isError"] is True
    payload = json.loads(call["content"][0]["text"])
    assert payload["error"]["code"] == "INPUT_NOT_FOUND"
    assert by_id[2]["result"] == {}


def test_unknown_method_returns_jsonrpc_error() -> None:
    responses = _talk([{"jsonrpc": "2.0", "id": 1, "method": "resources/list"}])
    assert responses[0]["error"]["code"] == -32601
