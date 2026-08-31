from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from gnomon.mcp_server import _handle

REPO_ROOT = Path(__file__).resolve().parent.parent


def _talk(messages: list[dict]) -> list[dict]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT / "src"), env.get("PYTHONPATH")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-m", "gnomon", "mcp", "serve"],
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
         "params": {"name": "gnomon_capabilities", "arguments": {}}},
    ])
    by_id = {response["id"]: response for response in responses}
    assert by_id[1]["result"]["serverInfo"]["name"] == "gnomon"
    tool_names = [tool["name"] for tool in by_id[2]["result"]["tools"]]
    # The frozen v0.2 tools come first, unchanged; registry-generated macro
    # and artifact tools follow.
    assert tool_names[:6] == [
            "gnomon_capabilities", "gnomon_inspect", "gnomon_describe",
            "gnomon_forecast", "gnomon_validate_covariates",
        "gnomon_submit_actuals",
    ]
    assert set(tool_names[6:]) == {
        "gnomon_investigate_change", "gnomon_detect_anomalies", "gnomon_decide", "gnomon_monitor",
        "gnomon_get_artifact", "gnomon_explain_run",
        "gnomon_status", "gnomon_resolve_outcome", "gnomon_route",
        # The bitemporal store, previously reachable only from the CLI.
        "gnomon_ingest", "gnomon_list_datasets",
        # Admission dry-run: rejection as a repair loop, not a post-mortem.
        "gnomon_preflight_context",
        # Detached sandbox install with polling — TSFMs were disclosed by
        # capabilities but installable only from a shell.
            "gnomon_install_tsfm",
            # Number-free reranking over an existing sealed publication.
            "gnomon_select_scenario",
        }
    call = by_id[3]["result"]
    assert call["isError"] is False
    payload = json.loads(call["content"][0]["text"])
    assert payload["interfaces"]["mcp"] is True


def test_tool_error_is_structured_not_fatal() -> None:
    responses = _talk([
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "gnomon_inspect", "arguments": {
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


def test_known_tool_hidden_by_profile_names_reachable_profiles(monkeypatch) -> None:
    monkeypatch.setenv("GNOMON_MCP_PROFILE", "evidence")
    result = _handle({
        "method": "tools/call",
        "params": {"name": "gnomon_monitor", "arguments": {}},
    })
    assert result is not None and result["isError"] is True
    error = result["structuredContent"]["error"]
    assert error["code"] == "TOOL_NOT_IN_PROFILE"
    assert error["details"] == {
        "tool": "gnomon_monitor",
        "profiles": ["core", "data", "decision", "describe", "full"],
        "active_profile": "evidence",
    }
    assert error["repair_options"][0]["action"] == "select_profile"


def test_truly_unknown_tool_remains_unknown(monkeypatch) -> None:
    monkeypatch.setenv("GNOMON_MCP_PROFILE", "core")
    result = _handle({
        "method": "tools/call",
        "params": {"name": "gnomon_time_machine", "arguments": {}},
    })
    assert result is not None
    assert result["structuredContent"]["error"]["code"] == "UNKNOWN_TOOL"
