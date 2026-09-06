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
    assert set(tool_names) == {
        "gnomon_capabilities", "gnomon_inspect", "gnomon_describe",
        "gnomon_forecast", "gnomon_evaluate", "gnomon_read",
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


def test_unknown_and_retired_tools_are_structured_errors():
    from gnomon import GnomonSession
    with GnomonSession.from_config() as session:
        for name in ("gnomon_install_tsfm", "gnomon_monitor", "not_a_tool"):
            result = _handle({"method": "tools/call", "params": {"name": name}}, session=session)
            assert result["isError"]
            assert result["structuredContent"]["error"]["code"] == "UNKNOWN_TOOL"
