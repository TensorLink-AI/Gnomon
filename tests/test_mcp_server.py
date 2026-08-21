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
    assert tool_names[:5] == [
            "gnomon_capabilities", "gnomon_inspect", "gnomon_forecast",
            "gnomon_validate_covariates",
        "gnomon_submit_actuals",
    ]
    assert set(tool_names[5:]) == {
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


def test_describe_volatility_question_serializes(monkeypatch) -> None:
    # Degenerate example data leaves the volatility direction baselines with
    # no samples; math.inf sentinels in the diagnostics used to make
    # json.dumps(allow_nan=False) raise, mislabelled as TRACKING_ERROR.
    from gnomon import mcp_server

    monkeypatch.delenv("GNOMON_MCP_PROFILE", raising=False)

    result = mcp_server._handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "gnomon_describe", "arguments": {
            "input": str(REPO_ROOT / "examples" / "daily_requests.csv"),
            "questions": [{"verb": "predict", "property": "volatility",
                           "target": "requests"}]}}})
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    answer = payload["answers"][0]
    assert answer["question"]["property"] == "volatility"
    assert "direction" in answer["answer"]


def test_json_safe_nulls_non_finite_floats() -> None:
    import math

    from gnomon.mcp_server import _json_safe

    assert _json_safe({
        "inf": math.inf, "nan": math.nan,
        "nested": [1.0, {"neg": -math.inf}, (math.nan,)],
    }) == {"inf": None, "nan": None, "nested": [1.0, {"neg": None}, [None]]}


def test_unserializable_tool_result_reports_internal_error() -> None:
    from gnomon.mcp_server import _tool_result

    result = _tool_result({"payload": object()}, False)
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["error"]["code"] == "INTERNAL_ERROR"
