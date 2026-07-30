"""A dependency-free MCP stdio server over the canonical runtime.

Implements the subset of the Model Context Protocol an agent host needs to
operate Aion's tools: initialize, ping, tools/list, and tools/call, as
newline-delimited JSON-RPC on stdio. Tool results carry the same JSON
payloads as the CLI — success and structured errors alike — so every
surface speaks one contract. Logs go to stderr; stdout carries protocol
messages only.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .contracts import AionError
from .toolspec import runner_for, visible_tools

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "aion", "version": "0.2.0"}


def _tool_result(payload: dict[str, Any], is_error: bool) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, allow_nan=False)}],
        "isError": is_error,
    }


def _handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    if method == "initialize":
        requested = (message.get("params") or {}).get("protocolVersion")
        return {
            "protocolVersion": requested or PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": tool["inputSchema"],
                }
                for tool in visible_tools()
            ]
        }
    if method == "tools/call":
        params = message.get("params") or {}
        runner = runner_for(str(params.get("name")))
        if runner is None:
            return _tool_result(
                AionError("UNKNOWN_TOOL", f"No such tool: {params.get('name')!r}").to_dict(),
                True,
            )
        arguments = params.get("arguments") or {}
        try:
            return _tool_result(runner(arguments), False)
        except AionError as exc:
            return _tool_result(exc.to_dict(), True)
        except KeyError as exc:
            return _tool_result(
                AionError("INVALID_ARGUMENTS", f"Missing required argument: {exc.args[0]}").to_dict(),
                True,
            )
        except (ValueError, FileNotFoundError) as exc:
            return _tool_result(
                AionError("TRACKING_ERROR", str(exc)).to_dict(), True,
            )
    return None


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    print("aion mcp server listening on stdio", file=sys.stderr)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        message_id = message.get("id")
        method = str(message.get("method", ""))
        if method.startswith("notifications/"):
            continue
        try:
            result = _handle(message)
        except Exception as exc:  # a protocol bug must not kill the server
            result = None
            if message_id is not None:
                _write(stdout, {
                    "jsonrpc": "2.0", "id": message_id,
                    "error": {"code": -32603, "message": f"internal error: {exc}"},
                })
                continue
        if message_id is None:
            continue
        if result is None:
            _write(stdout, {
                "jsonrpc": "2.0", "id": message_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            })
        else:
            _write(stdout, {"jsonrpc": "2.0", "id": message_id, "result": result})
    return 0


def _write(stdout: TextIO, message: dict[str, Any]) -> None:
    stdout.write(json.dumps(message, allow_nan=False) + "\n")
    stdout.flush()
