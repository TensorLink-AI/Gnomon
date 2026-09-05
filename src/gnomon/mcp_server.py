"""A dependency-free MCP stdio server over the canonical runtime.

Implements the subset of the Model Context Protocol an agent host needs to
operate Gnomon's tools: initialize, ping, tools/list, and tools/call, as
newline-delimited JSON-RPC on stdio. Tool results carry the same JSON
payloads as the CLI — success and structured errors alike — so every
surface speaks one contract. Logs go to stderr; stdout carries protocol
messages only.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from typing import Any, TextIO

from .contracts import GnomonError
from .product_contract import __version__

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "gnomon", "version": __version__}
MAX_REQUEST_BYTES = 1024 * 1024

#: The shape every tool result shares. Tools may publish something tighter
#: via an `outputSchema` key in their spec; this is the floor, and it is
#: what makes `structuredContent` checkable rather than merely present.
ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "status": {"type": "string"},
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "retryable": {"type": "boolean"},
                "details": {"type": "object"},
                "repair_options": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["code", "message"],
        },
        "support_assessment": {"type": "object"},
        "artifact_path": {"type": "string"},
    },
}


def _tool_result(payload: dict[str, Any], is_error: bool) -> dict[str, Any]:
    """A tool result in both shapes the advertised protocol supports.

    The text block stays exactly as it was, so a client that only reads
    `content` is unaffected. `structuredContent` carries the same payload
    as an object, which is what protocol 2025-06-18 exists to allow — an
    agent should not have to `JSON.parse` a string and validate it against
    nothing.
    """
    text = json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    return {
        "content": [{"type": "text", "text": text}],
        # Round-tripped rather than passed through, so the structured form
        # is byte-for-byte what the text block says — a tuple in the payload
        # would otherwise be an array in one and a tuple in the other.
        "structuredContent": json.loads(text),
        "isError": is_error,
    }


def runner_for(name):
    # Legacy tools remain lazy so an execution session does not import the
    # optional publication/context machinery merely to serve inference.
    from .toolspec import runner_for as legacy_runner
    return legacy_runner(name)


def _handle(message: dict[str, Any], *, session=None) -> dict[str, Any] | None:
    method = message.get("method")
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        if session is None:
            from .toolspec import visible_tools
            listed = visible_tools()
        else:
            listed = session.tools()
        return {
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": tool["inputSchema"],
                    # Published so a client can validate `structuredContent`
                    # rather than trusting it.
                    "outputSchema": tool.get("outputSchema", ENVELOPE_SCHEMA),
                }
                for tool in listed
            ]
        }
    if method == "tools/call":
        params = message.get("params") or {}
        name = str(params.get("name"))
        if session is not None:
            try:
                return _tool_result(session.call(name, params.get("arguments", {})), False)
            except GnomonError as exc:
                try:
                    return _tool_result(session.results.project(exc.to_dict()), True)
                except GnomonError:
                    return _tool_result(GnomonError("ERROR_DETAIL_LIMIT", "Error details exceed the result limit.").to_dict(), True)
            except Exception:
                # Provider exceptions can contain sensitive service details.
                # Do not send arbitrary exception strings back to the agent.
                logger.error("Execution session failed in tool %s", name)
                return _tool_result(GnomonError("EXECUTION_FAILED", "Provider or ledger execution failed.").to_dict(), True)
        from .toolspec import TOOLS, active_profile, profiles_for_tool
        runner = runner_for(name)
        if runner is None:
            known = next((tool for tool in TOOLS if tool["name"] == name), None)
            if known is not None:
                return _tool_result(
                    GnomonError(
                        "TOOL_NOT_IN_PROFILE",
                        f"Tool {name!r} is not exposed by the active "
                        f"{active_profile()!r} profile.",
                        {"tool": name, "profiles": profiles_for_tool(name),
                         "active_profile": active_profile()},
                    ).to_dict(),
                    True,
                )
            return _tool_result(
                GnomonError("UNKNOWN_TOOL", f"No such tool: {name!r}").to_dict(),
                True,
            )
        arguments = {**(params.get("arguments") or {}),
                     "_mcp_agent_boundary": True}
        try:
            return _tool_result(runner(arguments), False)
        except GnomonError as exc:
            return _tool_result(exc.to_dict(), True)
        except KeyError as exc:
            missing = str(exc.args[0])
            return _tool_result(
                GnomonError(
                    "INVALID_ARGUMENTS",
                    f"Missing required argument: {missing}",
                    {"missing_arguments": [missing], "tool": name},
                ).to_dict(),
                True,
            )
        except FileNotFoundError as exc:
            return _tool_result(
                GnomonError("INPUT_NOT_FOUND", str(exc)).to_dict(), True,
            )
        except ValueError as exc:
            return _tool_result(
                GnomonError("INVALID_ARGUMENTS", str(exc)).to_dict(), True,
            )
        except Exception as exc:
            # A bug in a tool must reach the model as a repairable result,
            # never as a transport error. Anything uncaught used to escape
            # to the outer handler and become JSON-RPC -32603, which gives
            # an agent no payload, no repair_options, and no way to
            # self-correct — a protocol failure standing in for a tool
            # failure.
            logger.exception("Unhandled error in tool %s", params.get("name"))
            return _tool_result(
                GnomonError(
                    "INTERNAL_ERROR",
                    f"{type(exc).__name__}: {exc}",
                    {"tool": params.get("name")},
                ).to_dict(),
                True,
            )
    return None


def _bounded_lines(stream):
    # TextIO bounds decoded characters; the extra UTF-8 check enforces bytes.
    # Caller-provided iterators already own their strings; do not claim to bound
    # their allocation. Oversized physical lines are drained before the next RPC.
    reader = getattr(stream, "buffer", stream)
    sentinel = b"" if reader is not stream else ""
    chunks = iter(lambda: reader.readline(MAX_REQUEST_BYTES + 1), sentinel) if hasattr(reader, "readline") else iter(reader)
    discarding = False
    for chunk in chunks:
        newline = chunk.endswith(b"\n" if isinstance(chunk, bytes) else "\n")
        if discarding:
            discarding = not newline
            continue
        try:
            raw = chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
            size = len(raw)
        except UnicodeEncodeError:
            size = MAX_REQUEST_BYTES + 1
        if size > MAX_REQUEST_BYTES:
            discarding = hasattr(reader, "readline") and not newline
            yield None
        else:
            try:
                yield raw.decode("utf-8")
            except UnicodeDecodeError:
                yield None


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None, *, session=None) -> int:
    if session is None:
        from .product_contract import resolve_mcp_profile
        if resolve_mcp_profile() == "execution":
            from .session import GnomonSession
            with GnomonSession.from_config() as owned:
                return serve(stdin, stdout, session=owned)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    print("gnomon mcp server listening on stdio", file=sys.stderr)
    for line in _bounded_lines(stdin):
        if line is None:
            _write(stdout, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Request exceeds the input byte limit or is not valid UTF-8."}})
            continue
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeError, RecursionError):
            _write(stdout, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON request."}})
            continue
        if not isinstance(message, dict):
            _write(stdout, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Request must be an object; batching is unsupported."}})
            continue
        message_id = message.get("id")
        if (type(message_id) not in (str, int, float, type(None))
                or (isinstance(message_id, float) and not math.isfinite(message_id))
                or len(str(message_id).encode("utf-8")) > 256):
            _write(stdout, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid or oversized request ID."}})
            continue
        method = message.get("method")
        if (not isinstance(method, str) or len(method.encode("utf-8")) > 256
                or (message.get("params") is not None and not isinstance(message["params"], dict))):
            _write(stdout, {"jsonrpc": "2.0", "id": message_id, "error": {"code": -32600, "message": "Invalid method or parameters."}})
            continue
        if method.startswith("notifications/"):
            continue
        try:
            result = _handle(message, session=session)
        except Exception:  # a protocol bug must not kill the server or expose secrets
            result = None
            if message_id is not None:
                _write(stdout, {
                    "jsonrpc": "2.0", "id": message_id,
                    "error": {"code": -32603, "message": "Internal protocol error."},
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
    stdout.write(json.dumps(message, allow_nan=False, ensure_ascii=False, separators=(",", ":")) + "\n")
    stdout.flush()
