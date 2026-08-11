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
import sys
from typing import Any, TextIO, get_args

from .contracts import GnomonError, Support, SupportStatus
from .toolspec import READ_ONLY_TOOLS, runner_for, visible_tools

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "gnomon", "version": "0.5.0"}

#: Both support vocabularies, published in the envelope schema so an agent
#: can branch on them without reading source. Derived from the contracts
#: Literals, so the schema cannot drift from the runtime.
SUPPORT_VALUES = list(get_args(Support))
SUPPORT_STATUS_VALUES = list(get_args(SupportStatus))

_SUPPORT_ASSESSMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": SUPPORT_STATUS_VALUES,
            "description": (
                "The harness support vocabulary. 'inconclusive' and "
                "'unsupported' are conclusions to be reported verbatim, "
                "never softened to 'low confidence' or replaced with an "
                "estimate."
            ),
        },
        "reasons": {"type": "array", "items": {"type": "object"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "recovery_actions": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Typed {code, message} next steps when a result is not fully supported.",
        },
        "disclosures": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Caveats that never change status; distinct from reasons.",
        },
    },
}

#: The shape every tool result shares. Tools may publish something tighter
#: via an `outputSchema` key in their spec; this is the floor, and it is
#: what makes `structuredContent` checkable rather than merely present.
ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "status": {
            "type": "string",
            # Not an enum: success values legitimately vary by tool
            # ('complete', 'ok', 'valid', 'supported', ...). Only 'error'
            # is load-bearing at the envelope level.
            "description": (
                "'error' accompanies a structured error; success values "
                "vary by tool. Not an abstention signal — a forecast that "
                "abstains still reports 'complete'; read each "
                "results[].support_assessment.status."
            ),
        },
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
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "support": {
                        "type": "string",
                        "enum": SUPPORT_VALUES,
                        "description": (
                            "Frozen v0.2 support enum, kept for "
                            "compatibility; prefer "
                            "support_assessment.status."
                        ),
                    },
                    "support_assessment": _SUPPORT_ASSESSMENT_SCHEMA,
                },
            },
        },
        "support_assessment": _SUPPORT_ASSESSMENT_SCHEMA,
        "artifact_path": {"type": "string"},
    },
}

#: Served from `initialize` so a bare MCP client gets the safe-use workflow
#: without the Hermes skill file. Condensed from skills/forecasting/SKILL.md.
INSTRUCTIONS = """\
Gnomon answers temporal questions; you formulate the objective and explain \
the evidence. Gnomon owns every number, interval, probability, selection, \
warning, and support status.

The four verbs: What changed? -> gnomon_investigate_change. What happens \
next? -> gnomon_forecast. What should we do? -> gnomon_decide. When should \
we intervene? -> gnomon_monitor.

Workflow: gnomon_capabilities once if unsure what is installed -> \
gnomon_inspect when mappings or data quality are uncertain -> the verb tool \
-> quote numbers verbatim from the response or the returned artifact_path \
(gnomon_get_artifact, gnomon_explain_run). Track consequential runs with a \
project; close the loop with gnomon_submit_actuals and gnomon_resolve_outcome.

Hard rules: never invent, adjust, or extrapolate any number — if it is not \
in a Gnomon artifact, it does not exist. Preserve abstention: 'unsupported' \
and 'inconclusive' are conclusions; report them with Gnomon's reasons, never \
soften them or substitute your own estimate. Surface every warning and the \
support_assessment even when the user only asked for numbers. Do not infer \
business thresholds or costs (gnomon_decide, gnomon_monitor) — ask the user, \
or omit that analysis. Never hand-clean data; use gnomon_inspect and the \
repair parameter so fixes stay disclosed. Errors carry repair_options: \
follow one or ask the user rather than improvising.

Tool calls never read gnomon.yaml: behaviour comes entirely from tool \
parameters and build defaults (discover them with gnomon_capabilities). A \
config file on the server's disk does not affect MCP results."""


def _tool_result(payload: dict[str, Any], is_error: bool) -> dict[str, Any]:
    """A tool result in both shapes the advertised protocol supports.

    The text block stays exactly as it was, so a client that only reads
    `content` is unaffected. `structuredContent` carries the same payload
    as an object, which is what protocol 2025-06-18 exists to allow — an
    agent should not have to `JSON.parse` a string and validate it against
    nothing.
    """
    text = json.dumps(payload, allow_nan=False)
    return {
        "content": [{"type": "text", "text": text}],
        # Round-tripped rather than passed through, so the structured form
        # is byte-for-byte what the text block says — a tuple in the payload
        # would otherwise be an array in one and a tuple in the other.
        "structuredContent": json.loads(text),
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
            "instructions": INSTRUCTIONS,
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
                    # Published so a client can validate `structuredContent`
                    # rather than trusting it.
                    "outputSchema": tool.get("outputSchema", ENVELOPE_SCHEMA),
                    # Writes are append-only (immutable artifacts, vintage
                    # accumulation), so nothing here is destructive — but a
                    # host should still know which tools mutate state at all.
                    "annotations": {
                        "readOnlyHint": tool["name"] in READ_ONLY_TOOLS,
                        "destructiveHint": False,
                    },
                }
                for tool in visible_tools()
            ]
        }
    if method == "tools/call":
        params = message.get("params") or {}
        runner = runner_for(str(params.get("name")))
        if runner is None:
            return _tool_result(
                GnomonError("UNKNOWN_TOOL", f"No such tool: {params.get('name')!r}").to_dict(),
                True,
            )
        arguments = params.get("arguments") or {}
        try:
            return _tool_result(runner(arguments), False)
        except GnomonError as exc:
            return _tool_result(exc.to_dict(), True)
        except KeyError as exc:
            # details.missing_arguments is what the repair option points at;
            # an empty details would make that guidance a dead reference.
            return _tool_result(
                GnomonError(
                    "INVALID_ARGUMENTS",
                    f"Missing required argument: {exc.args[0]}",
                    {"missing_arguments": [str(exc.args[0])]},
                ).to_dict(),
                True,
            )
        except (ValueError, FileNotFoundError) as exc:
            return _tool_result(
                GnomonError("TRACKING_ERROR", str(exc)).to_dict(), True,
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


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    print("gnomon mcp server listening on stdio", file=sys.stderr)
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
