"""``gnomon-mcp``: the treatment — the real MCP surface, the model driving.

The identical :class:`~benchmarks.decisionbench.arms.ToolLoopArm` loop as
the sandbox arm, with the interpreter swapped for every tool ``gnomon
mcp serve`` publishes, verbatim (the CiK arm's session and jail
screening, the TemporalBench arm's result bounding — reused as
libraries, not modified). The decision itself always belongs to the
model: Gnomon computes forecasts, exceedance risk, and support states;
``submit_decision`` is the same harness tool with the same schema as the
other arms, so the treatment differs only in what the agent can consult,
never in what it must produce.

The transcript records every tool call, so a run can be read afterwards
for *how* the decision was reached — including whether the engine's
abstention on an unsupportable series was followed by an escalation, a
hedge, or a confident number written past the refusal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.mcp_agent import (  # noqa: E402 — library reuse
    StdioMcpSession,
    jail_violations,
)
from benchmarks.temporalbench.mcp_agent import (  # noqa: E402 — library reuse
    bounded_tool_text,
)
from benchmarks.decisionbench.arms import (  # noqa: E402
    MAX_ROUNDS,
    MAX_TOOL_CALLS,
    MAX_TOOL_RESULT_CHARS,
    SUBMIT_TOOL,
    ToolLoopArm,
)
from benchmarks.decisionbench.tasks import DecisionTask  # noqa: E402

SYSTEM = """\
You are making one operational capacity decision. You have tools from
"gnomon", a deterministic temporal engine — backtested forecasting with
calibrated intervals (gnomon_forecast), threshold/exceedance analysis
(gnomon_monitor, gnomon_decide), change and anomaly inspection, and
disclosed repair of messy files. Use them or ignore them; the decision
and its responsibility are yours either way, and you finish by calling
submit_decision.

The engine abstains when a history cannot support its evaluation
protocol. Its abstention is not your abstention: you may still decide
from your own reading of the data, or escalate — escalation is priced
in the task, not free.

This run may only read and write inside {jail}. The task data file is
{csv_path}; pass output_dir under {jail} (for example {out_dir}).
Tool errors return typed codes and repair options; you may fix
arguments and retry within the caps ({max_rounds} rounds, {max_calls}
tool calls).
"""


class McpArm(ToolLoopArm):
    def __init__(self, task: DecisionTask, client: Any,
                 work_dir: str | None = None, session_factory: Any = None):
        super().__init__(task, client, work_dir=work_dir)
        self.session = (session_factory or StdioMcpSession)(self.jail)
        self.session.initialize()

    def system_prompt(self) -> str:
        return SYSTEM.format(
            jail=self.jail, csv_path=self.csv_path,
            out_dir=self.jail / "gnomon-output",
            max_rounds=MAX_ROUNDS, max_calls=MAX_TOOL_CALLS,
        )

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            }}
            for tool in self.session.list_tools()
        ] + [SUBMIT_TOOL]

    def screen_tool(self, name: str,
                    arguments: dict[str, Any]) -> str | None:
        violations = jail_violations(arguments, self.jail)
        if not violations:
            return None
        self.trace[-1]["jail_violations"] = violations
        return json.dumps({
            "code": "PATH_JAIL",
            "message": f"This run may only read and write inside "
                       f"{self.jail}. The data file is {self.csv_path}; "
                       f"write outputs under "
                       f"{self.jail / 'gnomon-output'}.",
            "violations": violations,
            "authored_by": "harness",
        })

    def dispatch_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            result = self.session.call_tool(name, arguments)
        except Exception as error:
            self.trace[-1]["transport_error"] = str(error)
            return json.dumps({
                "code": "MCP_TRANSPORT_FAILED",
                "message": f"the engine transport failed: {error}; decide "
                           f"from the data you have, or escalate",
                "authored_by": "harness",
            })
        self.trace[-1]["is_error"] = bool(result.get("isError"))
        structured = result.get("structuredContent") or {}
        if isinstance(structured, dict):
            code = ((structured.get("error") or {}).get("code")
                    or structured.get("code"))
            if code:
                self.trace[-1]["code"] = code
        content = result.get("content") or []
        text = (content[0].get("text", "") if content
                else json.dumps(structured))
        text, truncated = bounded_tool_text(text, MAX_TOOL_RESULT_CHARS)
        if truncated:
            self.trace[-1]["truncated"] = True
        return text

    def close(self) -> None:
        self.session.close()
