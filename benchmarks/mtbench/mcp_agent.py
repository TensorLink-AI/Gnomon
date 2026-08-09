"""MTBench's ``mcp`` mode: the real MCP surface, the model driving.

The ``tools`` mode (``tool_agent.py``) exposes Gnomon through four
hand-curated wrappers with the series pre-bound — no file paths, no
argument schemas, no way to fumble the interface. That measures routing
and selection, not whether a model can operate Gnomon as agents
actually meet it. This mode is the raw counterpart: the model holds
every tool ``gnomon mcp serve`` publishes, verbatim, plus one harness
tool (``submit_forecast``), and drives the engine itself. Running both
modes on the same samples isolates what the real surface's friction
costs — a product-relevant number in its own right.

The session, verbatim tool-spec conversion, and path jail are imported
from the CiK MCP arm (``benchmarks/cik/mcp_agent.py``, spec:
``docs/design/cik-mcp-tool-arm.md``) — used as a library, not modified.

The honesty contract is the same as the ``tools`` mode's:
``submit_forecast`` has exactly three exits — an ``artifact_path`` from
a ``gnomon_forecast`` call in THIS run, used byte-for-byte; the model's
own ``values``, one number per horizon step, labeled by the route
taxonomy (``informed-direct`` / ``direct``); or ``abstain``. An
artifact whose run abstained (``support: "unsupported"``) is rejected
at submission with the honest options restated, including retrying the
tool with ``best_effort: true`` — the model itself decides whether to
take the engine's labeled fallback, and the label travels into the
outcome. ``engine_abstentions`` counts the unsupported artifacts the
model's forecast calls produced, so a refusal answered past is
disclosed, never laundered. A breached cap abstains the sample with the
cap named — never a silent fallback.

Adapter decisions, disclosed: the bar history is written to the same
synthetic regular daily axis as every other MTBench condition (bar *k*
at epoch + *k* days; the metric compares values only); the article
reaches the model in the user message exactly as in ``tools`` mode.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.mcp_agent import (  # noqa: E402 — library reuse
    StdioMcpSession,
    _tool_calls_as_dicts,
    jail_violations,
)
from benchmarks.mtbench.gnomon_forecaster import EPOCH, write_bar_csv  # noqa: E402
from benchmarks.mtbench.tool_agent import MAX_TEXT_CHARS, USER  # noqa: E402

MAX_ROUNDS = 10
MAX_MCP_CALLS = 24
MAX_RUN_TOKENS = 250_000

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_forecast",
        "description": (
            "End the run with your answer. Exactly one of three exits: "
            "artifact_path (from a gnomon_forecast call in THIS run; its "
            "trajectory is used verbatim — you cannot edit it), values "
            "(your own point forecast, exactly one number per horizon "
            "step), or abstain=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact_path": {
                    "type": "string",
                    "description": "artifact_path from a gnomon_forecast result in this run.",
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Your own forecast: exactly one number per horizon step.",
                },
                "abstain": {"type": "boolean",
                            "description": "true to abstain honestly."},
                "reasoning": {"type": "string"},
            },
        },
    },
}

#: Symmetric by construction, like the CiK arm's prompt.
SYSTEM = """\
You are producing a {horizon}-step point forecast of a stock's price
from its recent history and the news article published over that
window. You have tools from "gnomon", a deterministic forecasting
engine, and you may use them or ignore them — your only obligation is
to end by calling `submit_forecast` with the best answer you can
produce.

What the engine offers: backtested model selection computed from the
numeric history, plus season and anomaly detection, and deterministic
gating of typed context events you extract from the article (never
numbers you invent). It abstains when the history cannot carry its
evaluation protocol; an abstention is not your abstention — you may
still reason from the history and the article and submit your own
values, ask the engine for its disclosed best-effort fallback
(`best_effort: true`; its rows are labeled, not supported forecasts),
or abstain.

The price history is at {csv_path} (columns: timestamp, value) on a
synthetic regular daily axis: bar k sits at {epoch} + k days (the
trading-bar convention; the metric compares values only, so the axis
never enters the score). Forecast horizon: {horizon} steps.

Three ways to finish:
- Submit `artifact_path` from a `gnomon_forecast` result: that
  artifact's trajectory becomes your answer verbatim.
- Submit your own `values` (exactly {horizon} numbers, one per step):
  your numbers, your responsibility.
- Submit `abstain: true` if you judge no defensible forecast exists.

Tool errors return typed codes and repair options; you may fix
arguments and retry within the caps ({max_rounds} rounds, {max_calls}
tool calls).
"""


def openai_tool_specs(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MCP tool entries as chat-completions specs, verbatim, plus the
    harness submit tool — the same no-pruning rule as the CiK arm
    (``outputSchema`` is dropped; the chat format has no slot for it)."""
    return [
        {"type": "function", "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["inputSchema"],
        }}
        for tool in mcp_tools
    ] + [SUBMIT_TOOL]


class _Run:
    """One sample's conversation: jail, server, loop, caps, submission."""

    def __init__(self, sample: dict[str, Any], client: Any,
                 session_factory: Any = None, work_dir: str | None = None):
        self.client = client
        self.values = [float(v) for v in sample["input_window"]]
        self.horizon = len(sample["output_window"])
        self.text = (sample.get("text") or "")[:MAX_TEXT_CHARS]
        self.jail = Path(tempfile.mkdtemp(prefix="mtbench-mcp-",
                                          dir=work_dir)).resolve()
        self.csv_path = self.jail / "history.csv"
        write_bar_csv(self.values, self.csv_path)
        self.session = (session_factory or StdioMcpSession)(self.jail)
        self.trace: list[dict[str, Any]] = []
        self.mcp_calls = 0
        #: artifact_path -> {"support", "selected_model", "rows"}; read
        #: once when the forecast call returns, reused at submission.
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.engine_abstentions = 0
        self.submission: dict[str, Any] | None = None
        self.tokens_at_start = (getattr(client, "total_prompt_tokens", 0)
                                + getattr(client, "total_completion_tokens", 0))

    # -- caps --------------------------------------------------------------
    def _run_tokens(self) -> int:
        return (getattr(self.client, "total_prompt_tokens", 0)
                + getattr(self.client, "total_completion_tokens", 0)
                - self.tokens_at_start)

    def _cap_breach(self) -> str | None:
        if self._run_tokens() > MAX_RUN_TOKENS:
            return f"cap:tokens exceeded {MAX_RUN_TOKENS}"
        return None

    def _common(self) -> dict[str, Any]:
        return {
            "forecasts_computed": len(self.artifacts),
            "engine_abstentions": self.engine_abstentions,
            "tool_calls": self.mcp_calls,
            "trace": [
                {key: value for key, value in entry.items()
                 if key in ("tool", "is_error", "code", "jail_violations",
                            "abstained")}
                for entry in self.trace
            ],
        }

    def _abstain_outcome(self, reason: str,
                         route: str | None = None) -> dict[str, Any]:
        self.trace.append({"abstained": reason})
        outcome = {"abstained": True, "reasons": [reason], **self._common()}
        if route:
            outcome["route"] = route
        return outcome

    # -- the loop ----------------------------------------------------------
    def drive(self) -> dict[str, Any]:
        self.session.initialize()
        tools = openai_tool_specs(self.session.list_tools())
        system = SYSTEM.format(
            csv_path=str(self.csv_path), epoch=EPOCH.date().isoformat(),
            horizon=self.horizon, max_rounds=MAX_ROUNDS,
            max_calls=MAX_MCP_CALLS,
        )
        history = ", ".join(f"{v:.4f}" for v in self.values)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": USER.format(
                n_obs=len(self.values), history=history,
                horizon=self.horizon, text=self.text or "(no article)",
            )},
        ]

        nudged = False
        for _round in range(MAX_ROUNDS):
            breach = self._cap_breach()
            if breach:
                return self._abstain_outcome(breach)
            response = self.client.chat(messages, n=1, tools=tools,
                                        tool_choice="auto")
            message = response.choices[0].message
            tool_calls = _tool_calls_as_dicts(message)
            if not tool_calls:
                messages.append({"role": "assistant",
                                 "content": message.content or ""})
                if self.submission:
                    break
                if nudged:
                    return self._abstain_outcome(
                        "no submission: prose answers twice after nudge")
                nudged = True
                messages.append({
                    "role": "user",
                    "content": "Finish by calling submit_forecast — an "
                               "artifact_path from a gnomon_forecast "
                               "result, your own values (one number per "
                               "step), or abstain=true.",
                })
                continue
            messages.append({"role": "assistant",
                             "content": message.content or None,
                             "tool_calls": tool_calls})
            for call in tool_calls:
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = None
                result = self._dispatch(call["function"]["name"], arguments)
                if isinstance(result, dict):  # a cap abstained the sample
                    return result
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": result})
            if self.submission:
                break
        if not self.submission:
            return self._abstain_outcome(
                f"cap:rounds {MAX_ROUNDS} rounds without a submission")
        return self._resolve_submission()

    # -- dispatch ----------------------------------------------------------
    def _dispatch(self, name: str,
                  arguments: dict[str, Any] | None) -> str | dict[str, Any]:
        entry: dict[str, Any] = {"tool": name}
        self.trace.append(entry)
        if arguments is None:
            entry["error"] = "unparseable arguments"
            return json.dumps({"code": "INVALID_ARGUMENTS",
                               "message": "Tool arguments were not valid JSON.",
                               "authored_by": "harness"})
        if name == "submit_forecast":
            payload = self._handle_submit(arguments)
            entry["result"] = {"accepted": payload.get("accepted"),
                               "message": payload.get("message")}
            return json.dumps(payload)

        violations = jail_violations(arguments, self.jail)
        if violations:
            entry["jail_violations"] = violations
            return json.dumps({
                "code": "PATH_JAIL",
                "message": "This run may only read and write inside its own "
                           "run directory. The history file is at "
                           f"{self.csv_path}.",
                "violations": violations,
                "authored_by": "harness",
            })
        if self.mcp_calls >= MAX_MCP_CALLS:
            return self._abstain_outcome(
                f"cap:tool_calls exceeded {MAX_MCP_CALLS}")
        self.mcp_calls += 1
        breach = self._cap_breach()
        if breach:
            return self._abstain_outcome(breach)
        try:
            result = self.session.call_tool(name, arguments)
        except Exception as error:
            # Transport death is a harness failure, disclosed as such.
            return self._abstain_outcome(f"mcp transport failed: {error}")
        entry["is_error"] = bool(result.get("isError"))
        structured = result.get("structuredContent") or {}
        if isinstance(structured, dict):
            code = ((structured.get("error") or {}).get("code")
                    or structured.get("code"))
            if code:
                entry["code"] = code
            if not result.get("isError") and structured.get("artifact_path"):
                self._register_artifact(str(structured["artifact_path"]))
        content = result.get("content") or []
        return content[0].get("text", "") if content else json.dumps(structured)

    def _register_artifact(self, artifact_path: str) -> None:
        """Read the artifact once: its support label feeds the
        engine-abstention count and the submission handler."""
        from gnomon.artifacts import read_artifact

        try:
            data = read_artifact(artifact_path)
        except Exception:
            return  # a submission naming it will report the read failure
        results = data.get("results") or []
        result = results[0] if results else {}
        record = {
            "support": result.get("support"),
            "selected_model": result.get("selected_model"),
            "rows": result.get("forecast") or [],
        }
        self.artifacts[artifact_path] = record
        if record["support"] == "unsupported" or not record["rows"]:
            self.engine_abstentions += 1

    # -- submission --------------------------------------------------------
    def _values_problems(self, values: Any) -> str | None:
        import math

        if not isinstance(values, list):
            return (f"values must be a list of exactly {self.horizon} "
                    f"numbers (one per horizon step); got "
                    f"{type(values).__name__}")
        if len(values) != self.horizon:
            return (f"values must be a list of exactly {self.horizon} "
                    f"numbers (one per horizon step); got {len(values)}")
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                return (f"values[{index}] must be a finite number; "
                        f"got {value!r:.60}")
        return None

    def _handle_submit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.submission is not None:
            return {"accepted": False, "authored_by": "harness",
                    "message": "A forecast was already submitted; the first "
                               "accepted submission stands."}
        artifact_path = arguments.get("artifact_path")
        values = arguments.get("values")
        abstain = arguments.get("abstain") is True
        if sum((artifact_path is not None, values is not None, abstain)) != 1:
            return {"accepted": False, "authored_by": "harness",
                    "message": "Provide exactly one of artifact_path, "
                               "values, or abstain=true."}
        if abstain:
            self.submission = {"route": "abstain",
                               "reasoning": arguments.get("reasoning")}
            return {"accepted": True, "abstained": True}
        if artifact_path is not None:
            record = self.artifacts.get(str(artifact_path))
            if record is None:
                return {"accepted": False, "authored_by": "harness",
                        "message": "artifact_path was not produced by a "
                                   "gnomon_forecast call in this run.",
                        "known_artifact_paths": sorted(self.artifacts)}
            if record["support"] == "unsupported" or not record["rows"]:
                return {"accepted": False, "authored_by": "harness",
                        "message": "That run abstained (support "
                                   "'unsupported') and published no "
                                   "trajectory. Submit your own values, "
                                   "abstain=true, or retry gnomon_forecast "
                                   "with best_effort=true for the engine's "
                                   "labeled fallback."}
            if len(record["rows"]) != self.horizon:
                return {"accepted": False, "authored_by": "harness",
                        "message": f"artifact horizon is "
                                   f"{len(record['rows'])}, task horizon is "
                                   f"{self.horizon}; re-run gnomon_forecast "
                                   f"with horizon={self.horizon}"}
            self.submission = {
                "route": "gnomon", "artifact_path": str(artifact_path),
                "prediction": [float(row.get("q50", row["point"]))
                               for row in record["rows"]],
                "support": record["support"],
                "selected_model": record["selected_model"],
                "reasoning": arguments.get("reasoning"),
            }
            return {"accepted": True, "route": "gnomon"}
        problems = self._values_problems(values)
        if problems:
            return {"accepted": False, "authored_by": "harness",
                    "message": problems}
        route = "direct" if self.mcp_calls == 0 else "informed-direct"
        self.submission = {
            "route": route,
            "prediction": [float(v) for v in values],
            "support": None, "selected_model": None,
            "reasoning": arguments.get("reasoning"),
        }
        return {"accepted": True, "route": route}

    # -- result ------------------------------------------------------------
    def _resolve_submission(self) -> dict[str, Any]:
        assert self.submission is not None
        if self.submission["route"] == "abstain":
            outcome = self._abstain_outcome("model abstained after tool use",
                                            route="abstain")
            outcome["submit_reasoning"] = self.submission["reasoning"]
            return outcome
        outcome = {
            "abstained": False,
            "prediction": self.submission["prediction"],
            "route": self.submission["route"],
            "support": self.submission["support"],
            "selected_model": self.submission["selected_model"],
            "context": None,
            "events": [],
            "submit_reasoning": self.submission["reasoning"],
            **self._common(),
        }
        if self.submission["route"] == "gnomon":
            outcome["artifact_path"] = self.submission["artifact_path"]
        return outcome

    def finish(self) -> None:
        self.session.close()


def run_sample_mcp(sample: dict[str, Any], client: Any, *,
                   session_factory: Any = None,
                   work_dir: str | None = None) -> dict[str, Any]:
    """Drive one sample through the real MCP surface; return the same
    outcome shape ``tool_agent.run_sample`` produces."""
    run = _Run(sample, client, session_factory=session_factory,
               work_dir=work_dir)
    try:
        return run.drive()
    finally:
        run.finish()
