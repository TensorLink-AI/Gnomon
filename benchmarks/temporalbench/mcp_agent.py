"""TemporalBench's ``gnomon-mcp`` condition: the real MCP surface, the
model driving.

The ``gnomon-agent`` condition computes Gnomon evidence in the harness
and pastes it into the prompt — the model never chooses to call
anything, so it measures evidence-grounded answering, not tool use.
This is the tool-use arm: the model holds every tool ``gnomon mcp
serve`` publishes, verbatim, plus one harness tool (``submit_answer``),
and drives the engine itself over the row's channels. The session,
verbatim tool-spec conversion, and path jail are imported from the CiK
MCP arm (``benchmarks/cik/mcp_agent.py``, spec:
``docs/design/cik-mcp-tool-arm.md``) — used as a library, not modified.

The honesty contract, per channel:

- ``artifact_path`` — a path returned by a ``gnomon_forecast`` call in
  THIS run, used byte-for-byte; the artifact's own ``target_column``
  must be the channel it is submitted for (a mismatch is a repairable
  rejection, not a silent mislabel), and an artifact whose run abstained
  (``support: "unsupported"``) is rejected with the honest options
  restated: submit your own values, abstain, or retry the tool with
  ``best_effort: true`` — whose rows come back labeled ``best_effort``
  and are disclosed like every other best-effort row.
- ``values`` — the model's own forecast, labeled ``model`` in the
  channel support map and routed ``informed-direct``/``direct``.
- an omitted channel, or ``{"abstain": true}``, is an abstention.

Per-channel routes and support labels travel into the details records,
the GnomonBench records, the summary, and ``score_per_channel``'s
support mix, so a model answer written past an engine refusal is always
labeled, never laundered.

Adapter decisions, disclosed: the row's channels are written to one
wide CSV on the same synthetic hourly axis as every other condition
(observation *k* at epoch + *k* hours, nulls omitted, channels laid
consecutively — exactly ``gnomon_runner``'s convention, so the values
the engine sees match the other Gnomon conditions); the official prompt
is the user message verbatim; a breached cap abstains the whole row
with the cap named — never a silent fallback.
"""

from __future__ import annotations

import csv
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
from benchmarks.temporalbench.gnomon_runner import EPOCH, STEP, _observed  # noqa: E402
from benchmarks.temporalbench.tasks import prompt_input_arrays  # noqa: E402

MAX_ROUNDS = 10
MAX_MCP_CALLS = 24
MAX_RUN_TOKENS = 250_000

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": (
            "End the run with your answer. For EACH forecast channel give "
            "exactly one of: artifact_path (from a gnomon_forecast call in "
            "THIS run whose target_column is that channel; its trajectory "
            "is used verbatim — you cannot edit it), values (your own "
            "forecast, exactly one number per horizon step), or "
            "{\"abstain\": true}. A channel you omit is recorded as an "
            "abstention. Include your answers to the task's multiple-"
            "choice questions in mcq."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "forecast": {
                    "type": "object",
                    "description": "channel name -> {artifact_path} or {values} or {abstain: true}",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "artifact_path": {"type": "string"},
                            "values": {"type": "array",
                                       "items": {"type": "number"}},
                            "abstain": {"type": "boolean"},
                        },
                    },
                },
                "mcq": {
                    "type": "object",
                    "description": "question key -> chosen option string, keyed as the task's output format specifies",
                    "additionalProperties": {"type": "string"},
                },
                "reasoning": {"type": "string"},
            },
        },
    },
}

#: Symmetric by construction, like the CiK arm's prompt: it names what
#: each path is good at and lets the transcript record the choice.
SYSTEM = """\
You are answering one time series task. You have tools from "gnomon", a
deterministic forecasting engine, and you may use them or ignore them —
your only obligation is to end by calling `submit_answer` with the best
answer you can produce.

What the engine offers: backtested model selection and calibrated
uncertainty computed from the numeric history. It abstains when a
channel's history cannot carry its evaluation protocol; an abstention
is not your abstention — you may still reason from the data and submit
your own values for that channel, ask the engine for its disclosed
best-effort fallback (`best_effort: true`; its rows are labeled, not
supported forecasts), or abstain.

The task's channel histories are in {csv_path} (columns: timestamp,
{channels}). The timestamps are a synthetic regular hourly axis —
observation k of a channel sits at {epoch} + k hours, recorded readings
laid consecutively; a shorter channel's trailing cells are blank. The
metrics are index-based, so the axis never enters the score. Forecast
horizon: {horizon} steps per channel.

Ways to finish each channel of `submit_answer.forecast`:
- artifact_path from a `gnomon_forecast` run on that channel: its
  trajectory becomes that channel's answer verbatim.
- your own values (exactly {horizon} numbers): your numbers, your
  responsibility.
- abstain (or omit the channel): an honest abstention.

Also answer every multiple-choice question of the task in
`submit_answer.mcq`. Tool errors return typed codes and repair options;
you may fix arguments and retry within the caps ({max_rounds} rounds,
{max_calls} tool calls).
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


def _write_wide_csv(channels: dict[str, list[float]], csv_path: Path) -> None:
    keys = list(channels)
    length = max((len(values) for values in channels.values()), default=0)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp"] + keys)
        for position in range(length):
            writer.writerow(
                [(EPOCH + position * STEP).isoformat()]
                + [repr(channels[key][position])
                   if position < len(channels[key]) else "" for key in keys]
            )


class _Run:
    """One row's conversation: jail, server, loop, caps, submission."""

    def __init__(self, row: dict[str, Any], client: Any,
                 session_factory: Any = None, work_dir: str | None = None):
        self.row = row
        self.client = client
        meta = row.get("meta") or {}
        self.horizon = int(meta.get("n_horizon") or 0)
        if self.horizon < 1:
            raise ValueError("row has no forecast horizon")
        ground_truth = row.get("ground_truth")
        self.target_keys = (list(ground_truth) if isinstance(ground_truth, dict)
                            else [meta.get("main_key")])
        arrays = prompt_input_arrays(row)
        observed = {key: _observed(arrays.get(key, []))
                    for key in self.target_keys}
        if "timestamp" in observed:
            raise ValueError("a channel named 'timestamp' collides with the "
                             "time column of the run CSV")
        self.jail = Path(tempfile.mkdtemp(prefix="tb-mcp-",
                                          dir=work_dir)).resolve()
        self.csv_path = self.jail / "history.csv"
        _write_wide_csv(observed, self.csv_path)
        self.session = (session_factory or StdioMcpSession)(self.jail)
        self.trace: list[dict[str, Any]] = []
        self.mcp_calls = 0
        self.artifact_paths: set[str] = set()
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

    def _abstain_outcome(self, reason: str) -> dict[str, Any]:
        """The whole row abstains with the cap named — never a fallback."""
        self.trace.append({"abstained": reason})
        return {
            "answer": {"forecast": {}, "mcq": {}},
            "abstained": [f"{key}: {reason}" for key in self.target_keys],
            "channel_support": {},
            "channel_route": {key: "abstain" for key in self.target_keys},
            "mcp": self._mcp_info(),
        }

    def _mcp_info(self) -> dict[str, Any]:
        return {
            "calls": self.mcp_calls,
            "run_tokens": self._run_tokens(),
            "tool_sequence": [
                {key: value for key, value in entry.items()
                 if key in ("tool", "is_error", "code", "jail_violations",
                            "abstained")}
                for entry in self.trace
            ],
        }

    # -- the loop ----------------------------------------------------------
    def drive(self) -> dict[str, Any]:
        self.session.initialize()
        tools = openai_tool_specs(self.session.list_tools())
        system = SYSTEM.format(
            csv_path=str(self.csv_path),
            channels=", ".join(self.target_keys),
            epoch=EPOCH.isoformat(), horizon=self.horizon,
            max_rounds=MAX_ROUNDS, max_calls=MAX_MCP_CALLS,
        )
        # The official prompt is the user message, verbatim: the
        # benchmark stays authoritative about the task and its output
        # format; the system message adds only the harness contract.
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": self.row["prompt"]},
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
                    "content": "Finish by calling submit_answer — per "
                               "channel an artifact_path from a "
                               "gnomon_forecast run, your own values, or "
                               "abstain; plus your mcq answers.",
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
                if isinstance(result, dict):  # a cap abstained the row
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
        """Route one tool call; return the tool-message text, or the
        row's abstention outcome when a cap ends the run."""
        entry: dict[str, Any] = {"tool": name}
        self.trace.append(entry)
        if arguments is None:
            entry["error"] = "unparseable arguments"
            return json.dumps({"code": "INVALID_ARGUMENTS",
                               "message": "Tool arguments were not valid JSON.",
                               "authored_by": "harness"})
        if name == "submit_answer":
            payload = self._handle_submit(arguments)
            entry["result"] = {"accepted": payload.get("accepted"),
                               "problems": payload.get("problems")}
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
                self.artifact_paths.add(str(structured["artifact_path"]))
        # Verbatim: the server's own text block, unedited.
        content = result.get("content") or []
        return content[0].get("text", "") if content else json.dumps(structured)

    # -- submission --------------------------------------------------------
    def _artifact_channel_rows(self, artifact_path: str,
                               channel: str) -> list[float] | tuple[str, str]:
        """The artifact's q50 trajectory for `channel`, or a
        ``(problem, support)`` pair explaining the rejection."""
        from gnomon.artifacts import read_artifact

        if artifact_path not in self.artifact_paths:
            return (f"{channel}: artifact_path was not produced by a "
                    f"gnomon_forecast call in this run; known: "
                    f"{sorted(self.artifact_paths)}", "")
        try:
            data = read_artifact(artifact_path)
        except Exception as error:
            return f"{channel}: artifact could not be read: {error}", ""
        target = ((data.get("task") or {}).get("schema") or {}).get(
            "target_column")
        if target != channel:
            return (f"{channel}: that artifact forecasts "
                    f"{target!r}, not {channel!r}; submit it for its own "
                    f"channel or run gnomon_forecast with "
                    f"target_column={channel!r}", "")
        results = data.get("results") or []
        result = results[0] if results else {}
        rows = result.get("forecast") or []
        support = result.get("support")
        if support == "unsupported" or not rows:
            return (f"{channel}: that run abstained (support "
                    f"'unsupported') and published no trajectory. Submit "
                    f"your own values, abstain on the channel, or retry "
                    f"gnomon_forecast with best_effort=true for the "
                    f"engine's labeled fallback.", "")
        if len(rows) != self.horizon:
            return (f"{channel}: artifact horizon is {len(rows)}, task "
                    f"horizon is {self.horizon}; re-run gnomon_forecast "
                    f"with horizon={self.horizon}", "")
        self._pending_support[channel] = str(support)
        return [float(row.get("q50", row["point"])) for row in rows]

    def _values_problems(self, channel: str, values: Any) -> str | None:
        import math

        if not isinstance(values, list):
            return (f"{channel}: values must be a list of exactly "
                    f"{self.horizon} numbers; got {type(values).__name__}")
        if len(values) != self.horizon:
            return (f"{channel}: values must be a list of exactly "
                    f"{self.horizon} numbers (one per horizon step); got "
                    f"{len(values)}")
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                return (f"{channel}: values[{index}] must be a finite "
                        f"number; got {value!r:.60}")
        return None

    def _handle_submit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.submission is not None:
            return {"accepted": False, "authored_by": "harness",
                    "message": "An answer was already submitted; the first "
                               "accepted submission stands."}
        forecast_spec = arguments.get("forecast") or {}
        if not isinstance(forecast_spec, dict):
            return {"accepted": False, "authored_by": "harness",
                    "problems": ["forecast must be an object mapping "
                                 "channel name to one exit"]}
        problems: list[str] = []
        resolved: dict[str, list[float]] = {}
        support: dict[str, str] = {}
        routes: dict[str, str] = {}
        self._pending_support: dict[str, str] = {}
        own_route = "direct" if self.mcp_calls == 0 else "informed-direct"
        for channel, entry in forecast_spec.items():
            if channel not in self.target_keys:
                problems.append(f"unknown channel {channel!r}; task "
                                f"channels: {sorted(self.target_keys)}")
                continue
            if not isinstance(entry, dict):
                problems.append(f"{channel}: entry must be an object with "
                                f"artifact_path, values, or abstain")
                continue
            exits = [key for key in ("artifact_path", "values", "abstain")
                     if entry.get(key) is not None]
            if entry.get("abstain") is True:
                if len(exits) > 1:
                    problems.append(f"{channel}: abstain cannot be combined "
                                    f"with artifact_path or values")
                    continue
                routes[channel] = "abstain"
                continue
            if len(exits) != 1:
                problems.append(f"{channel}: provide exactly one of "
                                f"artifact_path or values (or abstain)")
                continue
            if entry.get("artifact_path") is not None:
                rows = self._artifact_channel_rows(
                    str(entry["artifact_path"]), channel)
                if isinstance(rows, tuple):
                    problems.append(rows[0])
                    continue
                resolved[channel] = rows
                support[channel] = self._pending_support[channel]
                routes[channel] = "gnomon"
            else:
                problem = self._values_problems(channel, entry.get("values"))
                if problem:
                    problems.append(problem)
                    continue
                resolved[channel] = [float(v) for v in entry["values"]]
                support[channel] = "model"
                routes[channel] = own_route
        if problems:
            return {"accepted": False, "authored_by": "harness",
                    "problems": problems}
        for channel in self.target_keys:
            routes.setdefault(channel, "abstain")
        mcq = {str(key): str(value)
               for key, value in (arguments.get("mcq") or {}).items()}
        self.submission = {
            "forecast": resolved, "mcq": mcq, "support": support,
            "routes": routes, "reasoning": arguments.get("reasoning"),
        }
        return {"accepted": True, "routes": routes}

    # -- result ------------------------------------------------------------
    def _resolve_submission(self) -> dict[str, Any]:
        assert self.submission is not None
        abstained = [f"{channel}: abstained in submission"
                     for channel, route in
                     sorted(self.submission["routes"].items())
                     if route == "abstain"]
        return {
            "answer": {"forecast": self.submission["forecast"],
                       "mcq": self.submission["mcq"]},
            "abstained": abstained,
            "channel_support": self.submission["support"],
            "channel_route": self.submission["routes"],
            "submit_reasoning": self.submission["reasoning"],
            "mcp": self._mcp_info(),
        }

    def finish(self) -> None:
        self.session.close()


def run_row(row: dict[str, Any], client: Any, *,
            session_factory: Any = None,
            work_dir: str | None = None) -> dict[str, Any]:
    """Drive one T2/T4 row through the real MCP surface; return the same
    outcome shape ``answer_row`` produces for the other conditions."""
    run = _Run(row, client, session_factory=session_factory,
               work_dir=work_dir)
    try:
        return run.drive()
    finally:
        run.finish()
