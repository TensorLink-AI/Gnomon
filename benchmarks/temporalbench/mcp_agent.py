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

T1 and T3 carry no forecast channels, so they run the same session with
an MCQ-shaped submission (:func:`mcq_row`): the identical tool surface —
the arm's question is whether an agent that *can* interrogate the series
answers better, and pruning the tools to match the tier would answer it
by construction — but a system prompt that never mentions horizons or
channels, and a ``submit_answer`` whose schema is the tier's own answer
shape (T1: the row's label fields; T3: one choice per packed question),
so a right answer cannot be lost to JSON formatting.

Adapter decisions, disclosed: the row's channels are written to one
wide CSV on the same synthetic hourly axis as every other condition
(observation *k* at epoch + *k* hours, nulls omitted, channels laid
consecutively — exactly ``gnomon_runner``'s convention, so the values
the engine sees match the other Gnomon conditions); the official prompt
is the user message verbatim; tool results pass through verbatim until
they exceed :data:`MAX_TOOL_RESULT_CHARS`, past which the bulk arrays
are shrunk to head/tail plus counts under a harness-authored
``truncated`` marker naming the artifact that holds the full numbers —
support states, warnings, abstention reasons, recovery actions and
error codes are never what gets dropped. A breached cap does not void
work already done: a run that has submitted keeps its submission, and a
run that has not gets one final round to submit what it has, with the
cap named. Only after that is the row an abstention — never a silent
fallback, and never a zero scored as if the model had answered.
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
#: A multi-channel row is a long conversation: the official prompt is
#: large, every tool result is re-sent with each round, and a six-vital
#: MIMIC row legitimately spends several calls. At 250k this cap was
#: not bounding runaway agents, it was ending honest ones mid-answer —
#: every T2/T4 row of the first measured MCP sweep hit it after five to
#: eight tool calls. The bulk that made those runs expensive is now
#: bounded per tool result (:data:`MAX_TOOL_RESULT_CHARS`) and the
#: batched call path makes one run out of six, so this ceiling is a
#: guard again rather than the thing being measured.
MAX_RUN_TOKENS = 500_000
#: The one cap this arm lacked. Rounds, calls, and tokens all bound model
#: behaviour; none of them bounds a slow provider, so a run could sit for
#: hours inside retries without breaching anything. Mirrors the CiK arm's
#: reasoning (see benchmarks/cik/mcp_agent.py): far above what an honest
#: run needs, so it guards the harness rather than handicapping the arm.
MAX_WALL_SECONDS = 7200.0
#: Per tool message, the point past which a result is shrunk rather than
#: passed through. Measured against the case that matters: a six-channel
#: 29-step MIMIC row returns ~33k characters in either format, which
#: array-shrinking alone brings to ~18k (brief) or ~26k (full). At 16k
#: a brief batched result keeps its per-channel disclosures and most of
#: its numbers, a full one is pruned harder, and neither is paid for ten
#: times over as the conversation re-sends it each round.
MAX_TOOL_RESULT_CHARS = 16_000
#: Leading and trailing rows kept verbatim from a shrunk numeric array.
TRUNCATED_ARRAY_EDGE = 4


SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": (
            "End the run with your answer. For EACH forecast channel give "
            "exactly one of: artifact_path (from a gnomon_forecast call in "
            "THIS run that covers that channel — a multi-target run's "
            "artifact is submittable for every channel it forecast; its "
            "trajectory is used verbatim, you cannot edit it), values "
            "(your own forecast, exactly one number per horizon step), or "
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
#: What it adds beyond that is mechanical, not persuasive — where the
#: run may write, and that one call covers every channel. Both were
#: measured as pure waste: the first sweep spent a round per row on a
#: rejected `output_dir=/tmp/...` and six calls per row on what
#: `target_column: "a,b,c"` does in one. Neither sentence argues for
#: using the engine; they remove failure modes that were being scored
#: as tool-use quality.
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

{jail_rule}

One `gnomon_forecast` call covers every channel: `target_column` takes
a comma list ("{channel_list}"), which loads the file once, evaluates
the channels concurrently, and returns one artifact carrying a result
per channel — the same numbers per channel as separate calls, and a
channel that abstains is reported in its own result without blocking
the others. That artifact_path is submittable for each of its channels.
Passing `format: "brief"` keeps the response to the forecast paths and
the disclosures instead of the full evidence block.

Ways to finish each channel of `submit_answer.forecast`:
- artifact_path from a `gnomon_forecast` run covering that channel: its
  trajectory becomes that channel's answer verbatim.
- your own values (exactly {horizon} numbers): your numbers, your
  responsibility.
- abstain (or omit the channel): an honest abstention.

Also answer every multiple-choice question of the task in
`submit_answer.mcq`. Tool errors return typed codes and repair options;
you may fix arguments and retry within the caps ({max_rounds} rounds,
{max_calls} tool calls).
"""

#: T1/T3 hold no forecast channels, so the forecast contract — horizons,
#: artifacts, per-channel exits — is absent here rather than restated
#: and waived. The tool surface is NOT pruned to match: whether an agent
#: that can interrogate the series answers these questions better is the
#: measurement, and taking the tools away would settle it by
#: construction. What the prompt does instead is name, symmetrically,
#: what the engine can say about a history and that reading the numbers
#: directly is an equally admissible answer.
SYSTEM_MCQ = """\
You are answering one time series question set. You have tools from
"gnomon", a deterministic forecasting engine, and you may use them or
ignore them — your only obligation is to end by calling `submit_answer`
with your answers.

{data_rule}

Nothing here asks for a forecast. What the engine can contribute is
description of what the history did — `gnomon_inspect` for schema,
frequency, gaps and duplicates, `gnomon_detect_anomalies` for graded
anomalies with their support state. Reading the numbers in the task
yourself is an equally admissible way to answer.

{jail_rule}

{answer_rule} Tool errors return typed codes and repair options; you
may fix arguments and retry within the caps ({max_rounds} rounds,
{max_calls} tool calls).
"""

#: The path rule, stated once for both prompts: the first measured sweep
#: spent a whole round per row discovering it by rejection.
JAIL_RULE = """\
This run may only read and write inside {jail}. Paths outside it are
rejected before the server sees them, so pass an output_dir under that
directory — for example output_dir={out_dir}."""

#: Named when a cap is reached: the run ends after the next message, so
#: an answer already worked out is submitted rather than lost.
LAST_CALL = """\
This run is ending now ({cap}). This is the final message: call
submit_answer with the best answer you have — a partial answer counts,
an unsubmitted one does not. No other tool is available."""


def openai_tool_specs(mcp_tools: list[dict[str, Any]],
                      submit_tool: dict[str, Any] = SUBMIT_TOOL,
                      ) -> list[dict[str, Any]]:
    """MCP tool entries as chat-completions specs, verbatim, plus the
    harness submit tool — the same no-pruning rule as the CiK arm
    (``outputSchema`` is dropped; the chat format has no slot for it).
    ``submit_tool`` differs by tier: the per-channel forecast contract on
    T2/T4, the tier's own answer shape on T1/T3."""
    return [
        {"type": "function", "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["inputSchema"],
        }}
        for tool in mcp_tools
    ] + [submit_tool]


def mcq_submit_tool(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """The ``submit_answer`` spec for a T1/T3 row, and the prompt
    sentence describing it.

    The schema is the row's own answer shape — T1's label fields as
    named properties, T3's pack as a fixed-length list in question order
    — so a correct answer cannot be lost to output formatting, which is
    a property of the harness, not of the model's temporal reasoning.
    The option strings are not repeated into the schema: the official
    prompt already carries them verbatim, and paraphrasing them here
    would put this adapter between the benchmark and the model.
    """
    if row.get("tier") == "T3":
        count = len(row.get("pack") or [])
        answers = {
            "type": "array", "items": {"type": "string"},
            "description": (
                f"Exactly {count} answers, one per question of the task's "
                f"pack, in the order the task lists them; each the option "
                f"string that question offers."
            ),
        }
        if count:
            answers["minItems"] = answers["maxItems"] = count
        rule = (f"Answer all {count} questions of the task's pack in "
                f"`submit_answer.answers`, in the order the task lists "
                f"them, each the option string that question offers.")
    else:
        fields = list(row.get("labels") or {})
        answers = {
            "type": "object",
            "description": ("One entry per answer field the task asks "
                            "for: " + (", ".join(fields) or "the task's "
                                       "own field names") + "."),
            "properties": {field: {"type": "string"} for field in fields},
            "additionalProperties": {"type": "string"},
            **({"required": fields} if fields else {}),
        }
        rule = ("Give your answer for every field in "
                "`submit_answer.answers`" + (f" ({', '.join(fields)})"
                                             if fields else "")
                + ", each the option string the task offers for it.")
    return ({
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": ("End the run with your answers to the task's "
                            "questions. " + rule),
            "parameters": {
                "type": "object",
                "properties": {"answers": answers,
                               "reasoning": {"type": "string"}},
                "required": ["answers"],
            },
        },
    }, rule)


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


#: Keys whose values are never shrunk when a tool result is too large.
#: Everything Gnomon says about how far it will stand behind a number
#: lives under one of these; the bulk that makes a result large does
#: not. A budget is a reason to send fewer numbers, never a reason to
#: send numbers without their disclosures.
UNSHRINKABLE_KEYS = frozenset({
    "support", "support_assessment", "support_state", "warnings", "notes",
    "status", "code", "error", "message", "recovery", "recovery_actions",
    "disclosures", "disclosure", "abstention", "reason", "artifact_path",
    "forecast_id", "series", "selected_model", "interval_coverage",
})


def _shrink(value: Any, edge: int = TRUNCATED_ARRAY_EDGE) -> Any:
    """Bulk arrays as head/tail plus a count; disclosures untouched."""
    if isinstance(value, dict):
        return {key: (item if key in UNSHRINKABLE_KEYS else _shrink(item, edge))
                for key, item in value.items()}
    if isinstance(value, list) and len(value) > 2 * edge:
        return {
            "harness_truncated": True,
            "items_total": len(value),
            "head": [_shrink(item, edge) for item in value[:edge]],
            "tail": [_shrink(item, edge) for item in value[-edge:]],
        }
    if isinstance(value, list):
        return [_shrink(item, edge) for item in value]
    return value


def _holds_disclosure(value: Any) -> bool:
    """True when anything in this subtree is an :data:`UNSHRINKABLE_KEYS`
    entry — dropping it would take a disclosure with it."""
    if isinstance(value, dict):
        return any(key in UNSHRINKABLE_KEYS or _holds_disclosure(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_holds_disclosure(item) for item in value)
    return False


def _droppable(value: Any, path: str = "") -> list[tuple[dict, str, str, int]]:
    """``(parent, key, path, size)`` for every entry that may be dropped.

    An entry that holds a disclosure anywhere beneath it is descended
    into rather than dropped, so pruning takes the bulk out of a result
    (a forecast preview, an evidence block) and never the result itself:
    the per-channel support labels and warnings survive a hard squeeze
    even when the numbers do not.
    """
    found: list[tuple[dict, str, str, int]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in UNSHRINKABLE_KEYS or key.startswith("harness_"):
                continue
            where = f"{path}.{key}" if path else key
            if _holds_disclosure(item):
                found.extend(_droppable(item, where))
            else:
                found.append((value, key, where,
                              len(json.dumps(item, default=str))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_droppable(item, f"{path}[{index}]"))
    return found


def _prune_to_fit(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    """Drop the largest droppable entries until the payload fits, naming
    each one. Never drops :data:`UNSHRINKABLE_KEYS`, so what survives a
    hard squeeze is the epistemics and the path to the full numbers.

    Cutting the serialized text instead would be shorter code and a
    worse answer: the model would receive JSON it cannot parse, with the
    artifact_path it needs to submit possibly inside the part that was
    cut — the failure `bounded_evidence` exists to prevent, one layer
    down. Best-effort by construction: if the surviving skeleton is
    itself over the limit, it is returned whole rather than corrupted.
    """
    dropped: list[str] = []
    while len(json.dumps(payload, default=str)) > limit:
        candidates = _droppable(payload)
        if not candidates:
            break
        parent, key, where, _ = max(candidates, key=lambda item: (item[3], item[2]))
        del parent[key]
        dropped.append(where)
        payload["harness_dropped"] = dropped
    return payload


def bounded_tool_text(text: str, limit: int = MAX_TOOL_RESULT_CHARS,
                      ) -> tuple[str, bool]:
    """``(text, was_truncated)`` — the server's own text, or a shrunk
    rendering of it when it exceeds ``limit``.

    Long results are the run's largest cost: every round re-sends every
    earlier one, so one full-format multi-channel forecast is paid for
    again in each subsequent round. What gets shrunk is bulk — long
    numeric arrays, evidence blocks — to head/tail plus the full count,
    under an explicit ``harness_truncated`` marker so nothing looks like
    a complete list that is not one. What never gets shrunk is
    :data:`UNSHRINKABLE_KEYS`: support states, warnings, abstention
    reasons, recovery actions, error codes, and the artifact_path where
    the complete numbers are on disk and readable in full.

    A non-JSON body (a plain-text server message) is cut at the limit
    with the cut named — the alternative, dropping it entirely, tells the
    model less than a marked truncation does.
    """
    if len(text) <= limit:
        return text, False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return (text[:limit] + f"\n[harness: this result was cut at {limit} "
                f"characters of {len(text)}; the run's token budget is "
                f"shared across rounds]", True)
    shrunk = _shrink(payload)
    if not isinstance(shrunk, dict):
        return json.dumps(shrunk, default=str)[:limit], True
    shrunk["harness_note"] = (
        f"This result was {len(text)} characters; the harness shrank its "
        f"long arrays to their first and last {TRUNCATED_ARRAY_EDGE} "
        f"entries with the totals kept, and dropped whatever else it had "
        f"to (harness_dropped names it). Support states, warnings, "
        f"abstention reasons and recovery actions are verbatim. The "
        f"complete numbers are in the artifact directory on disk."
    )
    return json.dumps(_prune_to_fit(shrunk, limit), default=str), True


class _RunBase:
    """One row's conversation: jail, server, loop, caps, submission.

    Subclasses supply the tier's contract — the submit tool, the system
    prompt, what a submission resolves to — and inherit the parts that
    must be identical across tiers: the path jail, the caps, the
    last-call protocol, and the abstention bookkeeping.
    """

    #: Repeated verbatim when the model answers in prose instead of
    #: calling the submit tool.
    NUDGE = "Finish by calling submit_answer with your answer."

    def __init__(self, row: dict[str, Any], client: Any,
                 session_factory: Any = None, work_dir: str | None = None):
        import time

        self.row = row
        self.client = client
        self.started = time.time()
        self.channels = self._row_channels(row)
        if "timestamp" in self.channels:
            raise ValueError("a channel named 'timestamp' collides with the "
                             "time column of the run CSV")
        self.jail = Path(tempfile.mkdtemp(prefix="tb-mcp-",
                                          dir=work_dir)).resolve()
        self.csv_path: Path | None = None
        if self.channels:
            self.csv_path = self.jail / "history.csv"
            _write_wide_csv(self.channels, self.csv_path)
        self.session = (session_factory or StdioMcpSession)(self.jail)
        self.trace: list[dict[str, Any]] = []
        self.mcp_calls = 0
        self.artifact_paths: set[str] = set()
        self.submission: dict[str, Any] | None = None
        self.tokens_at_start = (getattr(client, "total_prompt_tokens", 0)
                                + getattr(client, "total_completion_tokens", 0))

    # -- tier contract -----------------------------------------------------
    def _row_channels(self, row: dict[str, Any]) -> dict[str, list[float]]:
        raise NotImplementedError

    def _submit_tool(self) -> dict[str, Any]:
        raise NotImplementedError

    def _system(self) -> str:
        raise NotImplementedError

    def _handle_submit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _resolve_submission(self) -> dict[str, Any]:
        raise NotImplementedError

    def _abstain_outcome(self, reason: str) -> dict[str, Any]:
        raise NotImplementedError

    # -- caps --------------------------------------------------------------
    def _run_tokens(self) -> int:
        return (getattr(self.client, "total_prompt_tokens", 0)
                + getattr(self.client, "total_completion_tokens", 0)
                - self.tokens_at_start)

    def _cap_breach(self) -> str | None:
        import time

        if time.time() - self.started > MAX_WALL_SECONDS:
            return f"cap:wall_clock exceeded {MAX_WALL_SECONDS:.0f}s"
        if self._run_tokens() > MAX_RUN_TOKENS:
            return f"cap:tokens exceeded {MAX_RUN_TOKENS}"
        return None

    def _jail_rule(self) -> str:
        return JAIL_RULE.format(jail=self.jail,
                                out_dir=self.jail / "gnomon-output")

    def _abstention_record(self, reason: str) -> dict[str, Any]:
        """The fields every tier's abstention outcome carries.

        ``row_abstained`` is what keeps a voided row out of the accuracy
        denominators: a run the harness ended has not answered the
        questions wrongly, it has not answered them, and scoring it zero
        would report a harness cap as a model failure.
        """
        self.trace.append({"abstained": reason})
        return {"abstained": [reason], "row_abstained": reason,
                "mcp": self._mcp_info()}

    def _mcp_info(self) -> dict[str, Any]:
        return {
            "calls": self.mcp_calls,
            "run_tokens": self._run_tokens(),
            "tool_sequence": [
                {key: value for key, value in entry.items()
                 if key in ("tool", "is_error", "code", "jail_violations",
                            "truncated", "last_call", "abstained")}
                for entry in self.trace
            ],
        }

    # -- the loop ----------------------------------------------------------
    def drive(self) -> dict[str, Any]:
        self.session.initialize()
        submit_tool = self._submit_tool()
        tools = openai_tool_specs(self.session.list_tools(), submit_tool)
        # The official prompt is the user message, verbatim: the
        # benchmark stays authoritative about the task and its output
        # format; the system message adds only the harness contract.
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": self.row["prompt"]},
        ]

        nudged = False
        for _round in range(MAX_ROUNDS):
            breach = self._cap_breach()
            if breach:
                return self._last_call(messages, submit_tool, breach)
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
                    return self._last_call(
                        messages, submit_tool,
                        "no submission: two prose answers after the nudge")
                nudged = True
                messages.append({"role": "user", "content": self.NUDGE})
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
                if isinstance(result, dict):  # transport death, disclosed
                    return result
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": result})
            if self.submission:
                break
        if not self.submission:
            return self._last_call(
                messages, submit_tool,
                f"cap:rounds {MAX_ROUNDS} rounds without a submission")
        return self._resolve_submission()

    def _last_call(self, messages: list[dict[str, Any]],
                   submit_tool: dict[str, Any], cap: str) -> dict[str, Any]:
        """One final round offering only ``submit_answer``.

        A cap used to void the row outright, which threw away work the
        model had already done — the measured cost was every T2/T4 row of
        a sweep scored as an abstention after five to eight completed
        tool calls. The engine tools are withdrawn (nothing more may be
        computed on a spent budget), the cap is named, and a submission
        that arrives is resolved normally. If none does, the row abstains
        exactly as before, with the cap still named.
        """
        self.trace.append({"last_call": cap})
        messages = messages + [{"role": "user",
                                "content": LAST_CALL.format(cap=cap)}]
        try:
            response = self.client.chat(messages, n=1, tools=[submit_tool],
                                        tool_choice="auto")
        except Exception as error:
            return self._abstain_outcome(f"{cap}; last call failed: {error}")
        for call in _tool_calls_as_dicts(response.choices[0].message):
            if call["function"]["name"] != "submit_answer":
                continue
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                continue
            self._dispatch("submit_answer", arguments)
            if self.submission:
                self.submission["last_call"] = cap
                return self._resolve_submission()
        return self._abstain_outcome(cap)

    # -- dispatch ----------------------------------------------------------
    def _dispatch(self, name: str,
                  arguments: dict[str, Any] | None) -> str | dict[str, Any]:
        """Route one tool call; return the tool-message text, or the
        row's abstention outcome when the transport dies."""
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
                "message": "This run may only read and write inside "
                           f"{self.jail}. The history file is at "
                           f"{self.csv_path}; write outputs under "
                           f"{self.jail / 'gnomon-output'}.",
                "violations": violations,
                "authored_by": "harness",
            })
        # A spent tool budget ends tool use, not the row: the model is
        # told to submit what it has, and the round loop's last call
        # collects it. Voiding here discarded completed work.
        if self.mcp_calls >= MAX_MCP_CALLS:
            entry["code"] = "TOOL_BUDGET_SPENT"
            return json.dumps({
                "code": "TOOL_BUDGET_SPENT",
                "message": f"This run's {MAX_MCP_CALLS} tool calls are "
                           f"spent; no further engine calls will run. "
                           f"Call submit_answer with your best answer.",
                "authored_by": "harness",
            })
        breach = self._cap_breach()
        if breach:
            entry["code"] = "TOKEN_BUDGET_SPENT"
            return json.dumps({
                "code": "TOKEN_BUDGET_SPENT",
                "message": f"This run has reached its {breach}; no further "
                           f"engine calls will run. Call submit_answer with "
                           f"your best answer.",
                "authored_by": "harness",
            })
        self.mcp_calls += 1
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
        # The server's own text block, unedited unless it is too large
        # to carry — then bulk-shrunk, marked, with every disclosure kept.
        content = result.get("content") or []
        text = content[0].get("text", "") if content else json.dumps(structured)
        text, truncated = bounded_tool_text(text, MAX_TOOL_RESULT_CHARS)
        if truncated:
            entry["truncated"] = True
        return text

    def finish(self) -> None:
        self.session.close()


class _Run(_RunBase):
    """T2/T4: the per-channel forecast contract."""

    NUDGE = ("Finish by calling submit_answer — per channel an "
             "artifact_path from a gnomon_forecast run, your own values, "
             "or abstain; plus your mcq answers.")

    def __init__(self, row: dict[str, Any], client: Any,
                 session_factory: Any = None, work_dir: str | None = None):
        meta = row.get("meta") or {}
        self.horizon = int(meta.get("n_horizon") or 0)
        if self.horizon < 1:
            raise ValueError("row has no forecast horizon")
        ground_truth = row.get("ground_truth")
        self.target_keys = (list(ground_truth) if isinstance(ground_truth, dict)
                            else [meta.get("main_key")])
        super().__init__(row, client, session_factory=session_factory,
                         work_dir=work_dir)

    def _row_channels(self, row: dict[str, Any]) -> dict[str, list[float]]:
        arrays = prompt_input_arrays(row)
        return {key: _observed(arrays.get(key, [])) for key in self.target_keys}

    def _submit_tool(self) -> dict[str, Any]:
        return SUBMIT_TOOL

    def _system(self) -> str:
        return SYSTEM.format(
            csv_path=str(self.csv_path),
            channels=", ".join(self.target_keys),
            channel_list=",".join(self.target_keys),
            epoch=EPOCH.isoformat(), horizon=self.horizon,
            jail_rule=self._jail_rule(),
            max_rounds=MAX_ROUNDS, max_calls=MAX_MCP_CALLS,
        )

    def _abstain_outcome(self, reason: str) -> dict[str, Any]:
        """The whole row abstains with the cap named — never a fallback."""
        record = self._abstention_record(reason)
        return {
            "answer": {"forecast": {}, "mcq": {}},
            **record,
            "abstained": [f"{key}: {reason}" for key in self.target_keys],
            "channel_support": {},
            "channel_route": {key: "abstain" for key in self.target_keys},
        }

    # -- submission --------------------------------------------------------
    def _artifact_channel_rows(self, artifact_path: str,
                               channel: str) -> list[float] | tuple[str, str]:
        """The artifact's q50 trajectory for `channel`, or a
        ``(problem, support)`` pair explaining the rejection.

        A multi-target run (``target_column: "hr,spo2"``) publishes one
        artifact with a result per channel, so the binding is to the
        result whose ``series`` is this channel — not to ``results[0]``,
        which would hand every channel the first one's numbers. A
        single-target run names its one result ``__default__`` (it has no
        series column), so there the binding is the artifact's own
        ``target_column``, as before. Either way an artifact that never
        forecast the channel is rejected, and the rejection names what it
        did forecast.
        """
        from gnomon.artifacts import read_artifact

        if artifact_path not in self.artifact_paths:
            return (f"{channel}: artifact_path was not produced by a "
                    f"gnomon_forecast call in this run; known: "
                    f"{sorted(self.artifact_paths)}", "")
        try:
            data = read_artifact(artifact_path)
        except Exception as error:
            return f"{channel}: artifact could not be read: {error}", ""
        results = data.get("results") or []
        target = ((data.get("task") or {}).get("schema") or {}).get(
            "target_column")
        targets = [name.strip() for name in str(target).split(",")]
        result = next((item for item in results
                       if item.get("series") == channel), None)
        if result is None and len(targets) == 1 and len(results) == 1 \
                and targets[0] == channel:
            result = results[0]
        if result is None:
            return (f"{channel}: that artifact forecasts "
                    f"{', '.join(repr(name) for name in targets)}, not "
                    f"{channel!r}; submit it for a channel it covers or run "
                    f"gnomon_forecast with target_column={channel!r}", "")
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
            **({"last_call": self.submission["last_call"]}
               if self.submission.get("last_call") else {}),
            "mcp": self._mcp_info(),
        }


class _McqRun(_RunBase):
    """T1/T3: no forecast, the tier's own answer shape.

    The answer object is exactly what this tier's scorer reads — T1's
    label fields at the top level, T3's list under ``answers`` — so
    nothing about the tool arm changes how a choice is graded relative
    to the other conditions.
    """

    NUDGE = "Finish by calling submit_answer with your answers."

    def __init__(self, row: dict[str, Any], client: Any,
                 session_factory: Any = None, work_dir: str | None = None):
        self.tool, self.answer_rule = mcq_submit_tool(row)
        self.expected_fields = list(row.get("labels") or {})
        self.expected_count = len(row.get("pack") or [])
        self.is_pack = row.get("tier") == "T3"
        self.rejections = 0
        super().__init__(row, client, session_factory=session_factory,
                         work_dir=work_dir)

    def _row_channels(self, row: dict[str, Any]) -> dict[str, list[float]]:
        try:
            arrays = prompt_input_arrays(row)
        except ValueError:
            # T1/T3 read their arrays out of the prompt's Input block; a
            # row without one simply has no CSV, and the prompt says so
            # rather than naming a file that is not there.
            return {}
        return {key: _observed(values) for key, values in arrays.items()
                if _observed(values)}

    def _submit_tool(self) -> dict[str, Any]:
        return self.tool

    def _system(self) -> str:
        if self.csv_path is not None:
            data_rule = (
                f"The task's series are also on disk at {self.csv_path} "
                f"(columns: timestamp, {', '.join(self.channels)}), on a "
                f"synthetic regular hourly axis — observation k of a "
                f"series sits at {EPOCH.isoformat()} + k hours, recorded "
                f"readings laid consecutively; a shorter series' trailing "
                f"cells are blank. It is the same data the task states, "
                f"in the shape the tools read."
            )
        else:
            data_rule = ("The task states its data in the prompt; nothing "
                         "was written to disk for this row, so a tool call "
                         "would need the numbers passed inline.")
        return SYSTEM_MCQ.format(
            data_rule=data_rule, jail_rule=self._jail_rule(),
            answer_rule=self.answer_rule,
            max_rounds=MAX_ROUNDS, max_calls=MAX_MCP_CALLS,
        )

    def _abstain_outcome(self, reason: str) -> dict[str, Any]:
        record = self._abstention_record(reason)
        return {"answer": {}, **record}

    def _handle_submit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.submission is not None:
            return {"accepted": False, "authored_by": "harness",
                    "message": "An answer was already submitted; the first "
                               "accepted submission stands."}
        answers = arguments.get("answers")
        problems = self._problems(answers)
        # One repair round, then the answer stands as given: a validator
        # that keeps rejecting turns an answer the model did produce into
        # an abstention, which is a worse lie than an incomplete answer.
        if problems and self.rejections == 0:
            self.rejections += 1
            return {"accepted": False, "authored_by": "harness",
                    "problems": problems}
        if self.is_pack:
            resolved: Any = {"answers": [str(item) for item in answers]
                             if isinstance(answers, list) else []}
        else:
            resolved = {str(key): str(value)
                        for key, value in (answers or {}).items()} \
                if isinstance(answers, dict) else {}
        self.submission = {"answer": resolved,
                           "reasoning": arguments.get("reasoning"),
                           "problems": problems}
        return {"accepted": True,
                **({"noted": problems} if problems else {})}

    def _problems(self, answers: Any) -> list[str]:
        if self.is_pack:
            if not isinstance(answers, list):
                return [f"answers must be a list of {self.expected_count} "
                        f"option strings, one per question of the pack"]
            if self.expected_count and len(answers) != self.expected_count:
                return [f"answers must hold exactly {self.expected_count} "
                        f"entries, one per question of the pack, in the "
                        f"task's order; got {len(answers)}"]
            return []
        if not isinstance(answers, dict):
            return ["answers must be an object with one entry per answer "
                    "field the task asks for: "
                    + (", ".join(self.expected_fields) or "(see the task)")]
        missing = [field for field in self.expected_fields
                   if field not in answers]
        return ([f"answers is missing the field(s) "
                 f"{', '.join(missing)}; the task asks for "
                 f"{', '.join(self.expected_fields)}"] if missing else [])

    def _resolve_submission(self) -> dict[str, Any]:
        assert self.submission is not None
        return {
            "answer": self.submission["answer"],
            "abstained": [],
            "submit_reasoning": self.submission["reasoning"],
            **({"submit_problems": self.submission["problems"]}
               if self.submission["problems"] else {}),
            **({"last_call": self.submission["last_call"]}
               if self.submission.get("last_call") else {}),
            "mcp": self._mcp_info(),
        }


def _drive(run: _RunBase) -> dict[str, Any]:
    try:
        return run.drive()
    finally:
        run.finish()


def run_row(row: dict[str, Any], client: Any, *,
            session_factory: Any = None,
            work_dir: str | None = None) -> dict[str, Any]:
    """Drive one T2/T4 row through the real MCP surface; return the same
    outcome shape ``answer_row`` produces for the other conditions."""
    return _drive(_Run(row, client, session_factory=session_factory,
                       work_dir=work_dir))


def mcq_row(row: dict[str, Any], client: Any, *,
            session_factory: Any = None,
            work_dir: str | None = None) -> dict[str, Any]:
    """Drive one T1/T3 row through the same surface with the tier's own
    answer shape; the answer object is what that tier's scorer reads."""
    return _drive(_McqRun(row, client, session_factory=session_factory,
                          work_dir=work_dir))
