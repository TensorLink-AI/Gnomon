"""TemporalBench's ``gnomon-mcp`` condition: the real MCP surface, the
model driving.

The ``gnomon-agent`` condition computes Gnomon evidence in the harness
and pastes it into the prompt — the model never chooses to call
anything, so it measures evidence-grounded answering, not tool use.
This is the tool-use arm: the model holds every tool ``gnomon mcp
serve`` publishes, verbatim, plus one harness tool (``submit_answer``),
and drives the engine itself over the row's channels. The session,
verbatim tool-spec conversion, path jail, and superseded-result
compaction are imported from the CiK MCP arm
(``benchmarks/cik/mcp_agent.py``, spec:
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
wide CSV anchored to the official row's history cutoff and next-step
timestamp (nulls omitted, channels laid
consecutively — exactly ``gnomon_runner``'s convention, so the values
the engine sees match the other Gnomon conditions); the official prompt
is the user message verbatim; tool results pass through verbatim until
they exceed :data:`MAX_TOOL_RESULT_CHARS`, past which the bulk arrays
are shrunk to head/tail plus counts under a harness-authored
``truncated`` marker naming the artifact that holds the full numbers —
support states, warnings, abstention reasons, recovery actions and
error codes are never what gets dropped. A result superseded by a later
call (same forecast channels, or an identical repeat) is additionally
compacted in the running history to its disclosures plus artifact_path
— dead bulk is not re-sent every remaining round, and what Gnomon said
about its numbers still is. A breached cap does not void
work already done: a run that has submitted keeps its submission, and a
run that has not gets one final round to submit what it has, with the
cap named. Only after that is the row an abstention — never a silent
fallback, and never a zero scored as if the model had answered.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.mcp_agent import (  # noqa: E402 — library reuse
    UNSHRINKABLE_KEYS,
    StdioMcpSession,
    ToolMessageLog,
    _tool_calls_as_dicts,
    jail_violations,
)
from benchmarks.common.openrouter import OpenRouterError  # noqa: E402
from benchmarks.temporalbench.gnomon_runner import EPOCH, STEP, _observed  # noqa: E402
from benchmarks.temporalbench.tasks import prompt_input_arrays  # noqa: E402

#: Measured, not guessed. At 10 this cap — not tokens, not wall clock —
#: is what ends the arm: in the 2026-08-11 deepseek-v4-flash pilots
#: 11 of 12 T2/T4 rows ended `cap:rounds` with the token ceiling never
#: breached (208k mean of 500k) and ~45s rows against 7200s. The rows
#: were not short of evidence; a clean forecast landed by round 3 and
#: the rest went to re-reading it. Raising to 30 alone took scored rows
#: 1/12 -> 4/12. Every round re-sends every earlier tool result, so this
#: trades against MAX_RUN_TOKENS: at 30 two rows hit the token ceiling.
MAX_ROUNDS = 10
# Product target and experiment guard: after four engine calls the model has
# had inspect + forecast + two reads. Measured traces then spent up to nine
# more calls rereading the same artifact. The final submit-only round preserves
# completed work, so this bounds browsing rather than forcing abstention.
MAX_MCP_CALLS = 4
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
                "choice_basis": {
                    "type": "object",
                    "description": (
                        "Required only when an mcq choice overrides a weak "
                        "canonical answer: question key -> typed opposing "
                        "evidence. task_context evidence must be an exact "
                        "quote from the task prompt."),
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": [
                                "computed_opposition", "task_context"]},
                            "evidence": {"type": "string"},
                        },
                        "required": ["kind", "evidence"],
                        "additionalProperties": False,
                    },
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

The task's channel histories are in {csv_path} ({data_layout}). The timestamps
are a synthetic regular hourly axis — observation k of a channel sits at
{epoch} + k hours and recorded readings are laid consecutively. The
metrics are index-based, so the axis never enters the score. Forecast
horizon: {horizon} steps per channel.

The harness has already resolved the schema: {schema_binding}. Do not call
capabilities or inspect to rediscover these supplied facts; call the forecast
verb directly with them.

{jail_rule}

One `gnomon_forecast` call covers every channel: {batching_contract}. It
returns one artifact carrying a result per channel; a channel that abstains is
reported independently. That artifact_path is submittable for each channel.
Passing `format: "brief"` keeps the response to the forecast paths and
the disclosures instead of the full evidence block.

If the task also asks about a temporal property (level, trend,
seasonality, volatility, regime, extremes, or dependence), include typed
`questions` in that same call rather than inferring the property from the
forecast path. Each object has `id`, `verb`, exact `target`, `property`, and
when predictive, `horizon`; Gnomon echoes the interpretation and may abstain
from that property without changing the primary forecast. Use target
`{{"kind":"each","members":[...]}}` for separate channel answers. For one
cross-unit volatility answer, use target
`{{"kind":"aggregate","members":[...],"aggregation":
"median_normalized_scale_ratio"}}`; the result retains every constituent.

Ways to finish each channel of `submit_answer.forecast`:
- artifact_path from a `gnomon_forecast` run covering that channel: its
  trajectory becomes that channel's answer verbatim.
- your own values (exactly {horizon} numbers): your numbers, your
  responsibility.
- abstain (or omit the channel): an honest abstention.

Also answer every multiple-choice question of the task in
`submit_answer.mcq`. Tool errors return typed codes and repair options;
When a typed answer has `support: abstained`, or says no categorical threshold
was supplied, preserve that limitation by choosing `Uncertain` when it is one
of the task's options; a baseline estimate is not a supported category.
Typed answers marked supported and automation-eligible are binding. A weak
canonical answer is advisory but remains the default: override it only when
independent task context or computed opposing evidence specifically supports a
different option, and provide `choice_basis[question]` with its typed basis.
For task context, `evidence` must be an exact quote from this task. Weakness or
uncertainty alone is not opposing evidence. The
host preserves both canonical and synthesized values and never rewrites the
immutable receipt.
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

#: Attempts allowed at the last call. Two, not one: the first measured
#: sweeps lost whole rows to an answer the model *had* produced and
#: mis-enveloped, with a correct repair message that had no round left
#: to land in. The extra attempt buys no new computation — the engine
#: tools are already withdrawn — so it repairs the envelope, never the
#: answer.
LAST_CALL_ATTEMPTS = 2

#: Sent when the last call's submission is rejected and one attempt
#: remains. Names the shape, because the observed failure was a
#: correct answer inside a JSON-encoded string.
LAST_CALL_REPAIR = """\
That submission was NOT accepted — the problems are in the tool result
above. This is your final attempt: call submit_answer once more with the
same content, in the shape the schema asks for. `forecast` and `mcq`
must be JSON objects, not strings containing JSON."""

#: Sent when the last call produced prose and no tool call. The answer
#: in that prose cannot be scored; only a submit_answer call can be.
LAST_CALL_PROSE = """\
Your answer was written as prose, which cannot be scored — only a
submit_answer tool call can. This is your final attempt: make that call
now with the answer you just described. Do not restate it in text."""

#: submit_answer fields whose value must be a container. A model that
#: double-encodes one — ``"forecast": "{\\"hr\\": ...}"`` instead of the
#: object — has produced the right answer inside the wrong envelope.
COERCIBLE_SUBMIT_FIELDS = ("forecast", "mcq", "answers")


def coerce_json_containers(
    arguments: dict[str, Any],
    fields: tuple[str, ...] = COERCIBLE_SUBMIT_FIELDS,
) -> tuple[dict[str, Any], list[str]]:
    """Parse double-encoded JSON in ``submit_answer``'s container fields.

    Tool-calling models routinely serialize a nested object into a
    string. The control condition's own parser is deliberately lenient —
    it scans free-form text for a JSON object and skips candidates that
    do not parse — so rejecting this arm's answer on the envelope alone
    holds the two conditions to different standards and discards an
    answer the model did produce.

    ONLY the envelope is repaired. A string that does not parse, or that
    parses to a scalar, is left exactly as it was for the validator to
    reject on its merits; no value is invented, reshaped, or filled in.
    Returns the arguments and the names of the fields repaired — the
    caller records them, because a repair this harness makes silently is
    a repair it is not entitled to make.
    """
    if not isinstance(arguments, dict):
        return arguments, []
    repaired: list[str] = []
    result = dict(arguments)
    for field in fields:
        value = result.get(field)
        if not isinstance(value, str):
            continue
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, (dict, list)):
            result[field] = parsed
            repaired.append(field)
    # One level deeper: a per-channel entry may itself arrive encoded.
    forecast = result.get("forecast")
    if isinstance(forecast, dict):
        channels = dict(forecast)
        for channel, entry in forecast.items():
            if not isinstance(entry, str):
                continue
            try:
                parsed = json.loads(entry)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                channels[channel] = parsed
                repaired.append(f"forecast.{channel}")
        result["forecast"] = channels
    return result, repaired


def openai_tool_specs(mcp_tools: list[dict[str, Any]],
                      submit_tool: dict[str, Any] | None = SUBMIT_TOOL,
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
    ] + ([submit_tool] if submit_tool is not None else [])


def preferred_execution_tool(
    profile: str, has_forecast_targets: bool, *, host_compiled: bool = False,
) -> str | None:
    """Return the host-compiled first verb for a forecast task.

    Profiles govern what else is available, not whether an agent must infer
    the obvious execution route from a larger menu. Leaving Core, Describe,
    and Full on ``auto`` measured tool-navigation noise rather than product
    capability and caused avoidable non-submissions.
    """
    if not has_forecast_targets:
        return None
    if profile == "mega":
        return "gnomon_run"
    if profile == "evidence" or host_compiled:
        return "gnomon_forecast"
    return None


def compile_row_context(row: dict[str, Any], client: Any,
                        receipt_dir: str | None = None) -> dict[str, Any]:
    """Run Gnomon's proposal/validation workflow over task narrative.

    Arrays and output instructions are excluded from the document: they are
    already bound through the execution compiler, and treating them as prose
    both wastes tokens and makes accidental numeric claims look like context.
    The LLM proposes only; :func:`parse_context_response` grounds every quote
    and validates every event before it can reach the forecast call.
    """
    precompiled = row.get("_validated_context")
    if isinstance(precompiled, dict):
        return {"attempted": True, **precompiled}

    from gnomon.workflows import (
        CONTEXT_RESPONSE_SCHEMA, DocumentRef,
        build_context_investigation_prompt, normalise_context_response_containers,
        parse_context_response,
    )

    narrative = str(row.get("prompt") or "").split("Input (JSON):", 1)[0].strip()
    receipt_path = None
    if receipt_dir:
        safe_id = "".join(character if character.isalnum() or character in "-_"
                          else "_" for character in str(
                              row.get("id") or "temporalbench"))
        receipt_path = Path(receipt_dir) / f"{safe_id}.json"
        if receipt_path.is_file():
            cached = json.loads(receipt_path.read_text(encoding="utf-8"))
            from gnomon.workflows import CONTEXT_COMPILER_CONTRACT_VERSION
            from gnomon.soft_context import content_fingerprint
            documents = ((cached.get("context_receipt") or {}).get(
                "documents") or [])
            if (cached.get("schema_version") !=
                    CONTEXT_COMPILER_CONTRACT_VERSION):
                cached = None
            elif (not documents or documents[0].get("content_fingerprint") !=
                  content_fingerprint(narrative)):
                raise ValueError("cached context receipt does not match task narrative")
            if cached is not None:
                return {**cached, "attempted": True, "compiler_called": False,
                        "receipt_reused": True}
    document = DocumentRef(
        name=f"{row.get('id', 'temporalbench')}-context",
        content=narrative,
        source_type="benchmark_prompt",
        reference=str(row.get("id") or "temporalbench"),
    )
    series = [str(item) for item in (row.get("meta") or {}).get(
        "target_keys", [])]
    workflow = build_context_investigation_prompt(
        [document], series, future_events=True)
    submit = {
        "type": "function",
        "function": {
            "name": "submit_context",
            "description": "Submit proposed context for deterministic validation.",
            "parameters": CONTEXT_RESPONSE_SCHEMA,
        },
    }
    try:
        response = client.chat(
            [{"role": "system", "content": workflow["instructions"]}],
            n=1, tools=[submit], tool_choice={"type": "function", "function": {
                "name": "submit_context"}}, max_tokens=4000)
        calls = _tool_calls_as_dicts(response.choices[0].message)
        call = next(item for item in calls
                    if item["function"]["name"] == "submit_context")
        raw = json.loads(call["function"]["arguments"] or "{}")
        raw, container_repairs = normalise_context_response_containers(raw)
        validated = parse_context_response(
            raw, [document], proposer={"kind": "llm",
                                      "model": getattr(client, "model", "unknown")})
        result = {"attempted": True, "compiler_called": True,
                  "container_repairs": container_repairs, **validated}
        if receipt_path is not None:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            rendered = json.dumps(validated, indent=2, sort_keys=True) + "\n"
            if receipt_path.exists() and receipt_path.read_text(
                    encoding="utf-8") != rendered:
                raise ValueError("immutable task receipt already has different content")
            receipt_path.write_text(rendered, encoding="utf-8")
        return result
    except Exception as error:
        # Context is optional evidence. A compiler/provider failure must be
        # visible but cannot turn readable numeric history into a failed row.
        return {"attempted": True, "compiler_called": True,
                "events": [], "hypotheses": [],
                "rejected": [], "error": str(error)}


def compile_row_temporal_questions(
    row: dict[str, Any], client: Any, targets: list[str],
    receipt_dir: str | None = None,
) -> dict[str, Any]:
    """Compile question text only; labels, options and futures stay sealed."""
    from gnomon.soft_context import content_fingerprint
    from gnomon.temporal_intent import (
        INTENT_COMPILER_MAX_TOKENS, INTENT_COMPILER_VERSION, INTENT_SCHEMA,
        compile_temporal_text_receipt,
    )

    if row.get("tier") == "T3":
        question_items = row.get("pack") or []
        text = "\n".join(str(item.get("question") or "")
                         for item in question_items
                         if isinstance(item, dict)).strip()
    elif row.get("tier") == "T1":
        # T1 carries its descriptive request in the public prompt rather
        # than ``mcq``.  Compile only the instruction section: the input
        # arrays are already host-bound and the output template/options are
        # answer vocabulary, not evidence about the requested property.
        instruction = str(row.get("prompt") or "").split(
            "Input (JSON):", 1)[0]
        lines = []
        for line in instruction.splitlines():
            match = re.match(r"^\s*\d+\)\s*([^:{]+)", line)
            if match:
                lines.append(f"Describe the {match.group(1).strip()} of "
                             f"{targets[0]}?" if len(targets) == 1
                             else f"Describe the {match.group(1).strip()}?")
        text = "\n".join(lines).strip()
    else:
        question_items = (row.get("mcq") or {}).values()
        text = "\n".join(str(item.get("question") or "")
                         for item in question_items
                         if isinstance(item, dict)).strip()
    if not text:
        return {"attempted": False, "questions": []}
    # A descriptive question concerns the observed history.  It must not
    # silently inherit an adjacent forecast task's future horizon; doing so
    # changes “what happened?” into “what will happen over N steps?”.
    descriptive = row.get("tier") in {"T1", "T3"}
    default_horizon = (1 if descriptive else
                       int((row.get("meta") or {}).get("n_horizon") or 1))
    fingerprint = content_fingerprint(json.dumps({
        "text": text, "targets": targets,
        "default_horizon": default_horizon,
        "compiler_version": INTENT_COMPILER_VERSION,
    }, sort_keys=True))
    receipt_path = None
    failed_cached_receipt: dict[str, Any] | None = None
    if receipt_dir:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_"
                          for c in str(row.get("id") or "temporalbench"))
        receipt_path = Path(receipt_dir) / f"{safe_id}.json"
        if receipt_path.is_file():
            cached = json.loads(receipt_path.read_text(encoding="utf-8"))
            if cached.get("input_fingerprint") != fingerprint:
                raise ValueError("cached temporal-intent receipt does not match input")
            if cached.get("questions"):
                return {**cached, "attempted": True, "compiler_called": False,
                        "receipt_reused": True}
            # A failed proposal is an immutable diagnostic, not a reusable
            # compilation result. Retry into a sibling receipt so one
            # malformed provider response cannot permanently erase all typed
            # questions from future/resumed benchmark runs.
            failed_cached_receipt = cached
            original_receipt_path = receipt_path
            retry_number = 1
            while True:
                suffix = ".retry" if retry_number == 1 else \
                    f".retry{retry_number}"
                receipt_path = original_receipt_path.with_name(
                    original_receipt_path.stem + suffix + ".json")
                if not receipt_path.is_file():
                    break
                retry = json.loads(receipt_path.read_text(encoding="utf-8"))
                if retry.get("input_fingerprint") != fingerprint:
                    raise ValueError(
                        "cached temporal-intent retry does not match input")
                if retry.get("questions"):
                    return {**retry, "attempted": True,
                            "compiler_called": False,
                            "receipt_reused": True,
                            "prior_failed_receipt": True}
                retry_number += 1

    class Adapter:
        def complete(self, prompt: str, response_schema: dict[str, Any]):
            submit = {"type": "function", "function": {
                "name": "submit_temporal_intent", "description": "Submit intent only.",
                "parameters": response_schema}}
            response = client.chat(
                [{"role": "system", "content": prompt}], n=1, tools=[submit],
                tool_choice={"type": "function", "function": {
                    "name": "submit_temporal_intent"}},
                max_tokens=INTENT_COMPILER_MAX_TOKENS)
            calls = _tool_calls_as_dicts(response.choices[0].message)
            call = next(item for item in calls
                        if item["function"]["name"] == "submit_temporal_intent")
            return json.loads(call["function"]["arguments"] or "{}")

    try:
        receipt = compile_temporal_text_receipt(
            text, available_targets=targets, adapter=Adapter(),
            default_verb=("describe" if descriptive
                          else "predict"),
            default_horizon=default_horizon)
        result = {"input_fingerprint": fingerprint,
                  "proposed": receipt["proposed"],
                  "questions": [item.to_dict() for item in receipt["accepted"]],
                  "rejected": receipt["rejected"], "compiler_called": True}
    except Exception as error:
        result = {"input_fingerprint": fingerprint, "questions": [],
                  "rejected": [{"type": type(error).__name__, "message": str(error),
                                "details": getattr(error, "details", {})}],
                  "compiler_called": True}
    if failed_cached_receipt is not None:
        result["prior_failed_receipt"] = True
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if receipt_path.exists() and receipt_path.read_text(encoding="utf-8") != rendered:
            raise ValueError("immutable temporal-intent receipt already differs")
        receipt_path.write_text(rendered, encoding="utf-8")
    return {"attempted": True, **result}


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


def _write_wide_csv(channels: dict[str, list[float]], csv_path: Path,
                    epoch: datetime = EPOCH,
                    step: timedelta = STEP) -> None:
    keys = list(channels)
    length = max((len(values) for values in channels.values()), default=0)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp"] + keys)
        for position in range(length):
            writer.writerow(
                [(epoch + position * step).isoformat()]
                + [repr(channels[key][position])
                   if position < len(channels[key]) else "" for key in keys]
            )


def _write_long_csv(channels: dict[str, list[float]], csv_path: Path,
                    epoch: datetime = EPOCH,
                    step: timedelta = STEP) -> None:
    """Lossless panel form for descriptive rows with unequal lengths."""
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "series", "value"])
        for series, values in channels.items():
            for position, value in enumerate(values):
                writer.writerow([(epoch + position * step).isoformat(),
                                 series, repr(value)])


#: The never-shrunk keys are shared with the CiK arm (imported above):
#: one definition of what a disclosure is, whether a result is being
#: size-bounded here or supersession-compacted by the shared
#: :class:`ToolMessageLog`.


#: Named inside every harness-shortened array. The shortening is the
#: harness's own doing, so the consequence of it is the harness's to
#: disclose: the measured failure was a model that read "truncated",
#: concluded the numbers were lost, and spent the rest of its rounds
#: re-inspecting the file to rebuild them by hand — 34 `gnomon_inspect`
#: calls across 9 rows that had already forecast successfully. This
#: states what the harness did and what still works; it says nothing
#: about the task, and nothing about whether to use the engine at all.
TRUNCATION_REMEDY = (
    "The harness shortened this array to fit the tool-result budget — "
    "Gnomon's own values are complete and unchanged. You do NOT need "
    "these numbers to answer: submit_answer with this run's "
    "artifact_path uses the full trajectory verbatim. Re-reading the "
    "file will not return more of it."
)


def _shrink(value: Any, edge: int = TRUNCATED_ARRAY_EDGE) -> Any:
    """Bulk arrays as head/tail plus a count; disclosures untouched."""
    if isinstance(value, dict):
        return {key: (item if key in UNSHRINKABLE_KEYS else _shrink(item, edge))
                for key, item in value.items()}
    if isinstance(value, list) and len(value) > 2 * edge:
        return {
            "harness_truncated": True,
            "items_total": len(value),
            "remedy": TRUNCATION_REMEDY,
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
                 session_factory: Any = None, work_dir: str | None = None,
                 profile: str = "full", compile_context: bool = False,
                 context_receipts_dir: str | None = None,
                 compile_questions: bool = False,
                 question_receipts_dir: str | None = None):
        import time

        # Evidence is the governed product arm: the model may explain and
        # preserve Gnomon's result, but it may not replace the published
        # trajectory with values of its own.  Historically the runner set a
        # private row flag for some call paths while the profile itself did
        # not enforce the promise.  That made ``informed-direct`` possible in
        # live Evidence runs even though the prompt and result contract called
        # the primary immutable.  Make the policy an affordance of the
        # profile, not an optional orchestration detail.  Copy the benchmark
        # row so this host-only marker never mutates the official corpus.
        self.row = dict(row)
        if profile == "evidence":
            self.row["_require_gnomon_execution"] = True
        self.profile = profile
        self.client = client
        self.channels = self._row_channels(self.row)
        # The jail is also the trust boundary for host-compiled context, so it
        # must exist before compilation receipts are materialised below.
        self.jail = Path(tempfile.mkdtemp(prefix="tb-mcp-",
                                          dir=work_dir)).resolve()
        # Evidence executes through a host-bound panel schema. Long form gives
        # each observation-indexed channel its own consecutive axis; a wide
        # table pads unequal histories and turns absent readings into gaps.
        self.panel_long_form = bool(
            profile == "evidence" or row.get("_host_compiled_forecast")
            or row.get("_require_gnomon_execution"))
        self.temporal_compilation = (
            compile_row_temporal_questions(
                row, client, list(self.channels), question_receipts_dir)
            if compile_questions and row.get("tier") in {"T1", "T2", "T3", "T4"}
            else {"attempted": False, "questions": []}
        )
        raw_origin = row.get("_time_origin")
        raw_step = row.get("_time_step_seconds")
        meta = row.get("meta") or {}
        history_end_raw = meta.get("history_end")
        history_end = (datetime.fromisoformat(str(history_end_raw))
                       if history_end_raw else None)
        if history_end is not None and history_end.tzinfo is None:
            history_end = history_end.replace(tzinfo=timezone.utc)
        # Official TemporalBench arrays are observation-indexed, not a
        # calendar grid. ``cluster_start - history_end`` is metadata about
        # the source sampling window and can be 4h even though the array has
        # consecutive observations. Inferring the CSV cadence from that gap
        # while declaring frequency=h manufactured an irregular grid. Only
        # synthetic/product rows that explicitly provide a step may override
        # the disclosed hourly observation axis.
        self.time_step = timedelta(seconds=float(
            raw_step if raw_step is not None else STEP.total_seconds()))
        longest = max((len(values) for values in self.channels.values()), default=1)
        self.epoch = (datetime.fromisoformat(str(raw_origin))
                      if raw_origin else
                      history_end - (longest - 1) * self.time_step
                      if history_end else EPOCH)
        self.frequency = str(row.get("_frequency", "h"))
        if self.epoch.tzinfo is None:
            raise ValueError("_time_origin must include an explicit timezone")
        self.context_compilation = (
            compile_row_context(row, client, context_receipts_dir)
            if compile_context and row.get("tier") in {"T3", "T4"}
            else {"attempted": False, "events": [], "hypotheses": [],
                  "rejected": []}
        )
        # TemporalBench supplies the narrative to the forecaster at the
        # history cutoff. Compiler outputs often copy the event's effective
        # time into known_at; for this benchmark source that would falsely
        # say the prompt becomes knowable only after prediction begins.
        # Preserve the immutable compiler receipt, but bind executable events
        # to the actual document availability time used by the experiment.
        if history_end and self.context_compilation.get("events"):
            aligned = []
            for original in self.context_compilation["events"]:
                event = dict(original)
                source = event.get("source") or {}
                if source.get("type") == "benchmark_prompt":
                    event["known_at"] = history_end.isoformat()
                    attributes = dict(event.get("attributes") or {})
                    attributes["benchmark_document_known_at"] = history_end.isoformat()
                    event["attributes"] = attributes
                aligned.append(event)
            self.context_compilation["events"] = aligned
        self.context_events_file = None
        if self.context_compilation.get("events"):
            # ``compile_context`` is a host-side integration step over the
            # benchmark document, not an assertion made by the tool-driving
            # model. Bind that reviewed receipt through the production file
            # trust boundary. Passing the same events inline correctly leaves
            # them unverified/scenario-only and would measure a different arm.
            context_file = self.jail / "compiled-context-events.json"
            context_file.write_text(json.dumps({
                "schema_version": "0.1",
                "events": self.context_compilation["events"],
            }, sort_keys=True) + "\n", encoding="utf-8")
            self.context_events_file = context_file
        self.started = time.time()
        self.covariate_arguments = self._row_covariates(row)
        if "timestamp" in self.channels:
            raise ValueError("a channel named 'timestamp' collides with the "
                             "time column of the run CSV")
        self.csv_path: Path | None = None
        if self.channels:
            self.csv_path = self.jail / "history.csv"
            if row.get("tier") in {"T1", "T3"} or self.panel_long_form:
                _write_long_csv(self.channels, self.csv_path, self.epoch,
                                self.time_step)
            else:
                _write_wide_csv(self.channels, self.csv_path, self.epoch,
                                self.time_step)
        self.session = (session_factory or StdioMcpSession)(self.jail)
        self.trace: list[dict[str, Any]] = []
        self.result_log = ToolMessageLog()
        self.mcp_calls = 0
        self.sufficient_at_call: int | None = None
        self.redundant_calls = 0
        self.schema_bytes = 0
        self.product_schema_bytes = 0
        self.harness_schema_bytes = 0
        self.artifact_paths: set[str] = set()
        # Descriptive typed answers are returned inline rather than written
        # to a forecast artifact.  Retain the engine-authored structures for
        # coverage reporting; the final model submission remains separate.
        self.descriptive_answer_receipts: list[dict[str, Any]] = []
        # Compiler acceptance is textual grounding, not permission to alter a
        # forecast.  Keep the engine's later admission/application decision
        # separately so the benchmark cannot report "context used" merely
        # because a quote survived compilation.
        self.context_execution: dict[str, dict[str, Any]] = {}
        self.covariate_execution: dict[str, dict[str, Any]] = {}
        self.complete_artifact_ready = False
        self.complete_description_ready = False
        self.submission: dict[str, Any] | None = None
        self.tokens_at_start = (getattr(client, "total_prompt_tokens", 0)
                                + getattr(client, "total_completion_tokens", 0))

    # -- tier contract -----------------------------------------------------
    def _row_channels(self, row: dict[str, Any]) -> dict[str, list[float]]:
        raise NotImplementedError

    def _row_covariates(self, row: dict[str, Any]) -> dict[str, Any]:
        """Compile official aligned future covariates into Gnomon rows.

        Each target uses its own compressed observation axis, matching the
        history CSV construction exactly.  This matters when channels have
        different missing positions: a single default covariate timeline
        would silently misalign at least one of them.
        """
        task_input = row.get("input") or {}
        history = task_input.get("history") or {}
        future = task_input.get("future_covariates") or {}
        declared_mapping = task_input.get("covariate_mapping") or []
        if not isinstance(history, dict) or not isinstance(future, dict):
            return {}
        names = sorted(name for name, values in future.items()
                       if isinstance(values, list))
        if not names:
            return {}
        rows: list[dict[str, Any]] = []
        for series, target_values in self.channels.items():
            # A one-target Gnomon artifact is keyed ``__default__`` even
            # when the source column has a public name. Multi-target runs
            # retain their column names. Covariate entity keys must follow
            # the artifact identity or every fold appears to have no values.
            covariate_series = ("__default__"
                                if len(self.channels) == 1 else series)
            # self.channels is already the observed-only sequence. Recover
            # the original target mask to compress historical covariates in
            # the same way.
            raw_target = history.get(series)
            if not isinstance(raw_target, list):
                continue
            keep = [index for index, value in enumerate(raw_target)
                    if isinstance(value, (int, float))]
            position = 0
            for index in keep:
                row_values = {}
                for name in names:
                    past = history.get(name)
                    if not isinstance(past, list) or index >= len(past) \
                            or not isinstance(past[index], (int, float)):
                        row_values = {}
                        break
                    row_values[name] = float(past[index])
                if not row_values:
                    continue
                rows.append({
                    "timestamp": (self.epoch + position * STEP).isoformat(),
                    "known_at": self.epoch.isoformat(), "series": covariate_series,
                    **row_values,
                })
                position += 1
            horizon = min(len(future[name]) for name in names)
            for offset in range(horizon):
                if not all(isinstance(future[name][offset], (int, float))
                           for name in names):
                    continue
                rows.append({
                    "timestamp": (self.epoch + (len(target_values) + offset) * STEP).isoformat(),
                    "known_at": self.epoch.isoformat(), "series": covariate_series,
                    **{name: float(future[name][offset]) for name in names},
                })
        if not rows:
            return {}
        declared_by_name = {
            str(item.get("name")): item for item in declared_mapping
            if isinstance(item, dict) and item.get("name")
        }
        return {
            "covariates": rows,
            "covariate_mapping": [{
                "name": name,
                "type": str(declared_by_name.get(name, {}).get(
                    "type") or ("cyclic_1440" if name == "time_position_in_day"
                                else "continuous")),
                "availability": str(declared_by_name.get(name, {}).get(
                    "availability") or "future_known"),
            } for name in names],
            "covariate_series_column": "series",
        }

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
        temporal_answer_receipts: list[dict[str, Any]] = [
            dict(receipt) for receipt in self.descriptive_answer_receipts
        ]
        artifact_forecast_identity: list[dict[str, Any]] = []
        for artifact_path in sorted(self.artifact_paths):
            artifact_json = Path(artifact_path) / "artifact.json"
            if artifact_json.is_file():
                try:
                    artifact = json.loads(artifact_json.read_text(
                        encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    artifact_forecast_identity.append({
                        "artifact_path": artifact_path,
                        "read_error": f"{type(error).__name__}: {error}",
                    })
                else:
                    final_identities = {
                        item.get("series"): item.get("payload")
                        for item in artifact.get("evidence", [])
                        if isinstance(item, dict)
                        and item.get("kind") == "final_candidate"
                    }
                    artifact_forecast_identity.append({
                        "artifact_path": artifact_path,
                        "artifact_id": artifact.get("artifact_id"),
                        "series": [{
                            "series": result.get("series"),
                            "selected_model": result.get("selected_model"),
                            "support": result.get("support"),
                            "candidate_identity": (
                                result.get("candidate_identity")
                                or final_identities.get(result.get("series"))),
                            "admission": result.get("admission"),
                        } for result in artifact.get("results", [])
                         if isinstance(result, dict)],
                    })
            receipt_path = Path(artifact_path) / "temporal_answers.json"
            if not receipt_path.is_file():
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                temporal_answer_receipts.append({
                    "artifact_path": artifact_path,
                    "read_error": f"{type(error).__name__}: {error}",
                })
            else:
                temporal_answer_receipts.append({
                    "artifact_path": artifact_path,
                    "artifact_id": receipt.get("artifact_id"),
                    "primary_forecast_unchanged": receipt.get(
                        "primary_forecast_unchanged"),
                    "answers": receipt.get("answers") or [],
                })
        return {
            "calls": self.mcp_calls,
            "surface_required_calls": (getattr(self, "sufficient_at_call", None)
                                       if getattr(self, "sufficient_at_call", None) is not None
                                       else self.mcp_calls),
            "redundant_calls": getattr(self, "redundant_calls", 0),
            "run_tokens": self._run_tokens(),
            "schema_bytes": self.schema_bytes,
            "product_schema_bytes": self.product_schema_bytes,
            "harness_schema_bytes": self.harness_schema_bytes,
            "compiler_calls": int(bool(self.context_compilation.get(
                "compiler_called"))),
            "context_receipt_id": self.context_compilation.get("receipt_id"),
            "context_receipt_reused": bool(
                self.context_compilation.get("receipt_reused")),
            "compiled_context": {
                "accepted_events": len(self.context_compilation.get("events", [])),
                "accepted_hypotheses": len(self.context_compilation.get("hypotheses", [])),
                "rejected": len(self.context_compilation.get("rejected", [])),
                **({"error": self.context_compilation["error"]}
                   if self.context_compilation.get("error") else {}),
            },
            "temporal_compilation": {
                "attempted": bool(self.temporal_compilation.get("attempted")),
                "compiler_called": bool(self.temporal_compilation.get("compiler_called")),
                "receipt_reused": bool(self.temporal_compilation.get("receipt_reused")),
                "accepted": len(self.temporal_compilation.get("questions", [])),
                "rejected": len(self.temporal_compilation.get("rejected", [])),
            },
            **({"temporal_answer_receipts": temporal_answer_receipts}
               if temporal_answer_receipts else {}),
            **({"artifact_forecast_identity": artifact_forecast_identity}
               if artifact_forecast_identity else {}),
            "tool_sequence": [
                {key: value for key, value in entry.items()
                 if key in ("tool", "is_error", "code", "jail_violations",
                            "message",
                            "truncated", "last_call", "abstained",
                            "superseded", "coerced", "submit_rejected",
                            "last_call_repair", "submission_fallback",
                            "host_submission", "typed_questions",
                            "compiled_questions", "engine_answers",
                            "host_data_binding")}
                for entry in self.trace
            ],
            # Harness-only provenance used by cross-benchmark adapters to
            # verify the artifact the agent actually submitted. Paths never
            # enter the model transcript or submission schema.
            "artifact_paths": sorted(self.artifact_paths),
        }

    # -- the loop ----------------------------------------------------------
    def drive(self) -> dict[str, Any]:
        self.session.initialize()
        submit_tool = self._submit_tool()
        product_tools = openai_tool_specs(self.session.list_tools(), None)
        if self.row.get("_host_compiled_forecast"):
            bound_name = preferred_execution_tool(
                self.profile, bool(getattr(self, "target_keys", None)),
                host_compiled=True)
            for spec in product_tools:
                function = spec.get("function") or {}
                if function.get("name") == bound_name:
                    function["description"] = (
                        "Execute the host-compiled forecast request. The "
                        "validated data, schema, horizon, context, and "
                        "covariates are already bound; pass no arguments.")
                    function["parameters"] = {
                        "type": "object", "additionalProperties": False}
        harness_tools = openai_tool_specs([], submit_tool)
        tools = [*product_tools, *harness_tools]
        self.product_schema_bytes = len(json.dumps(
            product_tools, separators=(",", ":")))
        self.harness_schema_bytes = len(json.dumps(
            harness_tools, separators=(",", ":")))
        self.schema_bytes = len(json.dumps(tools, separators=(",", ":")))
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
            tool_choice: Any = "auto"
            if self.mcp_calls == 0:
                preferred = preferred_execution_tool(
                    self.profile, bool(getattr(self, "target_keys", None)),
                    host_compiled=bool(self.row.get(
                        "_host_compiled_forecast")))
                if preferred:
                    tool_choice = {"type": "function", "function": {
                        "name": preferred}}
            response = self.client.chat(messages, n=1, tools=tools,
                                        tool_choice=tool_choice)
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
                # A result that retires an earlier one (same forecast
                # channels, or an identical repeat call) compacts the
                # older message to its disclosures + artifact_path, so
                # dead bulk stops being re-sent every remaining round.
                compacted = self.result_log.record(
                    call["function"]["name"], arguments, messages[-1])
                if compacted:
                    self.trace.append({"superseded": compacted})
                if self.complete_artifact_ready and not self.submission:
                    if (self.row.get("_host_compiled_forecast")
                            and self.row.get("_require_gnomon_execution")
                            and not self.row.get(
                                "_require_context_explanation")
                            and self.artifact_paths
                            and hasattr(self, "target_keys")
                            # Typed questions require the model's distinct
                            # synthesis submission. Auto-submitting the
                            # forecast artifact with an empty MCQ object would
                            # erase exactly the advisory lane being measured.
                            and not self.temporal_compilation.get("questions")):
                        artifact_path = sorted(self.artifact_paths)[-1]
                        accepted = self._handle_submit({
                            "forecast": {
                                channel: {"artifact_path": artifact_path}
                                for channel in self.target_keys},
                            "mcq": {},
                        })
                        if accepted.get("accepted") and self.submission:
                            self.trace.append({
                                "host_submission": "complete_artifact"})
                            return self._resolve_submission()
                    return self._last_call(
                        messages, submit_tool,
                        "forecast artifact ready; engine browsing closed")
                if self.complete_description_ready and not self.submission:
                    return self._last_call(
                        messages, submit_tool,
                        "typed descriptive answers ready; engine browsing closed")
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

        A rejected submission gets :data:`LAST_CALL_ATTEMPTS` - 1 repair
        rounds. One shot used to mean that an answer the model HAD
        produced, mis-enveloped, died together with the correct
        rejection message explaining how to fix it — the row recorded as
        an abstention it never made. The repair round withdraws the
        engine tools exactly as the first one does, so nothing new can
        be computed on a spent budget; only the envelope may change.
        """
        self.trace.append({"last_call": cap})
        messages = messages + [{"role": "user",
                                "content": LAST_CALL.format(cap=cap)}]
        for attempt in range(LAST_CALL_ATTEMPTS):
            try:
                response = self.client.chat(messages, n=1,
                                            tools=[submit_tool],
                                            tool_choice={
                                                "type": "function",
                                                "function": {
                                                    "name": "submit_answer"},
                                            })
            except OpenRouterError:
                # Transport exhaustion is not an agent abstention.  Let the
                # runner classify and retry it as infrastructure failure.
                raise
            except Exception as error:
                return self._abstain_outcome(
                    f"{cap}; last call failed: {error}")
            message = response.choices[0].message
            calls = _tool_calls_as_dicts(message)
            rejected: tuple[str, dict[str, Any]] | None = None
            if not calls:
                # Prose instead of a tool call is the same defect class
                # as a mis-enveloped one: the model answered, and only
                # the envelope is wrong. Repair it rather than record an
                # abstention the model never made.
                rejected = (json.dumps({
                    "accepted": False, "authored_by": "harness",
                    "problems": ["no submit_answer call — a prose answer "
                                 "cannot be scored"],
                }), None)
            for call in calls:
                if call["function"]["name"] != "submit_answer":
                    continue
                try:
                    arguments = json.loads(
                        call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    rejected = (json.dumps({
                        "accepted": False, "authored_by": "harness",
                        "problems": ["arguments were not valid JSON"],
                    }), call)
                    continue
                result = self._dispatch("submit_answer", arguments)
                if self.submission:
                    self.submission["last_call"] = cap
                    return self._resolve_submission()
                if isinstance(result, dict):  # transport death, disclosed
                    return result
                rejected = (result, call)
            if rejected is None or attempt + 1 >= LAST_CALL_ATTEMPTS:
                break
            text, call = rejected
            self.trace.append({"last_call_repair": attempt + 1})
            if call is None:  # prose: no tool_call_id to answer
                messages = messages + [
                    {"role": "assistant", "content": message.content or ""},
                    {"role": "user", "content": LAST_CALL_PROSE},
                ]
            else:
                messages = messages + [
                    {"role": "assistant", "content": message.content or None,
                     "tool_calls": calls},
                    {"role": "tool", "tool_call_id": call["id"],
                     "content": text},
                    {"role": "user", "content": LAST_CALL_REPAIR},
                ]
        if self.complete_artifact_ready and self.artifact_paths \
                and hasattr(self, "target_keys"):
            artifact_path = sorted(self.artifact_paths)[-1]
            fallback = {
                "forecast": {channel: {"artifact_path": artifact_path}
                             for channel in self.target_keys},
                "mcq": {},
                "reasoning": ("Harness preserved a complete verified artifact "
                              "after final submission formatting failed."),
            }
            accepted = self._handle_submit(fallback)
            if accepted.get("accepted") and self.submission:
                self.trace.append({"submission_fallback": "complete_artifact"})
                self.submission["last_call"] = cap
                return self._resolve_submission()
        if hasattr(self, "target_keys"):
            fallback = {
                "forecast": {channel: {"abstain": True}
                             for channel in self.target_keys},
                "mcq": {},
                "reasoning": ("Harness normalized the final malformed "
                              f"submission into typed abstentions: {cap}."),
            }
            accepted = self._handle_submit(fallback)
            if accepted.get("accepted") and self.submission:
                self.trace.append({"submission_fallback": "typed_abstention"})
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
            arguments, coerced = coerce_json_containers(arguments)
            if coerced:
                entry["coerced"] = coerced
            payload = self._handle_submit(arguments)
            entry["result"] = {"accepted": payload.get("accepted"),
                               "problems": payload.get("problems")}
            if payload.get("accepted") is False:
                entry["submit_rejected"] = payload.get("problems") or [
                    payload.get("message", "rejected")]
            return json.dumps(payload)

        # The harness has already resolved these fields from the official
        # task. Compile them into the execution call so the model chooses the
        # verb, not an accidental per-channel execution plan. This mirrors the
        # production host compiler and makes one wide-series request one run.
        if (name == "gnomon_describe" and self.profile == "evidence"
                and getattr(self, "channels", None)):
            # The host already resolved the task arrays into one lossless
            # long-form panel. Bind descriptive calls to that same data
            # contract, just as forecast calls are bound below. The model
            # still chooses the verb and typed questions; schema facts are
            # supplied by the host that owns them.
            questions = arguments.get("questions")
            compiled_questions = (getattr(
                self, "temporal_compilation", {}) or {}).get("questions") or []
            arguments = {
                "input": str(self.csv_path),
                "time_column": "timestamp",
                "target_column": "value",
                "series_column": "series",
                "frequency": getattr(self, "frequency", "h"),
                "format": "brief",
                **({"questions": compiled_questions or questions}
                   if compiled_questions or questions else {}),
            }
            entry["host_data_binding"] = "long_panel"
            entry["typed_questions"] = len(questions or [])
            entry["compiled_questions"] = len(compiled_questions)
        elif (name == "gnomon_forecast" and getattr(self, "target_keys", None)
                and (self.profile == "evidence"
                     or self.row.get("_host_compiled_forecast")
                     or self.row.get("_require_gnomon_execution"))):
            panel_long_form = getattr(
                self, "panel_long_form", self.profile == "evidence")
            questions = arguments.get("questions")
            requested_repair = arguments.get("repair")
            requested_regrid = arguments.get("regrid")
            arguments = {"input": str(self.csv_path),
                         "time_column": "timestamp",
                         "target_column": ("value" if panel_long_form
                                           else ",".join(self.target_keys)),
                         **({"series_column": "series"}
                            if panel_long_form else {}),
                         "frequency": getattr(self, "frequency", "h"),
                         "horizon": self.horizon,
                         "format": "brief",
                         **({"questions": questions} if questions else {})}
            if requested_repair is not None:
                arguments["repair"] = requested_repair
            if requested_regrid is not None:
                arguments["regrid"] = requested_regrid
            compiled_questions = (getattr(
                self, "temporal_compilation", {}) or {}).get("questions") or []
            if compiled_questions:
                arguments["questions"] = compiled_questions
            entry["typed_questions"] = len(questions or [])
            entry["compiled_questions"] = len(compiled_questions)
        elif (self.profile == "mega" and name == "gnomon_run"
              and getattr(self, "target_keys", None)):
            arguments = {"input": str(self.csv_path),
                         "time_column": "timestamp",
                         "target_column": ",".join(self.target_keys),
                         "frequency": getattr(self, "frequency", "h"),
                         "horizon": self.horizon,
                         "question": {"kind": "forecast"}}
        context_compilation = getattr(self, "context_compilation", {})
        if name in {"gnomon_forecast", "gnomon_run"} \
                and context_compilation.get("events"):
            arguments = {**arguments,
                         "context_events_file": str(self.context_events_file),
                         "future_events": True}
        if name in {"gnomon_forecast", "gnomon_run"} \
                and context_compilation.get("rejected"):
            rejections = []
            for index, rejected in enumerate(
                    context_compilation.get("rejected") or [], 1):
                if not isinstance(rejected, dict):
                    continue
                proposal = rejected.get("proposal") or {}
                problems = rejected.get("problems") or []
                rejections.append({
                    "context_id": str(
                        proposal.get("event_id") or
                        f"compiler-rejection-{index}"),
                    "reason_code": str(
                        rejected.get("reason_code") or "context_unresolved"),
                    "reason": str(problems[0] if problems else
                                  "Context failed deterministic validation."),
                    "source_span": str(
                        proposal.get("evidence_quote") or
                        "Compiler rejected context without a usable quote."),
                })
            if rejections:
                arguments = {**arguments, "context_rejections": rejections}
        covariate_arguments = getattr(self, "covariate_arguments", {})
        if name in {"gnomon_forecast", "gnomon_run"} \
                and covariate_arguments:
            arguments = {**arguments, **covariate_arguments}
        admission_registry = getattr(self, "model_evidence_registry", None)
        if name == "gnomon_forecast" and admission_registry:
            arguments = {
                **arguments,
                "model_admission": "evidence_weighted",
                "model_evidence_registry": str(admission_registry),
            }
            entry["model_admission"] = "evidence_weighted"

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
        if getattr(self, "sufficient_at_call", None) is not None:
            self.redundant_calls = getattr(self, "redundant_calls", 0) + 1
        self.mcp_calls += 1
        try:
            result = self.session.call_tool(name, arguments)
        except Exception as error:
            # Transport death is a harness failure, disclosed as such.
            return self._abstain_outcome(f"mcp transport failed: {error}")
        entry["is_error"] = bool(result.get("isError"))
        if (name == "gnomon_describe" and not entry["is_error"]
                and entry.get("host_data_binding") == "long_panel"
                and entry.get("compiled_questions", 0) > 0):
            self.complete_description_ready = True
        structured = result.get("structuredContent") or {}
        if isinstance(structured, dict):
            publication = structured.get("publication") or {}
            dispositions = publication.get("context_dispositions") or []
            context_summary = publication.get("context_summary") or {}
            if dispositions and getattr(self, "target_keys", None):
                rejected_codes = [
                    str(item.get("reason_code")) for item in dispositions
                    if isinstance(item, dict) and item.get("reason_code")]
                status = str(context_summary.get("status") or "rejected")
                for channel in self.target_keys:
                    self.context_execution[channel] = {
                        "status": status,
                        "considered": 1,
                        "admitted": int(status == "used"),
                        "rejected": sum(
                            1 for item in dispositions
                            if isinstance(item, dict)
                            and item.get("disposition") == "rejected"),
                        "scenario_only": sum(
                            1 for item in dispositions
                            if isinstance(item, dict)
                            and item.get("disposition") == "scenario"),
                        "applied": int(status == "used"),
                        "support": str(publication.get(
                            "recommended_support") or ""),
                        "automation_eligible": (
                            publication.get("automation") or {}).get(
                                "eligible"),
                        "canonical_primary_preserved": publication.get(
                            "primary_forecast_unchanged"),
                        "selected_projection_differs_from_primary": False,
                        "relationship_to_primary": context_summary.get(
                            "relationship_to_primary"),
                        "selected_output_role": context_summary.get(
                            "selected_output_role"),
                        "authority_summary": (
                            publication.get("recommendation_authority") or {}
                        ).get("reason"),
                        "admitted_event_ids": [],
                        "rejection_codes": rejected_codes,
                        "gate_reasons": [
                            str(item.get("reason")) for item in dispositions
                            if isinstance(item, dict) and item.get("reason")
                        ][:8],
                        "source_evidence": [
                            dict(item["source_evidence"])
                            for item in dispositions
                            if isinstance(item, dict)
                            and isinstance(item.get("source_evidence"), dict)
                        ][:8],
                        "scenario_consequence_summaries": [],
                    }
            sufficiency = ((structured.get("reasoning") or {}).get(
                "sufficiency") or {})
            if (getattr(self, "sufficient_at_call", None) is None
                    and sufficiency.get("requires_follow_up") is False
                    and sufficiency.get("sufficient_for")):
                self.sufficient_at_call = self.mcp_calls
            if (name == "gnomon_describe" and not result.get("isError")
                    and entry.get("compiled_questions", 0) > 0):
                answers = structured.get("answers") or []
                if isinstance(answers, list):
                    receipt = {
                        "source": "inline_describe",
                        # Describe creates no primary forecast, so forecast
                        # immutability is not applicable rather than a
                        # tautological pass.
                        "primary_forecast_unchanged": None,
                        "answers": [item for item in answers
                                    if isinstance(item, dict)],
                    }
                    # A repeated identical call must not multiply the
                    # denominator.  Keep one receipt per answer-id set.
                    answer_ids = tuple(str((item.get("question") or {}).get(
                        "id") or "") for item in receipt["answers"])
                    if not any(
                        tuple(str((item.get("question") or {}).get("id") or "")
                              for item in prior.get("answers") or [])
                        == answer_ids
                        for prior in self.descriptive_answer_receipts
                    ):
                        self.descriptive_answer_receipts.append(receipt)
                    entry["engine_answers"] = len(receipt["answers"])
            code = ((structured.get("error") or {}).get("code")
                    or structured.get("code"))
            if code:
                entry["code"] = code
                entry["message"] = ((structured.get("error") or {}).get("message")
                                    or structured.get("message"))
            if not result.get("isError") and structured.get("artifact_path"):
                artifact_path = str(structured["artifact_path"])
                self.artifact_paths.add(artifact_path)
                # A brief multi-series envelope contains only its notable
                # top-k rows.  The artifact is the publication contract and
                # retains every series, so response triage must never be
                # mistaken for an incomplete run (which otherwise prompts
                # the agent to repeat an already successful forecast).
                try:
                    from gnomon.artifacts import read_artifact

                    artifact = read_artifact(artifact_path)
                    result_names = {
                        str(row.get("series"))
                        for row in artifact.get("results", [])
                        if isinstance(row, dict)
                    }
                except Exception as error:
                    entry["artifact_validation_error"] = str(error)
                    result_names = set()
                targets = set(getattr(self, "target_keys", []))
                complete = targets <= result_names
                # The runtime's canonical identity for a one-target artifact
                # is __default__; the benchmark retains the source column's
                # public name. Treat this as the same resolved series here,
                # just as submission resolution already does.
                if len(targets) == 1 and result_names == {"__default__"}:
                    complete = True
                if complete:
                    self.complete_artifact_ready = True
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
                 session_factory: Any = None, work_dir: str | None = None,
                 profile: str = "full", compile_context: bool = False,
                 context_receipts_dir: str | None = None,
                 compile_questions: bool = False,
                 question_receipts_dir: str | None = None,
                 model_evidence_registry: str | None = None):
        meta = row.get("meta") or {}
        self.horizon = int(meta.get("n_horizon") or 0)
        if self.horizon < 1:
            raise ValueError("row has no forecast horizon")
        ground_truth = row.get("ground_truth")
        self.target_keys = (list(ground_truth) if isinstance(ground_truth, dict)
                            else [meta.get("main_key")])
        super().__init__(row, client, session_factory=session_factory,
                         work_dir=work_dir, profile=profile,
                         compile_context=compile_context,
                         context_receipts_dir=context_receipts_dir,
                         compile_questions=compile_questions,
                         question_receipts_dir=question_receipts_dir)
        self.model_evidence_registry = None
        if model_evidence_registry:
            import shutil
            source = Path(model_evidence_registry).resolve()
            target = self.jail / "model-evidence.json"
            shutil.copyfile(source, target)
            self.model_evidence_registry = target

    def _row_channels(self, row: dict[str, Any]) -> dict[str, list[float]]:
        arrays = prompt_input_arrays(row)
        return {key: _observed(arrays.get(key, [])) for key in self.target_keys}

    def _submit_tool(self) -> dict[str, Any]:
        if self.row.get("_require_gnomon_execution"):
            # Product-contract rows need only bind the already-produced
            # artifact. An open-ended reasoning field caused some providers
            # to spend thousands of tokens inside tool arguments and truncate
            # an otherwise trivial final submission.
            tool = json.loads(json.dumps(SUBMIT_TOOL))
            parameters = tool["function"]["parameters"]
            parameters["required"] = ["forecast"]
            if self.row.get("_require_context_explanation"):
                # A bounded synthesis is the object under test for context
                # surfaces. Keep the historical anti-runaway guard while
                # requiring enough text to verify that the model preserved
                # disposition, uncertainty, and authority.
                parameters["properties"]["reasoning"]["maxLength"] = 600
                parameters["properties"]["cited_context_gate_codes"] = {
                    "type": "array",
                    "description": (
                        "Exact failed_gate_codes and reason_code values only "
                        "from context_dispositions whose disposition is "
                        "rejected or scenario. Do not cite successful "
                        "reason_code values such as applied; use an empty "
                        "array when no admission/rejection gate failed."),
                    "items": {"type": "string"},
                    "maxItems": 8,
                }
                parameters["properties"]["context_automation_eligible"] = {
                    "type": "boolean",
                    "description": (
                        "Copy context_outcome.automation_eligible exactly. "
                        "False means context evidence alone cannot authorize "
                        "automation; do not weaken it to 'not requested'."),
                }
                parameters["properties"]["canonical_primary_preserved"] = {
                    "type": "boolean",
                    "description": (
                        "Copy context_outcome.canonical_primary_preserved "
                        "exactly; conditional projection changes do not mutate "
                        "the canonical primary."),
                }
                parameters["properties"]["cited_scenario_consequences"] = {
                    "type": "array",
                    "description": (
                        "Copy each consequence_summary exposed by Gnomon for "
                        "a represented numeric scenario; use an empty array "
                        "when no numeric scenario consequence is present."),
                    "items": {"type": "string"},
                    "maxItems": 8,
                }
                parameters["properties"]["cited_context_sources"] = {
                    "type": "array",
                    "description": (
                        "Copy every source.reference exposed in Gnomon's "
                        "context source_evidence for considered context. Use "
                        "an empty array when no validated source reference is "
                        "present; never infer or paraphrase a source."),
                    "items": {"type": "string"},
                    "maxItems": 8,
                }
                parameters["properties"]["reasoning"]["description"] = (
                    "Human-facing explanation of Gnomon's typed context "
                    "outcome. Scenario representation is not numeric "
                    "admission: say represented or retained as a scenario, "
                    "never admitted as a scenario, when admitted/applied "
                    "counts are zero.")
                parameters["required"].append("reasoning")
                parameters["required"].append("cited_context_gate_codes")
                parameters["required"].append(
                    "context_automation_eligible")
                parameters["required"].append(
                    "canonical_primary_preserved")
                parameters["required"].append(
                    "cited_scenario_consequences")
                parameters["required"].append("cited_context_sources")
            else:
                parameters["properties"].pop("reasoning", None)
            if getattr(self, "temporal_compilation", {}).get("questions"):
                # A forecast-only auto-submit would erase the agent layer.
                # Empty choice_basis is valid when every canonical answer is
                # retained, but making the field explicit forces the host and
                # model to acknowledge the dual contract.
                parameters["required"].extend(["mcq", "choice_basis"])
            parameters["additionalProperties"] = False
            return tool
        return SUBMIT_TOOL

    def _system(self) -> str:
        data_layout = (
            "long-form columns: timestamp, series, value"
            if self.panel_long_form else
            "wide columns: timestamp, " + ", ".join(self.target_keys))
        schema_binding = (
            "time_column is `timestamp`, target_column is `value`, "
            "series_column is `series`, and frequency is `h`"
            if self.panel_long_form else
            "time_column is `timestamp`, target_column is exactly `"
            + ",".join(self.target_keys) + "`, and frequency is `h`")
        batching_contract = (
            "the long-form `series` column identifies every channel while "
            "`target_column=value` and `series_column=series` bind the panel"
            if self.panel_long_form else
            "`target_column` takes the comma list \""
            + ",".join(self.target_keys) + "\"")
        text = SYSTEM.format(
            csv_path=str(self.csv_path),
            channels=", ".join(self.target_keys),
            channel_list=",".join(self.target_keys),
            data_layout=data_layout, schema_binding=schema_binding,
            batching_contract=batching_contract,
            epoch=self.epoch.isoformat(), horizon=self.horizon,
            jail_rule=self._jail_rule(),
            max_rounds=MAX_ROUNDS, max_calls=MAX_MCP_CALLS,
        )
        if self.profile == "mega":
            text = text.replace("gnomon_forecast", "gnomon_run with question.kind=forecast")
        elif self.profile == "evidence":
            text += ("\nThis is the precommitted evidence-pack arm: call "
                     "gnomon_forecast once with the history path, timestamp "
                     "column, all channels in one target_column comma list, "
                     "the stated horizon, and format brief; then submit that "
                     "artifact. No inspection call is available or needed.\n")
        if self.context_compilation.get("attempted"):
            text += ("\nThe host compiled the task narrative through Gnomon's "
                     "quoted-context validator. The complete authoritative "
                     "receipt is host-bound to execution; this bounded summary "
                     "is for routing and provenance only:\n"
                     + json.dumps(compact_context_compilation_for_prompt(
                         self.context_compilation), separators=(",", ":"))
                     + "\n")
        if self.row.get("_require_context_explanation"):
            text += (
                "\nThe final human-facing reasoning must preserve both "
                "authority facts from context_outcome. If "
                "canonical_primary_preserved is true, say the canonical "
                "primary remains preserved. If automation_eligible is false, "
                "say context evidence alone cannot authorize automation; "
                "'not requested' is not equivalent. The verifier rejects a "
                "submission that hides either fact.\n")
            text += (
                "If Gnomon exposes a scenario consequence_summary, copy it "
                "exactly into cited_scenario_consequences and communicate "
                "its first-affected and horizon-end primary-versus-scenario "
                "q50 deltas in the human-facing reasoning. "
                "Do not present those conditional values as the primary.\n")
            text += (
                "Scenario representation is not numeric admission. When "
                "Gnomon reports a scenario but no admitted or applied "
                "context, say represented or retained as a scenario; never "
                "say admitted as a scenario.\n")
            text += (
                "If failed_gate_codes is non-empty, name every cited code in "
                "the human-facing reasoning and explain its practical meaning. "
                "A code copied only into the structured submit field is not a "
                "human-visible explanation.\n")
            text += (
                "For a rejected or scenario context_disposition, treat its "
                "reason_code the same way: copy the exact engine-authored "
                "code and explain it. Never cite a successful `applied` "
                "reason_code as a failed gate. Use "
                "context_summary.context_evidence_automation_eligible for "
                "context authority; the separate global automation reason "
                "'not requested' does not grant context authority.\n")
            text += (
                "Copy each validated context source.reference exactly into "
                "cited_context_sources and name it in the human-facing "
                "reasoning. Do not invent a source when source_evidence is "
                "absent.\n")
        if self.row.get("_require_gnomon_execution"):
            text += ("\nThis product run delegates every published numeric "
                     "trajectory to Gnomon. Submit the forecast artifact "
                     "verbatim; model-authored values are not an allowed "
                     "fallback. If the artifact abstains, retry its disclosed "
                     "best-effort action or submit an abstention.\n")
        return text

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
        schema = ((data.get("task") or {}).get("schema") or {})
        target = schema.get("target_column")
        series_column = schema.get("series_column")
        targets = [name.strip() for name in str(target).split(",")]
        result = next((item for item in results
                       if item.get("series") == channel), None)
        # Long-form panels retain the loader's immutable ``target:group``
        # artifact identity.  Bind it from the artifact schema rather than
        # guessing or accepting arbitrary suffix matches.
        if result is None and series_column and len(targets) == 1:
            internal_name = f"{targets[0]}:{channel}"
            result = next((item for item in results
                           if item.get("series") == internal_name), None)
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
        scenarios = result.get("sensitivity_scenarios") or []
        if scenarios:
            scenario_rows = scenarios[0].get("forecast") or []
            if len(scenario_rows) == self.horizon:
                self._available_sensitivity[channel] = [
                    float(row.get("q50", row["point"]))
                    for row in scenario_rows
                ]
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
        gate = result.get("future_context") or {}
        disposition = result.get("context_outcome") or {}
        historical_gate = result.get("context") or {}
        covariate_gate = result.get("covariates") or {}
        if covariate_gate:
            self.covariate_execution[channel] = {
                "considered": bool(covariate_gate.get("considered")),
                "admitted": bool(covariate_gate.get("admitted")),
                "retained": list(covariate_gate.get("retained") or []),
                "rejected": list(covariate_gate.get("rejected") or []),
            }
        if gate or disposition or historical_gate:
            admitted = gate.get("admitted") or []
            rejected = gate.get("rejected") or []
            status = disposition.get("status")
            outcome_events = disposition.get("events") or []
            scenario_dispositions = sum(
                1 for item in (disposition.get("dispositions") or [])
                if isinstance(item, dict)
                and item.get("disposition") == "scenario"
            )
            self.context_execution[channel] = {
                "status": status,
                "considered": int(bool(gate.get("considered")
                                       or historical_gate.get("considered"))),
                "admitted": (len(admitted) if admitted else
                             len(outcome_events) if status == "applied" else 0),
                "rejected": (len(rejected) if rejected else
                             len(outcome_events) if status == "rejected" else 0),
                "scenario_only": (
                    len(outcome_events) if status == "scenario_only"
                    else scenario_dispositions),
                # The runtime makes influence explicit by lowering support to
                # context_trusted only when at least one admitted event was
                # actually applied to the emitted trajectory.
                "applied": (len(outcome_events) if status == "applied" else 0),
                "support": str(support),
                "automation_eligible": disposition.get(
                    "automation_eligible"),
                "canonical_primary_preserved": (
                    bool(disposition.get("canonical_primary_preserved", True))
                    if status else None),
                "selected_projection_differs_from_primary": (
                    disposition.get("selected_projection_differs_from_primary",
                                    disposition.get(
                                        "primary_forecast_changed", False))
                    if status else None),
                "relationship_to_primary": disposition.get(
                    "relationship_to_primary"),
                "selected_output_role": disposition.get(
                    "selected_output_role"),
                "authority_summary": disposition.get("authority_summary"),
                "admitted_event_ids": [
                    str(item.get("event_id")) for item in admitted
                    if isinstance(item, dict) and item.get("event_id")
                ],
                "rejection_codes": list(dict.fromkeys([
                    str(item.get("code")) for item in rejected
                    if isinstance(item, dict) and item.get("code")
                ] + [str(code) for code in
                     (disposition.get("failed_gate_codes") or [])])),
                "gate_reasons": [str(reason) for reason in
                                 (disposition.get("gate_reasons") or [])][:8],
                "source_evidence": [
                    dict(item) for item in
                    (disposition.get("context_evidence") or [])
                    if isinstance(item, dict)
                ][:8],
                "scenario_consequence_summaries": [
                    str(scenario.get("consequence_summary"))
                    for scenario in (result.get("sensitivity_scenarios") or [])
                    if isinstance(scenario, dict)
                    and scenario.get("consequence_summary")
                ][:8],
            }
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
        # ContextBench delegates every published number to Gnomon. Merely
        # calling a tool before substituting model-made values is still a
        # bypass, and was the sole source of publication-parity failures in
        # the natural-routing matrix.
        if (self.row.get("_require_gnomon_execution")
                or self.row.get("_host_compiled_forecast")):
            direct_channels = [
                str(channel) for channel, entry in forecast_spec.items()
                if isinstance(entry, dict) and entry.get("values") is not None
            ]
            if direct_channels:
                problems.append(
                    "host_execution_required: submit a Gnomon artifact or "
                    "abstain; model-authored forecast values bypass "
                    "the product contract for channel(s): "
                    + ", ".join(sorted(direct_channels))
                )
        resolved: dict[str, list[float]] = {}
        support: dict[str, str] = {}
        routes: dict[str, str] = {}
        self._pending_support: dict[str, str] = {}
        self._available_sensitivity: dict[str, list[float]] = {}
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
        raw_mcq = arguments.get("mcq")
        if raw_mcq is None:
            raw_mcq = {}
        if not isinstance(raw_mcq, Mapping):
            return {
                "accepted": False,
                "authored_by": "harness",
                "problems": [
                    "mcq must be an object mapping question ids to answers"
                ],
            }
        synthesized_mcq = {str(key): str(value)
                           for key, value in raw_mcq.items()}
        mcq = dict(synthesized_mcq)
        projected = self._project_receipt_choices()
        canonical_mcq = {key: value["display_value"]
                         for key, value in projected.items()}
        raw_basis = arguments.get("choice_basis") or {}
        if not isinstance(raw_basis, Mapping):
            return {"accepted": False, "authored_by": "harness",
                    "problems": ["choice_basis must be an object"]}
        accepted_overrides: dict[str, dict[str, str]] = {}
        binding = {key: value for key, value in projected.items()
                   if value["authority"] == "binding"}
        if binding:
            # A supported automation-eligible result remains software's
            # answer. Weak evidence is advisory: retain the model's explicit
            # synthesis and score it separately without mutating the receipt.
            mcq.update({key: value["display_value"]
                        for key, value in binding.items()})
        for key, value in projected.items():
            proposed = synthesized_mcq.get(key)
            differs = (proposed is not None and
                       str(proposed).strip().lower()
                       != str(value["display_value"]).strip().lower())
            if value["authority"] == "advisory" and differs:
                basis = raw_basis.get(key) if isinstance(raw_basis, Mapping) else None
                kind = str((basis or {}).get("kind") or "") \
                    if isinstance(basis, Mapping) else ""
                evidence = str((basis or {}).get("evidence") or "").strip() \
                    if isinstance(basis, Mapping) else ""
                matches_adjudicated_alternative = (
                    value.get("has_computed_opposition") is True
                    and str(proposed).strip().lower() == str(
                        value.get("computed_alternative") or ""
                    ).strip().lower()
                )
                valid = (
                    kind == "computed_opposition"
                    and matches_adjudicated_alternative
                    and bool(evidence)
                ) or (
                    kind == "task_context" and bool(evidence)
                    and evidence in str(self.row.get("prompt") or "")
                    # A quote establishes provenance, not predictive truth.
                    # Context may govern publication only after the engine's
                    # outcome-backed adjudicator validates the same choice.
                    and matches_adjudicated_alternative
                )
                if valid:
                    accepted_overrides[key] = {"kind": kind,
                                               "evidence": evidence}
                else:
                    mcq[key] = value["display_value"]
            if key not in mcq:
                # No synthesis was supplied. Canonical weak evidence remains
                # visible as the fallback, explicitly labelled advisory.
                mcq[key] = value["display_value"]
        if projected:
            self.trace.append({
                "tool": "host_choice_projection",
                "host_submission": "support_tiered_temporal_answer",
                "projected": projected,
            })
        self.submission = {
            "forecast": resolved, "mcq": mcq, "support": support,
            "routes": routes, "reasoning": arguments.get("reasoning"),
            "sensitivity_forecast": dict(self._available_sensitivity),
            "canonical_mcq": canonical_mcq,
            "synthesized_mcq": synthesized_mcq,
            "choice_authority": {
                key: ("advisory_override" if key in accepted_overrides else
                      "advisory_canonical_default"
                      if value["authority"] == "advisory" else "binding")
                for key, value in projected.items()},
            "choice_basis": accepted_overrides,
        }
        if self.row.get("_require_context_explanation"):
            supplied_codes = arguments.get("cited_context_gate_codes") or []
            if not isinstance(supplied_codes, list):
                supplied_codes = []
            expected_codes = sorted({
                str(code)
                for outcome in self.context_execution.values()
                for code in (outcome.get("rejection_codes") or [])
            })
            supplied = [str(code) for code in supplied_codes[:8]]
            self.submission["context_gate_citations"] = {
                "expected": expected_codes,
                "supplied": supplied,
                "matched": [code for code in supplied
                            if code in expected_codes],
                "invalid": [code for code in supplied
                            if code not in expected_codes],
            }
            engine_automation = {
                value.get("automation_eligible")
                for value in self.context_execution.values()
                if value.get("automation_eligible") is not None
            }
            supplied_automation = arguments.get(
                "context_automation_eligible")
            self.submission["context_automation_projection"] = {
                "engine": (next(iter(engine_automation))
                           if len(engine_automation) == 1 else None),
                "supplied": supplied_automation,
                "matched": (len(engine_automation) == 1 and
                            supplied_automation in engine_automation),
            }
            engine_primary = {
                value.get("canonical_primary_preserved")
                for value in self.context_execution.values()
                if value.get("canonical_primary_preserved") is not None
            }
            supplied_primary = arguments.get(
                "canonical_primary_preserved")
            self.submission["context_primary_projection"] = {
                "engine": (next(iter(engine_primary))
                           if len(engine_primary) == 1 else None),
                "supplied": supplied_primary,
                "matched": (len(engine_primary) == 1 and
                            supplied_primary in engine_primary),
            }
            expected_consequences = sorted({
                summary for outcome in self.context_execution.values()
                for summary in outcome.get(
                    "scenario_consequence_summaries", [])
            })
            supplied_consequences = [str(value) for value in
                                     (arguments.get(
                                         "cited_scenario_consequences") or [])[:8]]
            self.submission["context_consequence_projection"] = {
                "expected": expected_consequences,
                "supplied": supplied_consequences,
                "matched": [value for value in supplied_consequences
                            if value in expected_consequences],
                "invalid": [value for value in supplied_consequences
                            if value not in expected_consequences],
            }
            expected_sources = sorted({
                str((evidence.get("source") or {}).get("reference"))
                for outcome in self.context_execution.values()
                for evidence in outcome.get("source_evidence", [])
                if isinstance(evidence, dict)
                and isinstance(evidence.get("source"), dict)
                and (evidence.get("source") or {}).get("reference")
            })
            supplied_sources = [str(value) for value in
                                (arguments.get(
                                    "cited_context_sources") or [])[:8]]
            self.submission["context_source_projection"] = {
                "expected": expected_sources,
                "supplied": supplied_sources,
                "matched": [value for value in supplied_sources
                            if value in expected_sources],
                "invalid": [value for value in supplied_sources
                            if value not in expected_sources],
            }
            reasoning_text = str(arguments.get("reasoning") or "").lower()
            authority_problems: list[str] = []
            scenario_count = sum(
                int(outcome.get("scenario_only") or 0)
                for outcome in self.context_execution.values())
            admitted_count = sum(
                int(outcome.get("admitted") or 0)
                for outcome in self.context_execution.values())
            applied_count = sum(
                int(outcome.get("applied") or 0)
                for outcome in self.context_execution.values())
            scenario_admission_conflated = bool(re.search(
                r"(?:\badmitted\s+(?:only\s+)?as\s+(?:an?\s+)?scenario\b|"
                r"\bscenario\s+(?:was|is|has been)\s+admitted\b)",
                reasoning_text,
            ))
            if (scenario_count and not admitted_count and not applied_count
                    and scenario_admission_conflated):
                authority_problems.append(
                    "scenario_admission_conflated: scenario representation "
                    "is not numeric admission; say represented or retained "
                    "as a scenario, with the canonical primary unchanged")
            context_authority_stated = (
                "context_evidence_automation_eligible=false" in reasoning_text
                or "context_automation_eligible=false" in reasoning_text
                or ("context" in reasoning_text and any(
                    token in reasoning_text for token in (
                        "cannot automate", "cannot authorize",
                        "does not authorize", "no automation authority"))))
            if engine_automation == {False} and not context_authority_stated:
                authority_problems.append(
                    "context_authority_omitted: state that context evidence "
                    "alone cannot authorize automation; 'not requested' is "
                    "not equivalent")
            if (engine_automation == {False} and "not requested" in reasoning_text
                    and not context_authority_stated):
                authority_problems.append(
                    "context_authority_misattributed: automation is ineligible "
                    "because context evidence lacks authority, not because "
                    "automation was not requested")
            if (engine_primary == {True} and not (
                    "primary" in reasoning_text and any(
                        token in reasoning_text for token in (
                            "unchanged", "preserved", "remain",
                            "did not change")))):
                authority_problems.append(
                    "canonical_primary_omitted: state that the canonical "
                    "primary forecast remains preserved")
            citations = self.submission["context_gate_citations"]
            if citations["invalid"] or (
                    citations["expected"] and not citations["matched"]):
                authority_problems.append(
                    "context_gate_citation_invalid: cite exactly these "
                    "Gnomon context codes and no others: " +
                    (", ".join(citations["expected"]) or "(none)"))
            prose_missing_codes = [
                code for code in citations["expected"]
                if code.casefold() not in reasoning_text
            ]
            if prose_missing_codes:
                authority_problems.append(
                    "context_gate_not_human_visible: name these failed gate "
                    "codes in the reasoning and explain what they mean: " +
                    ", ".join(prose_missing_codes))
            consequences = self.submission["context_consequence_projection"]
            if consequences["invalid"] or (
                    consequences["expected"] and
                    set(consequences["matched"]) !=
                    set(consequences["expected"])):
                authority_problems.append(
                    "scenario_consequence_omitted: copy every Gnomon "
                    "consequence_summary exactly and communicate its "
                    "conditional q50 endpoints without calling them primary")
            sources = self.submission["context_source_projection"]
            if sources["invalid"] or (
                    sources["expected"] and
                    set(sources["matched"]) != set(sources["expected"])):
                authority_problems.append(
                    "context_source_omitted: copy every validated context "
                    "source.reference exactly and do not invent sources: "
                    + (", ".join(sources["expected"]) or "(none)"))
            prose_missing_sources = [
                source for source in sources["expected"]
                if source.casefold() not in reasoning_text
            ]
            if prose_missing_sources:
                authority_problems.append(
                    "context_source_not_human_visible: name these validated "
                    "context sources in the reasoning: "
                    + ", ".join(prose_missing_sources))
            if authority_problems:
                self.submission = None
                return {"accepted": False, "authored_by": "harness",
                        "problems": authority_problems,
                        "ready_to_retry": {
                            "tool": "submit_answer",
                            "reuse_forecast_artifact": True,
                            "rerun_gnomon": False,
                        }}
        return {"accepted": True, "routes": routes}

    def _project_receipt_choices(self) -> dict[str, dict[str, Any]]:
        """Project immutable engine answers into explicit task vocabulary."""
        from gnomon.temporal_vocabulary import project_temporal_choice

        question_keys = list((self.row.get("mcq") or {}).keys())
        aligned: dict[str, dict[str, Any]] = {}
        for artifact_path in sorted(self.artifact_paths):
            receipt_path = Path(artifact_path) / "temporal_answers.json"
            if not receipt_path.is_file():
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for answer in receipt.get("answers") or []:
                raw_id = str((answer.get("question") or {}).get("id") or "")
                base_id = raw_id.split(":", 1)[0]
                key = base_id if base_id in question_keys else None
                if key is None and base_id.startswith("q") and base_id[1:].isdigit():
                    index = int(base_id[1:]) - 1
                    if 0 <= index < len(question_keys):
                        key = question_keys[index]
                if key is not None:
                    aligned[key] = answer
        projected: dict[str, dict[str, str]] = {}
        for key, answer in aligned.items():
            best = answer.get("best_estimate") or {}
            # Supported automation-eligible answers bind the published
            # software answer. Weak estimates are advisory evidence for a
            # separately labelled model synthesis. A genuine abstention has
            # no projectable canonical value.
            options = ((self.row.get("mcq") or {}).get(key) or {}).get(
                "options") or []
            choice = project_temporal_choice(best.get("value"), options)
            if choice is not None:
                reasoning = (answer.get("answer") or {}).get("reasoning") or {}
                adjudication = reasoning.get("adjudication") or {}
                synthesis = adjudication.get("synthesis_eligibility") or {}
                alternative = adjudication.get("alternative") or {}
                alternative_choice = project_temporal_choice(
                    alternative.get("value"), options)
                computed_alternative = (
                    alternative_choice.get("display_value")
                    if alternative_choice else None)
                projected[key] = {
                    **choice,
                    "support": str(best.get("support") or "unknown"),
                    "automation_eligible": bool(
                        best.get("automation_eligible") is True),
                    "authority": ("binding" if
                                  best.get("support") == "supported"
                                  and best.get("automation_eligible") is True
                                  else "advisory"),
                    "has_computed_opposition": bool(
                        synthesis.get("eligible") is True
                        and computed_alternative is not None),
                    "computed_alternative": computed_alternative,
                    "adjudication_relationship": adjudication.get(
                        "relationship", "unresolved"),
                }
        return projected

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
            "context_execution": self.context_execution,
            "covariate_execution": getattr(self, "covariate_execution", {}),
            "sensitivity_forecast": self.submission.get(
                "sensitivity_forecast", {}),
            "submit_reasoning": self.submission["reasoning"],
            **({"context_gate_citations":
                self.submission["context_gate_citations"]}
               if self.submission.get("context_gate_citations") is not None
               else {}),
            **({"context_automation_projection":
                self.submission["context_automation_projection"]}
               if self.submission.get(
                   "context_automation_projection") is not None else {}),
            **({"context_primary_projection":
                self.submission["context_primary_projection"]}
               if self.submission.get(
                   "context_primary_projection") is not None else {}),
            **({"context_consequence_projection":
                self.submission["context_consequence_projection"]}
               if self.submission.get(
                   "context_consequence_projection") is not None else {}),
            **({"context_source_projection":
                self.submission["context_source_projection"]}
               if self.submission.get(
                   "context_source_projection") is not None else {}),
            "canonical_mcq": self.submission.get("canonical_mcq", {}),
            "synthesized_mcq": self.submission.get("synthesized_mcq", {}),
            "choice_authority": self.submission.get("choice_authority", {}),
            "choice_basis": self.submission.get("choice_basis", {}),
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
                 session_factory: Any = None, work_dir: str | None = None,
                 profile: str = "full", compile_context: bool = False,
                 context_receipts_dir: str | None = None,
                 compile_questions: bool = False,
                 question_receipts_dir: str | None = None):
        self.tool, self.answer_rule = mcq_submit_tool(row)
        self.expected_fields = list(row.get("labels") or {})
        self.expected_count = len(row.get("pack") or [])
        self.is_pack = row.get("tier") == "T3"
        self.rejections = 0
        super().__init__(row, client, session_factory=session_factory,
                         work_dir=work_dir, profile=profile,
                         compile_context=compile_context,
                         context_receipts_dir=context_receipts_dir,
                         compile_questions=compile_questions,
                         question_receipts_dir=question_receipts_dir)

    def _row_channels(self, row: dict[str, Any]) -> dict[str, list[float]]:
        try:
            arrays = prompt_input_arrays(row)
        except ValueError:
            # T1/T3 read their arrays out of the prompt's Input block; a
            # row without one simply has no CSV, and the prompt says so
            # rather than naming a file that is not there.
            return {}
        if row.get("tier") == "T1":
            # T1 asks about one series and may also include coordinate arrays
            # such as ``time_position_in_day``.  Those are not response
            # variables and must never be materialised as peer series.
            target = str((row.get("meta") or {}).get("main_key") or "")
            values = _observed(arrays.get(target, []))
            return {target: values} if target and values else {}
        return {key: _observed(values) for key, values in arrays.items()
                if _observed(values)}

    def _submit_tool(self) -> dict[str, Any]:
        return self.tool

    def _system(self) -> str:
        if self.csv_path is not None:
            data_rule = (
                f"The task's series are also on disk at {self.csv_path} "
                f"in lossless long form (columns: timestamp, series, value), on a "
                f"synthetic regular hourly axis — observation k of a "
                f"series sits at {self.epoch.isoformat()} + k hours, recorded "
                f"readings laid consecutively. For tools, target_column is "
                f"value and series_column is series. It is the same data the "
                f"task states, without padding shorter series with blanks."
            )
        else:
            data_rule = ("The task states its data in the prompt; nothing "
                         "was written to disk for this row, so a tool call "
                         "would need the numbers passed inline.")
        text = SYSTEM_MCQ.format(
            data_rule=data_rule, jail_rule=self._jail_rule(),
            answer_rule=self.answer_rule,
            max_rounds=MAX_ROUNDS, max_calls=MAX_MCP_CALLS,
        )
        if self.context_compilation.get("attempted"):
            text += ("\nThe complete validated context compiler receipt is "
                     "host-bound to execution. This bounded summary is for "
                     "routing and provenance only:\n"
                     + json.dumps(compact_context_compilation_for_prompt(
                         self.context_compilation), separators=(",", ":"))
                     + "\n")
        return text

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
            work_dir: str | None = None,
            profile: str = "full",
            compile_context: bool = False,
            context_receipts_dir: str | None = None,
            compile_questions: bool = False,
            question_receipts_dir: str | None = None,
            mcp_call_timeout: float | None = None,
            model_evidence_registry: str | None = None) -> dict[str, Any]:
    """Drive one T2/T4 row through the real MCP surface; return the same
    outcome shape ``answer_row`` produces for the other conditions."""
    if session_factory is None:
        _ensure_checkout_importable()
        session_factory = lambda jail: StdioMcpSession(
            jail, command=[sys.executable, "-m", "gnomon", "mcp", "serve",
                           "--profile", profile],
            call_timeout=mcp_call_timeout)
    return _drive(_Run(row, client, session_factory=session_factory,
                       work_dir=work_dir, profile=profile,
                       compile_context=compile_context,
                       context_receipts_dir=context_receipts_dir,
                       compile_questions=compile_questions,
                       question_receipts_dir=question_receipts_dir,
                       model_evidence_registry=model_evidence_registry))


def mcq_row(row: dict[str, Any], client: Any, *,
            session_factory: Any = None,
            work_dir: str | None = None,
            profile: str = "full",
            compile_context: bool = False,
            context_receipts_dir: str | None = None,
            compile_questions: bool = False,
            question_receipts_dir: str | None = None,
            mcp_call_timeout: float | None = None) -> dict[str, Any]:
    """Drive one T1/T3 row through the same surface with the tier's own
    answer shape; the answer object is what that tier's scorer reads."""
    if session_factory is None:
        _ensure_checkout_importable()
        session_factory = lambda jail: StdioMcpSession(
            jail, command=[sys.executable, "-m", "gnomon", "mcp", "serve",
                           "--profile", profile],
            call_timeout=mcp_call_timeout)
    return _drive(_McqRun(row, client, session_factory=session_factory,
                          work_dir=work_dir, profile=profile,
                          compile_context=compile_context,
                          context_receipts_dir=context_receipts_dir,
                          compile_questions=compile_questions,
                          question_receipts_dir=question_receipts_dir))


def _ensure_checkout_importable() -> None:
    """Keep a jailed MCP child bound to the checkout under evaluation."""
    import os

    repository = Path(__file__).resolve().parents[2]
    roots = [str(repository / "src"), str(repository)]
    for root in reversed(roots):
        if root not in sys.path:
            sys.path.insert(0, root)
    existing = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join([
        *[root for root in roots if root not in existing],
        *[item for item in existing if item],
    ])


# The host binds the complete compiler receipt to the execution call.  The
# tool-driving model needs routing and provenance facts, not a second inline
# copy of every event that will then be re-sent after the tool result.
def compact_context_compilation_for_prompt(
        compilation: Mapping[str, Any]) -> dict[str, Any]:
    events = [item for item in compilation.get("events", [])
              if isinstance(item, Mapping)]
    hypotheses = [item for item in compilation.get("hypotheses", [])
                  if isinstance(item, Mapping)]
    rejected = [item for item in compilation.get("rejected", [])
                if isinstance(item, Mapping)]
    event_types = sorted({str(item.get("event_type")) for item in events
                          if item.get("event_type")})
    known_times = sorted({str(item.get("known_at")) for item in events
                          if item.get("known_at")})
    rejection_codes: dict[str, int] = {}
    for item in rejected:
        code = str(item.get("reason_code") or "unspecified")
        rejection_codes[code] = rejection_codes.get(code, 0) + 1
    return {
        "receipt_id": compilation.get("receipt_id"),
        "accepted_event_count": len(events),
        "accepted_hypothesis_count": len(hypotheses),
        "rejected_count": len(rejected),
        "event_types": event_types[:8],
        **({"event_types_omitted": len(event_types) - 8}
           if len(event_types) > 8 else {}),
        **({"known_at_min": known_times[0], "known_at_max": known_times[-1]}
           if known_times else {}),
        **({"rejection_code_counts": rejection_codes}
           if rejection_codes else {}),
        "execution_binding": "host-bound complete receipt",
    }
