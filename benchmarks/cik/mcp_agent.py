"""The integrated arm: Gnomon as real MCP tools the model may use or skip.

Spec: ``docs/design/cik-mcp-tool-arm.md``. The decision whether to use
Gnomon is made implicitly and observably: the model holds every tool
``gnomon mcp serve``
publishes, verbatim, plus one harness tool (``submit_forecast``), and
whether it calls Gnomon at all is read from the transcript afterwards.

The honesty contract, restated for a free choice: the model never EDITS
a Gnomon number — a submitted artifact is used byte-for-byte or not at
all — and every self-written number is labeled by the route taxonomy
(``gnomon`` / ``direct`` / ``informed-direct``). A breached cap or a
missing submission is a disclosed abstention, never a silent fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.gnomon_forecaster import (  # noqa: E402
    GnomonAbstained,
    GnomonForecaster,
    build_context_text,
    samples_from_quantile_rows,
)
from benchmarks.common.openrouter import (  # noqa: E402
    OpenRouterClient,
    extract_json_objects,
)

MAX_ROUNDS = 10
MAX_MCP_CALLS = 24
MAX_RUN_TOKENS = 250_000
#: Bump when the system prompt, the caps, or the submit contract change:
#: the official cache reuses results by cache_name, and a cached run made
#: under an older contract is a different measurement wearing the same
#: name. Version 2 marks the first release where the cache name carries
#: this and the sampling temperature at all.
#: Version 3: the MCP surface itself changed under the arm — 17-tool
#: default surface, brief-by-default forecasts, the headline field, and
#: response-budget truncation — so rows cached under version 2 measured
#: a different contract.
#: Version 4: superseded tool results are compacted out of the running
#: message history (:class:`ToolMessageLog`), and the tool schemas the
#: model holds slimmed — the conversation a cached row was measured
#: under is not the conversation this version sends.
#: Version 5: the governed Evidence profile host-binds the first valid
#: Gnomon forecast artifact and forbids model-authored quantiles.  The agent
#: chooses the analysis; copying an artifact path is no longer a second,
#: failure-prone decision.
#: Version 6: Evidence compiles task context into a provenance receipt before
#: the agent loop and host-injects its validated events into the one governed
#: forecast call.  The agent still chooses whether to invoke forecasting, but
#: cannot silently omit context already gathered by its host.
#: Version 9: the compiler may preserve several stable typed hypotheses;
#: numerical influence is evaluated separately from semantic extraction.
#: Version 10: verified literal range claims are deterministically re-routed
#: through the ordinary constraint validator even when the model omits events.
#: Version 11: a weaker LLM-selected scenario cannot displace a deterministic
#: context_trusted path; selection remains autonomous among evidence peers.
MCP_CONTRACT_VERSION = 11
# A runaway agent is bounded by the three caps above; this one exists
# only to stop a hung endpoint from parking a worker forever, so it must
# sit above the latency an honest run can incur. At 600s it did not: it
# ended 29 of 355 runs in the two-arm comparison, and the runs it ended
# averaged 0-6 MCP calls out of 24 -- slow providers, not loops (one
# `route=direct` run spent 765s inside a single request). Meanwhile the
# CiK DirectPrompt baseline those runs are scored against carries no
# wall-clock cap at all and took a median 2965s per run, 90 of its 91
# scored runs exceeding 600s. A cap that the baseline would fail 99% of
# the time is not a budget guard, it is a handicap on one arm.
MAX_WALL_SECONDS = 7200.0

#: Arguments whose values are filesystem paths by contract: these must
#: resolve inside the jail no matter what. Every OTHER string argument
#: is rejected only if it resolves to an existing path outside the jail
#: (so a quoted source_span containing "/" stays admissible).
PATH_ARGUMENT_NAMES = frozenset({
    "input", "output_dir", "context_events_file", "covariates_file",
    "actuals_file", "store_path", "artifact_dir", "path", "file",
    "events_file", "plan_file",
})

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_forecast",
        "description": (
            "End the run with your answer. Exactly one of two exits: "
            "artifact_path (a path returned by a successful "
            "gnomon_forecast call in THIS run; its trajectory is used "
            "verbatim — you cannot edit it) or quantiles (your own "
            "per-step forecast, one {q10, q50, q90} object per horizon "
            "step). Not submitting is recorded as an abstention."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact_path": {
                    "type": "string",
                    "description": "artifact_path from a gnomon_forecast result in this run.",
                },
                "quantiles": {
                    "type": "array",
                    "description": "Your own forecast: exactly one object per horizon step.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "q10": {"type": "number"},
                            "q50": {"type": "number"},
                            "q90": {"type": "number"},
                        },
                        "required": ["q10", "q50", "q90"],
                    },
                },
                "reasoning": {"type": "string", "description": "Why this answer."},
            },
        },
    },
}

#: Symmetric by construction — the router arm's first prompt was a
#: Gnomon sales pitch and measured as a constant function. This one
#: names what each path is good at and lets the transcript record the
#: choice.
SYSTEM = """\
You are producing a probabilistic forecast for a time series task. You
have tools from "gnomon", a deterministic forecasting engine, and you
may use them or ignore them — your only obligation is to end by calling
`submit_forecast` with the best answer you can produce.

What the engine offers: backtested model selection and calibrated
uncertainty computed from the numeric history, and deterministic
verification of context sentences that STATE numbers (bounds, levels,
multiples of the usual level, stated trend cessations) supplied as
typed context events. What it cannot do: use qualitative context —
weather states, unquantified events, causal descriptions without
numbers — no matter how decisive; you can reason about those yourself.

Two ways to finish:
- Submit `artifact_path` from a `gnomon_forecast` result: that
  artifact's trajectory becomes your answer verbatim.
- Submit your own `quantiles` (one {{q10, q50, q90}} per step): your
  numbers, your responsibility.

The history file is at {csv_path} (columns: timestamp, value). Forecast
horizon: {horizon} steps, {future_start} to {future_end}. Tool errors
return typed codes and repair options; you may fix arguments and retry
within the caps ({max_rounds} rounds, {max_calls} tool calls).
"""

USER = """\
Task context:
{context}

Numeric history ({n_obs} observations, oldest first, also on disk at
{csv_path}):
{history}

Produce the best {horizon}-step probabilistic forecast, then call
submit_forecast.
"""

GOVERNED_CONTEXT_NOTE = """\
The host has already compiled the supplied task context into a governed
context receipt. Any gnomon_forecast call is automatically bound to that
receipt, the history file, target, and horizon; do not copy or reconstruct
events yourself. Call gnomon_forecast once when you want Gnomon's governed
answer. The host publishes the first valid artifact automatically.
"""

DOSSIER_INSTRUCTIONS = """\
You compile temporal context for a governed forecasting engine. Return ONLY
one JSON object with this shape:
{
  "events": [
    {"document_index": 0, "event_type": "short_label",
     "entity_scope": ["*"], "effective_start": "timezone-aware ISO",
     "effective_end": "timezone-aware ISO", "confidence": 0.0,
     "status": "confirmed | tentative",
     "evidence_quote": "verbatim context sentence",
     "effect_family": "level_shift | trend_change | variance_change | temporary_pulse | saturation_bound | seasonal_regime_change | unknown",
     "direction": "increase | decrease | unknown",
     "duration": "temporary | persistent | unknown",
     "entity_kind": "service | product | medication | procedure | calendar | capacity | price | environment | unknown"}
  ],
  "claims": [
    {"source_span": "verbatim context sentence",
     "relation": "supports_increase | supports_decrease | supports_stability | supports_higher_variance | supports_lower_variance | changes_seasonal_regime | constrains_range | unknown",
     "effective_start": "timezone-aware ISO", "effective_end": "timezone-aware ISO",
     "mechanism": "brief qualitative explanation", "confidence": 0.0}
  ],
  "forecast_candidate": {
    "quantiles": [{"timestamp": "exact requested timestamp", "q10": 0.0,
                   "q50": 0.0, "q90": 0.0}],
    "rationale": "how the cited claims modify the numeric history"
  },
  "effect_proposal": {
    "shape": "temporary_pulse | level_shift | trend_change | variance_change | ramp_recovery | seasonal_amplitude | seasonal_phase | cross_series_relationship | saturation_bound | custom_scenario",
    "unit": "target_units | fraction_of_level",
    "location": 0.0, "lower": 0.0, "upper": 0.0,
    "confidence": 0.0, "delay_steps": 0, "duration_steps": null,
    "period_steps": null,
    "scope": {"kind": "single_series", "series": ["*"]},
    "claim_ids": ["claim-1"], "rationale": "brief mechanism",
    "uncertainty_basis": "why this range is plausible"
  },
  "hypotheses": [
    {"kind": "absolute_value | bound | additive_change | multiplicative_change | regime_shift | relationship | historical_analogue | unsupported",
     "claim_ids": ["claim-1"], "target_series": ["*"],
     "predictor_series": null, "known_at": "history cutoff ISO",
     "lag_steps": 0, "direction": "increase | decrease | unknown",
     "rationale": "one bounded interpretation"}
  ],
  "covariate_tables": [
    {"name": "safe_snake_case", "type": "continuous | binary | cyclic_<period>",
     "rows": [{"document_index": 0,
                "timestamp": "normalised timezone-aware ISO",
                "source_time_span": "verbatim date/time token",
                "value": 0.0,
                "evidence_quote": "verbatim context span containing both the time and numeric value"}]}
  ]
}

Rules:
- Cite only exact spans present in context; never invent an event or source.
- Prefer effect_proposal over forecast_candidate: extract a cited temporal
  effect and let Gnomon compose the numbers. Use forecast_candidate only when
  no typed effect can express the relationship. Never claim numeric values
  came from text unless the cited span states them.
- Effect location/lower/upper are changes added to the primary path, not target
  values. If the text states an exact future value (for example, withdrawals
  become zero), encode it as an override:* event and omit effect_proposal;
  Gnomon parses and applies that value deterministically.
- Events are the narrow deterministic lane. Numeric bounds use
  event_type constraint:<label>; deterministic stated values use
  override:<label>. Other events carry qualitative classifications only;
  Gnomon estimates any magnitude from data or keeps them scenario-only.
- Put only events whose effective window overlaps one or more requested
  forecast timestamps in `events`. Historical events and regimes belong in
  cited `claims`; a dossier learned at the forecast cutoff cannot backdate
  them into historical folds.
- Claims are the richer interpretation lane. Put qualitative relationships
  there even when no deterministic event can represent them.
- When context permits more than one interpretation, emit up to six competing
  typed hypotheses rather than collapsing ambiguity into one numeric path.
  A relationship names its predictor and lag; an historical_analogue names
  only a cited analogue claim. Gnomon validates and evaluates these after the
  model response. Parsing confidence never upgrades support or automation.
- Covariate tables are extraction, never invention. Emit a row only when one
  verbatim quote contains both its time token and numeric value. Do not infer
  values from adjectives, interpolate missing rows, or supply known_at; the
  host owns knowledge time. Gnomon will test surviving tables out of sample
  before they may influence the canonical forecast.
- Use no observations after the history cutoff. Return empty arrays and null
  effect_proposal and forecast_candidate when context contains no
  forecast-relevant information.
"""


class StdioMcpSession:
    """Newline-delimited JSON-RPC client over a `gnomon mcp serve` child.

    The subprocess runs with the jail as its working directory, so any
    relative path that slips through argument screening still lands
    inside the jail.

    Every call carries a timeout: ``readline`` on a wedged server blocks
    forever, and the runners' own caps (rounds, calls, tokens, wall
    clock) are all checked *between* reads, so a single hung call used to
    stall an entire sequential run with no summary ever written. On
    timeout the child is killed and the call raises; the runners already
    treat transport death as a disclosed harness failure.
    """

    #: Generous per-call ceiling: a real gnomon_forecast over a large file
    #: (TSFM sandboxes included) finishes well inside this; only a hung
    #: server does not.
    DEFAULT_CALL_TIMEOUT_SECONDS = 600.0

    def __init__(self, cwd: str | Path, command: list[str] | None = None,
                 call_timeout: float | None = None,
                 profile: str | None = None):
        child_env = dict(os.environ)
        if profile:
            child_env["GNOMON_MCP_PROFILE"] = profile
        self._proc = subprocess.Popen(
            command or [sys.executable, "-m", "gnomon", "mcp", "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=str(cwd), text=True,
            env=child_env,
        )
        self._next_id = 0
        self.call_timeout = (self.DEFAULT_CALL_TIMEOUT_SECONDS
                             if call_timeout is None else float(call_timeout))

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        import threading

        self._next_id += 1
        request = {"jsonrpc": "2.0", "id": self._next_id,
                   "method": method, "params": params}
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        timed_out: list[bool] = []

        def _kill() -> None:
            timed_out.append(True)
            self._proc.kill()

        timer = threading.Timer(self.call_timeout, _kill)
        timer.start()
        try:
            line = self._proc.stdout.readline()
        finally:
            timer.cancel()
        if not line:
            if timed_out:
                raise RuntimeError(
                    f"gnomon mcp server did not answer {method} within "
                    f"{self.call_timeout:.0f}s and was killed"
                )
            raise RuntimeError("gnomon mcp server closed its stdout")
        message = json.loads(line)
        if "error" in message:
            raise RuntimeError(f"MCP {method} failed: {message['error']}")
        return message["result"]

    def initialize(self) -> dict[str, Any]:
        return self._rpc("initialize", {"protocolVersion": "2025-06-18"})

    def list_tools(self) -> list[dict[str, Any]]:
        return self._rpc("tools/list", {})["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


class InProcessMcpSession:
    """The same server code without a subprocess: calls the server's
    request handler directly. Used by the unit tests (and usable for
    smoke runs); the path jail is enforced by the harness either way.
    """

    def __init__(self, cwd: str | Path | None = None):
        self.cwd = cwd  # unused; documents interface parity
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _handle(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        from gnomon.mcp_server import _handle

        result = _handle({"method": method, "params": params})
        if result is None:
            raise RuntimeError(f"MCP method not handled: {method}")
        return result

    def initialize(self) -> dict[str, Any]:
        return self._handle("initialize", {})

    def list_tools(self) -> list[dict[str, Any]]:
        return self._handle("tools/list", {})["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return self._handle("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        pass


def openai_tool_specs(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MCP tool entries as chat-completions function specs, verbatim.

    Name, description, and inputSchema pass through untouched — pruning
    or paraphrasing would hide the confusion this arm exists to measure.
    ``outputSchema`` is dropped (the chat format has no slot for it);
    that loss is disclosed in the spec.
    """
    return [
        {"type": "function", "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["inputSchema"],
        }}
        for tool in mcp_tools
    ] + [SUBMIT_TOOL]


def _within(path: Path, jail: Path) -> bool:
    try:
        path.resolve().relative_to(jail.resolve())
        return True
    except ValueError:
        return False


def jail_violations(arguments: Any, jail: Path, key: str = "") -> list[str]:
    """Every argument value that would reach outside the jail.

    Path-named arguments must resolve inside the jail unconditionally.
    Any other string is a violation only if it resolves to an EXISTING
    filesystem entry outside the jail — so free text containing "/"
    (a quoted span, a unit like "km/h") stays admissible, while the
    cached benchmark datasets (future windows included) do not.
    """
    violations: list[str] = []
    if isinstance(arguments, dict):
        for name, value in arguments.items():
            violations.extend(jail_violations(value, jail, key=name))
    elif isinstance(arguments, list):
        for item in arguments:
            violations.extend(jail_violations(item, jail, key=key))
    elif isinstance(arguments, str) and arguments:
        raw = arguments[len("store:"):] if arguments.startswith("store:") else arguments
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = jail / candidate
        if key in PATH_ARGUMENT_NAMES:
            if not _within(candidate, jail):
                violations.append(f"{key}={arguments!r} resolves outside the run directory")
        elif ("/" in raw or "\\" in raw or raw.startswith("~")):
            try:
                exists = candidate.exists() or Path(raw).expanduser().exists()
            except OSError:
                exists = False
            if exists and not _within(candidate, jail):
                violations.append(f"{key}={arguments!r} names an existing path outside the run directory")
    return violations


def _task_series(task_instance: Any) -> tuple[list[str], list[float]]:
    """(timestamps, values) from a task's past window.

    Real CiK tasks carry pandas frames and go through the exact same
    conversion as the gnomon-agent arm; a task may instead supply
    ``past_time`` as a plain list of ``(iso_timestamp, value)`` pairs,
    which is what the pandas-free unit tests use.
    """
    past = task_instance.past_time
    if hasattr(past, "columns"):  # a pandas frame; lists have .index too
        history = GnomonForecaster._history_frame(task_instance)
        return ([ts.isoformat() for ts in history.index],
                [float(v) for v in history["value"].values])
    return ([str(ts) for ts, _ in past], [float(v) for _, v in past])


def _task_future_timestamps(task_instance: Any) -> list[str]:
    future = task_instance.future_time
    if hasattr(future, "columns"):
        import pandas as pd

        index = future.index
        if isinstance(index, pd.PeriodIndex):
            index = index.to_timestamp()
        index = pd.DatetimeIndex(index)
        if index.tz is None:
            index = index.tz_localize("UTC")
        return [ts.isoformat() for ts in index]
    return [str(ts) for ts in future]


def _write_history_csv(timestamps: list[str], values: list[float],
                       csv_path: Path) -> None:
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for timestamp, value in zip(timestamps, values):
            writer.writerow([timestamp, repr(float(value))])


def _tool_calls_as_dicts(message: Any) -> list[dict[str, Any]]:
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        calls.append({
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name,
                         "arguments": call.function.arguments},
        })
    return calls


#: Keys whose values are never dropped when a tool result is compacted —
#: by the TemporalBench arm's size bound or by supersession here.
#: Everything Gnomon says about how far it will stand behind a number
#: lives under one of these; the bulk that makes a result large does
#: not. A budget is a reason to send fewer numbers, never a reason to
#: send numbers without their disclosures.
UNSHRINKABLE_KEYS = frozenset({
    "support", "support_assessment", "support_state", "warnings", "notes",
    "status", "code", "error", "message", "recovery", "recovery_actions",
    "disclosures", "disclosure", "abstention", "reason", "artifact_path",
    "forecast_id", "series", "selected_model", "interval_coverage",
    "headline",
    "context_outcome",
})


def disclosure_skeleton(value: Any) -> Any:
    """The subtree of ``value`` reachable through disclosures alone.

    Keeps every :data:`UNSHRINKABLE_KEYS` entry verbatim — per-channel
    support states, warnings, abstention reasons, error envelopes, the
    artifact_path — and whatever dict/list structure is needed to reach
    them; everything else (forecast arrays, previews, evidence blocks)
    is dropped. This is what survives of a tool result once a later call
    has superseded it."""
    if isinstance(value, dict):
        kept: dict[str, Any] = {}
        for key, item in value.items():
            if key in UNSHRINKABLE_KEYS:
                kept[key] = item
            else:
                sub = disclosure_skeleton(item)
                if sub not in (None, {}, []):
                    kept[key] = sub
        return kept
    if isinstance(value, list):
        subs = [disclosure_skeleton(item) for item in value]
        return [item for item in subs if item not in (None, {}, [])]
    return None


class ToolMessageLog:
    """Compacts superseded tool results out of a running message history.

    The agent loop re-sends every prior tool result each round, so a
    result's cost is paid once per remaining round — quadratic in
    rounds, and the dominant token cost of a long run. Most of that
    re-sent bulk is dead: a model that calls ``gnomon_forecast`` five
    times over the same channels holds five full trajectories of which
    only the last is live. When a new result supersedes an older one,
    the older message's content is replaced in place with its
    :func:`disclosure_skeleton` plus a note naming the supersession —
    the support labels, warnings, abstention reasons, error codes, and
    ``artifact_path`` stay verbatim, and the full numbers remain on
    disk, re-readable via ``gnomon_get_artifact``.

    What supersedes what is deliberately narrow, because a wrongly
    compacted result deletes live evidence:

    - ``gnomon_forecast``: a result whose channels cover an earlier
      result's channels supersedes it only when every semantic argument
      also matches. Different horizons, cutoffs, datasets, thresholds,
      candidate pools, repair policies, or context are different questions.
    - every other tool: only a call with the *same arguments* supersedes
      (the retry/repeat pattern); a different-arguments call is new
      evidence, not a replacement.
    - errors never supersede anything and are never compacted: they are
      small, and their repair options are the recovery path.

    One further honest compression, because on a degraded run the
    epistemics ARE the bulk (six channels of identical short-history
    warnings dwarf the trimmed numbers): a stubbed disclosure whose text
    is character-identical in the live superseding result is replaced by
    a marker saying exactly that. The words are still in the
    conversation, one message down, attached to the numbers that are
    actually live; a disclosure that differs in any way stays verbatim
    in the stub.
    """

    #: The per-series disclosure fields eligible for the
    #: identical-in-the-later-result marker. Support labels are not
    #: listed: they are a handful of bytes and always ride verbatim.
    _DEDUP_FIELDS = ("warnings", "support_assessment", "notes")

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    @staticmethod
    def _key(tool: str, arguments: dict[str, Any],
             payload: dict[str, Any]) -> Any:
        if tool == "gnomon_forecast":
            channels = {item.get("series")
                        for item in payload.get("results") or []
                        if isinstance(item, dict) and item.get("series")}
            # A single-target run names its one result "__default__" —
            # the column identity lives in the arguments there, and
            # keying on the placeholder would collide forecasts of
            # DIFFERENT columns into one supersession chain.
            channels.discard("__default__")
            if not channels:
                target = str(arguments.get("target_column") or "")
                channels = {
                    name.strip() for name in target.split(",") if name.strip()
                }
            # target_column is represented by the channel set so an
            # equivalent batch can retire single-channel calls. `format`
            # changes inline verbosity and `output_dir` only relocates the
            # immutable artifact. Every argument capable of changing the
            # question or answer remains in the semantic key.
            semantic_arguments = {
                name: value for name, value in arguments.items()
                if name not in {"target_column", "format", "output_dir"}
            }
            return (
                frozenset(channels),
                json.dumps(semantic_arguments, sort_keys=True, default=str),
            )
        return json.dumps(arguments, sort_keys=True, default=str)

    @staticmethod
    def _supersedes(tool: str, new_key: Any, old_key: Any) -> bool:
        if tool == "gnomon_forecast":
            old_channels, old_semantics = old_key
            new_channels, new_semantics = new_key
            return (bool(old_channels) and old_channels <= new_channels
                    and old_semantics == new_semantics)
        return new_key == old_key

    @staticmethod
    def _results_by_channel(payload: dict[str, Any], key: Any,
                            ) -> dict[str, dict[str, Any]]:
        """Each result keyed by the channel it answers for, with a
        single-target run's ``__default__`` placeholder resolved to the
        one column its call named."""
        out: dict[str, dict[str, Any]] = {}
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            series = item.get("series")
            channels = key[0] if isinstance(key, tuple) else key
            if series in (None, "__default__") \
                    and isinstance(channels, frozenset) and len(channels) == 1:
                series = next(iter(channels))
            if series:
                out[str(series)] = item
        return out

    def _stub(self, entry: dict[str, Any], tool: str,
              new_payload: dict[str, Any], new_key: Any) -> dict[str, Any]:
        old_payload = entry["payload"]
        stub = disclosure_skeleton(old_payload)
        unchanged_note = (
            "character-identical in the later result below; nothing "
            "was reworded"
        )
        if stub == disclosure_skeleton(new_payload):
            # The repeat-call pattern: every disclosure of the old
            # result rides verbatim on the live one. Keep the identity
            # fields; say where the words are.
            stub = {field: stub[field]
                    for field in ("status", "artifact_path", "forecast_id",
                                  "headline")
                    if field in stub}
            stub["unchanged"] = f"all disclosures {unchanged_note}"
        elif isinstance(stub.get("results"), list):
            live = self._results_by_channel(new_payload, new_key)
            for result in stub["results"]:
                if not isinstance(result, dict):
                    continue
                channel = result.get("series")
                entry_key = entry["key"]
                channels = entry_key[0] if isinstance(entry_key, tuple) else entry_key
                if channel in (None, "__default__") \
                        and isinstance(channels, frozenset) \
                        and len(channels) == 1:
                    channel = next(iter(channels))
                counterpart = live.get(str(channel))
                if counterpart is None:
                    continue
                deduped = [field for field in self._DEDUP_FIELDS
                           if field in result
                           and result[field] == counterpart.get(field)]
                for field in deduped:
                    del result[field]
                if deduped:
                    result["unchanged"] = (
                        f"{', '.join(deduped)} {unchanged_note}")
        stub["harness_superseded"] = True
        stub["harness_note"] = (
            f"This {tool} result was superseded by a later {tool} "
            f"call; its bulk was removed from the conversation. "
            f"Support states and the disclosures above are verbatim "
            f"(an `unchanged` marker means the identical text rides on "
            f"the later result); the complete numbers are unchanged on "
            f"disk"
            + (" (re-read with gnomon_get_artifact)."
               if old_payload.get("artifact_path") else ".")
        )
        return stub

    def record(self, tool: str, arguments: dict[str, Any] | None,
               message: dict[str, Any]) -> int:
        """Note one appended tool message; compact the results it
        supersedes. Returns how many earlier messages were compacted."""
        if not isinstance(arguments, dict):
            return 0
        try:
            payload = json.loads(message.get("content") or "")
        except (json.JSONDecodeError, TypeError):
            return 0
        if not isinstance(payload, dict) or payload.get("code") \
                or payload.get("error") or payload.get("status") == "error":
            return 0
        key = self._key(tool, arguments, payload)
        compacted = 0
        for entry in self._entries:
            if entry["tool"] != tool or entry.get("compacted"):
                continue
            if not self._supersedes(tool, key, entry["key"]):
                continue
            stub = self._stub(entry, tool, payload, key)
            entry["message"]["content"] = json.dumps(stub, default=str)
            entry["compacted"] = True
            compacted += 1
        self._entries.append({"tool": tool, "key": key,
                              "payload": payload, "message": message,
                              "compacted": False})
        return compacted


class McpAgentForecaster:
    """CiK ``Baseline``-compatible callable: the model drives real MCP tools.

    ``session_factory`` and ``client`` are injectable for tests; the
    defaults spawn a real ``gnomon mcp serve`` subprocess per run and an
    OpenRouter chat client.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        openrouter_model: str,
        *,
        temperature: float = 1.0,
        work_dir: str | None = None,
        trace_dir: str | Path | None = None,
        client: Any = None,
        session_factory: Any = None,
        profile: str | None = None,
        output_role: str = "canonical",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.openrouter_model = openrouter_model
        self.temperature = temperature
        self.client = client or OpenRouterClient(
            openrouter_model, temperature=temperature,
            base_url=base_url, api_key=api_key)
        self.work_dir = work_dir
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.profile = (profile or os.environ.get(
            "GNOMON_MCP_PROFILE", "full")).strip().lower()
        if output_role not in {"canonical", "llm_candidate_shadow",
                               "publication_best_effort"}:
            raise ValueError("unknown output_role")
        if output_role != "canonical" and self.profile != "evidence":
            raise ValueError("output role requires the evidence profile")
        self.output_role = output_role
        # functools.partial remains pickleable when the official CiK runner
        # fans task-seeds out to worker processes; a closure does not.
        self.session_factory = session_factory or partial(
            StdioMcpSession, profile=self.profile)

    @property
    def cache_name(self) -> str:
        # Temperature and the arm's contract version are part of what a
        # cached result measures: without them, a rerun at a different
        # temperature — or after the prompt/caps changed — silently reused
        # the old runs' results under the same name.
        model = self.openrouter_model.replace("/", "-")
        return (f"McpAgentForecaster_model={model}"
                f"_temperature={self.temperature:g}"
                f"_profile={self.profile}"
                f"_output={self.output_role}"
                f"_endpoint={hashlib.sha256(str(getattr(self.client, 'base_url', 'injected-client')).encode()).hexdigest()[:10]}"
                f"_contract={MCP_CONTRACT_VERSION}")

    def __str__(self) -> str:
        return self.cache_name

    def __call__(self, task_instance: Any, n_samples: int):
        run = _Run(self, task_instance)
        try:
            submission, extra_info = run.drive()
        finally:
            run.finish()
        if self.output_role in {"llm_candidate_shadow",
                               "publication_best_effort"}:
            dossier = (run.context_compilation or {}).get("dossier") or {}
            candidate = dossier.get("forecast_candidate") or {}
            candidate_rows = candidate.get("quantiles")
            if (not candidate_rows and not dossier.get("effect_proposal")
                    and self.output_role == "llm_candidate_shadow"):
                raise GnomonAbstained([
                    "no admissible context candidate in the sealed dossier"])
            from gnomon.publication import publish_result, verify_publication
            artifact_result = getattr(run, "_submitted_result", None) or {
                "support": extra_info.get("support") or "best_effort",
                "forecast": submission,
            }
            selection = None
            selection_error = None
            if self.output_role == "publication_best_effort":
                from gnomon.publication import (build_scenario_catalog,
                                                scenario_selection_contract,
                                                validate_scenario_selection)
                from gnomon.temporal_state import build_temporal_state
                scenarios, _ = build_scenario_catalog(
                    artifact_result, dossiers=[dossier])
                if len(scenarios) > 1:
                    contract = scenario_selection_contract(
                        scenarios=scenarios, dossiers=[dossier],
                        temporal_state=build_temporal_state(
                            artifact_result, dossiers=[dossier]))
                    base_prompt = (
                        "Choose the most useful human-facing scenario under "
                        "this governed contract. Preserve every number and "
                        "support label. Return only the JSON response object.\n"
                        + json.dumps(contract))
                    last_error = None
                    for attempt in range(2):
                        prompt = base_prompt if attempt == 0 else (
                            base_prompt + "\nYour previous response was rejected: "
                            + str(last_error) + "\nRepair only that violation."
                        )
                        try:
                            response = self.client.completions(
                                [{"role": "user", "content": prompt}], n=1)[0]
                            objects = extract_json_objects(response)
                            if not objects:
                                raise ValueError("selector returned no JSON object")
                            selection = validate_scenario_selection(
                                objects[0], scenarios=scenarios,
                                dossiers=[dossier])
                            break
                        except Exception as error:
                            last_error = error
                            selection = None
                    if selection is None:
                        selection_error = f"selector rejected after repair: {last_error}"
            try:
                publication = publish_result(
                    artifact_result, mode="best_effort", dossiers=[dossier],
                    scenario_selection=selection)
            except ValueError as error:
                selection_error = f"selector rejected: {error}"
                selection = None
                publication = publish_result(
                    artifact_result, mode="best_effort", dossiers=[dossier])
            if not verify_publication(publication):
                raise RuntimeError("best-effort publication failed verification")
            submission = publication["recommended_forecast"]
            if self.output_role == "publication_best_effort":
                extra_info = {
                    **extra_info, "route": "publication_best_effort",
                    "publication": publication,
                    "scenario_selector": {
                        "attempted": selection is not None or selection_error is not None,
                        "accepted": publication.get("scenario_selection") is not None,
                        "error": selection_error,
                    },
                    "llm_usage": self.client.usage_summary,
                }
            extra_info = {
                **extra_info,
                "route": ("publication_best_effort"
                          if self.output_role == "publication_best_effort"
                          else "llm_candidate_shadow"),
                "candidate_support": dossier.get("candidate_support"),
                "candidate_seal_sha256": dossier.get("seal_sha256"),
                "automation_eligible": False,
                "primary_forecast_unchanged": True,
            }
        paths = samples_from_quantile_rows(submission, n_samples)
        try:
            import numpy as np
        except ModuleNotFoundError:
            # The official benchmark environment always has numpy; the
            # unit tests run without it and take the same [n, horizon, 1]
            # shape as nested lists.
            return [[[value] for value in path] for path in paths], extra_info
        return np.asarray(paths, dtype=float)[:, :, None], extra_info


class _Run:
    """One task-seed conversation: jail, server, loop, caps, submission."""

    def __init__(self, forecaster: McpAgentForecaster, task_instance: Any):
        self.forecaster = forecaster
        self.task = task_instance
        self.started = time.time()
        self.jail = Path(tempfile.mkdtemp(
            prefix="cik-mcp-", dir=forecaster.work_dir)).resolve()
        self.timestamps, self.values = _task_series(task_instance)
        self.csv_path = self.jail / "history.csv"
        _write_history_csv(self.timestamps, self.values, self.csv_path)
        self.horizon = len(task_instance.future_time)
        self.session = forecaster.session_factory(self.jail)
        self.trace: list[dict[str, Any]] = []
        self.result_log = ToolMessageLog()
        self.mcp_calls = 0
        self.artifact_paths: set[str] = set()
        self.submission: dict[str, Any] | None = None
        self.governed_evidence = forecaster.profile == "evidence"
        # Context compilation is part of this treatment's cost, so snapshot
        # usage before invoking it rather than hiding those tokens.
        self.tokens_at_start = (forecaster.client.total_prompt_tokens
                                + forecaster.client.total_completion_tokens)
        self.context_compilation = (
            self._compile_context() if self.governed_evidence else None)

    def _compile_context(self) -> dict[str, Any]:
        """Compile host-gathered prose once and retain an auditable receipt.

        The compiler proposes only typed events; Gnomon's normal context
        parser and admission machinery remain responsible for accepting and
        applying them.  No future target observations are exposed here.
        """
        from gnomon.context import event_from_dict, event_to_dict
        from gnomon.llm_dossier import (deterministic_events_from_claims,
                                        validate_temporal_dossier)
        from gnomon.workflows import DocumentRef, parse_context_response

        context = build_context_text(self.task)
        future_timestamps = _task_future_timestamps(self.task)
        history = "\n".join(
            f"{stamp},{value}" for stamp, value in
            zip(self.timestamps, self.values))
        prompt = (
            f"{DOSSIER_INSTRUCTIONS}\n"
            f"History cutoff: {self.timestamps[-1]}\n"
            f"Forecast timestamps: {json.dumps(future_timestamps)}\n"
            f"Numeric history (timestamp,value):\n{history}\n\n"
            f"Context:\n{context or '(none)'}\n"
        )
        raw: dict[str, Any] = {}
        compile_rejections: list[str] = []
        try:
            completion = self.forecaster.client.completions(
                [{"role": "user", "content": prompt}], n=1)[0]
            objects = extract_json_objects(completion)
            if objects:
                raw = objects[0]
            else:
                compile_rejections.append(
                    "no JSON object in temporal-dossier output")
        except Exception as error:
            compile_rejections.append(f"dossier compilation failed: {error}")

        # Exercise the product's bounded repair lane. The first response is
        # probed before event parsing so a corrected complete dossier (claims
        # plus effect) feeds every downstream validator consistently.
        if raw.get("effect_proposal") or raw.get("forecast_candidate"):
            probe, probe_rejections = validate_temporal_dossier(
                raw, context_text=context, cutoff=self.timestamps[-1],
                future_timestamps=future_timestamps, history=self.values,
                compiler_model=self.forecaster.openrouter_model)
            if not probe.get("effect_proposal") and not probe.get("forecast_candidate"):
                critique = probe.get("effect_proposal_critique") or {
                    "status": "rejected", "reasons": probe_rejections}
                try:
                    repair_completion = self.forecaster.client.completions([{
                        "role": "user", "content": (
                            prompt + "\nYour proposal was rejected by Gnomon:\n"
                            + json.dumps(critique)
                            + "\nReturn one complete corrected dossier JSON "
                              "including cited claims. This is the only repair round.")
                    }], n=1)[0]
                    repaired = extract_json_objects(repair_completion)
                    if repaired:
                        raw = repaired[0]
                    else:
                        compile_rejections.append(
                            "dossier repair returned no JSON object")
                except Exception as error:
                    compile_rejections.append(
                        f"dossier repair failed: {error}")

        # Exact stated values are safer than model-estimated deltas. Once the
        # span and window pass dossier validation, re-route them through the
        # normal override event validator; this never promotes qualitative
        # language or a model-authored number.
        final_probe, _ = validate_temporal_dossier(
            raw, context_text=context, cutoff=self.timestamps[-1],
            future_timestamps=future_timestamps, history=self.values,
            compiler_model=self.forecaster.openrouter_model)
        existing_spans = {str(item.get("evidence_quote") or item.get("source_span") or "")
                          for item in raw.get("events") or []
                          if isinstance(item, dict)}
        derived_events = [item for item in deterministic_events_from_claims(final_probe)
                          if item["evidence_quote"] not in existing_spans]
        if derived_events:
            raw = {**raw, "events": [*(raw.get("events") or []), *derived_events]}

        # Reuse Gnomon's product compiler contract rather than maintaining a
        # benchmark-only event dialect. The host, not the model, supplies the
        # document identity and the time at which it assembled the dossier.
        bound_raw = dict(raw)
        bound_events = []
        for proposal in list(raw.get("events") or []):
            if not isinstance(proposal, dict):
                bound_events.append(proposal)
                continue
            event = dict(proposal)
            event["document_index"] = 0
            event["known_at"] = self.timestamps[-1]
            event.setdefault("entity_scope", ["*"])
            if event.get("source_span") and not event.get("evidence_quote"):
                event["evidence_quote"] = event["source_span"]
            bound_events.append(event)
        bound_raw["events"] = bound_events
        compilation = parse_context_response(
            bound_raw,
            [DocumentRef(
                name="task_context", content=context,
                source_type="benchmark_task_context",
                reference=(f"cik:{getattr(self.task, 'name', self.task.__class__.__name__)}"
                           "#context"),
                known_at=self.timestamps[-1],
            )],
            proposer={"kind": "llm", "model": self.forecaster.openrouter_model},
            covariate_known_at=self.timestamps[-1],
            as_of=self.timestamps[-1],
        )
        events = [event_from_dict(event) for event in compilation["events"]]
        event_rejections = [
            "; ".join(str(problem) for problem in item.get("problems") or [])
            for item in compilation["rejected"]
        ]
        # The dossier is known only at the forecast cutoff. Historical event
        # descriptions may support interpretation, but cannot become
        # fold-admissible executable events retroactively. Keep only events
        # that can affect the requested future grid.
        from datetime import datetime

        forecast_start = datetime.fromisoformat(future_timestamps[0])
        forecast_end = datetime.fromisoformat(future_timestamps[-1])
        prospective_events = []
        for event in events:
            start = datetime.fromisoformat(event.effective_start)
            end = datetime.fromisoformat(event.effective_end)
            if end < forecast_start or start > forecast_end:
                event_rejections.append(
                    f"{event.event_id} rejected: event does not overlap the "
                    "requested forecast window; retain it as a cited claim")
            else:
                prospective_events.append(event)
        events = prospective_events
        dossier, dossier_rejections = validate_temporal_dossier(
            raw, context_text=context, cutoff=self.timestamps[-1],
            future_timestamps=future_timestamps, history=self.values,
            compiler_model=self.forecaster.openrouter_model,
        )
        covariate_receipt = compilation["covariates"]
        covariate_rejections = compilation["covariate_rejections"]
        rejections = [*compile_rejections, *event_rejections,
                      *dossier_rejections, *covariate_rejections]
        payload = {
            "schema_version": 1,
            "compiler": {
                "kind": "llm_proposes_gnomon_validates",
                "model": self.forecaster.openrouter_model,
            },
            "source": {
                "kind": "benchmark_task_context",
                "sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            },
            "events": [event_to_dict(event) for event in events],
            "context_receipt_id": compilation["receipt_id"],
            "hypotheses": compilation["hypotheses"],
            "dossier": dossier,
            "covariates": covariate_receipt,
            "rejections": list(rejections),
            "future_observations_exposed": False,
        }
        receipt_body = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                  default=str)
        payload["receipt_sha256"] = hashlib.sha256(
            receipt_body.encode("utf-8")).hexdigest()
        path = self.jail / "context-receipt.json"
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                        encoding="utf-8")
        retained_path = path
        if self.forecaster.trace_dir is not None:
            receipt_dir = self.forecaster.trace_dir / "context-receipts"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            retained_path = receipt_dir / (payload["receipt_sha256"] + ".json")
            rendered = path.read_text(encoding="utf-8")
            if retained_path.exists() and retained_path.read_text(
                    encoding="utf-8") != rendered:
                raise ValueError(
                    "sealed dossier path already contains different content")
            retained_path.write_text(rendered, encoding="utf-8")
        payload["path"] = str(retained_path)
        return payload

    # -- lifecycle ---------------------------------------------------------
    def finish(self) -> None:
        self.session.close()
        self._write_trace()

    def _write_trace(self) -> None:
        trace_dir = self.forecaster.trace_dir
        if trace_dir is None:
            return
        trace_dir.mkdir(parents=True, exist_ok=True)
        name = getattr(self.task, "name", self.task.__class__.__name__)
        seed = getattr(self.task, "seed", "x")
        payload = {
            "task": str(name), "seed": seed,
            "mcp_calls": self.mcp_calls,
            "submitted": (self.submission or {}).get("route"),
            "context_compilation": (
                {
                    "receipt_path": self.context_compilation["path"],
                    "source_sha256": self.context_compilation[
                        "source"]["sha256"],
                    "event_count": len(self.context_compilation["events"]),
                    "claim_count": len(
                        self.context_compilation["dossier"]["claims"]),
                    "hypothesis_count": len(
                        self.context_compilation["dossier"].get("hypotheses") or []),
                    "candidate_available": bool(
                        self.context_compilation["dossier"].get("effect_proposal")
                        or self.context_compilation["dossier"].get(
                            "forecast_candidate")),
                    "covariate_tables": len(
                        self.context_compilation["covariates"]["tables"]),
                    "covariate_tables_proposed": self.context_compilation[
                        "covariates"]["tables_proposed"],
                    "covariate_rows_proposed": self.context_compilation[
                        "covariates"]["rows_proposed"],
                    "covariate_rows_validated": self.context_compilation[
                        "covariates"]["rows_validated"],
                    "rejection_count": len(self.context_compilation["rejections"]),
                    "future_observations_exposed": False,
                }
                if self.context_compilation is not None else None
            ),
            "trace": self.trace,
            "total_time": time.time() - self.started,
        }
        # CiK task instances carry no `seed` attribute and the forecaster
        # is one object shared across every task-seed, so `seed` is "x"
        # for all five runs of a task: writing to one name silently kept
        # the last run and discarded four (103 traces survived 355 runs).
        # A trace is diagnostic evidence; losing it costs a diagnosis.
        path = trace_dir / f"{name}-seed{seed}.json"
        suffix = 1
        while path.exists():
            suffix += 1
            path = trace_dir / f"{name}-seed{seed}.{suffix}.json"
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                        encoding="utf-8")

    # -- caps --------------------------------------------------------------
    def _run_tokens(self) -> int:
        client = self.forecaster.client
        return (client.total_prompt_tokens + client.total_completion_tokens
                - self.tokens_at_start)

    def _check_budget_caps(self) -> None:
        if time.time() - self.started > MAX_WALL_SECONDS:
            self._abstain(f"cap:wall_clock exceeded {MAX_WALL_SECONDS:.0f}s")
        if self._run_tokens() > MAX_RUN_TOKENS:
            self._abstain(f"cap:tokens exceeded {MAX_RUN_TOKENS}")

    def _abstain(self, reason: str) -> None:
        self.trace.append({"abstained": reason})
        raise GnomonAbstained([reason])

    # -- the loop ----------------------------------------------------------
    def drive(self) -> tuple[list[dict[str, float]], dict[str, Any]]:
        self.session.initialize()
        mcp_tools = self.session.list_tools()
        if self.governed_evidence:
            # This benchmark asks one known verb: forecast. Presenting a
            # descriptive detour measures tool distraction, not useful agent
            # autonomy; discovery belongs to hosts where the intent is unknown.
            mcp_tools = [tool for tool in mcp_tools
                         if tool.get("name") == "gnomon_forecast"]
        tools = openai_tool_specs(mcp_tools)
        future_index = _task_future_timestamps(self.task)
        system = SYSTEM.format(
            csv_path=str(self.csv_path), horizon=self.horizon,
            future_start=future_index[0], future_end=future_index[-1],
            max_rounds=MAX_ROUNDS, max_calls=MAX_MCP_CALLS,
        )
        if self.governed_evidence:
            system += "\n" + GOVERNED_CONTEXT_NOTE
        history = "\n".join(
            f"{ts},{value}" for ts, value in zip(self.timestamps, self.values))
        context = build_context_text(self.task)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": USER.format(
                context=context or "(no textual context)",
                n_obs=len(self.values), csv_path=str(self.csv_path),
                history=history, horizon=self.horizon,
            )},
        ]

        nudged = False
        for _round in range(MAX_ROUNDS):
            self._check_budget_caps()
            response = self.forecaster.client.chat(
                messages, n=1, tools=tools, tool_choice="auto")
            message = response.choices[0].message
            tool_calls = _tool_calls_as_dicts(message)
            if not tool_calls:
                messages.append({"role": "assistant",
                                 "content": message.content or ""})
                if self.submission:
                    break
                if nudged:
                    self._abstain("no submission: prose answers twice after nudge")
                nudged = True
                messages.append({
                    "role": "user",
                    "content": "Finish by calling submit_forecast — either "
                               "an artifact_path from a gnomon_forecast "
                               "result, or your own quantiles.",
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
                result = self._dispatch(
                    call["function"]["name"], arguments)
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": result})
                compacted = self.result_log.record(
                    call["function"]["name"], arguments, messages[-1])
                if compacted:
                    self.trace.append({"superseded": compacted})
            if self.submission:
                break
        if not self.submission:
            self._abstain(f"cap:rounds {MAX_ROUNDS} rounds without a submission")
        return self._resolve_submission()

    # -- dispatch ----------------------------------------------------------
    def _dispatch(self, name: str, arguments: dict[str, Any] | None) -> str:
        """Route one tool call; return the tool-message text the model sees."""
        entry: dict[str, Any] = {"tool": name}
        self.trace.append(entry)
        if arguments is None:
            entry["error"] = "unparseable arguments"
            return json.dumps({"code": "INVALID_ARGUMENTS",
                               "message": "Tool arguments were not valid JSON.",
                               "authored_by": "harness"})
        if name == "submit_forecast":
            payload = self._handle_submit(arguments)
            entry["result"] = payload
            return json.dumps(payload)

        if self.governed_evidence and name == "gnomon_forecast":
            receipt = self.context_compilation or {}
            from gnomon.llm_covariates import inline_covariate_arguments
            covariate_arguments = inline_covariate_arguments(
                receipt.get("covariates") or {})
            # These are host-owned bindings. A model can choose the verb but
            # cannot swap the data, horizon, or omit admitted context.
            arguments = {
                **arguments,
                "input": str(self.csv_path),
                "time_column": "timestamp",
                "target_column": "value",
                "horizon": self.horizon,
                "context_events": receipt.get("events", []),
                **covariate_arguments,
                "future_events": True,
                "structural_events": True,
                "output_dir": str(self.jail / "gnomon-output"),
                "format": "brief",
            }
            entry["host_context_binding"] = {
                "receipt_sha256": (receipt.get("source") or {}).get("sha256"),
                "events": len(receipt.get("events") or []),
                "covariate_tables_proposed": len(
                    (receipt.get("covariates") or {}).get("tables") or []),
                "covariate_table_bound": bool(covariate_arguments),
                "rejections": len(receipt.get("rejections") or []),
            }

        violations = jail_violations(arguments, self.jail)
        if violations:
            entry["jail_violations"] = violations
            payload = {
                "code": "PATH_JAIL",
                "message": "This run may only read and write inside its own "
                           "run directory. The history file is at "
                           f"{self.csv_path}.",
                "violations": violations,
                "authored_by": "harness",
            }
            return json.dumps(payload)
        if self.mcp_calls >= MAX_MCP_CALLS:
            self._abstain(f"cap:tool_calls exceeded {MAX_MCP_CALLS}")
        self.mcp_calls += 1
        self._check_budget_caps()
        try:
            result = self.session.call_tool(name, arguments)
        except Exception as error:
            # Transport death is a harness failure, disclosed as such.
            self._abstain(f"mcp transport failed: {error}")
        entry["is_error"] = bool(result.get("isError"))
        structured = result.get("structuredContent") or {}
        if isinstance(structured, dict):
            code = (structured.get("error") or {}).get("code") or structured.get("code")
            if code:
                entry["code"] = code
            if not result.get("isError") and structured.get("artifact_path"):
                artifact_path = str(structured["artifact_path"])
                self.artifact_paths.add(artifact_path)
                entry["artifact_path"] = artifact_path
                results = structured.get("results") or []
                if results and isinstance(results[0], dict):
                    entry["context_outcome"] = results[0].get("context_outcome")
                    entry["support"] = results[0].get("support")
                if (self.governed_evidence and name == "gnomon_forecast"
                        and self.submission is None):
                    # Evidence is a governed product arm. Once the agent has
                    # chosen Gnomon's forecast verb and it produced a valid
                    # artifact, publication is a deterministic host action.
                    # Requiring another model turn merely tests whether the
                    # model copies a path and caused measured 10-round loops.
                    bound = self._handle_submit({"artifact_path": artifact_path})
                    entry["host_bound_submission"] = bound
        # Verbatim: the server's own text block, unedited.
        content = result.get("content") or []
        text = content[0].get("text", "") if content else json.dumps(structured)
        return text

    # -- submission --------------------------------------------------------
    def _handle_submit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.submission is not None:
            return {"accepted": False, "authored_by": "harness",
                    "message": "A forecast was already submitted; the first "
                               "accepted submission stands."}
        artifact_path = arguments.get("artifact_path")
        quantiles = arguments.get("quantiles")
        if self.governed_evidence and quantiles is not None:
            return {
                "accepted": False,
                "authored_by": "harness",
                "message": "governed Evidence requires a Gnomon forecast "
                           "artifact; model-authored quantiles are disabled.",
            }
        if bool(artifact_path) == bool(quantiles):
            return {"accepted": False, "authored_by": "harness",
                    "message": "Provide exactly one of artifact_path or quantiles."}
        if artifact_path:
            if str(artifact_path) not in self.artifact_paths:
                return {"accepted": False, "authored_by": "harness",
                        "message": "artifact_path was not produced by a "
                                   "gnomon_forecast call in this run.",
                        "known_artifact_paths": sorted(self.artifact_paths)}
            rows = self._artifact_rows(str(artifact_path))
            if isinstance(rows, str):
                return {"accepted": False, "authored_by": "harness",
                        "message": rows}
            self.submission = {"route": "gnomon", "rows": rows,
                               "artifact_path": str(artifact_path),
                               "reasoning": arguments.get("reasoning")}
            return {"accepted": True, "route": "gnomon"}
        problems = self._quantile_problems(quantiles)
        if problems:
            return {"accepted": False, "authored_by": "harness",
                    "message": problems}
        route = "direct" if self.mcp_calls == 0 else "informed-direct"
        self.submission = {
            "route": route,
            "rows": [{"q10": float(row["q10"]), "q50": float(row["q50"]),
                      "q90": float(row["q90"]),
                      "point": float(row["q50"])} for row in quantiles],
            "reasoning": arguments.get("reasoning"),
        }
        return {"accepted": True, "route": route}

    def _artifact_rows(self, artifact_path: str) -> list[dict[str, Any]] | str:
        from gnomon.artifacts import read_artifact

        try:
            data = read_artifact(artifact_path)
        except Exception as error:
            return f"artifact could not be read: {error}"
        results = data.get("results") or []
        if not results:
            return "artifact holds no results"
        rows = results[0].get("forecast") or []
        if len(rows) != self.horizon:
            return (f"artifact horizon is {len(rows)}, task horizon is "
                    f"{self.horizon}; re-run gnomon_forecast with "
                    f"horizon={self.horizon}")
        self._submitted_support = results[0].get("support")
        self._submitted_model = results[0].get("selected_model")
        self._submitted_result = results[0]
        return rows

    def _quantile_problems(self, quantiles: Any) -> str | None:
        # A rejection must name what was received, or the model retries
        # the same shape until the rounds cap converts it to an
        # abstention (observed: 10 identical rejections on a dry run).
        if isinstance(quantiles, dict):
            keys = ", ".join(sorted(str(key) for key in quantiles))
            return (f"quantiles must be a list of exactly {self.horizon} "
                    f"{{q10, q50, q90}} objects (one per horizon step); got "
                    f"a single object with keys [{keys}]. Parallel per-"
                    f"quantile arrays are not accepted — send "
                    f'[{{"q10": ..., "q50": ..., "q90": ...}}, ...] with '
                    f"one entry per step.")
        if not isinstance(quantiles, list):
            return (f"quantiles must be a list of exactly {self.horizon} "
                    f"objects (one per horizon step); got "
                    f"{type(quantiles).__name__}")
        if len(quantiles) != self.horizon:
            return (f"quantiles must be a list of exactly {self.horizon} "
                    f"objects (one per horizon step); got {len(quantiles)}")
        import math

        for index, row in enumerate(quantiles):
            if not isinstance(row, dict):
                return f"quantiles[{index}] is not an object"
            for key in ("q10", "q50", "q90"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    return (f"quantiles[{index}].{key} must be a finite "
                            f"number; got {value!r:.60}")
        return None

    # -- result ------------------------------------------------------------
    def _resolve_submission(self) -> tuple[list[dict[str, float]], dict[str, Any]]:
        assert self.submission is not None
        extra_info: dict[str, Any] = {
            "adapter": self.forecaster.cache_name,
            "route": self.submission["route"],
            "mcp_calls": self.mcp_calls,
            "run_tokens": self._run_tokens(),
            "tool_sequence": [
                {key: value for key, value in entry.items()
                 if key in ("tool", "is_error", "code", "jail_violations")}
                for entry in self.trace
            ],
            "submit_reasoning": self.submission.get("reasoning"),
            "llm_usage": self.forecaster.client.usage_summary,
            "total_time": time.time() - self.started,
        }
        if self.context_compilation is not None:
            dossier = self.context_compilation["dossier"]
            extra_info["context_compilation"] = {
                "receipt_path": self.context_compilation["path"],
                "source_sha256": self.context_compilation["source"]["sha256"],
                "event_count": len(self.context_compilation["events"]),
                "claim_count": len(
                    self.context_compilation["dossier"]["claims"]),
                "hypothesis_count": len(dossier.get("hypotheses") or []),
                "hypothesis_status": (dossier.get("hypothesis_critique") or {}).get(
                    "status"),
                "candidate_available": bool(
                    self.context_compilation["dossier"].get("effect_proposal")
                    or self.context_compilation["dossier"].get("forecast_candidate")),
                "covariate_tables": len(
                    self.context_compilation["covariates"]["tables"]),
                "covariate_tables_proposed": self.context_compilation[
                    "covariates"]["tables_proposed"],
                "covariate_rows_proposed": self.context_compilation[
                    "covariates"]["rows_proposed"],
                "covariate_rows_validated": self.context_compilation[
                    "covariates"]["rows_validated"],
                "rejection_count": len(self.context_compilation["rejections"]),
                "future_observations_exposed": False,
            }
            if dossier.get("forecast_candidate") or dossier.get("effect_proposal"):
                # Retained for matched shadow scoring against this exact
                # compiler generation. It is never sent back into the agent
                # conversation and never replaces the canonical submission.
                extra_info["llm_candidate_shadow"] = {
                    "support": dossier["candidate_support"],
                    "seal_sha256": dossier["seal_sha256"],
                    "forecast_candidate": dossier["forecast_candidate"],
                    "effect_proposal": dossier.get("effect_proposal"),
                    "automation_eligible": False,
                    "primary_forecast_unchanged": True,
                }
        if self.submission["route"] == "gnomon":
            extra_info["artifact_path"] = self.submission["artifact_path"]
            extra_info["support"] = getattr(self, "_submitted_support", None)
            extra_info["selected_model"] = getattr(self, "_submitted_model", None)
        return self.submission["rows"], extra_info
