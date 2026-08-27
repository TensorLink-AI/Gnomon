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
import math
import os
import re
import statistics
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
MAX_CONTEXT_COMPILATION_SECONDS = max(1.0, min(
    300.0, float(os.environ.get("GNOMON_CONTEXT_COMPILATION_SECONDS", "60"))))
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
#: Version 12: the host skips the selector call when the product contract says
#: evidence already makes the recommendation non-discretionary.
#: Version 20: future schedules may cite multiple verified spans; each value
#: remains source-entailed and every source belongs to the transformation.
#: Version 21: invalid transformations receive one bounded provenance repair
#: before execution; the repair cannot remove already verified claims.
#: Version 22: bounded pre-cutoff companion channels remain available as
#: citable evidence instead of being discarded by target projection.
#: Version 23: transformation repair receives constant-specific verbatim
#: source hints, while the model still has to cite and bind them explicitly.
#: Version 24: exact relational context asks for a sealed probabilistic
#: fallback alongside the safer executable transformation.
#: Version 25: a directionally supported candidate inside a cited numeric
#: bound may leave the history scale as a warned, non-automatable scenario.
#: Version 26: that exception requires a cited quantitative relationship;
#: direction plus a loose bound is not enough to license a large jump.
#: Version 27: self-declared placeholders and zero-width model forecasts are
#: rejected rather than becoming human-facing prior-assisted answers.
#: Version 28: dossier compilation, repair, and governed selection use
#: deterministic-temperature calls independently of the conversational agent.
#: Version 29: genuine zero-width point candidates receive a disclosed robust
#: history-based uncertainty floor; self-declared placeholders remain rejected.
#: Version 30: the sole dossier repair receives every failed lane; an effect
#: critique can no longer hide a malformed probabilistic candidate.
#: Version 31: historical companion evidence is explicitly excluded from
#: future-known covariate tables, eliminating a noisy invalid-output loop.
#: Version 32: the LLM receives full-history deterministic summaries plus a
#: bounded target tail; the forecasting engine still consumes the full CSV.
#: Version 33: structured compilation, repair, and selection explicitly disable
#: hidden reasoning while the conversational agent retains its configured mode.
#: Version 34: a transformation can bind stale model-side citations to the sole
#: verified claim; all ordinary entailment checks still run after rebinding.
#: Version 35: extraction stays non-reasoning; the sole failed-numeric-lane
#: repair may use low reasoning to resolve implicit relationships adaptively.
#: Version 36: transformation preflight executes dummy-primary paths to catch
#: malformed, unentailed, or horizon-mismatched inputs before the live call.
#: Version 37: every governed structured call has a bounded transport deadline
#: with no hidden retry multiplier; repair is non-reasoning after the adaptive
#: reasoning experiment exceeded a production-acceptable wall time.
#: Version 38: mixed-unit linear equations use the governed
#: ``linear_combination`` macro with engine-derived coefficient units.
#: Version 39: a sealed candidate whose claimed governed derivation fails
#: preflight remains visible but is ineligible for recommendation selection.
#: Version 40: ordinary additive equations receive deterministic, disclosed
#: coefficient-unit normalization without requiring a special macro spelling.
#: Version 41: recursive linear equations bind trusted pre-cutoff state and
#: feed prior outputs back inside the sealed executor, never through LLM data.
#: Version 42: an explicit zero intercept is treated as additive identity, and
#: transformation driver schedules are not duplicated as covariate proposals.
#: Version 43: verbose cited lag-array equations canonicalize deterministically
#: to recursion; invented target arrays are discarded, not executed.
#: Version 44: exact lag claims without a numeric lane receive one focused,
#: bounded sufficiency repair instead of being silently interpretation-only.
#: Version 45: known-intent Evidence invokes its sole host-bound forecast
#: directly; open-intent profiles retain bounded conversational tool routing.
#: Version 46: numeric dataframe columns use the context's X_n aliases;
#: structured tool errors retain details and invalid recurrence history cannot
#: destroy the immutable primary forecast.
#: Version 47: unambiguous future/schedule driver aliases bind to governed
#: observed identities; collisions and inconsistent schedules still reject.
#: Version 48: recursive uncertainty uses linear state covariance, and
#: explosive dynamics remain visible but cannot lead a recommendation.
#: Version 49: nested lag(series) and pre-shifted lag-array spellings share one
#: deterministic recurrence canonical form, including future-prefix aliases.
#: Version 50: canonical recursive drivers accept the same unambiguous
#: future-/schedule-/forecast alias forms as verbose equations.
#: Version 51: recurrences must beat last-value in aligned pre-cutoff replay
#: before best-effort publication may recommend them.
#: Version 52: traces retain compact publication authority and recurrence
#: admission diagnostics, so recommendation behavior is auditable per case.
#: Version 53: redundant driver names such as X_0_lag2 are safely rebound to
#: X_0 only when their typed lag agrees and their future schedules match.
#: Version 54: a model-authored path cannot bypass the authority of a governed
#: transformation over the same claims, including a failed replay gate.
#: Version 55: compilation and its sole repair share one end-to-end deadline;
#: traces disclose per-stage latency instead of hiding stacked waits.
#: Version 56: the interactive default deadline is 60 seconds and is explicitly
#: configurable for offline evaluation.
#: Version 57: explicit lag equations use a compact typed extraction contract
#: and an eight-row evidence tail instead of the universal dossier packet.
#: Version 58: the host grounds explicit-equation documents when the compiler
#: omits a verbatim claim; entailment and replay still govern every number.
#: Version 59: compact recursive_linear output is deterministically normalized
#: to the public transformation envelope instead of spending a repair call.
#: Version 60: host-verified plural claim IDs take precedence over stale
#: compiler-authored singular IDs during compact normalization.
#: Version 61: timestamp/value schedule rows collapse to numeric arrays only
#: when their timestamps exactly match the host-owned forecast grid.
#: Version 62: traces retain replay sample size, skill, and both candidate and
#: baseline errors so a demotion is statistically diagnosable.
#: Version 63: cited historical driver ranges can bridge explicitly documented
#: semantics to encoded structured columns without inferred rescaling.
#: Version 64: typed recurrence objects and cited future range schedules are
#: normalized only under exact source/grid coverage.
#: Version 65: source dates entail midnight ISO endpoints on daily grids, and
#: traces retain typed execution-stage context dispositions.
#: Version 66: the host, not the compiler, binds transformation and supplied
#: series knowledge time to the sealed context receipt cutoff.
#: Version 67: explicit named piecewise-constant schedules are extracted from
#: source text when the compiler omits a recurrence driver's typed schedule.
#: Version 68: replay diagnostics populate generic evidence authority, and
#: best-effort publication prioritizes historically admitted transformations.
#: Version 69: malformed model-authored range schedules no longer mask an
#: exact, source-cited schedule that the host can extract deterministically.
#: Version 70: recurrence traces distinguish origin-safe observations from a
#: specification supplied only at the current cutoff.
#: Version 71: compiler-level context rejection remains visible in the public
#: typed disposition instead of disappearing between receipt and publication.
#: Version 72: traces retain the bounded recovery path for every rejected
#: context item, matching the human- and agent-facing publication envelope.
#: Version 73: traces distinguish an independently selected recommendation
#: from the default prior-assisted lane and disclose mandatory human review.
#: Version 74: historical observation semantics use a focused compiler lane;
#: fitted counterfactual replay and truthful publication authority are visible.
#: Version 75: plain model-authored paths require an explicit governed
#: selection; replay-admitted paths remain evidence-dominant. This changes the
#: returned forecast and therefore must invalidate benchmark prediction caches.
#: Version 76: candidate distributions must beat the raw comparator under
#: fold-safe probabilistic replay, not point MAE alone.
#: Version 77: cited semantic zeros may expose a sharply separated noisy-zero
#: sensitivity, while outcome-inferred membership remains non-admissible.
#: Version 78: zero-cluster separation is keyed to the tight censored component
#: while allowing ordinary activity to remain naturally broad.
#: Version 79: overlapping noisy-zero regimes use a bounded deterministic
#: two-component sensitivity, still categorically ineligible for self-admission.
#: Version 80: the governed selector sees compact candidate derivation and
#: admission facts rather than inferring authority from scenario IDs.
#: Version 81: literal exact-value overrides with source-cited endpoints apply
#: to every boundary quantile instead of inheriting timing uncertainty.
#: Version 82: source-cited recurring daily clock windows compile into a
#: fold-replayed historical observation counterfactual.
#: Version 93: bounded effects preserve a stated plateau and multiply every
#: conditional quantile rather than shifting all bounds by the median effect.
#: Version 94: fold-starved forecasts may admit seasonal-naive over last-value
#: from six or more independent seasonal probes with chronological stability.
#: Version 95: exact cited multipliers repair vague custom shape labels, and a
#: governed host executes one server-authored capped grid repair without an LLM
#: improvisation turn.
#: Version 96: numeric context cannot disappear through an empty successful
#: compile; one bounded sufficiency repair must type it or reject it explicitly.
#: Version 97: past-tense yearless month/day references bind to the latest
#: occurrence at or before the host-owned cutoff, with provenance retained.
#: Version 98: inapplicable null lags no longer discard analogue hypotheses,
#: and sealed model candidates may use compact host-interpolated quantile anchors.
#: Version 99: numeric-context repair distinguishes best-effort prior-assisted
#: anchor scenarios from strict evidence admission instead of defaulting to refusal.
#: Version 100: explicit compiler-confidence ranges retain grounded claims at
#: their conservative endpoint; parsing confidence still grants no authority.
#: Version 101: sparse timestamped quantile rows use the same strict,
#: provenance-disclosed interpolation contract as explicit quantile anchors.
#: Version 102: the formal candidate schema made the anchor contract explicit.
#: Version 103: meaningful interior anchors may omit horizon edges; Gnomon
#: completes only those unanchored edges from the immutable primary and records
#: that provenance instead of asking the model to invent endpoints.
#: Version 104: malformed compiler-confidence metadata retains a verbatim
#: grounded claim at a disclosed conservative floor; confidence remains
#: non-authoritative, while explicit out-of-range numeric values still fail.
MCP_CONTRACT_VERSION = 104
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

_VERBOSE_DOSSIER_CONTRACT_REFERENCE = """\
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
  "observation_interpretations": [
    {"kind": "historical_contamination", "claim_ids": ["claim-1"],
     "predicate": {"op": "equals", "value": 0.0},
     "window": "cited_window | all_observed_history",
     "rationale": "why these readings are not future-generating behavior"}
  ],
  "forecast_candidate": {
    "quantiles": [{"timestamp": "exact requested timestamp", "q10": 0.0,
                   "q50": 0.0, "q90": 0.0}],
    "quantile_anchors": [
      {"timestamp": "FIRST requested timestamp", "q10": 0.0,
       "q50": 0.0, "q90": 0.0},
      {"timestamp": "optional exact turning-point timestamp", "q10": 0.0,
       "q50": 0.0, "q90": 0.0},
      {"timestamp": "LAST requested timestamp", "q10": 0.0,
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
  ,"transformations": [
    {"transformation": {
       "known_at": "history cutoff ISO", "claim_ids": ["claim-1"],
       "lane": "historically_testable | prior_assisted | scenario_only",
       "output_unit": "declared target unit",
       "expression": {"op": "add", "args": [
          {"op": "primary", "quantile": "q50"},
          {"op": "series", "name": "future_input"}]}},
     "units": {"primary": "declared target unit",
               "future_input": "declared target unit"},
     "series_values": {
       "future_input": {"values": [0.0], "known_at": "history cutoff ISO",
                        "source_claim_ids": ["claim-1"]}
    }
  ]
}

Rules:
- Cite only exact spans present in context; never invent an event or source.
- Prefer effect_proposal for a simple cited shift or pulse and let Gnomon
  compose those numbers. For a precise multi-input relationship, physical
  law, or piecewise schedule, provide the safe transformation and ALSO a
  forecast_candidate as a sealed fallback when you can compute a useful
  probabilistic path. The fallback remains prior_assisted, cannot automate,
  and never replaces the immutable primary. Never claim numeric values came
  from text unless the cited span states them; derived candidate values must
  explain their arithmetic in the rationale.
- Effect location/lower/upper are changes added to the primary path, not target
  values. A stated level of "4 times usual" is therefore an additive
  fraction_of_level change of 3.0, not 4.0; Gnomon rechecks this arithmetic
  against the cited span. If the text states an exact future value (for example, withdrawals
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
- A numeric effect must cite both its magnitude and its timing. Prefer one
  verbatim claim span containing both. If the source separates them, emit one
  claim for the dated window and one for the numeric relationship, then cite
  both claim IDs from the effect/transform. Never attach an uncited timestamp
  to a magnitude-only sentence.
- When context permits more than one interpretation, emit up to six competing
  typed hypotheses rather than collapsing ambiguity into one numeric path.
  A relationship names its predictor and lag; an historical_analogue names
  only a cited analogue claim. Gnomon validates and evaluates these after the
  model response. Parsing confidence never upgrades support or automation.
- Covariate tables are extraction, never invention. Emit a row only when one
  verbatim quote contains both its time token and numeric value. Do not infer
  values from adjectives, interpolate missing rows, or supply known_at; the
  host owns knowledge time. Gnomon will test surviving tables out of sample
  before they may influence the canonical forecast. `Observed companion-series
  history` is historical evidence for hypotheses, transformations, and
  candidate reasoning; NEVER copy those historical rows into covariate_tables.
  A covariate row must fall on an exact requested forecast timestamp and its
  future value must be explicitly stated in the source context.
- Transformations are a restricted declarative lane, never code. Approved
  operators are literal, primary, series, add, subtract, multiply, divide,
  power, lag, difference, percent_change, rolling_mean, clip, and quantile. Use a
  transformation only when cited context states a precise prospective rule or
  supplies every future input value. Constants and future values must be
  verbatim-entailable from cited claims; do not extrapolate or fill them. Use
  `historically_testable` only when the relationship can be replayed from
  point-in-time history, `prior_assisted` for a precise stated future rule,
  and `scenario_only` when it is merely conditional. The engine validates and
  seals the AST; it never executes generated Python, SQL, or expressions.
- AST grammar is exact: binary/variadic arithmetic uses
  `{\"op\":\"add|subtract|multiply|divide\",\"args\":[NODE,...]}`;
  series uses `{\"op\":\"series\",\"name\":\"future_input\"}`; lag uses
  `{\"op\":\"lag\",\"args\":[NODE],\"steps\":2}`; bounded powers use
  `{\"op\":\"power\",\"args\":[NODE,{\"op\":\"literal\",\"value\":2}]}`.
  Unary change/rolling/clip/quantile nodes likewise put the child in a
  one-element `args` array and their parameter beside it. Every referenced
  series name must exactly match a `series_values` key, whose values array has
  exactly one item per forecast timestamp. When a schedule is split across
  spans, `source_claim_ids` cites every claim needed to entail all distinct
  values, and the transformation's own `claim_ids` includes them. Use
  canonical claim IDs (`claim-1`, `claim-2`, ...) in verified claim order.
- For a cited reference law use the compact safe macro
  `{"op":"reference_power","series":"driver","input_reference":{"value":3000,"unit":"rpm"},"output_reference":{"value":37.5,"unit":"Pa"},"exponent":2}`.
  It means `37.5 Pa * (driver / 3000 rpm)^2`; Gnomon expands and seals it as
  ordinary arithmetic. Do not add it to the primary forecast unless the cited
  source explicitly states a delta rather than an absolute relationship.
- For a cited linear equation across differently-unitized series, use
  `{"op":"linear_combination","output_unit":"target units","terms":[{"coefficient":1.2,"series":"driver_lag1"},{"coefficient":-0.4,"series":"target_lag1"}],"intercept":0.0}`.
  Gnomon derives each coefficient's target/input conversion unit, validates
  every coefficient against the cited claims, and expands the macro into
  ordinary sealed arithmetic. Omit `intercept` when the equation has none.
- For a cited recurrence such as an ARX equation, never invent future target
  lag values. Use `{"op":"recursive_linear","output_unit":"target units","intercept":0,"autoregressive_terms":[{"lag":1,"coefficient":0.5}],"driver_terms":[{"series":"driver","lag":1,"coefficient":2.0}]}`.
  Gnomon binds pre-cutoff target/driver history from the governed snapshot,
  recursively feeds prior outputs back itself, and propagates uncertainty.
  `series_values` supplies only the cited future driver schedule; do not put
  target lags or historical observations in `series_values`. Do not duplicate
  a schedule already consumed by a transformation in `covariate_tables`.
- Use no observations after the history cutoff. Return empty arrays and null
  effect_proposal and forecast_candidate when context contains no
  forecast-relevant information.
"""

DOSSIER_INSTRUCTIONS = """\
Compile supplied temporal context into one governed JSON dossier. Return ONLY
JSON with these eight keys; use [] or null when absent:
{"events":[],"claims":[],"hypotheses":[],"covariate_tables":[],
"transformations":[],"observation_interpretations":[],
"effect_proposal":null,"forecast_candidate":null}

Shapes:
- event: {document_index:0,event_type,entity_scope:["*"],effective_start,
effective_end,confidence,status:"confirmed|tentative",evidence_quote,
effect_family:"level_shift|trend_change|variance_change|temporary_pulse|
saturation_bound|seasonal_regime_change|unknown",direction:"increase|decrease|
unknown",duration:"temporary|persistent|unknown",entity_kind:"service|product|
medication|procedure|calendar|capacity|price|environment|unknown"}.
- claim: {source_span,relation:"supports_increase|supports_decrease|
supports_stability|supports_higher_variance|supports_lower_variance|
changes_seasonal_regime|constrains_range|unknown",effective_start,effective_end,
mechanism,confidence}.
- observation_interpretation: {kind:"historical_contamination",claim_ids:
["claim-1"],predicate:{op:"equals",value:0}|{op:"recurring_window",start,
duration_steps,period_steps},window:
"cited_window|all_observed_history",rationale}; emit only for a literally
stated historical zero/absence caused by a data-generating disruption.
- hypothesis: {kind:"absolute_value|bound|additive_change|
multiplicative_change|regime_shift|relationship|historical_analogue|
unsupported",claim_ids:["claim-1"],target_series:["*"],predictor_series,
known_at,lag_steps,direction,rationale}; emit at most six.
- effect_proposal: {shape:"temporary_pulse|level_shift|trend_change|
variance_change|ramp_recovery|seasonal_amplitude|seasonal_phase|
cross_series_relationship|saturation_bound|custom_scenario",unit:
"target_units|fraction_of_level",location,lower,upper,confidence,delay_steps,
duration_steps,period_steps,scope:{kind:"single_series",series:["*"]},
claim_ids,rationale,uncertainty_basis}.
- forecast_candidate: {quantiles:[{timestamp,q10,q50,q90}],rationale}; include
every exact requested forecast timestamp, OR use quantile_anchors with one or
more ordered {timestamp,q10,q50,q90} rows at meaningful requested timestamps.
Gnomon interpolates between anchors and completes only unanchored horizon edges
from the immutable primary, with that provenance disclosed. Never invent edge
anchors merely to cover the grid. Omit the candidate when neither
representation is supportable.
- covariate table: {name,type:"continuous|binary|cyclic_<period>",rows:[
{document_index:0,timestamp,source_time_span,value,evidence_quote}]}.
- transformation wrapper: {transformation:{known_at,claim_ids,lane:
"historically_testable|prior_assisted|scenario_only",output_unit,expression},
units:{primary:"unit",series_name:"unit"},series_values:{series_name:
{values:[],known_at,source_claim_ids:[]}}}.

Rules:
1. Every source_span/evidence_quote is verbatim context. Never invent a fact,
timestamp, magnitude, unit, source, or future target observation. Numeric
influence cites both timing and magnitude. Claim IDs follow verified order.
2. Events overlap the requested future window. Use constraint:<label> for a
stated bound and override:<label> for an exact future value. Other events are
qualitative; historical events/regimes are claims, never backdated events.
3. Prefer effect_proposal for a cited shift or pulse. location/lower/upper are
changes to the primary, not target levels ("4 times usual" is an additive
fraction_of_level change of 3). Exact target values use override events.
Never infer an exact zero from words such as closed, outage, or unavailable;
without a stated target value, use a qualitative event/hypothesis and an
explicitly prior_assisted forecast_candidate only if you can derive one.
4. Preserve ambiguity as competing hypotheses. Confidence never upgrades
support or automation. A model-authored candidate is prior_assisted only,
never automation-eligible, and its rationale explains derived arithmetic.
Historical statements that readings were corrupted by an outage, closure,
stockout, censoring, or reporting failure remain forecast-relevant even when
the event will not recur. Emit a verbatim claim plus an `unsupported` or
`regime_shift` hypothesis describing the observation semantics. You may emit a
sealed forecast_candidate estimated from unaffected history; label its
counterfactual arithmetic and uncertainty. Do not rewrite history or create a
future event when the source says the disruption has ended.
When the source literally says zero or no target events were recorded because
of that historical disruption, add an observation_interpretation with the
exact-zero predicate. Use cited_window when dates are stated; use
all_observed_history only when the prose explicitly describes the corruption
as historical but supplies no exact dates. Gnomon applies this predicate to a
copy, discloses retained/excluded counts, and may derive a non-automatable
counterfactual; it never cleans or mutates the immutable primary.
5. Covariates are verbatim future extraction: each quote contains its time and
value, and timestamp is an exact requested forecast timestamp. Never infer or
interpolate. NEVER copy those historical rows into covariate_tables; observed
companion history is evidence only.
6. Transformations are cited declarative ASTs, never code. Operators: literal,
primary,series,add,subtract,multiply,divide,power,lag,difference,
percent_change,rolling_mean,clip,quantile,reference_power,
linear_combination,recursive_linear. Arithmetic uses args:[NODE,...]; series
uses {op:"series",name}; lag uses {op:"lag",args:[NODE],steps:N}. Every future
series has exactly one cited series_values item per forecast timestamp. Never
fill, extrapolate, or duplicate it as a covariate.
7. historically_testable requires point-in-time replay; prior_assisted is a
precise prospective rule; scenario_only is conditional. For recursive_linear,
supply cited future drivers only—Gnomon binds history and prior outputs.
8. For a precise cited law or schedule prefer a safe transformation; also add
a sealed forecast_candidate only when you can calculate a useful probabilistic
path. Gnomon validates citations, units, arithmetic and seals. Invalid content
is rejected without changing the primary.
9. Use nothing after the history cutoff. If context is irrelevant, return the
seven empty/null fields.
"""


def _expects_historical_zero_interpretation(context: str) -> bool:
    """Whether prose explicitly supports asking for the typed zero lane.

    This only schedules a bounded repair; it does not create a claim, choose a
    window, or grant numeric authority. The repaired dossier still passes the
    ordinary citation, entailment, filtering, and sealing boundary.
    """
    text = " ".join(context.casefold().split())
    disruption = any(token in text for token in (
        "maintenance", "outage", "closure", "stockout", "reporting failure"))
    exact_absence = bool(
        re.search(r"\b(?:zero|no)\b.{0,50}\b(?:recorded|withdrawal|sale|order|request|reading|transaction|event)s?\b", text)
        or re.search(r"\b(?:recorded|withdrawal|sale|order|request|reading|transaction|event)s?\b.{0,50}\b(?:zero|none)\b", text))
    ended = any(token in text for token in (
        "no future", "has ended", "had ended", "will not recur",
        "will no longer", "does not continue")) or bool(re.search(
            r"\bwill not be (?:in|under|on) (?:maintenance|an? outage|closure)\b",
            text))
    return disruption and exact_absence and ended


OBSERVATION_INSTRUCTIONS = """\
Compile historical observation semantics into one governed JSON dossier.
Return ONLY these eight keys:
{"events":[],"claims":[],"hypotheses":[],"covariate_tables":[],
"transformations":[],"observation_interpretations":[],
"effect_proposal":null,"forecast_candidate":null}

Copy the exact sentence saying a historical disruption caused zero or no
recorded target activity into one claim. Historical claims use their stated
dates; if no dates are stated use the supplied history bounds. Because the
source says the disruption ended, emit no future event or effect.

If the sentence states a recurrence, add:
{"kind":"historical_contamination","claim_ids":["claim-1"],
 "predicate":{"op":"recurring_window","start":"stated date",
 "duration_steps":1,"period_steps":2},"window":"cited_window",
 "rationale":"brief observation semantics"}
using only verbatim numbers. If dates/schedule are absent, do not guess a mask.
For a source-stated daily clock window, use
{"op":"recurring_clock_window","start_time":"20:00","end_time":"00:00"}
with the two times copied verbatim.

For best-effort human use, you may instead author a sealed forecast_candidate
from the supplied numeric history and the fact that the disruption will not
recur. Use either quantiles with one row per exact forecast timestamp, or
{"constant_quantiles":{"q10":0,"q50":0,"q90":0}} when the same distribution
applies to every step; Gnomon expands that compact form onto its host-owned
grid. It must have non-zero uncertainty, cite no unseen outcomes, and explain
its arithmetic.
This path is prior_assisted, cannot edit the immutable primary, upgrade
support, or authorize automation. Return null if you cannot compute it.
"""

RELATIONSHIP_INSTRUCTIONS = """\
Extract one explicit lagged numeric relationship for Gnomon's safe recurrence
executor. Return ONLY JSON with `claims` and `transformations`; set both to []
if the text does not state an exact equation. Claims quote the equation and any
future driver schedule verbatim. The transformation uses `recursive_linear`
with numeric `intercept`, `autoregressive_terms` ({lag, coefficient}), and
`driver_terms` ({series, lag, coefficient}). Put cited future driver values in
`series_values`. If the text states historical driver ranges, put them in
`historical_series_segments` keyed by series, with rows
{start, end, value, source_claim_ids}; do not infer or rescale them. Never
supply future target values or executable code. Use the
history cutoff as `known_at`, `historically_testable` as the lane, and preserve
source series names. Each transformation claim_id refers to the 1-based order
of its claim (`claim-1`, ...).
"""


def _has_explicit_lag_relationship(text: str) -> bool:
    """Recognize explicit equation syntax, not a benchmark task family."""
    numeric = bool(re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text))
    lag = bool(
        re.search(r"\blag\s*[-_ ]?\d+\b", text, re.I)
        or re.search(r"\b[A-Za-z_]\w*\s*[\[(]\s*t\s*-\s*\d+\s*[\])]", text)
        # Scientific and business documents commonly render equations in
        # LaTeX-ish superscript notation: X_1^{t-2}. Treating that as generic
        # prose sends an exact relationship through the much larger universal
        # dossier contract and makes extraction slower and less reliable.
        or re.search(r"\b[A-Za-z_]\w*\s*\^\s*\{\s*t\s*-\s*\d+\s*\}", text))
    equation = "=" in text or bool(re.search(r"\b(coefficient|affects?)\b", text, re.I))
    return numeric and lag and equation


def _extract_explicit_driver_schedule(
    text: str, *, series: str, cutoff: str, future_timestamps: list[str],
    claim_id: str = "claim-1",
) -> tuple[list[dict[str, Any]], list[float]] | None:
    """Extract verbatim piecewise-constant ranges for one named driver."""
    ranges = []
    pattern = re.compile(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+from\s+"
        r"(\d{4}-\d{2}-\d{2})(?:T[^\s,;]+)?\s+to\s+"
        r"(\d{4}-\d{2}-\d{2})(?:T[^\s,;]+)?", re.I)
    for line in text.splitlines():
        if series.casefold() not in line.casefold():
            continue
        for match in pattern.finditer(line):
            value, start, end = float(match.group(1)), match.group(2), match.group(3)
            ranges.append({"start": start, "end": end, "value": value,
                           "source_claim_ids": [claim_id]})
    if not ranges:
        return None
    cutoff_date = cutoff[:10]
    historical = [item for item in ranges if item["end"] <= cutoff_date]
    future_ranges = [item for item in ranges if item["end"] > cutoff_date]
    future = []
    for timestamp in sorted(future_timestamps):
        day = timestamp[:10]
        matches = [item for item in future_ranges
                   if item["start"] <= day <= item["end"]]
        if len(matches) != 1:
            return None
        future.append(float(matches[0]["value"]))
    return historical, future


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


def _task_target_name(task_instance: Any) -> str:
    """Preserve the source column's semantic identity for context binding."""
    past = task_instance.past_time
    if hasattr(past, "columns") and len(past.columns):
        name = _semantic_column_name(past.columns[-1])
        return name or "value"
    return "value"


def _semantic_column_name(value: Any) -> str:
    """Match CiK's numeric dataframe columns to the names in its context."""
    text = str(value).strip()
    return f"X_{text}" if re.fullmatch(r"\d+", text) else text


def _task_companion_evidence(task_instance: Any, *, limit: int = 32) -> str:
    """Render bounded, pre-cutoff companion channels as citable evidence.

    CiK's multivariate tasks put predictors beside the target in ``past_time``.
    The forecast itself remains univariate, but dropping those observed
    channels before context compilation makes legitimate relationships
    impossible to ground.  Keep only a bounded tail and label it as observed
    history; no future target or predictor values are introduced here.
    """
    past = task_instance.past_time
    if not hasattr(past, "columns") or len(past.columns) < 2:
        return ""
    target = past.columns[-1]
    companions = [column for column in past.columns if column != target]
    if not companions:
        return ""
    tail = past.loc[:, companions].tail(limit)
    rows = ["Observed companion-series history (known before the cutoff):"]
    rows.append("timestamp," + ",".join(
        _semantic_column_name(column) for column in companions))
    for timestamp, values in tail.iterrows():
        stamp = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
        rows.append(stamp + "," + ",".join(
            repr(float(value)) for value in values.values))
    return "\n".join(rows)


def _task_companion_histories(task_instance: Any) -> dict[str, list[float]]:
    """Return full pre-cutoff companion histories for engine execution only."""
    past = task_instance.past_time
    if not hasattr(past, "columns") or len(past.columns) < 2:
        return {}
    target = past.columns[-1]
    return {_semantic_column_name(column): [float(value) for value in past[column].values]
            for column in past.columns if column != target}


def _compiler_target_evidence(timestamps: list[str], values: list[float],
                              *, limit: int = 64) -> str:
    """Compact target evidence for the LLM; the engine retains the full CSV."""
    if not values:
        return "Numeric target history: unavailable"
    differences = [abs(right - left)
                   for left, right in zip(values, values[1:])]
    summary = {
        "observations": len(values),
        "first": values[0], "last": values[-1],
        "minimum": min(values), "maximum": max(values),
        "median": statistics.median(values),
        "median_absolute_first_difference": (
            statistics.median(differences) if differences else 0.0),
    }
    start = max(0, len(values) - limit)
    rows = [
        "Numeric target history summary (computed over the complete pre-cutoff series):",
        json.dumps(summary, sort_keys=True),
        f"Recent target tail (last {len(values) - start} observations):",
        "timestamp,value",
    ]
    rows.extend(f"{timestamp},{value}" for timestamp, value in
                zip(timestamps[start:], values[start:]))
    return "\n".join(rows)


def _transformation_repair_hints(
        failures: list[dict[str, Any]], context: str) -> list[str]:
    """Return verbatim context lines that may entail rejected constants."""
    constants: set[str] = set()
    for failure in failures:
        for violation in failure.get("violations") or []:
            match = re.search(
                r"Transformation constant ([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
                str(violation.get("message") or ""))
            if match:
                constants.add(match.group(1))
    hints = []
    for line in context.splitlines():
        stripped = line.strip()
        if stripped and any(re.search(
                rf"(?<![\d.]){re.escape(value)}(?![\d.])", stripped)
                            for value in constants):
            hints.append(stripped)
    return list(dict.fromkeys(hints))[:6]


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
                       csv_path: Path, target_name: str = "value",
                       companions: dict[str, list[float]] | None = None) -> None:
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        names = list((companions or {}).keys())
        writer.writerow(["timestamp", *names, target_name])
        for index, (timestamp, value) in enumerate(zip(timestamps, values)):
            writer.writerow([timestamp, *[repr(float(companions[name][index]))
                                           for name in names],
                             repr(float(value))])


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
            live_publication = getattr(run, "_publication", None)
            selection = None
            selection_error = None
            if self.output_role == "publication_best_effort":
                from gnomon.publication import (build_scenario_catalog,
                                                scenario_selection_contract,
                                                select_publication,
                                                validate_scenario_selection)
                from gnomon.temporal_state import build_temporal_state
                scenarios, _ = build_scenario_catalog(
                    artifact_result, dossiers=[dossier])
                if len(scenarios) > 1:
                    contract = scenario_selection_contract(
                        scenarios=scenarios, dossiers=[dossier],
                        temporal_state=build_temporal_state(
                            artifact_result, dossiers=[dossier]))
                    if contract.get("selection_required") is False:
                        # A host should not pay for a choice the verifier would
                        # reject. Emptying this local list only skips the model
                        # call; publish_result rebuilds the full portfolio.
                        selection_error = (
                            "selector skipped: governed evidence dominance")
                        scenarios = []
                if len(scenarios) > 1:
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
                                [{"role": "user", "content": prompt}], n=1,
                                temperature=0, reasoning_effort="none",
                                request_timeout=120,
                                transport_retries=0)[0]
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
            if self.output_role == "publication_best_effort" and live_publication:
                # This is the product result returned by the real MCP call,
                # including typed transformation use/rejection. Re-point its
                # sealed recommendation only when the governed selector earns
                # a valid choice; never rebuild or reforecast the artifact.
                if selection is not None:
                    publication = select_publication(
                        live_publication, selection)
                else:
                    publication = live_publication
                    if selection_error is None:
                        selection_error = "live MCP publication used without selection"
            else:
                if self.output_role == "llm_candidate_shadow":
                    # Shadow is an explicit evaluation instrument: score the
                    # sealed model candidate without making it the product
                    # default. Express that choice through the same governed
                    # selector contract so forecast values remain untouchable.
                    from gnomon.publication import build_scenario_catalog
                    shadow_scenarios, _ = build_scenario_catalog(
                        artifact_result, dossiers=[dossier])
                    shadow = next((item for item in shadow_scenarios
                                   if item.get("role") == "model_authored"), None)
                    if shadow is not None:
                        remaining = [item["scenario_id"]
                                     for item in shadow_scenarios
                                     if item["scenario_id"] != shadow["scenario_id"]]
                        claim_ids = list(shadow.get("claim_ids") or [])
                        selection = {
                            "selected_scenario_id": shadow["scenario_id"],
                            "ranking": [shadow["scenario_id"], *remaining],
                            "cited_claim_ids": claim_ids,
                            "counterevidence_claim_ids": [],
                            "confidence": .5,
                            "rationale": "Explicit shadow evaluation of the sealed candidate.",
                            "what_would_change_selection": "Resolved outcomes score this candidate.",
                        }
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
            run.final_submission = {
                "route": extra_info.get("route"),
                "recommended_scenario_id": publication.get(
                    "recommended_scenario_id"),
                "primary_forecast_unchanged": publication.get(
                    "primary_forecast_unchanged"),
                "automation_eligible": (
                    publication.get("automation") or {}).get("eligible"),
                "recommendation_authority": publication.get(
                    "recommendation_authority"),
                "scenario_selector": extra_info.get("scenario_selector"),
            }
            # ``drive`` closes the MCP process before governed selection to
            # avoid holding an idle server during the second model call. Rewrite
            # the same trace after selection so the diagnostic names the output
            # that was actually scored rather than only the initial MCP result.
            run._write_trace()
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
        self.target_name = _task_target_name(task_instance)
        self.companion_evidence = _task_companion_evidence(task_instance)
        self.companion_histories = _task_companion_histories(task_instance)
        self.csv_path = self.jail / "history.csv"
        _write_history_csv(
            self.timestamps, self.values, self.csv_path, self.target_name,
            self.companion_histories)
        self.horizon = len(task_instance.future_time)
        self.session = forecaster.session_factory(self.jail)
        self.trace: list[dict[str, Any]] = []
        self.result_log = ToolMessageLog()
        self.mcp_calls = 0
        self.artifact_paths: set[str] = set()
        self.submission: dict[str, Any] | None = None
        self.final_submission: dict[str, Any] | None = None
        self._trace_path: Path | None = None
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
        from gnomon.llm_dossier import (
            deterministic_historical_observation_claim,
            validate_temporal_dossier,
        )
        from gnomon.workflows import DocumentRef, parse_context_response

        narrative_context = build_context_text(self.task)
        context = "\n\n".join(part for part in (
            narrative_context, self.companion_evidence) if part)
        future_timestamps = _task_future_timestamps(self.task)
        relationship_contract = _has_explicit_lag_relationship(context)
        observation_contract = (
            not relationship_contract
            and _expects_historical_zero_interpretation(context))
        compiler_context = (narrative_context if relationship_contract else context)
        history = _compiler_target_evidence(
            self.timestamps, self.values,
            limit=8 if relationship_contract else
            128 if observation_contract else 64)
        instructions = (RELATIONSHIP_INSTRUCTIONS if relationship_contract
                        else OBSERVATION_INSTRUCTIONS if observation_contract
                        else DOSSIER_INSTRUCTIONS)
        prompt = (
            f"{instructions}\n"
            f"Forecast target series: {self.target_name}\n"
            f"History cutoff: {self.timestamps[-1]}\n"
            f"Forecast timestamps: {json.dumps(future_timestamps)}\n"
            f"{history}\n\n"
            f"Context:\n{compiler_context or '(none)'}\n"
        )
        raw: dict[str, Any] = {}
        compile_rejections: list[str] = []
        repair_used = False
        compilation_started = time.monotonic()
        compilation_deadline = (compilation_started
                                + MAX_CONTEXT_COMPILATION_SECONDS)
        compiler_calls: list[dict[str, Any]] = []

        def bind_active_target(candidate: dict[str, Any]) -> dict[str, Any]:
            """Attach host-owned series identity without granting semantics."""
            return {**candidate, "series": [self.target_name]}

        def complete(content: str, stage: str) -> str:
            remaining = compilation_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "context workflow deadline exhausted before " + stage)
            started = time.monotonic()
            try:
                return self.forecaster.client.completions(
                    [{"role": "user", "content": content}], n=1,
                    temperature=0, reasoning_effort="none",
                    request_timeout=max(1, min(
                        120, math.ceil(remaining))),
                    transport_retries=0)[0]
            finally:
                compiler_calls.append({
                    "stage": stage,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                })
        try:
            completion = complete(prompt, "initial_compile")
            objects = extract_json_objects(completion)
            if objects:
                raw = bind_active_target(objects[0])
            else:
                compile_rejections.append(
                    "no JSON object in temporal-dossier output")
        except Exception as error:
            compile_rejections.append(f"dossier compilation failed: {error}")

        # In the compact equation contract the host can ground the document
        # without asking the model to copy a long formula byte-for-byte. This
        # grants no semantic or numeric authority: AST constants still need
        # source entailment and the executable must still pass replay.
        if relationship_contract and raw.get("transformations"):
            claims = [item for item in raw.get("claims") or []
                      if isinstance(item, dict)]
            grounded = any(str(item.get("source_span") or "") in context
                           and str(item.get("source_span") or "").strip()
                           for item in claims)
            if not grounded and narrative_context.strip():
                raw["claims"] = [{
                    "source_span": narrative_context,
                    "relation": "unknown",
                    "effective_start": future_timestamps[0],
                    "effective_end": future_timestamps[-1],
                    "mechanism": "host-grounded explicit equation document",
                    "confidence": 1.0,
                }]
                for wrapper in raw.get("transformations") or []:
                    transformation = (wrapper.get("transformation", wrapper)
                                      if isinstance(wrapper, dict) else {})
                    if isinstance(transformation, dict):
                        transformation["claim_ids"] = ["claim-1"]
                    for supplied in ((wrapper.get("series_values") or {}).values()
                                     if isinstance(wrapper, dict) else []):
                        if isinstance(supplied, dict):
                            supplied["source_claim_ids"] = ["claim-1"]

        # The host assembled this receipt at the cutoff. Compiler-authored
        # knowledge times are neither trusted nor useful; bind all numeric
        # lanes to the host-owned timestamp before any validation or repair.
        for wrapper in raw.get("transformations") or []:
            if not isinstance(wrapper, dict):
                continue
            transformation = wrapper.get("transformation", wrapper)
            if isinstance(transformation, dict):
                transformation["known_at"] = self.timestamps[-1]
            for supplied in (wrapper.get("series_values") or {}).values():
                if isinstance(supplied, dict):
                    supplied["known_at"] = self.timestamps[-1]

        # Exercise the product's bounded repair lane. The first response is
        # probed before event parsing so a corrected complete dossier (claims
        # plus effect) feeds every downstream validator consistently.
        proposed_any_lane = any(raw.get(key) not in (None, [], {}) for key in (
            "events", "claims", "hypotheses", "covariate_tables",
            "transformations", "observation_interpretations",
            "effect_proposal", "forecast_candidate"))
        observation_lane_missing = (
            _expects_historical_zero_interpretation(context)
            and not raw.get("observation_interpretations"))
        # An empty response to numeric context is not a successful compile.
        # It is common for useful references (a comparable site's peak, a
        # budget, a dated rate) to be informative but not deterministic. Give
        # the existing single repair round a chance to represent that evidence
        # as a cited hypothesis or sealed prior-assisted scenario. The model
        # may still explicitly classify it unsupported; it may not silently
        # erase supplied information.
        unresolved_numeric_context = bool(
            context.strip() and not proposed_any_lane
            and re.search(r"(?<!\w)\d+(?:\.\d+)?", context))
        if (proposed_any_lane or observation_lane_missing
                or unresolved_numeric_context):
            probe, probe_rejections = validate_temporal_dossier(
                raw, context_text=context, cutoff=self.timestamps[-1],
                future_timestamps=future_timestamps, history=self.values,
                history_timestamps=self.timestamps,
                compiler_model=self.forecaster.openrouter_model)
            effect_failed = (probe.get("effect_proposal_critique") or {}).get(
                "status") == "rejected"
            candidate_failed = (probe.get("candidate_critique") or {}).get(
                "status") == "rejected"
            hypothesis_failures = (probe.get("hypothesis_critique") or {}).get(
                "rejected") or []
            observation_failures = (
                probe.get("observation_interpretation_critique") or {}).get(
                    "rejected") or []
            if (probe_rejections or effect_failed or candidate_failed
                    or hypothesis_failures or observation_failures
                    or observation_lane_missing or unresolved_numeric_context):
                # Do not let one failed lane hide another. In particular, an
                # effect critique used to mask a malformed candidate, causing
                # the sole repair round to return another placeholder path.
                critique = {
                    "effect_proposal": probe.get("effect_proposal_critique"),
                    "forecast_candidate": probe.get("candidate_critique"),
                    "hypotheses": probe.get("hypothesis_critique"),
                    "observation_interpretations": probe.get(
                        "observation_interpretation_critique"),
                    "semantic_sufficiency": ({
                        "code": "MISSING_HISTORICAL_ZERO_INTERPRETATION",
                        "message": (
                            "The cited prose explicitly describes historical "
                            "zero-recording corruption that has ended. Include "
                            "a verbatim claim and the typed exact-zero "
                            "observation_interpretation; do not create a future "
                            "event or mutate history."),
                    } if observation_lane_missing else {
                        "code": "NUMERIC_CONTEXT_UNRESOLVED",
                        "message": (
                            "The supplied context contains numeric information "
                            "but the dossier represented none of it. Return a "
                            "verbatim cited claim and typed hypothesis. Add a "
                            "sealed probabilistic forecast_candidate only when "
                            "the history plus cited reference supports one; for "
                            "a long smooth path use quantile_anchors at the first, "
                            "last, and meaningful turning-point timestamps; "
                            "this is a best_effort human-review lane, so a "
                            "reasonable temporal/domain prior may supply anchor "
                            "numbers when the rationale labels them model-prior "
                            "rather than source-stated. Such a candidate remains "
                            "prior_assisted and can never automate. Use an "
                            "unsupported hypothesis only when no bounded useful "
                            "conditional path can be formed, and name the missing "
                            "evidence."
                        ),
                    } if unresolved_numeric_context else None),
                    "all_rejections": probe_rejections,
                }
                try:
                    repair_used = True
                    repair_completion = complete(
                        (
                            prompt + "\nYour proposal was rejected by Gnomon:\n"
                            + json.dumps(critique)
                            + "\nReturn one complete corrected dossier JSON "
                              "including cited claims. If you propose numeric "
                              "quantiles, they must be a computed probabilistic "
                              "path with non-zero uncertainty, never placeholders. "
                              "This is the only repair round."),
                        "dossier_repair")
                    repaired = extract_json_objects(repair_completion)
                    if repaired:
                        raw = bind_active_target(repaired[0])
                    else:
                        compile_rejections.append(
                            "dossier repair returned no JSON object")
                except Exception as error:
                    compile_rejections.append(
                        f"dossier repair failed: {error}")

        # Literal, high-precision observation semantics should not disappear
        # merely because a stochastic compiler dropped the optional wrapper
        # or malformed its claim. This fallback copies one exact source
        # sentence and stated date only; all filtering and candidate creation
        # still pass through Gnomon's normal validator below.
        if _expects_historical_zero_interpretation(context):
            literal_claim = deterministic_historical_observation_claim(
                context, history_start=self.timestamps[0],
                cutoff=self.timestamps[-1])
            if literal_claim is not None:
                remaining_claims = [
                    item for item in raw.get("claims") or []
                    if not isinstance(item, dict) or
                    str(item.get("source_span") or "") !=
                    literal_claim["source_span"]]
                raw = {**raw, "claims": [literal_claim, *remaining_claims]}

        # A broad extraction prompt sometimes identifies every exact equation
        # yet emits no executable lane. For humans this looks like “Gnomon read
        # my context and ignored it.” Spend the one existing repair round on a
        # focused compilation only when the model itself returned verbatim,
        # numeric lag-relationship claims. This is a sufficiency repair, not a
        # benchmark-family rule, and remains bounded by the same deadline.
        claim_spans = [str(item.get("source_span") or "")
                       for item in raw.get("claims") or []
                       if isinstance(item, dict)]
        exact_lag_claims = [span for span in claim_spans
                            if re.search(r"\blag\s*\d+\b", span, re.I)
                            and re.search(r"[+-]?\d+(?:\.\d+)?\s*\*", span)]
        numeric_lane_missing = not (
            raw.get("transformations") or raw.get("effect_proposal")
            or raw.get("forecast_candidate"))
        if exact_lag_claims and numeric_lane_missing and not repair_used:
            try:
                repair_used = True
                focused = (
                    prompt + "\nGnomon found exact cited lag equations but no "
                    "executable numeric lane. Return one complete corrected "
                    "dossier JSON. Preserve the verbatim claims and express "
                    "their equation as one recursive_linear transformation. "
                    "Use autoregressive_terms for target lags, driver_terms "
                    "for companion lags, and only cited future driver values "
                    "in series_values. Do not emit target-lag arrays or "
                    "duplicate the schedule in covariate_tables. This is the "
                    "only repair round.\nExact claims:\n"
                    + json.dumps(exact_lag_claims))
                repair_completion = complete(
                    focused, "relationship_sufficiency_repair")
                repaired = extract_json_objects(repair_completion)
                if repaired:
                    raw = bind_active_target(repaired[0])
                else:
                    compile_rejections.append(
                        "relationship sufficiency repair returned no JSON object")
            except Exception as error:
                compile_rejections.append(
                    f"relationship sufficiency repair failed: {error}")

        # Claims remain evidence for dossiers and transformations. Do not
        # synthesize wildcard numeric events from them: only explicitly
        # target-bound event proposals may change the numeric path.
        final_probe, _ = validate_temporal_dossier(
            raw, context_text=context, cutoff=self.timestamps[-1],
            future_timestamps=future_timestamps, history=self.values,
            history_timestamps=self.timestamps,
            compiler_model=self.forecaster.openrouter_model)

        # Claim IDs are host-assigned after validation. When exactly one claim
        # survives, a stale or omitted model-side ID is unambiguous and can be
        # rebound before AST validation. Entailment checks still verify every
        # constant and supplied value against that sole source span.
        verified_claims = final_probe.get("claims") or []
        if len(verified_claims) == 1:
            sole_id = str(verified_claims[0]["claim_id"])
            known_ids = {sole_id}
            effect = raw.get("effect_proposal")
            if isinstance(effect, dict):
                effect = dict(effect)
                cited = {str(value) for value in effect.get("claim_ids") or []}
                if not cited or not cited.issubset(known_ids):
                    effect["claim_ids"] = [sole_id]
                raw = {**raw, "effect_proposal": effect}
            normalized_hypotheses = []
            for item in raw.get("hypotheses") or []:
                if not isinstance(item, dict):
                    normalized_hypotheses.append(item)
                    continue
                hypothesis = dict(item)
                cited = {str(value) for value in
                         hypothesis.get("claim_ids") or []}
                if not cited or not cited.issubset(known_ids):
                    hypothesis["claim_ids"] = [sole_id]
                normalized_hypotheses.append(hypothesis)
            raw = {**raw, "hypotheses": normalized_hypotheses}
            normalized_observations = []
            for item in raw.get("observation_interpretations") or []:
                if not isinstance(item, dict):
                    normalized_observations.append(item)
                    continue
                interpretation = dict(item)
                cited = {str(value) for value in
                         interpretation.get("claim_ids") or []}
                if not cited or not cited.issubset(known_ids):
                    interpretation["claim_ids"] = [sole_id]
                normalized_observations.append(interpretation)
            raw = {**raw,
                   "observation_interpretations": normalized_observations}
            normalized_transformations = []
            for item in raw.get("transformations") or []:
                if not isinstance(item, dict):
                    normalized_transformations.append(item)
                    continue
                wrapper = dict(item)
                transformation = dict(wrapper.get("transformation", wrapper))
                cited = {str(value) for value in
                         transformation.get("claim_ids") or []}
                if not cited or not cited.issubset(known_ids):
                    transformation["claim_ids"] = [sole_id]
                    transformation["citation_binding"] = (
                        "single_verified_claim")
                wrapper["transformation"] = transformation
                series_values = {}
                for name, payload in (wrapper.get("series_values") or {}).items():
                    value_payload = dict(payload) if isinstance(payload, dict) else payload
                    if isinstance(value_payload, dict):
                        source_ids = {str(value) for value in
                                      value_payload.get("source_claim_ids") or []}
                        if not source_ids or not source_ids.issubset(known_ids):
                            value_payload["source_claim_ids"] = [sole_id]
                            value_payload["citation_binding"] = (
                                "single_verified_claim")
                    series_values[name] = value_payload
                wrapper["series_values"] = series_values
                normalized_transformations.append(wrapper)
            raw = {**raw, "transformations": normalized_transformations}

        # Literal bounds and absolute future states do not need model-authored
        # effect arithmetic. The model locates and dates the verbatim claim;
        # Gnomon's parser independently recovers its number and sends the
        # resulting event back through the ordinary future-context admission
        # path. This is deterministic extraction, not a semantic guess.
        from gnomon.llm_dossier import deterministic_events_from_claims
        derived_events = deterministic_events_from_claims(final_probe)
        existing_events = [item for item in raw.get("events") or []
                           if isinstance(item, dict)]
        existing_keys = {(str(item.get("event_type") or ""),
                          str(item.get("evidence_quote") or
                              item.get("source_span") or ""))
                         for item in existing_events}
        for event in derived_events:
            event = {**event, "entity_scope": ["__default__"],
                     "host_target_binding": "single_target_verified_claim"}
            key = (str(event.get("event_type") or ""),
                   str(event.get("evidence_quote") or ""))
            if key not in existing_keys:
                existing_events.append(event)
                existing_keys.add(key)
        raw = {**raw, "events": existing_events}

        # Transformations share the same single repair budget. Preflight the
        # sealed AST before the forecast call so a near-correct proposal can
        # add missing verbatim claims or repair only the transformation lane;
        # accepted events/effects/covariates are never replaced by this pass.
        from gnomon.context_intelligence import canonicalize_recursive_wrapper

        def canonicalize_transformations(candidate_raw: dict[str, Any]) -> dict[str, Any]:
            normalized = []
            for item in candidate_raw.get("transformations") or []:
                # Normalize a common compact spelling returned by small/fast
                # compilers. It contains the same typed fields but nests the
                # recurrence beside metadata instead of under expression.
                embedded = (item.get("transformation")
                            if isinstance(item, dict) else None)
                compact = (item.get("recursive_linear")
                           if isinstance(item, dict) else None)
                if not isinstance(compact, dict) and isinstance(embedded, dict):
                    compact = embedded.get("recursive_linear")
                if (not isinstance(compact, dict) and isinstance(embedded, dict)
                        and embedded.get("type") == "recursive_linear"
                        and "expression" not in embedded):
                    compact = {key: embedded.get(key) for key in (
                        "intercept", "autoregressive_terms", "driver_terms",
                        "series_values", "historical_series_segments")
                               if key in embedded}
                if isinstance(item, dict) and isinstance(compact, dict):
                    metadata = embedded if isinstance(embedded, dict) else item
                    recurrence = dict(compact)
                    schedules = recurrence.pop("series_values", {}) or {}
                    historical_segments = recurrence.pop(
                        "historical_series_segments", None) or metadata.get(
                            "historical_series_segments")
                    claim_ids = list(metadata.get("claim_ids") or [])
                    claim_id = str((claim_ids[0] if claim_ids else None)
                                   or metadata.get("claim_id") or "claim-1")
                    series_values = {}
                    for name, schedule in schedules.items():
                        if isinstance(schedule, dict):
                            values = [schedule[key] for key in sorted(schedule)]
                        elif (isinstance(schedule, list) and schedule
                              and all(isinstance(row, dict)
                                      and "timestamp" in row and "value" in row
                                      for row in schedule)):
                            rows = sorted(schedule,
                                          key=lambda row: str(row["timestamp"]))
                            stamps = [str(row["timestamp"]) for row in rows]
                            # Only erase timestamp structure after proving it
                            # is exactly the host-owned requested grid.
                            values = ([row["value"] for row in rows]
                                      if stamps == sorted(future_timestamps)
                                      else schedule)
                        elif (isinstance(schedule, list) and schedule
                              and all(isinstance(row, dict)
                                      and {"start", "end", "value"}.issubset(row)
                                      for row in schedule)):
                            expanded = []
                            valid = True
                            for stamp in sorted(future_timestamps):
                                matches = [row for row in schedule
                                           if str(row["start"]) <= stamp
                                           <= str(row["end"])]
                                if len(matches) != 1:
                                    valid = False
                                    break
                                expanded.append(matches[0]["value"])
                            cited = all(str(row["start"]) in narrative_context
                                        and str(row["end"]) in narrative_context
                                        for row in schedule)
                            values = expanded if valid and cited else schedule
                        else:
                            values = schedule
                        series_values[str(name)] = {
                            "values": values, "known_at": metadata.get("known_at"),
                            "source_claim_ids": [claim_id],
                        }
                    output_unit = "target_units"
                    item = {
                        "transformation": {
                            "known_at": metadata.get("known_at"),
                            "claim_ids": [claim_id],
                            "lane": "historically_testable",
                            "output_unit": output_unit,
                            "expression": {"op": "recursive_linear",
                                           "output_unit": output_unit,
                                           **recurrence},
                        },
                        "units": {"primary": output_unit, **{
                            name: output_unit for name in series_values}},
                        "series_values": series_values,
                        **({"historical_series_segments": historical_segments}
                           if historical_segments else {}),
                    }
                canonical, status = canonicalize_recursive_wrapper(
                    item, target_name=self.target_name,
                    driver_names=list(self.companion_histories))
                if isinstance(canonical, dict) and status.get("status") != "not_applicable":
                    canonical = {**canonical, "syntax_canonicalization": status}
                if isinstance(canonical, dict):
                    transformation = canonical.get("transformation", canonical)
                    expression = (transformation.get("expression")
                                  if isinstance(transformation, dict) else None)
                    if isinstance(expression, dict) and expression.get(
                            "op") == "recursive_linear":
                        values = dict(canonical.get("series_values") or {})
                        histories = dict(canonical.get(
                            "historical_series_segments") or {})
                        units = dict(canonical.get("units") or {})
                        claim_ids = list(transformation.get("claim_ids") or [
                            "claim-1"])
                        for name in sorted({str(term.get("series")) for term in
                                            expression.get("driver_terms") or []
                                            if term.get("series")}):
                            supplied = values.get(name)
                            supplied_values = (supplied.get("values")
                                               if isinstance(supplied, dict)
                                               else None)
                            schedule_is_executable = (
                                isinstance(supplied_values, list)
                                and len(supplied_values) == len(future_timestamps)
                                and all(isinstance(value, (int, float))
                                        and not isinstance(value, bool)
                                        for value in supplied_values)
                            )
                            if (schedule_is_executable
                                    and name in histories):
                                continue
                            extracted = _extract_explicit_driver_schedule(
                                narrative_context, series=name,
                                cutoff=self.timestamps[-1],
                                future_timestamps=future_timestamps,
                                claim_id=str(claim_ids[0]))
                            if extracted is None:
                                continue
                            historical, future = extracted
                            # The source-owned extractor is authoritative over
                            # a malformed representation proposed by the model.
                            # It only succeeds under exact named range and
                            # host-grid coverage, so this is normalization, not
                            # interpolation or model-authored repair.
                            values[name] = {
                                "values": future,
                                "known_at": self.timestamps[-1],
                                "source_claim_ids": [str(claim_ids[0])],
                                "syntax_canonicalization": "cited_range_schedule",
                            }
                            histories.setdefault(name, historical)
                            units.setdefault(name, str(
                                transformation.get("output_unit") or "target_units"))
                        canonical = {**canonical, "series_values": values,
                                     "historical_series_segments": histories,
                                     "units": units}
                normalized.append(canonical)
            return {**candidate_raw, "transformations": normalized}

        raw = canonicalize_transformations(raw)

        def transformation_violations(
                candidate_raw: dict[str, Any], dossier: dict[str, Any]) -> list[dict[str, Any]]:
            from gnomon.context_intelligence import (
                TransformationError, compile_transformation,
                execute_transformation,
            )

            claims = dossier.get("claims") or []
            claim_ids = [str(claim.get("claim_id")) for claim in claims]
            spans = {str(claim.get("claim_id")): str(
                claim.get("source_span") or "") for claim in claims}
            failures = []
            for index, item in enumerate(candidate_raw.get("transformations") or [], 1):
                wrapper = item if isinstance(item, dict) else {}
                compiled, critique = compile_transformation(
                    wrapper.get("transformation", wrapper),
                    series=list((wrapper.get("series_values") or {}).keys()),
                    claim_ids=claim_ids, cutoff=self.timestamps[-1],
                    units=wrapper.get("units"), repair=wrapper.get("repair"),
                    claim_spans=spans)
                if compiled is None:
                    failures.append({"index": index,
                                     "violations": critique["violations"]})
                    continue
                # Syntax-only validation missed malformed series payloads
                # until live publication, too late for the bounded repair.
                # A zero-valued dummy primary exercises every input/provenance
                # contract without observing or fabricating future targets.
                dummy_primary = [{"q10": 0.0, "q50": 0.0, "q90": 0.0,
                                  "point": 0.0}
                                 for _ in future_timestamps]
                try:
                    execute_transformation(
                        compiled, primary=dummy_primary,
                        series_values=wrapper.get("series_values") or {},
                        claim_spans=spans,
                        history_values=self.values,
                        history_series=self.companion_histories)
                except TransformationError as error:
                    failures.append({"index": index,
                                     "violations": [error.as_dict()]})
            return failures

        transform_failures = transformation_violations(raw, final_probe)
        if transform_failures and not repair_used:
            try:
                repair_used = True
                repair_hints = _transformation_repair_hints(
                    transform_failures, context)
                repair_completion = complete(
                    (
                        prompt + "\nGnomon rejected the transformation lane:\n"
                        + json.dumps(transform_failures)
                        + "\nVerbatim source lines that may support rejected "
                          "constants (cite them only if relevant):\n"
                        + json.dumps(repair_hints)
                        + "\nReturn one complete corrected dossier JSON. "
                          "You may add verbatim cited claims and replace "
                          "transformations only; this is the sole repair round."),
                    "transformation_repair")
                repaired_objects = extract_json_objects(repair_completion)
                if repaired_objects:
                    repaired = repaired_objects[0]
                    prior_spans = {str(item.get("source_span") or "")
                                   for item in raw.get("claims") or []
                                   if isinstance(item, dict)}
                    repaired_spans = {str(item.get("source_span") or "")
                                      for item in repaired.get("claims") or []
                                      if isinstance(item, dict)}
                    if not prior_spans.issubset(repaired_spans):
                        compile_rejections.append(
                            "transformation repair attempted to remove prior verified claims")
                    else:
                        raw = {**raw,
                               "claims": repaired.get("claims") or [],
                               "transformations": repaired.get("transformations") or []}
                        raw = canonicalize_transformations(raw)
                        final_probe, _ = validate_temporal_dossier(
                            raw, context_text=context, cutoff=self.timestamps[-1],
                            future_timestamps=future_timestamps,
                            history=self.values,
                            compiler_model=self.forecaster.openrouter_model)
                else:
                    compile_rejections.append(
                        "transformation repair returned no JSON object")
            except Exception as error:
                compile_rejections.append(
                    f"transformation repair failed: {error}")
        for wrapper in raw.get("transformations") or []:
            if not isinstance(wrapper, dict):
                continue
            transformation = wrapper.get("transformation", wrapper)
            if isinstance(transformation, dict):
                transformation["known_at"] = self.timestamps[-1]
            for supplied in (wrapper.get("series_values") or {}).values():
                if isinstance(supplied, dict):
                    supplied["known_at"] = self.timestamps[-1]
        raw = canonicalize_transformations(raw)
        final_probe, _ = validate_temporal_dossier(
            raw, context_text=context, cutoff=self.timestamps[-1],
            future_timestamps=future_timestamps, history=self.values,
            compiler_model=self.forecaster.openrouter_model)
        remaining_transform_failures = transformation_violations(raw, final_probe)
        if remaining_transform_failures:
            compile_rejections.append(
                "transformation_preflight_rejected: "
                + json.dumps(remaining_transform_failures, sort_keys=True))

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
            active_target=self.target_name,
        )
        runtime_events = []
        for event in compilation["events"]:
            normalized = dict(event)
            if normalized.get("entity_scope") == [self.target_name]:
                # The context validator reasons in semantic column names; an
                # ungrouped runtime publishes that one column as __default__.
                # Convert only after quote/target validation has succeeded.
                normalized["entity_scope"] = ["__default__"]
            runtime_events.append(event_from_dict(normalized))
        events = runtime_events
        event_rejections = [{
            "context_id": f"event-proposal-{index}",
            "reason_code": str(item.get("reason_code") or
                               "event_proposal_rejected"),
            "reason": "; ".join(str(problem) for problem in
                                item.get("problems") or []) or
                      "Event proposal was rejected.",
        } for index, item in enumerate(compilation["rejected"], 1)]
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
                event_rejections.append({
                    "context_id": str(event.event_id),
                    "reason_code": "event_outside_forecast_window",
                    "reason": (
                        "Event does not overlap the requested forecast window; "
                        "retain it as a cited claim."),
                })
            else:
                prospective_events.append(event)
        events = prospective_events
        dossier, dossier_rejections = validate_temporal_dossier(
            raw, context_text=context, cutoff=self.timestamps[-1],
            future_timestamps=future_timestamps, history=self.values,
            history_timestamps=self.timestamps,
            compiler_model=self.forecaster.openrouter_model,
            validated_events=events,
            candidate_selection_eligible=not bool(
                remaining_transform_failures and raw.get("forecast_candidate")
                and raw.get("transformations")),
            candidate_selection_reason=(
                "Accompanying governed transformation failed preflight; the "
                "sealed model path remains visible as a scenario but cannot "
                "become the default recommendation."
                if remaining_transform_failures and raw.get("forecast_candidate")
                and raw.get("transformations") else None),
        )
        covariate_receipt = compilation["covariates"]
        covariate_rejections = compilation["covariate_rejections"]
        rejections = [*compile_rejections, *event_rejections,
                      *dossier_rejections, *covariate_rejections]
        if (context.strip() and not events and not dossier.get("claims")
                and not dossier.get("forecast_candidate")
                and not dossier.get("effect_proposal")
                and not compilation.get("hypotheses")
                and not (covariate_receipt or {}).get("tables")
                and not raw.get("transformations")):
            rejections.append(
                "context_unresolved: the compiler returned no grounded event, "
                "claim, covariate, transformation, or candidate; the immutable "
                "primary remains visible and the context did not influence it")
        payload = {
            "schema_version": 1,
            "compiler": {
                "kind": "llm_proposes_gnomon_validates",
                "model": self.forecaster.openrouter_model,
                "contract": ("explicit_lag_relationship"
                             if relationship_contract else
                             "historical_observation_semantics"
                             if observation_contract else "universal_dossier"),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "workflow_budget_seconds": MAX_CONTEXT_COMPILATION_SECONDS,
                "elapsed_seconds": round(
                    time.monotonic() - compilation_started, 6),
                "calls": compiler_calls,
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
            "transformations": list(raw.get("transformations") or [])[:6],
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
                    "transformations_proposed": len(
                        self.context_compilation.get("transformations") or []),
                    "rejection_count": len(self.context_compilation["rejections"]),
                    "future_observations_exposed": False,
                    "compiler_timing": self.context_compilation["compiler"],
                }
                if self.context_compilation is not None else None
            ),
            "trace": self.trace,
            "final_submission": self.final_submission,
            "total_time": time.time() - self.started,
        }
        # CiK task instances carry no `seed` attribute and the forecaster
        # is one object shared across every task-seed, so `seed` is "x"
        # for all five runs of a task: writing to one name silently kept
        # the last run and discarded four (103 traces survived 355 runs).
        # A trace is diagnostic evidence; losing it costs a diagnosis.
        path = self._trace_path or trace_dir / f"{name}-seed{seed}.json"
        if self._trace_path is None:
            suffix = 1
            while path.exists():
                suffix += 1
                path = trace_dir / f"{name}-seed{seed}.{suffix}.json"
            self._trace_path = path
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
            # The host already knows this benchmark's intent, and every
            # forecast argument is host-bound below. Asking the LLM to choose
            # the only visible tool adds no reasoning signal and can cost a
            # multi-minute provider round. This is the production composition:
            # model compiles qualitative context; host invokes the typed verb.
            self._dispatch("gnomon_forecast", {})
            if not self.submission:
                self._retry_governed_safe_repair()
            if not self.submission:
                self._abstain(
                    "governed forecast did not produce a publishable artifact")
            return self._resolve_submission()
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
                messages, n=1, tools=tools, tool_choice="auto",
                request_timeout=120, transport_retries=0)
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

    def _retry_governed_safe_repair(self) -> bool:
        """Execute one server-authored, bounded repair without another LLM turn.

        The server may identify an interior grid defect and return a literal
        aggressive-repair call. That repair is already capped and disclosed by
        Gnomon. A production host should not turn this deterministic recovery
        into agent improvisation, nor silently abstain while an executable
        recovery is present. Only the named repair action is honored; arbitrary
        returned arguments are deliberately not replayed.
        """
        if not self.trace:
            return False
        error = self.trace[-1]
        details = error.get("error_details") or {}
        options = details.get("repair_options") or []
        eligible = any(
            isinstance(option, dict)
            and option.get("action") == "retry_with_aggressive_repair"
            and isinstance(option.get("tool_call"), dict)
            and option["tool_call"].get("name") == "gnomon_forecast"
            and (option["tool_call"].get("arguments") or {}).get("repair")
                == "aggressive"
            for option in options
        )
        if not eligible:
            return False
        self.trace.append({
            "governed_recovery": "server_authored_aggressive_repair",
            "source_error_code": error.get("code"),
            "model_turn_required": False,
        })
        self._dispatch("gnomon_forecast", {"repair": "aggressive"})
        return self.submission is not None

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
                "target_column": self.target_name,
                "horizon": self.horizon,
                "context_events": receipt.get("events", []),
                **covariate_arguments,
                "future_events": True,
                "structural_events": True,
                "output_dir": str(self.jail / "gnomon-output"),
                "format": "brief",
            }
            if self.forecaster.output_role == "publication_best_effort":
                arguments.update({
                    "publication_mode": "best_effort",
                    "temporal_dossiers": [receipt.get("dossier") or {}],
                    "context_submission": {
                        "known_at": self.timestamps[-1],
                        "transformations": receipt.get("transformations") or [],
                        "rejections": receipt.get("rejections") or [],
                    },
                })
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
                entry["error_details"] = structured.get("error") or structured
            if not result.get("isError") and structured.get("artifact_path"):
                artifact_path = str(structured["artifact_path"])
                self.artifact_paths.add(artifact_path)
                entry["artifact_path"] = artifact_path
                results = structured.get("results") or []
                if results and isinstance(results[0], dict):
                    entry["context_outcome"] = results[0].get("context_outcome")
                    entry["support"] = results[0].get("support")
                if structured.get("publication"):
                    publication = structured["publication"]
                    if publication.get("projection") == "compact":
                        receipt = structured.get("publication_path")
                        if not receipt:
                            self._abstain(
                                "compact publication omitted its receipt path")
                        try:
                            publication = json.loads(
                                Path(str(receipt)).read_text(encoding="utf-8"))
                        except (OSError, ValueError, TypeError) as error:
                            self._abstain(
                                f"publication receipt could not be read: {error}")
                    self._publication = publication
                    portfolio = publication.get("candidate_portfolio") or []
                    entry["publication"] = {
                        "mode": publication.get("mode"),
                        "recommended_scenario_id": publication.get(
                            "recommended_scenario_id"),
                        "primary_forecast_unchanged": publication.get(
                            "primary_forecast_unchanged"),
                        "automation_eligible": (
                            publication.get("automation") or {}).get("eligible"),
                        "recommendation_authority": publication.get(
                            "recommendation_authority"),
                        "context_dispositions": [{
                            "context_id": item.get("context_id"),
                            "disposition": item.get("disposition"),
                            "reason_code": item.get("reason_code"),
                            "reason": item.get("reason"),
                            "recovery_action": ({
                                key: (item.get("recovery_action") or {}).get(key)
                                for key in ("code", "message",
                                            "required_evidence",
                                            "automation_eligible")
                            } if item.get("recovery_action") else None),
                        } for item in publication.get(
                            "context_dispositions") or []],
                        "candidates": [{
                            "scenario_id": item.get("scenario_id"),
                            "role": item.get("role"),
                            "support": item.get("support"),
                            "selection_eligible": item.get("selection_eligible"),
                            "conditional_replay": (
                                item.get("effect") or {}).get(
                                    "conditional_replay"),
                            "recurrence_replay": (
                                (item.get("effect") or {}).get("validation") or {}
                            ).get("recurrence_replay_reason"),
                            "recurrence_replay_points": (
                                (item.get("effect") or {}).get("validation") or {}
                            ).get("recurrence_replay_points"),
                            "recurrence_replay_skill": (
                                (item.get("effect") or {}).get("validation") or {}
                            ).get("recurrence_replay_skill"),
                            "recurrence_candidate_mae": (
                                (item.get("effect") or {}).get("validation") or {}
                            ).get("recurrence_replay_candidate_mae"),
                            "recurrence_baseline_mae": (
                                (item.get("effect") or {}).get("validation") or {}
                            ).get("recurrence_replay_baseline_mae"),
                            "per_origin_observation_availability_checked": (
                                (item.get("effect") or {}).get("validation") or {}
                            ).get("per_origin_observation_availability_checked"),
                            "specification_known_at_each_origin": (
                                (item.get("effect") or {}).get("validation") or {}
                            ).get("specification_known_at_each_origin"),
                            "validation_interpretation": (
                                (item.get("effect") or {}).get("validation") or {}
                            ).get("validation_interpretation"),
                        } for item in portfolio],
                    }
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
                "transformations_proposed": len(
                    self.context_compilation.get("transformations") or []),
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
