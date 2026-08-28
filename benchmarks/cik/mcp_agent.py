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
from datetime import datetime
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
from gnomon.agent_context import (  # noqa: E402
    build_relationship_prior_prompt,
    build_sampled_context_prior_prompt,
    candidate_from_relationship_prior_specs,
    candidate_from_sampled_paths,
    recommended_initial_sample_count,
    recommended_sample_count,
    sampled_prior_sufficiency,
)

MAX_ROUNDS = 10
MAX_MCP_CALLS = 24
MAX_RUN_TOKENS = 250_000
MAX_CONTEXT_COMPILATION_SECONDS = max(1.0, min(
    300.0, float(os.environ.get("GNOMON_CONTEXT_COMPILATION_SECONDS", "60"))))
MIN_CONTEXT_REPAIR_SECONDS = 10.0
MODEL_PRIOR_PATH_SAMPLES = 5
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
#: Version 105: long regular forecast grids are described by exact boundaries,
#: step size, and count instead of retransmitting hundreds of timestamps.
#: Version 106: safe transformation literals may bind additional verbatim
#: source lines containing those exact constants instead of spending the sole
#: LLM repair round on citation bookkeeping.
#: Version 107: those static parameter claims bind to the host cutoff rather
#: than masquerading as future-window events.
#: Version 108: transformations bind literals to every already-verified
#: verbatim parameter claim, not only claims added during host normalization.
#: Version 109: missing literal units bind only from exact source-adjacent units
#: already declared by the transformation; arithmetic remains unchanged.
#: Version 110: timestamp/value future-driver rows canonicalize to values only
#: after exact identity with the complete host forecast grid is proved.
#: Version 111: missing dates on transformation-specification claims bind to
#: the cutoff; valid supplied windows and future driver vintages remain intact.
#: Version 112: a source-adjacent denominator unit may type both sides of an
#: explicit series/literal normalization ratio, making the ratio dimensionless.
#: Version 113: numeric provenance matching treats integral `298` and explicit
#: decimal `298.0` as equivalent source spellings without fuzzy tolerance.
#: Version 114: verified driver-schedule claims join both the series payload
#: and its parent transformation claim set.
#: Version 115: piecewise-constant drivers may use one initial value plus
#: source-cited change points resolved uniquely onto the host grid.
#: Version 116: exact algebraic identity operations are removed before
#: provenance validation; every material constant remains cited and checked.
#: Version 117: the latest cited change at or before the forecast boundary
#: establishes a compact schedule's initial state; future changes stay on-grid.
#: Version 118: a proposed transformation owns the one repair budget instead
#: of malformed optional side lanes; derived constants remain source-shaped.
#: Version 119: exact numeric citations allow sentence-ending punctuation but
#: still reject longer decimal prefixes.
#: Version 120: AST constants may bind exact semantic word forms already
#: recognized by the core entailment validator from separate source sentences.
#: Version 121: a transformation repair replaces only rejected ASTs; prior
#: claims are immutable and new verbatim claims append without rewriting them.
#: Version 122: repaired transformations repeat host provenance binding, and
#: an exactly cited square/cube may restore a prematurely evaluated literal.
#: Version 123: universal fraction/ratio aliases are dimensionless in the
#: governed unit algebra; percent remains a distinct unit requiring scaling.
#: Version 124: an exact cited multiplier may form a large scenario-only path;
#: approximate or model-authored extremes still hit the scale guard.
#: Version 125: one accepted executable lane prevents malformed optional lanes
#: from consuming the sole repair call or replacing a useful dossier.
#: Version 126: sealed receipts record why each bounded repair stage did or did
#: not trigger, including accepted and rejected lane statuses.
#: Version 127: repair telemetry includes typed effect violation codes and
#: bounded candidate rejection reasons for diagnosis without hidden reasoning.
#: Version 128: rejected effect telemetry retains only typed scalar field
#: values/types, never free-form model reasoning or unbounded payloads.
#: Version 129: effect distributions parse independently from confidence, and
#: tentative/confirmed confidence labels normalize without authority effects.
#: Version 130: a unique operative correction marker may resolve conflicting
#: baseline multipliers split across separately verified claims.
#: Version 131: one exact cited operative scenario is human-facing
#: evidence-dominant; its hypothetical support and automation ban remain.
#: Version 132: selector telemetry distinguishes an intentional evidence-
#: dominance skip from an attempted selector rejection or a needless call.
#: Version 133: conditional effect distributions expose source-stated versus
#: model-authored provenance and never repeat unsupported empirical claims.
#: Version 134: sealed model-authored forecast paths likewise separate their
#: unverified rationale from the bounded public provenance statement.
#: Version 135: publication exposes one authoritative aggregate context
#: disposition while retaining parser- and executable-lane diagnostics.
#: Version 136: compact MCP responses and traces retain that aggregate context
#: disposition instead of leaving it only in the full publication receipt.
#: Version 137: a live evidence-dominant publication records an intentional
#: selector skip rather than a generic no-selection diagnostic.
#: Version 138: recovery actions disclose when they concern only a rejected
#: side representation and no further call is needed for the recommendation.
#: Version 139: relative multipliers cannot masquerade as absolute overrides,
#: and deterministic absolute claims own their numeric scenario representation.
#: Version 140: model-authored full paths over deterministic absolute/range
#: claims remain outcome-scored shadows but cannot compete for recommendation.
#: Version 141: recurring-contamination replay uses origin-safe rotated placebo
#: blocks and resolves ambiguous schedule endpoints from pre-cutoff evidence.
#: Version 142: isolated multi-seed runs bind the runner's authoritative seed
#: into trace identity instead of overwriting every case as `seedx`.
#: Version 156: structured companion context preserves a governed executable
#: and a separately sealed model-authored path; selection receives compact
#: replay evidence, one bounded repair, and auditable portfolio telemetry.
#: Version 157: qualitative context uses a compact claim/event contract and an
#: empty compile receives one bounded repair instead of being silently erased.
#: Version 158: explicit categorical state histories and future schedules use
#: a deterministic parser plus a fold-replayed state-level executable.
#: Version 159: a failed governed alternative is deterministically dominated
#: by the primary, so the adapter does not spend an LLM call re-ranking it.
#: Version 160: categorical transition citations bind to the exact host grid
#: timestamp, preserving timezone provenance without model normalization.
#: Version 161: when the fold-replayed categorical executable is rejected, a
#: best-effort surface may request one sealed model-authored shadow candidate;
#: the failed replay remains counterevidence and automation stays disabled.
#: Version 162: the selector contract explicitly ranks every ineligible path
#: below every eligible path while retaining it as counterevidence.
#: Version 163: an out-of-sample-winning structured seasonal baseline may
#: enter best-effort selection through the non-automatable model-assisted lane.
#: Version 164: complete-cycle prequential evidence makes a strongly winning
#: seasonal assisted path deterministic instead of leaving it to LLM ranking.
#: Version 165: categorical best-effort priors aggregate five independently
#: sampled, host-grid-bound point paths instead of asking one model turn to
#: self-report calibrated quantiles.
#: Version 166: host-observed sampled-path provenance is sealed into the model
#: dossier and exposed to selection; shadow scoring follows that dossier seal.
#: Version 167: partial-holdout model assistance stays visible but cannot own
#: the recommendation without full-horizon or complete-cycle evidence.
#: Version 168: explicit best-effort publication deterministically recommends
#: one host-sampled prior consensus when no governed path dominates.
#: Version 169: recommendation receipts distinguish explicit best-effort policy
#: selection from an independent LLM selector call.
#: Version 170: sampled numeric priors use a direct compact forecast prompt
#: instead of the semantic compiler's diagnostic dump.
#: Version 171: numeric elicitation explicitly applies background, constraints
#: and scenarios, matching ordinary forecast intent without weakening host
#: validation, provenance, or publication authority.
#: Version 172: use a conventional timestamp-value forecast response and a
#: forecasting system role, then host-validate and convert it into the sealed
#: candidate schema. JSON value paths remain accepted for compatibility.
#: Version 173: numeric forecasting inherits the model/provider reasoning mode
#: and normal forecast budget instead of inheriting the deterministic semantic
#: compiler's non-reasoning, low-token policy.
#: Version 174: sampled priors use concurrent single-sample requests instead
#: of potentially correlated choices from one provider batch.
#: Version 175: explicit reference-normalized power laws and future driver
#: transitions use a deterministic typed front door before sealed prior
#: elicitation, and host-observed companion names are available to hypothesis
#: validation. This materially changes the context treatment and cache key.
#: Version 176: sealed numeric-path elicitation disables hidden reasoning and
#: uses a horizon-sized output budget after the typed dossier has already
#: isolated the forecasting argument.
#: Version 177: explicit nonlinear reference relationships request the full
#: five-path prior concurrently because a late initial request otherwise
#: consumes the shared deadline before adaptive expansion can run.
#: Version 178: transport failures are recorded separately from semantic path
#: validity, so an unavailable provider response cannot masquerade as
#: contradictory forecast evidence.
#: Version 179: sampled paths for explicit reference laws represent the
#: uncertain future driver; Gnomon applies the cited power law to create the
#: target path, preventing driver facts from becoming target overrides or
#: model arithmetic from silently changing the relationship.
#: Version 180: publication rechecks deterministic claim authority against the
#: actual target identity, so a cited driver transition cannot block or
#: supersede a separately sealed target scenario.
#: Version 181: the MCP publication boundary carries the caller's semantic
#: target identity through ungrouped ``__default__`` artifacts, making the
#: driver/target authority check effective in the live product response.
#: Version 182: the jailed MCP subprocess is pinned to the host's working-tree
#: source instead of silently importing an older installed wheel.
#: Version 183: named but numerically incomplete driver laws receive a typed
#: prior-only front door, preserving useful model knowledge without inventing
#: coefficients, support, or automation authority.
#: Version 184: a cited driver transition beginning exactly at the observation
#: cutoff is future-relevant; deterministic relationship routing no longer
#: discards that boundary case.
#: Version 185: supporting citations must belong to the selected sealed
#: scenario; rejected/unattached context can appear only as counterevidence.
#: Version 186: named driver relationships use a prefix-replayed, complexity-
#: penalized relationship family before any model-authored path is requested.
#: Version 187: when replay cannot identify a useful mapping, the host model
#: may supply only a repeated declarative family/exponent prior; Gnomon fits
#: scale, constructs uncertainty, seals the path, and keeps automation off.
#: Version 188: all typed hypothesis identities survive sealed publication
#: reranking, not only hypotheses marked as mandatory counterevidence.
#: Version 189: one explicit future zero-activity window is compiled without
#: an LLM round trip; its stated duration is bound to the host forecast grid.
#: Version 190: an unambiguous dated qualitative target direction is retained
#: as a non-numeric event before bounded model-candidate elicitation.
MCP_CONTRACT_VERSION = 190
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
     "mechanism": "brief qualitative explanation", "confidence": 0.0,
     "timing_status": "resolved | unresolved_trigger | atemporal_context"}
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
- Preserve a relevant general rule whose trigger is not dated (for example,
  demand falls on holidays without naming a holiday in the horizon) as a
  claim with null effective dates and `timing_status:"unresolved_trigger"`,
  plus an `unsupported` hypothesis. Do not turn it into an event, effect,
  transformation, or automated recommendation. It exists to explain what is
  missing and what dated evidence would make the rule executable.
- Use `timing_status:"atemporal_context"` for historical summaries and
  timeless relationships that have no event onset (for example, a historical
  annual average or a correlation). Use null effective dates. Preserve them as
  hypotheses; do not invent a trigger date or apply them as a deterministic
  effect. They may support a labelled prior-assisted candidate only when its
  assumptions and uncertainty are explicit.
- Correlation, co-occurrence, and "move together" statements are
  associational, not intervention evidence. Preserve them as relationship
  hypotheses. Do not cite them to justify a model-authored numeric path; only
  a fold-validated relationship executable may turn them into selection
  authority.
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
  candidate reasoning. When one companion series spans both target history
  and the requested future grid, include its verbatim historical rows in the
  same covariate table so Gnomon can fit and replay a relationship; never copy
  target outcomes. Every row must fall on an exact target-history or requested
  forecast timestamp and its value must be explicitly stated in context.
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
mechanism,confidence,timing_status:"resolved|unresolved_trigger|
atemporal_context"}. Use null effective dates with unresolved_trigger or
atemporal_context.
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
{values:[],known_at,source_claim_ids:[]}}}. A piecewise-constant driver may
replace values with initial_value and change_points:[{timestamp,value}]; use
exact host timestamps or unique HH:MM:SS clock times copied from context.

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
General temporal rules with an unidentified or undated trigger are still
useful context: preserve the verbatim claim with timing_status
`unresolved_trigger` and an unsupported hypothesis, but never apply it
numerically or authorize automation.
Historical summaries and timeless relationships are not missing triggers. Mark
them `atemporal_context`; preserve their interpretation and ask for applicable
driver observations, a comparison period, or an explicit bounded scenario—not
an invented event date.
Correlation and co-occurrence do not imply that intervening on one series
changes another. Keep such claims as hypotheses; do not use them as causal
authority for forecast-candidate selection.
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
5. Covariates are verbatim extraction: each quote contains its time and value,
and timestamp is an exact target-history or requested forecast timestamp.
Never infer or interpolate. For a companion series spanning history and the
future grid, include both portions in one table so Gnomon can fit the mapping;
never copy or reconstruct target outcomes.
6. Transformations are cited declarative ASTs, never code. Operators: literal,
primary,series,add,subtract,multiply,divide,power,lag,difference,
percent_change,rolling_mean,clip,quantile,reference_power,
linear_combination,recursive_linear. Arithmetic uses args:[NODE,...]; series
uses {op:"series",name}; lag uses {op:"lag",args:[NODE],steps:N}. Every future
series has either one cited value per forecast timestamp or a cited
piecewise-constant initial value plus change points. Gnomon resolves change
times to its exact grid and performs the forward fill. Never invent an initial
value or change point, extrapolate beyond the declared schedule, or duplicate
it as a covariate.
Preserve source-stated literals and operations in the AST. Never precompute a
derived constant such as replacing `3000^2` with `9000000`, because every
material literal must be entailed by a cited source.
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

QUALITATIVE_INSTRUCTIONS = """\
Compile qualitative temporal context into a governed JSON dossier. Return ONLY
JSON with exactly these keys:
{"events":[],"claims":[],"hypotheses":[],"covariate_tables":[],
"transformations":[],"observation_interpretations":[],
"effect_proposal":null,"forecast_candidate":null}

Use this compact lane when the source supplies states, direction, timing, or a
relationship but no numeric magnitude. Preserve every forecast-relevant fact:
- Future event: {document_index:0,event_type:"short source label",
  entity_scope:["*"],effective_start:"exact ISO",effective_end:"exact ISO",
  confidence:0.0,status:"confirmed|tentative",evidence_quote:"verbatim span",
  effect_family:"level_shift|trend_change|variance_change|temporary_pulse|
  saturation_bound|seasonal_regime_change|unknown",
  direction:"increase|decrease|unknown",duration:"temporary|persistent|unknown",
  entity_kind:"service|product|medication|procedure|calendar|capacity|price|
  environment|unknown"}.
- Claim: {source_span:"verbatim span",relation:"supports_increase|
  supports_decrease|supports_stability|supports_higher_variance|
  supports_lower_variance|changes_seasonal_regime|constrains_range|unknown",
  effective_start:null,effective_end:null,mechanism:"brief interpretation",
  confidence:0.0,timing_status:"resolved|unresolved_trigger|atemporal_context"}.
- Hypothesis: {kind:"regime_shift|relationship|historical_analogue|unsupported",
  claim_ids:["claim-1"],target_series:["*"],predictor_series:null,known_at:
  "history cutoff ISO",lag_steps:0,direction:"increase|decrease|unknown",
  rationale:"bounded competing interpretation"}.

Rules: quote only the source. Put only future-overlapping states in events;
historical states and general relationships are claims. Preserve ambiguous
interpretations as up to six hypotheses. Do not invent numeric encodings,
effect sizes, target paths, dates, or causal authority. Leave numeric lanes
empty: this contract describes what is known and what evidence is missing; it
cannot edit the immutable primary, upgrade support, or authorize automation.
If the context is irrelevant, return the empty object shown above.
"""

COMPANION_INSTRUCTIONS = """\
Extract source-grounded companion time series for a governed forecast. Return
ONLY JSON with these eight keys:
{"events":[],"claims":[],"hypotheses":[],"covariate_tables":[],
"transformations":[],"observation_interpretations":[],
"effect_proposal":null,"forecast_candidate":null}

For every named companion series that spans target history and the requested
forecast grid, emit one covariate table:
{"name":"source_grounded_snake_case","type":"continuous","rows":[
{"document_index":0,"timestamp":"timezone-aware ISO","source_time_span":
"exact source time token","value":0.0,"evidence_quote":"exact source row"}]}

Include both historical overlap and future-grid rows. Each timestamp and value
must occur together in the quoted source row. Never copy target observations,
infer a missing value, interpolate, author target events, or calculate a
forecast candidate. Preserve series identity in the table name using only
words present in its source heading/description. Gnomon, not the model, fits
and replays mappings against last value. If no companion has at least four
overlap rows plus complete future coverage, return empty arrays/nulls.
"""

COMPANION_CANDIDATE_INSTRUCTIONS = """\
Author one bounded probabilistic forecast candidate from the supplied target
history and source-grounded companion paths. Return ONLY:
{"forecast_candidate":{"quantiles":[{"timestamp":"exact requested ISO",
"q10":0.0,"q50":0.0,"q90":0.0}],"rationale":"brief arithmetic and competing
interpretation"}}

Provide exactly one row per requested forecast timestamp, ordered by time.
Use only target history and context known at the cutoff. Do not claim that a
companion is the target, hide uncertainty, or invent observations. The path is
a human-review-only prior-assisted alternative: it cannot edit the immutable
primary, upgrade support, or authorize automation. Because this is the
explicitly requested best-effort lane, produce the best bounded estimate when
the supplied paths cover the complete grid; represent ambiguity with wider
quantiles and state the competing interpretation rather than withholding.
"""

def _empirical_quantile(values: list[float], probability: float) -> float:
    """Dependency-free linear empirical quantile for bounded LLM path draws."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a quantile from no values")
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    weight = position - left
    return ordered[left] + (ordered[right] - ordered[left]) * weight


def _sample_path_stability(
    paths: list[list[float]], history_values: list[float] | None,
) -> dict[str, Any]:
    """Summarize elicitation dispersion without treating it as forecast skill.

    All quantities are host-computed and scale-free.  They intentionally make
    no admission decision: agreement among repeated model draws is useful
    uncertainty evidence, but it is not historical validation.
    """
    history = [float(value) for value in history_values or []
               if math.isfinite(float(value))]
    increments = [abs(right - left) for left, right in zip(history, history[1:])
                  if abs(right - left) > 0]
    if increments:
        scale = statistics.median(increments)
        scale_basis = "median_nonzero_history_increment"
    elif len(history) >= 2 and max(history) > min(history):
        scale = max(history) - min(history)
        scale_basis = "history_range"
    else:
        scale = max(1.0, abs(statistics.median(history)) * .01) if history else 1.0
        scale_basis = "level_floor"
    scale = max(float(scale), 1e-12)

    widths = []
    for index in range(len(paths[0])):
        values = [path[index] for path in paths]
        widths.append((_empirical_quantile(values, .9) -
                       _empirical_quantile(values, .1)) / scale)
    pairwise = []
    for left_index, left in enumerate(paths):
        for right in paths[left_index + 1:]:
            pairwise.append(statistics.mean(
                abs(a - b) for a, b in zip(left, right)) / scale)

    direction_agreement = []
    tolerance = scale * 1e-6
    for index in range(1, len(paths[0])):
        signs = []
        for path in paths:
            change = path[index] - path[index - 1]
            signs.append(1 if change > tolerance else
                         -1 if change < -tolerance else 0)
        direction_agreement.append(max(signs.count(-1), signs.count(0),
                                       signs.count(1)) / len(signs))
    return {
        "version": "0.1",
        "interpretation": "stability_not_historical_skill",
        "scale_basis": scale_basis,
        "path_count": len(paths),
        "horizon": len(paths[0]),
        "median_pointwise_q80_width_scaled": statistics.median(widths),
        "p90_pointwise_q80_width_scaled": _empirical_quantile(widths, .9),
        "median_pairwise_mae_scaled": (
            statistics.median(pairwise) if pairwise else 0.0),
        "max_pairwise_mae_scaled": max(pairwise) if pairwise else 0.0,
        "mean_direction_agreement": (
            statistics.mean(direction_agreement)
            if direction_agreement else 1.0),
        "unanimous_direction_fraction": (
            sum(value == 1.0 for value in direction_agreement) /
            len(direction_agreement) if direction_agreement else 1.0),
    }


def _candidate_from_sampled_paths(
    outputs: list[str], future_timestamps: list[str],
    *, history_values: list[float] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate sampled point paths and aggregate their marginal quantiles.

    Timestamps remain host-owned: completions return only values in the exact
    requested order. Invalid draws are rejected independently so one malformed
    completion cannot erase otherwise useful bounded prior evidence.
    """
    accepted: list[list[float]] = []
    rejection_reasons: list[str] = []
    rationales: list[str] = []
    expected = len(future_timestamps)
    display_timestamps = []
    for timestamp in future_timestamps:
        try:
            display_timestamps.append(datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")).strftime(
                    "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            display_timestamps.append(timestamp)
    for output in outputs:
        objects = extract_json_objects(output)
        first = objects[0] if objects else {}
        raw = first.get("forecast_path") if isinstance(first, dict) else None
        values = raw.get("values") if isinstance(raw, dict) else None
        if not isinstance(values, list):
            match = re.search(
                r"<forecast>\s*(.*?)\s*</forecast>", output,
                flags=re.IGNORECASE | re.DOTALL)
            parsed_values: list[float] = []
            parsed_timestamps: list[str] = []
            if match:
                for line in match.group(1).splitlines():
                    line = line.strip().strip("(), ")
                    if not line:
                        continue
                    parts = line.rsplit(",", 1)
                    if len(parts) != 2:
                        parsed_values = []
                        break
                    parsed_timestamps.append(parts[0].strip(" '\""))
                    try:
                        parsed_values.append(float(parts[1].strip()))
                    except ValueError:
                        parsed_values = []
                        break
            if parsed_values and parsed_timestamps == display_timestamps:
                values = parsed_values
        if not isinstance(values, list) or len(values) != expected:
            rejection_reasons.append(
                f"forecast response requires {expected} host-grid-bound values")
            continue
        try:
            path = [float(value) for value in values]
        except (TypeError, ValueError):
            rejection_reasons.append("forecast_path contains a non-number")
            continue
        if not all(math.isfinite(value) for value in path):
            rejection_reasons.append("forecast_path contains a non-finite value")
            continue
        accepted.append(path)
        rationale = " ".join(str(
            raw.get("rationale") or "" if isinstance(raw, dict) else ""
        ).split())
        if rationale:
            rationales.append(rationale[:300])
    diagnostics = {
        "requested": len(outputs), "accepted": len(accepted),
        "rejected": len(outputs) - len(accepted),
        "rejection_reasons": rejection_reasons[:8],
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "timestamp_binding": "host_grid_order",
        "request_mode": "concurrent_single_sample_requests",
    }
    if not accepted:
        return None, diagnostics
    diagnostics["stability"] = _sample_path_stability(
        accepted, history_values)
    rows = []
    for index, timestamp in enumerate(future_timestamps):
        values = [path[index] for path in accepted]
        rows.append({
            "timestamp": timestamp,
            "q10": _empirical_quantile(values, .1),
            "q50": _empirical_quantile(values, .5),
            "q90": _empirical_quantile(values, .9),
        })
    candidate = {
        "quantiles": rows,
        # Private hand-off into the host sealing boundary. It is removed before
        # the model-authored candidate is validated and never appears in the
        # compact compiler diagnostics.
        "_validated_sample_paths": accepted,
        "rationale": (
            f"Host-aggregated {len(accepted)} sampled model-authored "
            "point paths into empirical marginal quantiles. "
            + ("Sample rationales: " + " | ".join(rationales[:2])
               if rationales else "")),
    }
    return candidate, diagnostics


def _sampled_context_prior_prompt(
    *, timestamps: list[str], values: list[float],
    future_timestamps: list[str], context: str,
) -> str:
    """Build a compact indexed numeric prompt separate from compilation.

    The host owns both grids. Repeating a timestamp beside every historical
    value and asking the model to echo every future timestamp wastes the
    bounded workflow budget without adding forecast information. Regular
    grids are therefore represented by their endpoints, step and count; an
    irregular future grid remains explicit.
    """
    history_values = ",".join(f"{float(value):.12g}" for value in values)

    def grid_summary(grid: list[str]) -> str:
        parsed = [datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")) for timestamp in grid]
        if len(parsed) < 2:
            return f"timestamps={grid!r}"
        steps = [(right - left).total_seconds()
                 for left, right in zip(parsed, parsed[1:])]
        if steps and max(steps) == min(steps):
            return (
                f"start={grid[0]}, end={grid[-1]}, "
                f"step_seconds={steps[0]:.12g}, count={len(grid)}")
        return "timestamps=" + json.dumps(grid, separators=(",", ":"))

    history_grid = grid_summary(timestamps)
    future_grid = grid_summary(future_timestamps)
    return f"""\
I have a time series forecasting task for you.

Here is context known at the forecast cutoff. Factor in relevant background
knowledge, satisfy any stated constraints, and respect any stated scenarios.
<context>
{context}
</context>

The host owns the historical grid ({history_grid}). Values below are in exact
grid order:
<history>
[{history_values}]
</history>

Predict the future grid ({future_grid}).

Return only compact JSON with exactly one finite value per future grid point,
in order. Do not echo timestamps:

{{"forecast_path":{{"values":[1.0,2.0],"rationale":"brief basis"}}}}

Use no observations after the cutoff.
"""


def _has_material_numeric_context(text: str) -> bool:
    """Whether prose contains a quantity beyond calendar/clock notation.

    Dates establish knowledge and effect windows but do not by themselves
    justify a numeric scenario.  Strip their common representations before
    deciding whether the compiler must preserve a material quantity.  A
    duration, level, rate, bound, or coefficient remains visible.
    """
    stripped = re.sub(
        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\b",
        " ", text)
    stripped = re.sub(
        r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?\b",
        " ", stripped, flags=re.IGNORECASE)
    return bool(re.search(r"(?<!\w)[+-]?(?:\d+(?:\.\d*)?|\.\d+)", stripped))


def _future_numeric_path_needs_executable(
        text: str, future_timestamps: list[str], raw: dict[str, Any],
) -> bool:
    """Detect a supplied dated driver path without an executable mapping.

    This schedules the existing bounded repair only. It neither chooses the
    relevant series nor grants numeric authority. The model must still quote
    exact rows and the ordinary validators must seal or reject its proposal.
    """
    if any(raw.get(key) for key in (
            "transformations", "forecast_candidate", "effect_proposal")):
        return False
    matched = 0
    for timestamp in future_timestamps:
        date = str(timestamp).split("T", 1)[0]
        if re.search(
                rf"{re.escape(date)}[^\n]{{0,40}}[+-]?(?:\d+(?:\.\d*)?|\.\d+)",
                text):
            matched += 1
    return matched >= min(2, len(future_timestamps))


def _looks_like_structured_companion_context(
        text: str, future_timestamps: list[str],
) -> bool:
    """Route source-labelled reference tables without benchmark metadata."""
    if not _future_numeric_path_needs_executable(text, future_timestamps, {}):
        return False
    marker = re.search(
        r"\b(for reference|companion|predictor|driver|peer|related series|"
        r"external series)\b", text, re.IGNORECASE)
    rows = re.findall(
        r"\d{4}-\d{2}-\d{2}[^\n]{0,40}[+-]?(?:\d+(?:\.\d*)?|\.\d+)",
        text)
    return bool(marker and len(rows) >= len(future_timestamps) + 4)


def _extract_structured_companion_tables(
        text: str, history_timestamps: list[str],
        future_timestamps: list[str],
) -> list[dict[str, Any]]:
    """Parse explicit heading/separator/time-value blocks without an LLM."""
    grid = {timestamp.split("T", 1)[0]: timestamp
            for timestamp in [*history_timestamps, *future_timestamps]}
    if len(grid) != len(history_timestamps) + len(future_timestamps):
        return []
    lines = text.splitlines()
    tables = []
    row_pattern = re.compile(
        r"^\s*\(?\s*(\d{4}-\d{2}-\d{2}(?:[ T][^,)]*)?)\s*,\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)?\s*$")
    for index in range(1, len(lines) - 1):
        if not re.fullmatch(r"\s*-{3,}\s*", lines[index]):
            continue
        heading = lines[index - 1].strip()
        if not heading:
            continue
        rows = []
        cursor = index + 1
        while cursor < len(lines):
            match = row_pattern.fullmatch(lines[cursor])
            if match is None:
                break
            date = match.group(1)[:10]
            timestamp = grid.get(date)
            if timestamp is None:
                rows = []
                break
            value = float(match.group(2))
            if not math.isfinite(value):
                rows = []
                break
            rows.append({
                "document_index": 0, "timestamp": timestamp,
                "source_time_span": match.group(1).strip(), "value": value,
                "evidence_quote": lines[cursor].strip(),
            })
            cursor += 1
        covered = {row["timestamp"] for row in rows}
        if (len(covered.intersection(history_timestamps)) < 4
                or not set(future_timestamps).issubset(covered)):
            continue
        name = re.sub(r"[^a-z0-9_]+", "_", heading.casefold()).strip("_")
        if not name or not re.match(r"^[a-z_]", name):
            continue
        tables.append({"name": name[:64], "type": "continuous", "rows": rows})
    names = [table["name"] for table in tables]
    return tables if len(names) == len(set(names)) else []


def _extract_categorical_state_schedule(
        text: str, history_timestamps: list[str],
        future_timestamps: list[str],
) -> dict[str, Any] | None:
    """Parse an explicit timestamped state log without inventing magnitudes.

    The accepted grammar is intentionally narrow and domain-neutral:
    ``At the beginning ... <subject> was <state>`` followed by timestamped
    ``became``/``will become`` transitions. Labels remain categorical; their
    numeric effect is learned separately by fold-safe replay.
    """
    lines = [re.sub(r"^(?:Background|Scenario|Constraints):\s*", "",
                    line.strip(), flags=re.IGNORECASE)
             for line in text.splitlines() if line.strip()]
    beginning_pattern = re.compile(
        r"^At the beginning of (?:the )?series,\s+(?:the\s+)?"
        r"(?P<subject>[A-Za-z][A-Za-z _-]*?)\s+was\s+"
        r"(?P<state>[A-Za-z][A-Za-z_-]*)\.$", re.IGNORECASE)
    transition_pattern = re.compile(
        r"^At\s+(?P<time>[^,]+),\s+(?:we expect that\s+)?(?:the\s+)?"
        r"(?P<subject>[A-Za-z][A-Za-z _-]*?)\s+"
        r"(?:(?:will\s+)?become|became)\s+"
        r"(?P<state>[A-Za-z][A-Za-z_-]*)\.$", re.IGNORECASE)
    initial = None
    transitions: list[dict[str, Any]] = []
    for line in lines:
        match = beginning_pattern.fullmatch(line)
        if match and initial is None:
            initial = {
                "subject": "_".join(match.group("subject").casefold().split()),
                "state": match.group("state").casefold(), "quote": line,
            }
            continue
        match = transition_pattern.fullmatch(line)
        if not match:
            continue
        try:
            stamp = datetime.fromisoformat(
                match.group("time").strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        transitions.append({
            "subject": "_".join(match.group("subject").casefold().split()),
            "state": match.group("state").casefold(), "quote": line,
            "time": stamp, "source_time": match.group("time").strip(),
        })
    if initial is None or not transitions:
        return None
    if any(item["subject"] != initial["subject"] for item in transitions):
        return None

    def comparable(value: str) -> datetime:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None)

    grid = [*history_timestamps, *future_timestamps]
    try:
        grid_times = [comparable(value) for value in grid]
    except ValueError:
        return None
    grid_set = set(grid_times)
    canonical_by_time = dict(zip(grid_times, grid))
    normalized = [{**item, "time": item["time"].replace(tzinfo=None)}
                  for item in transitions]
    normalized.sort(key=lambda item: item["time"])
    if any(item["time"] not in grid_set for item in normalized):
        return None
    state, quote = initial["state"], initial["quote"]
    schedule, cursor = [], 0
    for timestamp, stamp in zip(grid, grid_times):
        while cursor < len(normalized) and normalized[cursor]["time"] <= stamp:
            state = normalized[cursor]["state"]
            quote = normalized[cursor]["quote"]
            cursor += 1
        schedule.append({"timestamp": timestamp, "state": state,
                         "evidence_quote": quote})
    history_count = len(history_timestamps)
    history_states = [item["state"] for item in schedule[:history_count]]
    future_states = [item["state"] for item in schedule[history_count:]]
    if len(set(history_states)) < 2 or not future_states:
        return None

    cited = [initial, *normalized]
    claims = []
    for item in cited:
        is_transition = "time" in item
        timestamp = (canonical_by_time.get(item["time"])
                     if is_transition else None)
        claims.append({
            "source_span": item["quote"], "relation": "unknown",
            "effective_start": timestamp, "effective_end": timestamp,
            "timing_status": ("resolved" if is_transition else
                              "atemporal_context"),
            "mechanism": "source-stated categorical state transition",
            "confidence": 1.0,
        })
    claim_ids = [f"claim-{index}" for index in range(1, len(claims) + 1)]
    raw = {
        "events": [], "claims": claims,
        "hypotheses": [{
            "kind": "unsupported", "claim_ids": claim_ids,
            "target_series": ["*"], "predictor_series": None,
            "known_at": history_timestamps[-1], "lag_steps": 0,
            "direction": "unknown",
            "rationale": (
                "The source supplies an observed categorical state history "
                "and a known future state schedule; estimate its target-level "
                "association by expanding-origin replay."),
        }],
        "covariate_tables": [], "transformations": [],
        "observation_interpretations": [], "effect_proposal": None,
        "forecast_candidate": None,
    }
    return {
        "name": initial["subject"], "raw": raw,
        "history_states": history_states, "future_states": future_states,
        "claim_ids": claim_ids, "schedule": schedule,
    }


def _validated_item_count(value: Any) -> int:
    """Normalize validator count/list shapes for diagnostic receipts."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 0


def _bounded_context_rejections(
        rejections: Any, *, limit: int = 16,
) -> tuple[list[Any], int]:
    """Project full receipt diagnostics into the bounded MCP wire contract.

    The context receipt is the authoritative, lossless record. This helper
    only bounds the host-authored request so diagnostic volume cannot make an
    otherwise valid forecast call fail schema validation.
    """
    if not isinstance(rejections, list):
        return [], 0
    if len(rejections) <= limit:
        return list(rejections), 0
    retained = list(rejections[:max(0, limit - 1)])
    omitted = len(rejections) - len(retained)
    retained.append({
        "context_id": "context-submission-overflow",
        "reason_code": "ADDITIONAL_CONTEXT_REJECTIONS_RETAINED",
        "reason": (
            f"{omitted} additional rejection(s) are retained in the sealed "
            "context receipt; this wire summary changes no claim, support "
            "state, or forecast number."
        ),
    })
    return retained, omitted


def _canonicalize_unreferenced_covariate_names(
        candidate: dict[str, Any],
) -> dict[str, Any]:
    """Make display-style table labels safe when no expression references them.

    Names are internal identifiers, not source evidence. We only normalize
    them when no transformation exists, avoiding any possibility of silently
    rebinding executable semantics.
    """
    if candidate.get("transformations"):
        return candidate
    tables = candidate.get("covariate_tables")
    if not isinstance(tables, list):
        return candidate
    used: set[str] = set()
    normalized: list[Any] = []
    for index, item in enumerate(tables, 1):
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        table = dict(item)
        original = str(table.get("name") or "")
        name = re.sub(r"[^a-z0-9_]+", "_", original.casefold()).strip("_")
        if not name or not re.match(r"^[a-z_]", name):
            name = f"context_driver_{index}"
        base = name[:64]
        name = base
        suffix = 2
        while name in used:
            tail = f"_{suffix}"
            name = base[:64-len(tail)] + tail
            suffix += 1
        used.add(name)
        table["name"] = name
        normalized.append(table)
    return {**candidate, "covariate_tables": normalized}


def _demote_covariate_duplicate_events(
        candidate: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Prefer companion-data semantics over a duplicate target override.

    A verbatim row cannot simultaneously be a predictor observation and an
    exact intervention on the target. Keeping the covariate is the strictly
    less authoritative interpretation; values still require downstream
    admission before they may influence a forecast.
    """
    quotes = {
        str(row.get("evidence_quote") or "").strip()
        for table in candidate.get("covariate_tables") or []
        if isinstance(table, dict)
        for row in table.get("rows") or [] if isinstance(row, dict)
        if str(row.get("evidence_quote") or "").strip()
    }
    if not quotes:
        return candidate, 0
    retained, demoted = [], 0
    for item in candidate.get("events") or []:
        quote = (str(item.get("evidence_quote") or
                     item.get("source_span") or "").strip()
                 if isinstance(item, dict) else "")
        if quote in quotes:
            demoted += 1
        else:
            retained.append(item)
    return {**candidate, "events": retained}, demoted


def _bind_covariate_row_claims(
        candidate: dict[str, Any], receipt: dict[str, Any],
        future_timestamps: list[str], *, table_name: str | None = None,
        maximum_claims: int = 16,
) -> dict[str, Any]:
    """Bind validator-proven companion rows as narrow numeric claims."""
    claims = [dict(item) for item in candidate.get("claims") or []
              if isinstance(item, dict)]
    existing = {str(item.get("source_span") or "").strip() for item in claims}
    future = set(future_timestamps)
    for table in receipt.get("tables") or []:
        if table_name is not None and table.get("name") != table_name:
            continue
        for row in table.get("rows") or []:
            if len(claims) >= maximum_claims:
                break
            provenance = row.get("provenance") or {}
            quote = str(provenance.get("evidence_quote") or "").strip()
            timestamp = str(row.get("timestamp") or "")
            if not quote or quote in existing:
                continue
            is_future = timestamp in future
            claims.append({
                "source_span": quote, "relation": "unknown",
                "effective_start": timestamp if is_future else None,
                "effective_end": timestamp if is_future else None,
                "timing_status": "resolved" if is_future else "atemporal_context",
                "mechanism": "host-verified companion-series observation",
                "confidence": 1.0,
            })
            existing.add(quote)
    return {**candidate, "claims": claims}


def _fit_governed_companion_from_receipt(
        receipt: dict[str, Any], *, context: str,
        history_timestamps: list[str], history_values: list[float],
        future_timestamps: list[str], claims: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Fit and multiplicity-guard host-verified companion paths."""
    tables = list(receipt.get("tables") or [])
    if not tables:
        return None
    source = " ".join(context.casefold().split())
    generic = {"rate", "value", "values", "forecast", "grid", "data",
               "series", "metric", "measure"}
    claim_by_span = {str(item.get("source_span") or ""): str(item["claim_id"])
                     for item in claims if item.get("claim_id")}
    history_by_time = dict(zip(history_timestamps, history_values))
    from gnomon.context_intelligence import fit_companion_level_candidate
    primary = [{"timestamp": timestamp} for timestamp in future_timestamps]
    fitted = []
    for table in tables:
        name = str(table.get("name") or "")
        tokens = [token for token in name.casefold().split("_") if token]
        meaningful = [token for token in tokens if token not in generic]
        if (not meaningful or not all(
                re.search(rf"\b{re.escape(token)}\b", source)
                for token in tokens)):
            continue
        by_time, quote_by_time = {}, {}
        for row in table.get("rows") or []:
            timestamp = str(row.get("timestamp") or "")
            if name not in row:
                continue
            by_time[timestamp] = float(row[name])
            quote_by_time[timestamp] = str(
                (row.get("provenance") or {}).get("evidence_quote") or "")
        overlap = [timestamp for timestamp in history_timestamps
                   if timestamp in by_time]
        required = [*overlap, *future_timestamps]
        if len(overlap) < 4 or any(timestamp not in by_time
                                   for timestamp in future_timestamps):
            continue
        claim_ids = [claim_by_span[quote_by_time[timestamp]]
                     for timestamp in required
                     if quote_by_time.get(timestamp) in claim_by_span]
        if claims and len(claim_ids) != len(required):
            continue
        candidate = fit_companion_level_candidate(
            [float(history_by_time[timestamp]) for timestamp in overlap],
            [by_time[timestamp] for timestamp in overlap],
            [by_time[timestamp] for timestamp in future_timestamps],
            primary=primary, claim_ids=claim_ids,
            hypothesis_id=f"host-verified-companion:{name}")
        candidate["source_table_name"] = name
        fitted.append(candidate)
    if not fitted:
        return None
    selected = max(fitted, key=lambda item: (
        float((item.get("validation") or {}).get("skill") or -math.inf),
        str(item.get("source_table_name"))))
    threshold = min(.25, .02 + .02 * math.log2(max(1, len(fitted))))
    validation = dict(selected["validation"])
    eligible = bool(validation["validation_points"] >= 3
                    and validation["skill"] >= threshold)
    validation.update({
        "candidate_tables": len(fitted),
        "multiplicity_adjusted_threshold": threshold,
        "beats_baseline": eligible,
    })
    selected["validation"] = validation
    selected["selection_eligible"] = eligible
    return selected


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
Extract one explicit lagged relationship for Gnomon's safe recurrence
executor. Return ONLY JSON with `claims` and `transformations`; set both to []
if the text states neither an exact equation nor explicit parent/lag structure.
Claims quote the relationship and any driver schedule verbatim. If every
coefficient is stated, use `recursive_linear` with numeric `intercept`,
`autoregressive_terms` ({lag, coefficient}), and `driver_terms`
({series, lag, coefficient}). If the source states only variables and lags,
use `fit_recursive_linear` with `autoregressive_lags` and `driver_lags`
({series,lags}); NEVER invent coefficients. Gnomon fits them from history and
admits the result only when expanding-origin replay beats last value. Put cited
future driver values in `series_values`. If the text states historical ranges, put them in
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
    relationship = ("=" in text or bool(re.search(
        r"\b(coefficient|affects?|parents?|depends?\s+on)\b", text, re.I)))
    return numeric and lag and relationship


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
            # An empty generated range can be rendered with start after end
            # when a piecewise segment receives zero observations. It covers
            # no timestamp and therefore carries no values to reconstruct;
            # ignore it rather than poisoning otherwise complete cited ranges.
            if start > end:
                continue
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
        # The child deliberately runs from a disposable jail, so its normal
        # import path would resolve the environment's last installed wheel.
        # Benchmarks must exercise the same working-tree revision as the host
        # adapter. Pin that source explicitly and retain any caller path after
        # it; the manifest/cache contract then describes one coherent build.
        repository_root = Path(__file__).resolve().parents[2]
        source_paths = [str(repository_root / "src"), str(repository_root)]
        inherited_path = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = os.pathsep.join(
            [*source_paths, *([inherited_path] if inherited_path else [])])
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
    explicit = str(getattr(task_instance, "target_name", "") or "").strip()
    if explicit:
        return _semantic_column_name(explicit)
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


def _transformation_literal_values(wrapper: Any) -> list[float]:
    """Return only executable AST literals, never dates, lags, or metadata."""
    transformation = (wrapper.get("transformation", wrapper)
                      if isinstance(wrapper, dict) else {})
    expression = (transformation.get("expression")
                  if isinstance(transformation, dict) else None)
    values: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("op") == "literal":
                value = node.get("value")
                if (isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))):
                    values.append(float(value))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(expression)
    return list(dict.fromkeys(values))


def _numeric_token_pattern(value: float) -> str:
    token = re.escape(format(float(value), ".15g"))
    if float(value).is_integer():
        token += r"(?:\.0+)?"
    return rf"(?<![\d.]){token}(?!\d|\.\d)"


def _verbatim_constant_lines(wrapper: Any, context: str,
                             existing_spans: list[str]) -> list[str]:
    """Find source lines needed to entail explicit AST literals."""
    existing = "\n".join(existing_spans).replace(",", "")
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    output = []
    for value in _transformation_literal_values(wrapper):
        pattern = _numeric_token_pattern(value)
        if re.search(pattern, existing.replace(",", "")):
            continue
        match = next((line for line in lines if re.search(
            pattern, line.replace(",", ""))), None)
        if match is not None:
            output.append(match)
    return list(dict.fromkeys(output))[:6]


def _verbatim_semantic_constant_lines(wrapper: Any, context: str,
                                      existing_spans: list[str]) -> list[str]:
    """Find exact word forms already accepted by constant entailment."""
    existing = "\n".join(existing_spans).casefold()
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    patterns = {
        0.25: r"\b(?:a\s+quarter|one\s+quarter|quarter)\b",
        0.5: r"\b(?:a\s+half|one\s+half|half)\b",
        2.0: r"\b(?:twice|double|square|squared|quadratic)\b",
        3.0: r"\b(?:triple|cube|cubed|cubic)\b",
    }
    output = []
    for value in _transformation_literal_values(wrapper):
        pattern = patterns.get(value)
        if not pattern or re.search(pattern, existing):
            continue
        match = next((line for line in lines
                      if re.search(pattern, line.casefold())), None)
        if match is not None:
            output.append(match)
    return list(dict.fromkeys(output))[:6]


def _verbatim_literal_claim_ids(wrapper: Any,
                                claims: list[dict[str, Any]]) -> list[str]:
    """Bind AST literals to already-verified verbatim parameter claims."""
    values = _transformation_literal_values(wrapper)
    output = []
    for claim in claims:
        span = str(claim.get("source_span") or "").replace(",", "")
        if any(re.search(_numeric_token_pattern(value), span)
               for value in values):
            output.append(str(claim.get("claim_id") or ""))
    return [claim_id for claim_id in dict.fromkeys(output) if claim_id]


def _future_series_values(wrapper: Any) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    if not isinstance(wrapper, dict):
        return output
    supplied = wrapper.get("series_values") or {}
    if not isinstance(supplied, dict):
        return output
    for name, payload in supplied.items():
        rows = payload.get("values") if isinstance(payload, dict) else None
        values = []
        for row in rows if isinstance(rows, list) else []:
            value = row.get("value") if isinstance(row, dict) else row
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value))):
                values.append(float(value))
        if values:
            output[str(name)] = list(dict.fromkeys(values))
    return output


def _verbatim_series_lines(wrapper: Any, context: str,
                           existing_spans: list[str]) -> list[str]:
    existing = "\n".join(existing_spans).replace(",", "")
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    output = []
    for values in _future_series_values(wrapper).values():
        for value in values:
            pattern = _numeric_token_pattern(value)
            if re.search(pattern, existing) or any(
                    re.search(pattern, line.replace(",", ""))
                    for line in output):
                continue
            match = next((line for line in lines if re.search(
                pattern, line.replace(",", ""))), None)
            if match is not None:
                output.append(match)
    return list(dict.fromkeys(output))[:12]


def _verbatim_series_claim_ids(wrapper: Any,
                               claims: list[dict[str, Any]]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for name, values in _future_series_values(wrapper).items():
        ids = []
        for claim in claims:
            span = str(claim.get("source_span") or "").replace(",", "")
            if any(re.search(_numeric_token_pattern(value), span)
                   for value in values):
                ids.append(str(claim.get("claim_id") or ""))
        output[name] = [claim_id for claim_id in dict.fromkeys(ids) if claim_id]
    return output


def _bind_verbatim_literal_units(wrapper: Any,
                                 claims: list[dict[str, Any]]) -> tuple[Any, int]:
    """Attach only source-adjacent units already declared by the wrapper."""
    if not isinstance(wrapper, dict):
        return wrapper, 0
    output = json.loads(json.dumps(wrapper))
    transformation = output.get("transformation", output)
    if not isinstance(transformation, dict):
        return output, 0
    expression = (transformation.get("expression")
                  if isinstance(transformation, dict) else None)
    raw_units = output.get("units") or {}
    declared = {str(value) for value in (
                    raw_units.values() if isinstance(raw_units, dict) else [])
                if str(value) and str(value) != "unknown"}
    if isinstance(transformation, dict):
        unit = str(transformation.get("output_unit") or "")
        if unit and unit != "unknown":
            declared.add(unit)
    spans = "\n".join(str(claim.get("source_span") or "") for claim in claims)
    changes = 0

    def adjacent_units(value: float, *, declared_only: bool) -> list[str]:
        token = _numeric_token_pattern(value)
        matches = re.findall(
            rf"{token}\s*([A-Za-z][A-Za-z0-9_/*^.-]*)",
            spans)
        units = [match.rstrip(".,;:") for match in matches]
        if declared_only:
            units = [unit for unit in units if unit in declared]
        return list(dict.fromkeys(units))

    def walk(node: Any) -> None:
        nonlocal changes
        if isinstance(node, dict):
            args = node.get("args") or []
            if node.get("op") == "divide" and len(args) == 2:
                numerator, denominator = args
                if (isinstance(numerator, dict)
                        and numerator.get("op") == "series"
                        and isinstance(denominator, dict)
                        and denominator.get("op") == "literal"
                        and not denominator.get("unit")):
                    value = denominator.get("value")
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        units = adjacent_units(float(value), declared_only=False)
                        if len(units) == 1:
                            unit = units[0]
                            denominator["unit"] = unit
                            output.setdefault("units", {})[
                                str(numerator.get("name") or "")] = unit
                            declared.add(unit)
                            changes += 1
            if node.get("op") == "literal" and not node.get("unit"):
                value = node.get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    units = adjacent_units(float(value), declared_only=True)
                    if len(units) == 1:
                        node["unit"] = units[0]
                        changes += 1
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(expression)
    if changes and isinstance(transformation, dict):
        transformation["literal_unit_binding"] = "verbatim_source_adjacency"
    return output, changes


def _canonicalize_timestamped_series_values(
        wrapper: Any, future_timestamps: list[str]) -> tuple[Any, int]:
    """Remove timestamp wrappers only after exact host-grid identity proof."""
    if not isinstance(wrapper, dict):
        return wrapper, 0
    output = json.loads(json.dumps(wrapper))
    changes = 0
    supplied = output.get("series_values") or {}
    if not isinstance(supplied, dict):
        return output, 0
    for payload in supplied.values():
        if not isinstance(payload, dict):
            continue
        rows = payload.get("values")
        if not (isinstance(rows, list) and rows
                and all(isinstance(row, dict)
                        and {"timestamp", "value"}.issubset(row) for row in rows)):
            continue
        ordered = sorted(rows, key=lambda row: str(row["timestamp"]))
        if [str(row["timestamp"]) for row in ordered] != sorted(future_timestamps):
            continue
        payload["values"] = [row["value"] for row in ordered]
        payload["syntax_canonicalization"] = (
            "timestamped_rows_exact_host_grid")
        changes += 1
    return output, changes


def _expand_change_point_series_values(
        wrapper: Any, future_timestamps: list[str]) -> tuple[Any, int]:
    """Expand a compact piecewise-constant schedule on one exact host grid."""
    if not isinstance(wrapper, dict):
        return wrapper, 0
    output = json.loads(json.dumps(wrapper))
    index_by_stamp = {stamp: index for index, stamp in enumerate(future_timestamps)}
    changes = 0

    first_dt = datetime.fromisoformat(
        future_timestamps[0].replace("Z", "+00:00"))

    def resolve(value: Any) -> tuple[int, float] | None:
        stamp = str(value or "")
        if stamp in index_by_stamp:
            exact = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            return index_by_stamp[stamp], exact.timestamp()
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None and parsed.tzinfo is not None and parsed < first_dt:
            return 0, parsed.timestamp()
        clock = stamp + (":00" if re.fullmatch(r"\d{2}:\d{2}", stamp) else "")
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?", clock):
            return None
        matches = [index for index, host in enumerate(future_timestamps)
                   if re.search(rf"T{re.escape(clock)}(?:[+-]|Z$)", host)]
        if len(matches) == 1:
            exact = datetime.fromisoformat(
                future_timestamps[matches[0]].replace("Z", "+00:00"))
            return matches[0], exact.timestamp()
        return None

    supplied = output.get("series_values") or {}
    if not isinstance(supplied, dict):
        return output, 0
    for payload in supplied.values():
        if not isinstance(payload, dict) or "change_points" not in payload:
            continue
        points = payload.get("change_points")
        initial = payload.get("initial_value")
        if not (isinstance(points, list)
                and isinstance(initial, (int, float))
                and not isinstance(initial, bool)):
            continue
        parsed = []
        valid = True
        for point in points:
            if not isinstance(point, dict):
                valid = False
                break
            resolved = resolve(point.get("timestamp"))
            value = point.get("value")
            if (resolved is None or not isinstance(value, (int, float))
                    or isinstance(value, bool)):
                valid = False
                break
            index, order = resolved
            parsed.append((index, order, float(value)))
        grouped: dict[int, tuple[float, float]] = {}
        for index, order, value in parsed:
            prior = grouped.get(index)
            if prior is not None and prior[0] == order:
                valid = False
                break
            if prior is None or order > prior[0]:
                grouped[index] = (order, value)
        if not valid:
            continue
        values = [float(initial)] * len(future_timestamps)
        resolved_points = [(index, item[1]) for index, item in grouped.items()]
        for index, value in sorted(resolved_points):
            values[index:] = [value] * (len(values) - index)
        payload["values"] = values
        payload["syntax_canonicalization"] = (
            "cited_piecewise_constant_change_points")
        payload["resolved_change_points"] = [
            {"timestamp": future_timestamps[index], "value": value}
            for index, value in sorted(resolved_points)]
        changes += 1
    return output, changes


def _bind_missing_transformation_claim_windows(
        raw: dict[str, Any], cutoff: str) -> tuple[dict[str, Any], int]:
    """Bind undated specification claims to cutoff, never future outcomes."""
    if not raw.get("transformations"):
        return raw, 0

    def aware(value: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        return parsed if parsed.tzinfo is not None else None

    claims = []
    changes = 0
    for item in raw.get("claims") or []:
        if not isinstance(item, dict):
            claims.append(item)
            continue
        claim = dict(item)
        start = aware(claim.get("effective_start"))
        end = aware(claim.get("effective_end"))
        if start is None or end is None or end < start:
            claim["effective_start"] = cutoff
            claim["effective_end"] = cutoff
            claim["effective_window_binding"] = (
                "undated_transformation_specification_at_cutoff")
            changes += 1
        claims.append(claim)
    return {**raw, "claims": claims}, changes


def _simplify_identity_literals(wrapper: Any) -> tuple[Any, int]:
    """Remove exact algebraic identities without changing numeric meaning."""
    if not isinstance(wrapper, dict):
        return wrapper, 0
    output = json.loads(json.dumps(wrapper))
    transformation = output.get("transformation", output)
    if not isinstance(transformation, dict):
        return output, 0
    expression = (transformation.get("expression")
                  if isinstance(transformation, dict) else None)
    changes = 0

    def literal(node: Any, value: float) -> bool:
        return (isinstance(node, dict) and node.get("op") == "literal"
                and isinstance(node.get("value"), (int, float))
                and not isinstance(node.get("value"), bool)
                and float(node["value"]) == value)

    def simplify(node: Any) -> Any:
        nonlocal changes
        if not isinstance(node, dict):
            return node
        normalized = {key: ([simplify(child) for child in value]
                            if key == "args" and isinstance(value, list)
                            else value) for key, value in node.items()}
        args = normalized.get("args") or []
        op = normalized.get("op")
        if len(args) == 2:
            left, right = args
            if op in {"multiply", "divide"} and literal(right, 1.0):
                changes += 1
                return left
            if op == "multiply" and literal(left, 1.0):
                changes += 1
                return right
            if op in {"add", "subtract"} and literal(right, 0.0):
                changes += 1
                return left
            if op == "add" and literal(left, 0.0):
                changes += 1
                return right
            if op == "power" and literal(right, 1.0):
                changes += 1
                return left
        return normalized

    if isinstance(transformation, dict) and isinstance(expression, dict):
        transformation["expression"] = simplify(expression)
        if changes:
            transformation["identity_simplification"] = (
                "exact_algebraic_identities_removed")
    return output, changes


def _restore_cited_power_literals(
        wrapper: Any, claims: list[dict[str, Any]]) -> tuple[Any, int]:
    """Restore an exact cited power that a compiler prematurely evaluated.

    This is deliberately narrower than algebraic inference: the cited claim
    must contain both the base literal and the matching square/cube semantics,
    the derived value must not itself appear in any cited span, and exactly one
    base/exponent pair may match.  It lets provenance validation inspect the
    source-shaped equation instead of an uncited arithmetic intermediate.
    """
    if not isinstance(wrapper, dict):
        return wrapper, 0
    output = json.loads(json.dumps(wrapper))
    transformation = output.get("transformation", output)
    if not isinstance(transformation, dict):
        return output, 0
    expression = (transformation.get("expression")
                  if isinstance(transformation, dict) else None)
    cited_ids = {str(value) for value in
                 (transformation.get("claim_ids") or [])}
    cited = [claim for claim in claims
             if str(claim.get("claim_id") or "") in cited_ids]
    replacements: dict[float, tuple[float, int]] = {}
    all_spans = "\n".join(str(claim.get("source_span") or "")
                           for claim in cited).replace(",", "")
    candidates: list[tuple[float, int]] = []
    for claim in cited:
        span = str(claim.get("source_span") or "").replace(",", "")
        lowered = span.casefold()
        exponents = []
        if re.search(r"\b(?:square|squared|quadratic)\b", lowered):
            exponents.append(2)
        if re.search(r"\b(?:cube|cubed|cubic)\b", lowered):
            exponents.append(3)
        numbers = [float(token) for token in re.findall(
            r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])", span)]
        candidates.extend((base, exponent) for base in numbers
                          for exponent in exponents)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("op") == "literal":
                value = node.get("value")
                if (isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and not re.search(_numeric_token_pattern(float(value)),
                                          all_spans)):
                    matches = [(base, exponent) for base, exponent in candidates
                               if math.isclose(base ** exponent, float(value),
                                               rel_tol=1e-12, abs_tol=1e-12)]
                    if len(matches) == 1:
                        replacements[float(value)] = matches[0]
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(expression)
    changes = 0

    def rewrite(node: Any) -> Any:
        nonlocal changes
        if isinstance(node, dict):
            if node.get("op") == "literal":
                value = node.get("value")
                match = replacements.get(float(value)) if isinstance(
                    value, (int, float)) and not isinstance(value, bool) else None
                if match is not None:
                    changes += 1
                    base, exponent = match
                    return {"op": "power", "args": [
                        {"op": "literal", "value": base},
                        {"op": "literal", "value": exponent},
                    ]}
            return {key: rewrite(value) for key, value in node.items()}
        if isinstance(node, list):
            return [rewrite(value) for value in node]
        return node

    if isinstance(transformation, dict):
        transformation["expression"] = rewrite(expression)
        if changes:
            transformation["syntax_canonicalization"] = (
                "exact_cited_power_restoration")
    return output, changes


def _bind_transformation_provenance(
        raw: dict[str, Any], verified_claims: list[dict[str, Any]],
        ) -> dict[str, Any]:
    """Bind host-assigned claim IDs to literals and future series values."""
    if not verified_claims or not raw.get("transformations"):
        return raw
    host_grounded = [str(claim["claim_id"]) for claim in verified_claims
                     if claim.get("mechanism") ==
                     "host-grounded explicit transformation constant"]
    rebound = []
    for item in raw.get("transformations") or []:
        if not isinstance(item, dict):
            rebound.append(item)
            continue
        wrapper = dict(item)
        transformation_raw = wrapper.get("transformation", wrapper)
        if not isinstance(transformation_raw, dict):
            # Leave malformed model output intact for the typed compiler
            # rejection. Provenance binding must never make an untrusted shape
            # executable—or crash before validation can explain the problem.
            rebound.append(wrapper)
            continue
        transformation = dict(transformation_raw)
        raw_series_values = wrapper.get("series_values") or {}
        if not isinstance(raw_series_values, dict):
            rebound.append(wrapper)
            continue
        literal_ids = _verbatim_literal_claim_ids(wrapper, verified_claims)
        series_ids = _verbatim_series_claim_ids(wrapper, verified_claims)
        transformation["claim_ids"] = list(dict.fromkeys([
            *[str(value) for value in transformation.get("claim_ids") or []],
            *literal_ids, *host_grounded,
            *[claim_id for ids in series_ids.values() for claim_id in ids],
        ]))
        if literal_ids:
            transformation["citation_binding"] = (
                "model_semantics_plus_verbatim_constant_lines")
        series_values = dict(raw_series_values)
        for name, claim_ids in series_ids.items():
            payload = series_values.get(name)
            if isinstance(payload, dict) and claim_ids:
                payload = dict(payload)
                payload["source_claim_ids"] = list(dict.fromkeys([
                    *[str(value) for value in
                      payload.get("source_claim_ids") or []], *claim_ids]))
                payload["citation_binding"] = (
                    "verbatim_future_series_value_lines")
                series_values[name] = payload
        wrapper["series_values"] = series_values
        wrapper["transformation"] = transformation
        rebound.append(wrapper)
    return {**raw, "transformations": rebound}


def _select_publication_fail_closed(
        publication: dict[str, Any], selection: dict[str, Any] | None,
        ) -> tuple[dict[str, Any], str | None]:
    """Apply a model ranking or retain the already verified publication."""
    if selection is None:
        from gnomon.publication import dominant_scenario_id
        portfolio = list(publication.get("candidate_portfolio") or [])
        if len(portfolio) <= 1:
            return publication, None
        dominant = dominant_scenario_id(portfolio)
        if dominant == publication.get("recommended_scenario_id"):
            return publication, "selector skipped: governed evidence dominance"
        return publication, "live MCP publication used without selection"
    from gnomon.publication import select_publication
    try:
        live_ids = [str(item.get("scenario_id")) for item in
                    publication.get("candidate_portfolio") or []]
        selected = str(selection.get("selected_scenario_id") or "")
        if selected not in live_ids:
            raise ValueError("selected scenario is absent from live portfolio")
        proposed = [str(item) for item in selection.get("ranking") or []]
        # The pre-call catalog can be a strict subset of the live MCP
        # portfolio (for example, the product may add a typed sensitivity
        # lane). Complete only the number-free ordering; never synthesize or
        # remove a candidate, and retain the model's relative order.
        ranking = list(dict.fromkeys([
            selected,
            *[item for item in proposed
              if item in live_ids and item != selected],
            *[item for item in live_ids if item != selected],
        ]))
        completed = {**selection, "selected_scenario_id": selected,
                     "ranking": ranking,
                     "host_completed_live_portfolio": ranking != proposed}
        return select_publication(publication, completed), None
    except (TypeError, ValueError) as error:
        return publication, f"selector incompatible with live portfolio: {error}"


def _canonicalize_scenario_selection_evidence(
        raw: Any, scenarios: list[dict[str, Any]], *,
        known_claim_ids: set[str] | None = None,
        known_hypothesis_ids: set[str] | None = None) -> Any:
    """Resolve only mechanically duplicated or stale evidence references."""
    if not isinstance(raw, dict):
        return raw
    output = json.loads(json.dumps(raw))
    cited = [str(item) for item in output.get("cited_claim_ids") or []]
    counter = [str(item) for item in
               output.get("counterevidence_claim_ids") or []]
    overlap = set(cited).intersection(counter)
    selected_id = str(output.get("selected_scenario_id") or "")
    selected_claims = set(next((item.get("claim_ids") or []
                                for item in scenarios
                                if str(item.get("scenario_id")) == selected_id), []))
    output["cited_claim_ids"] = list(dict.fromkeys(
        item for item in cited
        if item not in overlap or item in selected_claims))
    output["counterevidence_claim_ids"] = list(dict.fromkeys(
        item for item in counter
        if item not in overlap or item not in selected_claims))
    if known_claim_ids is not None:
        output["cited_claim_ids"] = [
            item for item in output["cited_claim_ids"]
            if item in known_claim_ids]
        output["counterevidence_claim_ids"] = [
            item for item in output["counterevidence_claim_ids"]
            if item in known_claim_ids]
        # The selected sealed scenario already owns its provenance. Repair a
        # stale/model-invented alias by citing those exact IDs; this cannot add
        # a claim, change a number, or alter support.
        if selected_claims and not selected_claims.intersection(
                output["cited_claim_ids"]):
            output["cited_claim_ids"].extend(sorted(
                selected_claims.intersection(known_claim_ids)))
    if known_hypothesis_ids is not None:
        output["counterevidence_hypothesis_ids"] = [
            str(item) for item in
            output.get("counterevidence_hypothesis_ids") or []
            if str(item) in known_hypothesis_ids]
    return output


def _merge_transformation_repair(raw: dict[str, Any],
                                 repaired: dict[str, Any]) -> dict[str, Any]:
    """Preserve prior claims; allow the bounded repair to replace only ASTs."""
    merged_claims = list(raw.get("claims") or [])
    known_spans = {str(item.get("source_span") or "")
                   for item in merged_claims if isinstance(item, dict)}
    for claim in repaired.get("claims") or []:
        span = (str(claim.get("source_span") or "")
                if isinstance(claim, dict) else "")
        if span and span not in known_spans:
            merged_claims.append(claim)
            known_spans.add(span)
    return {**raw, "claims": merged_claims,
            "transformations": repaired.get("transformations") or []}


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


def _forecast_grid_prompt(timestamps: list[str], *, compact_after: int = 32) -> str:
    """Describe a regular host grid without retransmitting every timestamp."""
    if len(timestamps) <= compact_after:
        return json.dumps(timestamps)
    try:
        parsed = [datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                  for stamp in timestamps]
        steps = [(right - left).total_seconds()
                 for left, right in zip(parsed, parsed[1:])]
    except (TypeError, ValueError):
        return json.dumps(timestamps)
    if not steps or any(not math.isclose(step, steps[0], abs_tol=1e-9)
                        for step in steps[1:]):
        # Never describe an irregular grid as regular. It is safer to pay the
        # schema cost than let a model form off-grid anchors.
        return json.dumps(timestamps)
    return json.dumps({
        "kind": "regular_host_grid",
        "first": timestamps[0],
        "last": timestamps[-1],
        "steps": len(timestamps),
        "step_seconds": steps[0],
        "anchor_rule": (
            "quantile anchors must use first, last, or another timestamp "
            "obtained by adding an integer number of step_seconds to first"),
    }, separators=(",", ":"))


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
        if output_role not in {"canonical", "immutable_primary",
                               "llm_candidate_shadow",
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
        governed_distribution_paths: list[list[float]] | None = None
        try:
            submission, extra_info = run.drive()
        finally:
            run.finish()
        if self.output_role == "immutable_primary":
            artifact_result = getattr(run, "_submitted_result", None) or {}
            primary = artifact_result.get("primary_forecast")
            context_changed = any(
                bool((entry.get("context_outcome") or {}).get(
                    "selected_projection_differs_from_primary",
                    (entry.get("context_outcome") or {}).get(
                        "primary_forecast_changed")))
                for entry in run.trace)
            if (not primary and not context_changed
                    and isinstance(artifact_result.get("forecast"), list)):
                # Uninfluenced artifacts do not duplicate the public path
                # under primary_forecast; the public path is the primary.
                primary = artifact_result["forecast"]
            if not isinstance(primary, list) or not primary:
                raise GnomonAbstained([
                    "verified artifact did not retain an immutable primary"])
            submission = primary
            extra_info = {
                **extra_info,
                "route": "immutable_primary_diagnostic",
                "primary_forecast_unchanged": True,
                "context_recommendation_ignored": True,
                "diagnostic_only": True,
            }
            run.final_submission = {
                "route": "immutable_primary_diagnostic",
                "primary_forecast_unchanged": True,
                "context_recommendation_ignored": True,
                "diagnostic_only": True,
            }
            run._write_trace()
        if self.output_role in {"llm_candidate_shadow",
                               "publication_best_effort"}:
            compilation = run.context_compilation or {}
            dossier = compilation.get("dossier") or {}
            dossiers = [item for item in compilation.get("dossiers") or
                        [dossier] if isinstance(item, dict) and item]
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
            selection_attempted = False
            selection_policy_applied = False
            if self.output_role == "publication_best_effort":
                from gnomon.publication import (build_scenario_catalog,
                                                best_effort_prior_selection,
                                                scenario_selection_contract,
                                                validate_scenario_selection)
                from gnomon.temporal_state import build_temporal_state
                scenarios, _ = build_scenario_catalog(
                    artifact_result, dossiers=dossiers)
                policy_selection = best_effort_prior_selection(
                    scenarios=scenarios, dossiers=dossiers)
                if policy_selection is not None:
                    selection = policy_selection
                    selection_policy_applied = True
                    scenarios = []
                if len(scenarios) > 1:
                    contract = scenario_selection_contract(
                        scenarios=scenarios, dossiers=dossiers,
                        temporal_state=build_temporal_state(
                            artifact_result, dossiers=dossiers))
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
                            selection_attempted = True
                            response = self.client.completions(
                                [{"role": "user", "content": prompt}], n=1,
                                temperature=0, reasoning_effort="none",
                                request_timeout=120,
                                transport_retries=0)[0]
                            objects = extract_json_objects(response)
                            if not objects:
                                raise ValueError("selector returned no JSON object")
                            normalized_selection = (
                                _canonicalize_scenario_selection_evidence(
                                    objects[0], scenarios,
                                    known_claim_ids={
                                        str(item.get("claim_id"))
                                        for item in contract.get("claims") or []
                                        if item.get("claim_id")
                                        and item.get("relation") !=
                                        "counterevidence"},
                                    known_hypothesis_ids={
                                        str(item.get("claim_id"))
                                        for item in contract.get("claims") or []
                                        if item.get("claim_id")
                                        and (item.get("relation") ==
                                             "counterevidence"
                                             or str(item.get("relation") or
                                                    "").startswith(
                                                        "hypothesis:"))}))
                            selection = validate_scenario_selection(
                                normalized_selection, scenarios=scenarios,
                                dossiers=dossiers)
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
                publication, live_selection_error = (
                    _select_publication_fail_closed(
                        live_publication, selection))
                if (live_selection_error is not None
                        and selection_error is None):
                    selection_error = live_selection_error
            else:
                if self.output_role == "llm_candidate_shadow":
                    # Shadow is an explicit evaluation instrument: score the
                    # sealed model candidate without making it the product
                    # default. Express that choice through the same governed
                    # selector contract so forecast values remain untouchable.
                    from gnomon.publication import build_scenario_catalog
                    shadow_scenarios, _ = build_scenario_catalog(
                        artifact_result, dossiers=dossiers)
                    shadow = next((item for item in shadow_scenarios
                                   if item.get("role") == "model_authored"), None)
                    if shadow is not None:
                        remaining = [item["scenario_id"]
                                     for item in shadow_scenarios
                                     if item["scenario_id"] != shadow["scenario_id"]]
                        claim_ids = list(shadow.get("claim_ids") or [])
                        counter_ids = list(dict.fromkeys(
                            str(hypothesis.get("hypothesis_id"))
                            for item in dossiers
                            for hypothesis in item.get("hypotheses") or []
                            if hypothesis.get("kind") == "unsupported"
                            and hypothesis.get("hypothesis_id")))
                        selection = {
                            "selected_scenario_id": shadow["scenario_id"],
                            "ranking": [shadow["scenario_id"], *remaining],
                            "cited_claim_ids": claim_ids,
                            "counterevidence_claim_ids": [],
                            "counterevidence_hypothesis_ids": counter_ids,
                            "confidence": .5,
                            "rationale": "Explicit shadow evaluation of the sealed candidate.",
                            "what_would_change_selection": "Resolved outcomes score this candidate.",
                        }
                try:
                    publication = publish_result(
                        artifact_result, mode="best_effort", dossiers=dossiers,
                        scenario_selection=selection)
                except ValueError as error:
                    selection_error = f"selector rejected: {error}"
                    selection = None
                    publication = publish_result(
                        artifact_result, mode="best_effort", dossiers=dossiers)
            if not verify_publication(publication):
                raise RuntimeError("best-effort publication failed verification")
            submission = publication["recommended_forecast"]
            selected_item = next((
                item for item in publication.get("candidate_portfolio") or []
                if item.get("scenario_id") == publication.get(
                    "recommended_scenario_id")), None)
            model_distribution_dossiers = [
                item for item in dossiers
                if (item.get("candidate_critique") or {}).get(
                    "candidate_origin") == "model_authored"
                and isinstance((item.get("forecast_candidate") or {}).get(
                    "sample_paths"), list)
                and (item.get("forecast_candidate") or {}).get("sample_paths")
            ]
            selected_dossier = None
            if (isinstance(selected_item, dict)
                    and selected_item.get("role") == "model_authored"):
                source_seal = selected_item.get("source_seal_sha256")
                selected_dossier = next((
                    item for item in model_distribution_dossiers
                    if item.get("seal_sha256") == source_seal), None)
            elif (self.output_role == "llm_candidate_shadow"
                    and len(model_distribution_dossiers) == 1):
                # The shadow route is an explicit evaluation instrument. Its
                # distribution is the uniquely sealed model candidate even
                # when product publication correctly refuses to promote it.
                selected_dossier = model_distribution_dossiers[0]
                source_seal = selected_dossier.get("seal_sha256")
            if selected_dossier is not None:
                raw_paths = ((selected_dossier or {}).get(
                    "forecast_candidate") or {}).get("sample_paths")
                if isinstance(raw_paths, list) and raw_paths:
                    governed_distribution_paths = [
                        [float(value) for value in path] for path in raw_paths]
            if self.output_role == "publication_best_effort":
                extra_info = {
                    **extra_info, "route": "publication_best_effort",
                    "publication": publication,
                    "scenario_selector": {
                        "attempted": selection_attempted,
                        "accepted": publication.get("scenario_selection") is not None,
                        "disposition": (
                            "policy_applied" if selection_policy_applied else
                            "accepted" if publication.get(
                                "scenario_selection") is not None else
                            "rejected" if selection_attempted else
                            "skipped_evidence_dominance"
                            if selection_error and selection_error.startswith(
                                "selector skipped:") else "not_required"),
                        "error": selection_error,
                        **({"policy": "best_effort_sampled_prior_policy"}
                           if selection_policy_applied else {}),
                    },
                    "llm_usage": self.client.usage_summary,
                    **({"governed_distribution": {
                        "kind": "sealed_empirical_model_paths",
                        "sample_count": len(governed_distribution_paths),
                        "horizon": len(governed_distribution_paths[0]),
                        "source_seal_sha256": source_seal,
                        "compact_summary": "recommended_forecast",
                    }} if governed_distribution_paths else {}),
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
                **({"governed_distribution": {
                    "kind": "sealed_empirical_model_paths",
                    "sample_count": len(governed_distribution_paths),
                    "horizon": len(governed_distribution_paths[0]),
                    "source_seal_sha256": source_seal,
                    "compact_summary": "recommended_forecast",
                }} if governed_distribution_paths else {}),
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
                "context_summary": publication.get("context_summary"),
                "scenario_selector": extra_info.get("scenario_selector"),
                "scenario_selection": publication.get("scenario_selection"),
                "candidate_portfolio": [{
                    "scenario_id": item.get("scenario_id"),
                    "role": item.get("role"),
                    "support": item.get("support"),
                    "human_selection_eligible": item.get(
                        "human_selection_eligible"),
                    "effect": item.get("effect"),
                } for item in publication.get("candidate_portfolio") or []],
                **({"governed_distribution": {
                    "kind": "sealed_empirical_model_paths",
                    "sample_count": len(governed_distribution_paths),
                    "horizon": len(governed_distribution_paths[0]),
                }} if governed_distribution_paths else {}),
            }
            # ``drive`` closes the MCP process before governed selection to
            # avoid holding an idle server during the second model call. Rewrite
            # the same trace after selection so the diagnostic names the output
            # that was actually scored rather than only the initial MCP result.
            run._write_trace()
        if governed_distribution_paths:
            paths = [list(governed_distribution_paths[index % len(
                governed_distribution_paths)]) for index in range(n_samples)]
        else:
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
        from gnomon.calibration_counterfactual import (
            deterministic_additive_drift_claim,
        )
        from gnomon.llm_dossier import (
            deterministic_dated_multiplier_dossier,
            deterministic_dated_directional_event_dossier,
            deterministic_dated_zero_window_dossier,
            deterministic_ended_recurring_disruption_dossier,
            deterministic_historical_observation_claim,
            deterministic_named_driver_relationship_dossier,
            deterministic_reference_power_dossier,
            validate_temporal_dossier,
        )
        from gnomon.workflows import DocumentRef, parse_context_response

        narrative_context = build_context_text(self.task)
        context = "\n\n".join(part for part in (
            narrative_context, self.companion_evidence) if part)
        future_timestamps = _task_future_timestamps(self.task)
        deterministic_reference_power = (
            deterministic_reference_power_dossier(
                context, cutoff=self.timestamps[-1],
                driver_names=list(self.companion_histories)))
        reference_power_spec = None
        if deterministic_reference_power is not None:
            deterministic_reference_power = dict(deterministic_reference_power)
            reference_power_spec = deterministic_reference_power.pop(
                "_reference_power_spec", None)
        deterministic_named_relationship = (
            deterministic_named_driver_relationship_dossier(
                context, cutoff=self.timestamps[-1],
                driver_names=list(self.companion_histories))
            if deterministic_reference_power is None else None)
        named_relationship_spec = None
        if deterministic_named_relationship is not None:
            deterministic_named_relationship = dict(
                deterministic_named_relationship)
            named_relationship_spec = deterministic_named_relationship.pop(
                "_named_driver_relationship", None)
        categorical_schedule = _extract_categorical_state_schedule(
            narrative_context, self.timestamps, future_timestamps)
        relationship_contract = bool(
            categorical_schedule is None
            and _has_explicit_lag_relationship(context))
        observation_contract = (
            not relationship_contract
            and _expects_historical_zero_interpretation(context))
        companion_contract = bool(
            not relationship_contract and not observation_contract
            and deterministic_reference_power is None
            and deterministic_named_relationship is None
            and _looks_like_structured_companion_context(
                context, future_timestamps))
        deterministic_ended_disruption = (
            deterministic_ended_recurring_disruption_dossier(
                context, cutoff=self.timestamps[-1])
            if (categorical_schedule is None and not relationship_contract
                and not observation_contract and not companion_contract)
            else None)
        if (categorical_schedule is not None or relationship_contract
                or observation_contract or deterministic_ended_disruption is not None):
            deterministic_reference_power = None
            deterministic_named_relationship = None
            named_relationship_spec = None
        deterministic_calibration_claim = (
            deterministic_additive_drift_claim(
                context, history_start=self.timestamps[0],
                cutoff=self.timestamps[-1])
            if (deterministic_ended_disruption is None
                and deterministic_reference_power is None
                and categorical_schedule is None and not relationship_contract
                and not observation_contract and not companion_contract)
            else None)
        deterministic_zero_window = (
            deterministic_dated_zero_window_dossier(
                context, cutoff=self.timestamps[-1],
                future_timestamps=future_timestamps,
                target_name=self.target_name)
            if (deterministic_ended_disruption is None
                and deterministic_reference_power is None
                and deterministic_named_relationship is None
                and deterministic_calibration_claim is None
                and categorical_schedule is None and not relationship_contract
                and not observation_contract and not companion_contract)
            else None)
        deterministic_multiplier = (
            deterministic_dated_multiplier_dossier(
                context, cutoff=self.timestamps[-1],
                future_timestamps=future_timestamps,
                target_name=self.target_name)
            if (deterministic_calibration_claim is None
                and deterministic_zero_window is None
                and categorical_schedule is None and not relationship_contract
                and not observation_contract and not companion_contract)
            else None)
        deterministic_directional_event = (
            deterministic_dated_directional_event_dossier(
                context, cutoff=self.timestamps[-1],
                future_timestamps=future_timestamps,
                target_name=self.target_name)
            if (deterministic_calibration_claim is None
                and deterministic_zero_window is None
                and deterministic_multiplier is None
                and categorical_schedule is None and not relationship_contract
                and not observation_contract and not companion_contract)
            else None)
        compiler_context = (narrative_context if relationship_contract else context)
        history = _compiler_target_evidence(
            self.timestamps, self.values,
            limit=8 if relationship_contract else
            128 if observation_contract else 64)
        material_numeric_context = _has_material_numeric_context(
            compiler_context)
        instructions = (RELATIONSHIP_INSTRUCTIONS if relationship_contract
                        else OBSERVATION_INSTRUCTIONS if observation_contract
                        else COMPANION_INSTRUCTIONS if companion_contract
                        else DOSSIER_INSTRUCTIONS if material_numeric_context
                        else QUALITATIVE_INSTRUCTIONS)
        # A governed dossier has a deliberately bounded schema. Leaving the
        # completion uncapped lets verbose providers consume the entire shared
        # workflow deadline and starve the separately sealed candidate lane.
        compiler_max_tokens = (2_000 if instructions is QUALITATIVE_INSTRUCTIONS
                               else 5_000)
        numeric_routing_note = (
            "\nHost routing note: the context contains at least one material "
            "numeric quantity beyond dates or clock times. Do not return an "
            "empty dossier. Preserve it in a verbatim claim and at least one "
            "typed hypothesis. If the history plus cited quantity supports a "
            "bounded useful conditional forecast, include that sealed "
            "prior_assisted candidate in this first response; otherwise "
            "classify it unsupported and name the missing evidence.\n"
            if material_numeric_context and not relationship_contract
            and not companion_contract else "")
        prompt = (
            f"{instructions}{numeric_routing_note}\n"
            f"Forecast target series: {self.target_name}\n"
            f"History cutoff: {self.timestamps[-1]}\n"
            f"Forecast grid: {_forecast_grid_prompt(future_timestamps)}\n"
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
        repair_decisions: list[dict[str, Any]] = []
        model_candidate_proposal: dict[str, Any] | None = None
        model_candidate_prompt_bytes = 0
        model_candidate_status = "not_requested"
        model_candidate_sampling: dict[str, Any] | None = None
        model_candidate_sample_paths: list[list[float]] | None = None
        deterministic_companion_tables = (
            _extract_structured_companion_tables(
                context, self.timestamps, future_timestamps)
            if companion_contract else [])

        def bind_active_target(candidate: dict[str, Any]) -> dict[str, Any]:
            """Attach host-owned target and observed companion identities.

            The model cannot introduce a series name here.  Exposing the
            columns already present in the governed snapshot lets a typed
            relationship survive validation without granting it numeric
            authority.
            """
            return {**candidate, "series": [
                self.target_name, *sorted(self.companion_histories)]}

        def bind_host_knowledge_time(candidate: dict[str, Any]) -> dict[str, Any]:
            """Bind receipt-time metadata the compiler has no authority over."""
            for hypothesis in candidate.get("hypotheses") or []:
                if isinstance(hypothesis, dict):
                    hypothesis["known_at"] = self.timestamps[-1]
            return candidate

        def complete(content: str, stage: str) -> str:
            remaining = compilation_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "context workflow deadline exhausted before " + stage)
            started = time.monotonic()
            # The one bounded repair is part of the public behavior, not a
            # theoretical fallback. Do not let an empty or malformed initial
            # completion consume the entire workflow budget and leave the
            # repair a one-second request that cannot possibly complete.
            request_budget = remaining
            if stage == "initial_compile" and remaining > 2:
                reserve = min(MIN_CONTEXT_REPAIR_SECONDS, remaining * .25)
                request_budget = max(1.0, remaining - reserve)
            try:
                return self.forecaster.client.completions(
                    [{"role": "user", "content": content}], n=1,
                    temperature=0, max_tokens=compiler_max_tokens,
                    reasoning_effort="none",
                    request_timeout=max(1, min(
                        120, math.floor(request_budget))),
                    transport_retries=0)[0]
            finally:
                compiler_calls.append({
                    "stage": stage,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                })

        def complete_many(content: str, stage: str, *, n: int) -> list[str]:
            """Bounded concurrent single-sample elicitation under one deadline."""
            remaining = compilation_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "context workflow deadline exhausted before " + stage)
            started = time.monotonic()
            try:
                from concurrent.futures import ThreadPoolExecutor
                messages = [
                    {"role": "system", "content":
                     "You are a useful forecasting assistant."},
                    {"role": "user", "content": content},
                ]
                timeout = max(1, min(120, math.floor(remaining)))

                def one_sample(_: int) -> tuple[str, str | None]:
                    try:
                        # The dossier has already isolated the temporal
                        # argument. This call supplies one numeric path on a
                        # host-owned grid; hidden reasoning adds latency and
                        # can consume the workflow deadline without improving
                        # the auditable output.
                        path_token_budget = min(
                            6_000, max(1_500,
                                       500 + 30 * len(future_timestamps)))
                        response = self.forecaster.client.completions(
                            messages, n=1, temperature=1,
                            max_tokens=path_token_budget,
                            reasoning_effort="none", request_timeout=timeout,
                            transport_retries=0)[0]
                        return response, None
                    except Exception as error:  # independent sampled draws
                        # One slow or failed provider request must not erase
                        # the other sealed paths. Transport availability is
                        # recorded separately from semantic path validity.
                        return "", str(error)[:300]

                with ThreadPoolExecutor(max_workers=min(n, 8)) as pool:
                    outcomes = list(pool.map(one_sample, range(n)))
                failures = [error for _, error in outcomes if error]
                if failures:
                    compiler_calls.append({
                        "stage": stage + "_partial_failures",
                        "failed_completions": len(failures),
                        "failure_reasons": failures[:3],
                    })
                return [response for response, _ in outcomes]
            finally:
                compiler_calls.append({
                    "stage": stage,
                    "requested_completions": n,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                })

        def normalize_relationship(candidate: dict[str, Any]) -> dict[str, Any]:
            """Reapply host-owned relationship bindings after any LLM turn."""
            candidate = _canonicalize_unreferenced_covariate_names(candidate)
            if not relationship_contract or not candidate.get("transformations"):
                return candidate
            claims = [item for item in candidate.get("claims") or []
                      if isinstance(item, dict)]
            grounded = any(str(item.get("source_span") or "") in context
                           and str(item.get("source_span") or "").strip()
                           for item in claims)
            host_fallback = not grounded and bool(narrative_context.strip())
            if host_fallback:
                claims = [{
                    "source_span": narrative_context, "relation": "unknown",
                    "effective_start": future_timestamps[0],
                    "effective_end": future_timestamps[-1],
                    "mechanism": "host-grounded explicit equation document",
                    "confidence": 1.0,
                }]
            for claim in claims:
                if str(claim.get("source_span") or "").strip() in context:
                    claim["effective_start"] = future_timestamps[0]
                    claim["effective_end"] = future_timestamps[-1]
                    claim.pop("timing_status", None)
                    claim["effective_window_binding"] = (
                        "host_forecast_grid_for_relationship_specification")
            candidate = {**candidate, "claims": claims}
            for wrapper in candidate.get("transformations") or []:
                if not isinstance(wrapper, dict):
                    continue
                transformation = wrapper.get("transformation", wrapper)
                if host_fallback and isinstance(transformation, dict):
                    transformation["claim_ids"] = ["claim-1"]
                supplied = wrapper.get("series_values") or {}
                for payload in (supplied.values()
                                if isinstance(supplied, dict) else []):
                    if host_fallback and isinstance(payload, dict):
                        payload["source_claim_ids"] = ["claim-1"]
            return candidate
        if categorical_schedule is not None:
            raw = bind_active_target(categorical_schedule["raw"])
            compiler_calls.append({
                "stage": "deterministic_categorical_state_parse",
                "elapsed_seconds": 0.0,
            })
        elif deterministic_companion_tables:
            raw = bind_active_target({
                "events": [], "claims": [], "hypotheses": [],
                "covariate_tables": deterministic_companion_tables,
                "transformations": [], "observation_interpretations": [],
                "effect_proposal": None, "forecast_candidate": None,
            })
            compiler_calls.append({
                "stage": "deterministic_structured_companion_parse",
                "elapsed_seconds": 0.0,
            })
            # Parsing and reasoning are separate jobs. Exact rows do not need
            # model transcription, but best-effort publication may still gain
            # from a separately sealed model interpretation of their temporal
            # relationship. This candidate never replaces the governed
            # executable in its dossier and can never authorize automation.
            if self.forecaster.output_role in {
                    "publication_best_effort", "llm_candidate_shadow"}:
                model_candidate_status = "requested"
                candidate_prompt = (
                    f"{COMPANION_CANDIDATE_INSTRUCTIONS}\n"
                    f"Forecast target series: {self.target_name}\n"
                    f"History cutoff: {self.timestamps[-1]}\n"
                    f"Forecast grid: {_forecast_grid_prompt(future_timestamps)}\n"
                    f"{history}\n\nContext:\n{context}\n")
                model_candidate_prompt_bytes = len(
                    candidate_prompt.encode("utf-8"))
                def accept_model_candidate(content: str) -> bool:
                    nonlocal model_candidate_proposal, model_candidate_status
                    objects = extract_json_objects(content)
                    if not objects:
                        model_candidate_status = "no_json"
                        return False
                    first = objects[0]
                    wrapped = first.get("forecast_candidate")
                    if isinstance(wrapped, dict):
                        model_candidate_proposal = wrapped
                        model_candidate_status = "proposed"
                        return True
                    if isinstance(first.get("quantiles"), list):
                        # Bounded schema normalization: accept the exact
                        # candidate body when the model omitted only its outer
                        # key. Validators still own timestamps, quantiles,
                        # citations and plausibility.
                        model_candidate_proposal = first
                        model_candidate_status = "proposed_unwrapped"
                        return True
                    model_candidate_status = (
                        "withheld" if wrapped is None else "invalid_shape")
                    return False
                try:
                    candidate_completion = complete(
                        candidate_prompt, "model_companion_candidate")
                    accepted = accept_model_candidate(candidate_completion)
                    if not accepted and not repair_used:
                        repair_used = True
                        repair_completion = complete(
                            candidate_prompt
                            + "\nYour previous response was not an executable "
                              "candidate (status=" + model_candidate_status
                            + "). This is the single repair: return only the "
                              "requested JSON with every grid row.",
                            "model_companion_candidate_repair")
                        accepted = accept_model_candidate(repair_completion)
                        if accepted:
                            model_candidate_status = "proposed_after_repair"
                    if not accepted:
                        compile_rejections.append(
                            "model companion candidate unavailable after "
                            f"bounded repair: {model_candidate_status}")
                except Exception as error:
                    model_candidate_status = "request_failed"
                    compile_rejections.append(
                        f"model companion candidate failed: {error}")
        elif deterministic_ended_disruption is not None:
            raw = bind_active_target(deterministic_ended_disruption)
            compiler_calls.append({
                "stage": "deterministic_ended_recurring_disruption_parse",
                "elapsed_seconds": 0.0,
            })
        elif deterministic_reference_power is not None:
            raw = bind_active_target(deterministic_reference_power)
            compiler_calls.append({
                "stage": "deterministic_reference_power_parse",
                "elapsed_seconds": 0.0,
            })
        elif deterministic_named_relationship is not None:
            raw = bind_active_target(deterministic_named_relationship)
            compiler_calls.append({
                "stage": "deterministic_named_driver_relationship_parse",
                "elapsed_seconds": 0.0,
            })
        elif deterministic_calibration_claim is not None:
            raw = bind_active_target({
                "events": [], "claims": [deterministic_calibration_claim],
                "hypotheses": [], "covariate_tables": [],
                "transformations": [], "observation_interpretations": [],
                "effect_proposal": None, "forecast_candidate": None,
            })
            compiler_calls.append({
                "stage": "deterministic_additive_drift_parse",
                "elapsed_seconds": 0.0,
            })
        elif deterministic_zero_window is not None:
            raw = bind_active_target(deterministic_zero_window)
            compiler_calls.append({
                "stage": "deterministic_dated_zero_window_parse",
                "elapsed_seconds": 0.0,
            })
        elif deterministic_multiplier is not None:
            raw = bind_active_target(deterministic_multiplier)
            compiler_calls.append({
                "stage": "deterministic_dated_multiplier_parse",
                "elapsed_seconds": 0.0,
            })
        elif deterministic_directional_event is not None:
            raw = bind_active_target(deterministic_directional_event)
            compiler_calls.append({
                "stage": "deterministic_dated_directional_event_parse",
                "elapsed_seconds": 0.0,
            })
        else:
            try:
                completion = complete(prompt, "initial_compile")
                objects = extract_json_objects(completion)
                if objects:
                    raw = bind_host_knowledge_time(bind_active_target(objects[0]))
                else:
                    compile_rejections.append(
                        "no JSON object in temporal-dossier output")
            except Exception as error:
                compile_rejections.append(f"dossier compilation failed: {error}")

        # Host bindings are idempotent and must be applied after every model
        # turn, not only the initial completion.
        raw = normalize_relationship(raw)

        # The host assembled this receipt at the cutoff. Compiler-authored
        # knowledge times are neither trusted nor useful; bind all numeric
        # lanes to the host-owned timestamp before any validation or repair.
        for wrapper in raw.get("transformations") or []:
            if not isinstance(wrapper, dict):
                continue
            transformation = wrapper.get("transformation", wrapper)
            if isinstance(transformation, dict):
                transformation["known_at"] = self.timestamps[-1]
            raw_series = wrapper.get("series_values") or {}
            for supplied in (raw_series.values()
                             if isinstance(raw_series, dict) else []):
                if isinstance(supplied, dict):
                    supplied["known_at"] = self.timestamps[-1]
        raw, _ = _bind_missing_transformation_claim_windows(
            raw, self.timestamps[-1])

        # High-precision historical observation semantics do not need an LLM
        # to copy the same sentence twice.  Bind the deterministic verbatim
        # claim before probing so the ordinary dossier validator can derive
        # and replay its contamination interpretation immediately.  A repair
        # remains available when the literal extractor cannot establish the
        # required source evidence.
        literal_observation_claim = None
        if _expects_historical_zero_interpretation(context):
            literal_observation_claim = (
                deterministic_historical_observation_claim(
                    context, history_start=self.timestamps[0],
                    cutoff=self.timestamps[-1]))
            if literal_observation_claim is not None:
                remaining_claims = [
                    item for item in raw.get("claims") or []
                    if not isinstance(item, dict) or
                    str(item.get("source_span") or "") !=
                    literal_observation_claim["source_span"]]
                raw = {**raw, "claims": [
                    literal_observation_claim, *remaining_claims]}

        # Exercise the product's bounded repair lane. The first response is
        # probed before event parsing so a corrected complete dossier (claims
        # plus effect) feeds every downstream validator consistently.
        proposed_any_lane = any(raw.get(key) not in (None, [], {}) for key in (
            "events", "claims", "hypotheses", "covariate_tables",
            "transformations", "observation_interpretations",
            "effect_proposal", "forecast_candidate"))
        observation_lane_missing = (
            _expects_historical_zero_interpretation(context)
            and literal_observation_claim is None
            and not raw.get("observation_interpretations"))
        # An empty response to numeric context is not a successful compile.
        # It is common for useful references (a comparable site's peak, a
        # budget, a dated rate) to be informative but not deterministic. Give
        # the existing single repair round a chance to represent that evidence
        # as a cited hypothesis or sealed prior-assisted scenario. The model
        # may still explicitly classify it unsupported; it may not silently
        # erase supplied information.
        unresolved_context = bool(context.strip() and not proposed_any_lane)
        unresolved_numeric_context = bool(
            unresolved_context and _has_material_numeric_context(context))
        future_path_needs_executable = _future_numeric_path_needs_executable(
            context, future_timestamps, raw)
        if categorical_schedule is not None:
            # State labels are categorical observations, not numeric paths.
            # The deterministic fitted executable below owns their influence.
            future_path_needs_executable = False
        companion_mapping_pending = bool(
            companion_contract and raw.get("covariate_tables")
            and future_path_needs_executable)
        transformation_proposed = bool(raw.get("transformations"))
        if ((proposed_any_lane or observation_lane_missing
                or unresolved_context) and not transformation_proposed):
            probe, probe_rejections = validate_temporal_dossier(
                raw, context_text=context, cutoff=self.timestamps[-1],
                future_timestamps=future_timestamps, history=self.values,
                history_timestamps=self.timestamps,
                compiler_model=self.forecaster.openrouter_model)
            effect_critique = probe.get("effect_proposal_critique") or {}
            candidate_critique = probe.get("candidate_critique") or {}
            effect_failed = effect_critique.get("status") == "rejected"
            effect_accepted = effect_critique.get("status") in {
                "accepted", "accepted_after_repair"}
            candidate_failed = candidate_critique.get("status") == "rejected"
            candidate_accepted = candidate_critique.get("status") in {
                "accepted", "accepted_after_repair"}
            hypothesis_failures = (probe.get("hypothesis_critique") or {}).get(
                "rejected") or []
            hypothesis_violation_codes = list(dict.fromkeys(
                str(violation.get("code"))
                for failure in hypothesis_failures
                for violation in failure.get("violations") or []
                if violation.get("code")))
            observation_failures = (
                probe.get("observation_interpretation_critique") or {}).get(
                    "rejected") or []
            observation_accepted = bool((
                probe.get("observation_interpretation_critique") or {}).get(
                    "accepted"))
            accepted_executable = (
                effect_accepted or candidate_accepted or observation_accepted)
            verified_claims = probe.get("claims") or []
            retained_unresolved_interpretation = bool(
                verified_claims
                and all(claim.get("timing_status") == "unresolved_trigger"
                        for claim in verified_claims))
            retained_atemporal_interpretation = bool(
                verified_claims
                and all(claim.get("timing_status") == "atemporal_context"
                        for claim in verified_claims)
                and probe.get("hypotheses"))
            retained_reference_interpretation = bool(
                (deterministic_reference_power is not None
                 or deterministic_named_relationship is not None)
                and verified_claims and probe.get("hypotheses")
                and not hypothesis_failures)
            # Once one numeric lane is valid, malformed optional lanes remain
            # visible in the dossier critique but must not replace a useful
            # result or consume another LLM call. Required observation
            # semantics are the exception because answering a different data
            # interpretation would be materially wrong. Likewise, when every
            # retained claim has unresolved trigger timing, another model call
            # cannot create the missing source evidence: preserve the useful
            # hypotheses and return the typed recovery instead of retrying a
            # categorically ineligible numeric lane.
            repair_required = (
                observation_lane_missing or unresolved_context
                or (future_path_needs_executable
                    and not companion_mapping_pending
                    and not retained_reference_interpretation
                    and not accepted_executable)
                or (not accepted_executable
                    and not retained_unresolved_interpretation
                    and not retained_atemporal_interpretation and (
                    probe_rejections or effect_failed or candidate_failed
                    or hypothesis_failures or observation_failures)))
            rejected_effect_fields = {}
            if effect_failed and isinstance(raw.get("effect_proposal"), dict):
                for key in ("shape", "unit", "location", "lower", "upper",
                            "confidence", "delay_steps", "duration_steps",
                            "period_steps"):
                    if key not in raw["effect_proposal"]:
                        continue
                    value = raw["effect_proposal"][key]
                    rejected_effect_fields[key] = (
                        {"type": type(value).__name__, "value": value}
                        if isinstance(value, (str, int, float, bool))
                        or value is None else
                        {"type": type(value).__name__})
            repair_decisions.append({
                "stage": "dossier_probe",
                "triggered": bool(repair_required),
                "accepted_executable": bool(accepted_executable),
                "effect_status": effect_critique.get("status"),
                "effect_violation_codes": list(dict.fromkeys(
                    str(violation.get("code"))
                    for attempt in effect_critique.get("attempts") or []
                    for violation in attempt.get("violations") or []
                    if violation.get("code"))),
                "rejected_effect_fields": rejected_effect_fields,
                "candidate_status": candidate_critique.get("status"),
                "candidate_reasons": [str(reason) for reason in
                                      candidate_critique.get("reasons") or []][
                                          :6],
                "accepted_observation_interpretations": _validated_item_count((
                    probe.get("observation_interpretation_critique") or {}).get(
                        "accepted")),
                "rejected_hypotheses": len(hypothesis_failures),
                "hypothesis_violation_codes": hypothesis_violation_codes,
                "rejected_observation_interpretations": len(
                    observation_failures),
                "required_observation_lane_missing": bool(
                    observation_lane_missing),
                "numeric_context_unresolved": bool(
                    unresolved_numeric_context),
                "context_unresolved": bool(unresolved_context),
                "future_numeric_path_required": bool(
                    future_path_needs_executable),
                "future_numeric_path_needs_executable": bool(
                    future_path_needs_executable
                    and not companion_mapping_pending
                    and not accepted_executable),
                "governed_companion_mapping_pending": companion_mapping_pending,
                "top_level_rejections": len(probe_rejections),
                **({
                    "retained_unresolved_interpretation": True,
                    "skip_reason": "missing_trigger_evidence_not_repairable",
                } if retained_unresolved_interpretation else {}),
                **({
                    "retained_atemporal_interpretation": True,
                    "skip_reason": "typed_background_already_preserved",
                } if retained_atemporal_interpretation else {}),
                **({
                    "retained_reference_interpretation": True,
                    "skip_reason": (
                        "typed_reference_law_preserved_for_sealed_prior"),
                } if retained_reference_interpretation else {}),
            })
            if repair_required:
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
                        "code": "FUTURE_NUMERIC_PATH_NEEDS_EXECUTABLE",
                        "message": (
                            "The source supplies dated numeric companion-series "
                            "values on the requested forecast grid, but the "
                            "dossier flattened them into prose. Extract the "
                            "single most target-relevant path as verbatim "
                            "covariate rows. If the short target overlap can "
                            "support a bounded level/scale mapping, also return "
                            "a sealed probabilistic forecast_candidate whose "
                            "rationale states the mapping and uncertainty. It "
                            "is prior_assisted, human-review-only, and cannot "
                            "authorize automation. Otherwise preserve a typed "
                            "rejection naming why the supplied path is not "
                            "identifiable. Do not copy a target outcome or "
                            "invent missing rows."
                        ),
                    } if future_path_needs_executable else {
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
                    } if unresolved_numeric_context else {
                        "code": "CONTEXT_UNRESOLVED",
                        "message": (
                            "The supplied context was silently represented by "
                            "no claim, hypothesis, executable, or rejection. "
                            "Return at least one verbatim cited claim and a "
                            "typed hypothesis describing what it could imply. "
                            "If it cannot safely influence a forecast, preserve "
                            "it as scenario-only or reject it with the specific "
                            "missing evidence. Propose a sealed candidate only "
                            "when a bounded path is defensible. Never invent an "
                            "effect size or alter the immutable primary."
                        ),
                    } if unresolved_context else None),
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
                        raw = bind_host_knowledge_time(
                            bind_active_target(repaired[0]))
                        raw = normalize_relationship(raw)
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
            literal_claim = literal_observation_claim or (
                deterministic_historical_observation_claim(
                    context, history_start=self.timestamps[0],
                    cutoff=self.timestamps[-1]))
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
            repair_decisions.append({
                "stage": "relationship_sufficiency_probe",
                "triggered": True,
                "exact_lag_claims": len(exact_lag_claims),
                "numeric_lane_missing": True,
            })
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
                    raw = normalize_relationship(raw)
                else:
                    compile_rejections.append(
                        "relationship sufficiency repair returned no JSON object")
            except Exception as error:
                compile_rejections.append(
                    f"relationship sufficiency repair failed: {error}")

        # Claims remain evidence for dossiers and transformations. Do not
        # synthesize wildcard numeric events from them: only explicitly
        # target-bound event proposals may change the numeric path.
        normalized_schedules = []
        for item in raw.get("transformations") or []:
            simplified, _ = _simplify_identity_literals(item)
            expanded, _ = _expand_change_point_series_values(
                simplified, future_timestamps)
            if isinstance(expanded, dict):
                transformation = expanded.get("transformation", expanded)
                if isinstance(transformation, dict):
                    transformation["known_at"] = self.timestamps[-1]
                raw_series = expanded.get("series_values") or {}
                for payload in (raw_series.values()
                                if isinstance(raw_series, dict) else []):
                    if isinstance(payload, dict):
                        payload["known_at"] = self.timestamps[-1]
            normalized_schedules.append(expanded)
        raw = {**raw, "transformations": normalized_schedules}

        existing_claims = [item for item in raw.get("claims") or []
                           if isinstance(item, dict)]
        existing_spans = [str(item.get("source_span") or "")
                          for item in existing_claims]
        constant_claim_spans = []
        for wrapper in raw.get("transformations") or []:
            constant_claim_spans.extend(_verbatim_constant_lines(
                wrapper, context, existing_spans + constant_claim_spans))
            constant_claim_spans.extend(_verbatim_semantic_constant_lines(
                wrapper, context, existing_spans + constant_claim_spans))
            constant_claim_spans.extend(_verbatim_series_lines(
                wrapper, context, existing_spans + constant_claim_spans))
        for span in dict.fromkeys(constant_claim_spans):
            existing_claims.append({
                "source_span": span,
                "relation": "unknown",
                "effective_start": self.timestamps[-1],
                "effective_end": self.timestamps[-1],
                "mechanism": "host-grounded explicit transformation constant",
                "confidence": 1.0,
            })
        if constant_claim_spans:
            raw = {**raw, "claims": existing_claims}

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
        raw = _bind_transformation_provenance(raw, verified_claims)
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
                transformation_raw = wrapper.get("transformation", wrapper)
                if not isinstance(transformation_raw, dict):
                    normalized_transformations.append(wrapper)
                    continue
                transformation = dict(transformation_raw)
                cited = {str(value) for value in
                         transformation.get("claim_ids") or []}
                if not cited or not cited.issubset(known_ids):
                    transformation["claim_ids"] = [sole_id]
                    transformation["citation_binding"] = (
                        "single_verified_claim")
                wrapper["transformation"] = transformation
                raw_series_values = wrapper.get("series_values") or {}
                if not isinstance(raw_series_values, dict):
                    normalized_transformations.append(wrapper)
                    continue
                series_values = {}
                for name, payload in raw_series_values.items():
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
        def normalize_deterministic_events(
                candidate_raw: dict[str, Any],
                probe: dict[str, Any]) -> dict[str, Any]:
            """Apply host numeric parsing after every compiler/repair pass."""
            existing_events = [item for item in
                               candidate_raw.get("events") or []
                               if isinstance(item, dict)]
            single_target_event_spans = {
                str(item.get("evidence_quote") or
                    item.get("source_span") or "")
                for item in existing_events
                if str(item.get("evidence_quote") or
                       item.get("source_span") or "").strip()
            }
            # Deterministic zero-window compilation already verified target
            # ownership against the host-bound single target. Preserve that
            # authority when symbolic CiK column names (for example ``0``)
            # would otherwise obscure ordinary nouns such as withdrawals.
            if deterministic_zero_window is not None:
                single_target_event_spans.update(
                    str(item.get("source_span") or "")
                    for item in deterministic_zero_window.get("claims") or []
                    if str(item.get("source_span") or "").strip())
            derived_events = deterministic_events_from_claims(
                {**probe, "events": existing_events},
                target_name=self.target_name,
                target_verified_spans=single_target_event_spans)
            existing_keys = {(str(item.get("event_type") or ""),
                              str(item.get("evidence_quote") or
                                  item.get("source_span") or ""))
                             for item in existing_events}
            for derived in derived_events:
                event = {
                    # Validate against the semantic input column first. The
                    # runtime conversion to ``__default__`` happens only
                    # after ``parse_context_response`` accepts this binding.
                    **derived, "entity_scope": [self.target_name],
                    "host_target_binding": "single_target_verified_claim",
                }
                quote = str(event.get("evidence_quote") or
                            event.get("source_span") or "")
                shadowed_types = [str(item.get("event_type") or "")
                                  for item in existing_events
                                  if str(item.get("evidence_quote") or
                                         item.get("source_span") or "") == quote
                                  and str(item.get("event_type") or "") != str(
                                      event.get("event_type") or "")]
                if shadowed_types:
                    # One source quote gets one numeric authority. A broad
                    # model label must not shadow the stricter host parser.
                    existing_events = [
                        item for item in existing_events
                        if str(item.get("evidence_quote") or
                               item.get("source_span") or "") != quote]
                    existing_keys = {
                        (str(item.get("event_type") or ""),
                         str(item.get("evidence_quote") or
                             item.get("source_span") or ""))
                        for item in existing_events
                    }
                    event["attributes"] = {
                        **(event.get("attributes") or {}),
                        "host_normalization": {
                            "supersedes_model_event_types": sorted(set(
                                shadowed_types)),
                            "basis": "same_verified_source_quote",
                        },
                    }
                key = (str(event.get("event_type") or ""), quote)
                if key not in existing_keys:
                    existing_events.append(event)
                    existing_keys.add(key)
            return {**candidate_raw, "events": existing_events}

        raw = normalize_deterministic_events(raw, final_probe)

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
                if (not isinstance(embedded, dict) and isinstance(item, dict)
                        and item.get("type") in {
                            "recursive_linear", "fit_recursive_linear"}):
                    embedded = item
                compact = (item.get("recursive_linear")
                           if isinstance(item, dict) else None)
                if not isinstance(compact, dict) and isinstance(embedded, dict):
                    compact = embedded.get("recursive_linear")
                embedded_type = (str(embedded.get("type") or "")
                                 if isinstance(embedded, dict) else "")
                if (not isinstance(compact, dict) and isinstance(embedded, dict)
                        and embedded_type in {
                            "recursive_linear", "fit_recursive_linear"}
                        and "expression" not in embedded):
                    compact = {key: embedded.get(key) for key in (
                        "intercept", "autoregressive_terms", "driver_terms",
                        "autoregressive_lags", "driver_lags",
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
                            "expression": {"op": (embedded_type or
                                                   "recursive_linear"),
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
                            "op") in {"recursive_linear", "fit_recursive_linear"}:
                        values = dict(canonical.get("series_values") or {})
                        histories = dict(canonical.get(
                            "historical_series_segments") or {})
                        units = dict(canonical.get("units") or {})
                        claim_ids = list(transformation.get("claim_ids") or [
                            "claim-1"])
                        relationship_terms = (
                            expression.get("driver_terms") or []
                            if expression.get("op") == "recursive_linear"
                            else expression.get("driver_lags") or [])
                        for name in sorted({str(term.get("series")) for term in
                                            relationship_terms
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
                            # Exact cited range extraction owns this
                            # representation. A model-authored history may
                            # carry stale claim IDs, overlaps, or impossible
                            # endpoints; retaining it with ``setdefault`` made
                            # deterministic recovery depend on compiler shape.
                            histories[name] = historical
                            units.setdefault(name, str(
                                transformation.get("output_unit") or "target_units"))
                        canonical = {**canonical, "series_values": values,
                                     "historical_series_segments": histories,
                                     "units": units}
                normalized.append(canonical)
            return {**candidate_raw, "transformations": normalized}

        raw = canonicalize_transformations(raw)

        unit_bound_transformations = []
        for item in raw.get("transformations") or []:
            source_shaped, _ = _restore_cited_power_literals(
                item, verified_claims)
            canonical_rows, _ = _canonicalize_timestamped_series_values(
                source_shaped, future_timestamps)
            bound, _ = _bind_verbatim_literal_units(
                canonical_rows, verified_claims)
            unit_bound_transformations.append(bound)
        raw = {**raw, "transformations": unit_bound_transformations}

        def transformation_violations(
                candidate_raw: dict[str, Any], dossier: dict[str, Any]) -> list[dict[str, Any]]:
            from gnomon.context_intelligence import (
                TransformationError, compile_transformation,
                execute_transformation,
            )

            claims = dossier.get("claims") or []
            claim_ids = [str(claim.get("claim_id")) for claim in claims]
            unresolved_ids = {
                str(claim.get("claim_id")) for claim in claims
                if claim.get("timing_status") == "unresolved_trigger"
            }
            spans = {str(claim.get("claim_id")): str(
                claim.get("source_span") or "") for claim in claims}
            failures = []
            for index, item in enumerate(candidate_raw.get("transformations") or [], 1):
                wrapper = item if isinstance(item, dict) else {}
                transformation = wrapper.get("transformation", wrapper)
                cited = {str(value) for value in
                         (transformation.get("claim_ids") or []
                          if isinstance(transformation, dict) else [])}
                if cited.intersection(unresolved_ids):
                    failures.append({"index": index, "violations": [{
                        "code": "UNRESOLVED_TRIGGER_TIMING",
                        "message": (
                            "A transformation cannot execute until every "
                            "cited trigger has a dated effective window."),
                    }]})
                    continue
                raw_series = wrapper.get("series_values") or {}
                compiled, critique = compile_transformation(
                    transformation,
                    series=list(raw_series.keys())
                    if isinstance(raw_series, dict) else [],
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
                expression = compiled.get("expression") or {}
                if expression.get("op") == "fit_recursive_linear":
                    # Fit admission needs real aligned history and may use
                    # cited historical segments that are reconstructed only
                    # inside the governed forecast boundary. Syntax, claims,
                    # future provenance and the eventual fit are still
                    # fail-closed there; a dummy fit here would falsely reject
                    # a valid document-supplied driver.
                    continue
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
        effect_status = (final_probe.get("effect_proposal_critique") or {}).get(
            "status")
        accepted_observations = (
            final_probe.get("observation_interpretation_critique") or {}).get(
                "accepted") or []
        observation_count = _validated_item_count(accepted_observations)
        non_transform_executable = (
            effect_status in {"accepted", "accepted_after_repair"}
            or observation_count > 0)
        unresolved_spans = [
            " ".join(str(claim.get("source_span") or "").split()).casefold()
            for claim in final_probe.get("claims") or []
            if claim.get("timing_status") == "unresolved_trigger"
            and claim.get("source_span")]
        event_timing_join_possible = any(
            span and span in " ".join(str(
                event.get("evidence_quote") or event.get("source_span") or
                "").split()).casefold()
            for span in unresolved_spans
            for event in raw.get("events") or []
            if isinstance(event, dict)
            and event.get("effective_start") and event.get("effective_end"))
        transform_repair_eligible = bool(
            transform_failures and not repair_used
            and not non_transform_executable
            and not event_timing_join_possible)
        if raw.get("transformations"):
            transform_violation_codes = sorted({
                str(violation.get("code"))
                for failure in transform_failures
                for violation in failure.get("violations", [])
                if violation.get("code")
            })
            repair_decisions.append({
                "stage": "transformation_preflight",
                "triggered": transform_repair_eligible,
                "failure_count": len(transform_failures),
                "violation_codes": transform_violation_codes,
                "repair_already_used": bool(repair_used),
                "alternative_executable_available": bool(
                    non_transform_executable),
                "skip_reason": (
                    "valid_non_transform_executable"
                    if transform_failures and non_transform_executable
                    else "validated_event_join_pending"
                    if transform_failures and event_timing_join_possible
                    else None),
            })
        if transform_repair_eligible:
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
                    raw = _merge_transformation_repair(raw, repaired)
                    raw, _ = _bind_missing_transformation_claim_windows(
                        raw, self.timestamps[-1])
                    repaired_transforms = []
                    for item in raw.get("transformations") or []:
                        simplified, _ = _simplify_identity_literals(item)
                        expanded, _ = _expand_change_point_series_values(
                            simplified, future_timestamps)
                        repaired_transforms.append(expanded)
                    raw = {**raw, "transformations": repaired_transforms}
                    final_probe, _ = validate_temporal_dossier(
                        raw, context_text=context, cutoff=self.timestamps[-1],
                        future_timestamps=future_timestamps,
                        history=self.values,
                        history_timestamps=self.timestamps,
                        compiler_model=self.forecaster.openrouter_model)
                    verified_claims = final_probe.get("claims") or []
                    raw = _bind_transformation_provenance(
                        raw, verified_claims)
                    raw = canonicalize_transformations(raw)
                    repaired_bound = []
                    for item in raw.get("transformations") or []:
                        source_shaped, _ = _restore_cited_power_literals(
                            item, verified_claims)
                        canonical_rows, _ = (
                            _canonicalize_timestamped_series_values(
                                source_shaped, future_timestamps))
                        bound, _ = _bind_verbatim_literal_units(
                            canonical_rows, verified_claims)
                        repaired_bound.append(bound)
                    raw = {**raw, "transformations": repaired_bound}
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
            raw_series = wrapper.get("series_values") or {}
            for supplied in (raw_series.values()
                             if isinstance(raw_series, dict) else []):
                if isinstance(supplied, dict):
                    supplied["known_at"] = self.timestamps[-1]
        raw = canonicalize_transformations(raw)
        raw, duplicate_events_demoted = _demote_covariate_duplicate_events(raw)
        final_probe, _ = validate_temporal_dossier(
            raw, context_text=context, cutoff=self.timestamps[-1],
            future_timestamps=future_timestamps, history=self.values,
            compiler_model=self.forecaster.openrouter_model)
        # The bounded repair can add the grounded claim or event shape that
        # the initial pass omitted. Numeric normalization must therefore run
        # after repair as well; otherwise equivalent compiler representations
        # produce materially different forecasts.
        raw = normalize_deterministic_events(raw, final_probe)
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
        transform_failure_codes = {
            str(violation.get("code") or "")
            for failure in remaining_transform_failures
            for violation in failure.get("violations") or []
            if isinstance(violation, dict)
        }
        cited_semantic_constant_available = bool(re.search(
            r"\b(?:a\s+quarter|one\s+quarter|quarter|a\s+half|one\s+half|"
            r"half|twice|double|square|squared|quadratic|triple|cube|cubed|"
            r"cubic)\b", context, re.IGNORECASE))
        # Best-effort may expose a sealed model prior when deterministic
        # execution failed solely because the source omitted a needed
        # constant (for example, a named domain law without its exponent).
        # This is categorically different from contradicting an explicit
        # source quantity or failing timing/provenance/safety validation.
        prior_only_semantic_gap = bool(
            remaining_transform_failures
            and transform_failure_codes == {
                "UNENTAILED_TRANSFORMATION_CONSTANT"}
            and not cited_semantic_constant_available
        )
        candidate_blocked_by_transform = bool(
            remaining_transform_failures and raw.get("forecast_candidate")
            and raw.get("transformations") and not prior_only_semantic_gap)
        covariate_receipt = compilation["covariates"]
        covariate_rejections = compilation["covariate_rejections"]
        provisional_companion = _fit_governed_companion_from_receipt(
            covariate_receipt, context=context,
            history_timestamps=self.timestamps, history_values=self.values,
            future_timestamps=future_timestamps, claims=[])
        raw = _bind_covariate_row_claims(
            raw, covariate_receipt, future_timestamps,
            table_name=(provisional_companion or {}).get("source_table_name"))
        preliminary_dossier, _ = validate_temporal_dossier(
            raw, context_text=context, cutoff=self.timestamps[-1],
            future_timestamps=future_timestamps, history=self.values,
            history_timestamps=self.timestamps,
            compiler_model=self.forecaster.openrouter_model,
            validated_events=events)
        governed_categorical = None
        if categorical_schedule is not None:
            from gnomon.context_intelligence import (
                fit_categorical_state_candidate)
            governed_categorical = fit_categorical_state_candidate(
                self.values, categorical_schedule["history_states"],
                categorical_schedule["future_states"],
                primary=[{"timestamp": timestamp}
                         for timestamp in future_timestamps],
                claim_ids=[str(item["claim_id"]) for item in
                           preliminary_dossier.get("claims") or []],
                hypothesis_id=(
                    "host-verified-categorical-state:"
                    + str(categorical_schedule["name"])))
        governed_named_relationship = None
        named_future_driver = None
        if named_relationship_spec is not None:
            from gnomon.context_intelligence import (
                fit_companion_relationship_candidate)
            driver_name = str(named_relationship_spec["driver"])
            transition_time = datetime.fromisoformat(str(
                named_relationship_spec["transition_timestamp"]
            ).replace("Z", "+00:00"))
            transition_value = float(named_relationship_spec[
                "transition_value"])
            observed_driver = self.companion_histories[driver_name]
            named_future_driver = [
                transition_value if datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")) >= transition_time
                else float(observed_driver[-1])
                for timestamp in future_timestamps]
            try:
                governed_named_relationship = (
                    fit_companion_relationship_candidate(
                        self.values, observed_driver, named_future_driver,
                        primary=[{"timestamp": timestamp}
                                 for timestamp in future_timestamps],
                        claim_ids=[str(item["claim_id"]) for item in
                                   preliminary_dossier.get("claims") or []],
                        hypothesis_id=(
                            "host-fitted-named-relationship:" + driver_name)))
                governed_named_relationship["future_driver_assumption"] = {
                    "driver": driver_name,
                    "transition_timestamp": transition_time.isoformat(),
                    "transition_value": transition_value,
                    "shape": named_relationship_spec[
                        "future_path_assumption"],
                }
            except ValueError:
                governed_named_relationship = None
        governed_companion = _fit_governed_companion_from_receipt(
            covariate_receipt, context=context,
            history_timestamps=self.timestamps, history_values=self.values,
            future_timestamps=future_timestamps,
            claims=preliminary_dossier.get("claims") or [])
        if governed_companion and provisional_companion:
            provisional_validation = provisional_companion.get("validation") or {}
            governed_validation = dict(governed_companion.get("validation") or {})
            for key in ("candidate_tables", "multiplicity_adjusted_threshold"):
                if key in provisional_validation:
                    governed_validation[key] = provisional_validation[key]
            governed_validation["beats_baseline"] = bool(
                provisional_companion.get("selection_eligible"))
            governed_companion["validation"] = governed_validation
            governed_companion["selection_eligible"] = bool(
                provisional_companion.get("selection_eligible"))
        governed_candidate = (governed_categorical or governed_companion
                              or governed_named_relationship)
        relationship_prior_needed = bool(
            named_relationship_spec is not None
            and not (governed_named_relationship or {}).get(
                "selection_eligible")
            and self.forecaster.output_role in {
                "publication_best_effort", "llm_candidate_shadow"})
        if relationship_prior_needed and named_future_driver is not None:
            relationship_prompt = build_relationship_prior_prompt(
                context=context, target_name=self.target_name,
                driver_name=str(named_relationship_spec["driver"]))
            model_candidate_prompt_bytes = len(
                relationship_prompt.encode("utf-8"))
            try:
                responses = complete_many(
                    relationship_prompt,
                    "model_relationship_prior_samples", n=5)
                relationship_prior, model_candidate_sampling = (
                    candidate_from_relationship_prior_specs(
                        [response for response in responses if response.strip()],
                        target_history=self.values,
                        driver_history=self.companion_histories[str(
                            named_relationship_spec["driver"])],
                        future_driver=named_future_driver,
                        future_timestamps=future_timestamps,
                        claim_ids=[str(item["claim_id"]) for item in
                                   preliminary_dossier.get("claims") or []]))
                model_candidate_sampling["transport"] = {
                    "requested": 5,
                    "returned": sum(bool(item.strip()) for item in responses),
                    "failed": sum(not item.strip() for item in responses),
                }
                if relationship_prior is not None:
                    relationship_prior["validation"][
                        "historical_mapping_counterevidence"] = (
                            (governed_named_relationship or {}).get(
                                "validation"))
                    for hypothesis in raw.get("hypotheses") or []:
                        if hypothesis.get("kind") == "relationship":
                            hypothesis["rationale"] = (
                                "Repeated model elicitation supplied one "
                                "stable declarative relationship family. "
                                "Gnomon fitted its scale and path; the prior "
                                "has no historical skill authority, and the "
                                "failed/weak replay remains counterevidence.")
                    governed_candidate = relationship_prior
                    model_candidate_status = (
                        "sealed_declarative_relationship_prior")
                else:
                    model_candidate_status = (
                        "withheld_unstable_relationship_prior")
            except Exception as error:
                model_candidate_status = "relationship_prior_request_failed"
                compile_rejections.append(
                    f"model relationship prior failed: {error}")
        categorical_prior_needed = bool(
            categorical_schedule is not None
            and governed_categorical is not None
            and not governed_categorical.get("selection_eligible"))
        numeric_interpretation_hypotheses = [
            item for item in preliminary_dossier.get("hypotheses") or []
            if item.get("kind") in {
                "absolute_value", "bound", "additive_change",
                "multiplicative_change", "regime_shift", "relationship",
                "historical_analogue"}]
        qualitative_future_event_prior_needed = bool(
            categorical_schedule is None
            and model_candidate_proposal is None
            and governed_candidate is None
            and events
            and not raw.get("effect_proposal")
            and not raw.get("transformations")
            and preliminary_dossier.get("forecast_candidate") is None
            and any(
                str((event.attributes or {}).get(
                    "soft_context", {}).get("direction") or "unknown")
                in {"increase", "decrease"}
                and not str(event.event_type or "").startswith(
                    ("constraint:", "override:"))
                for event in events))
        interpretation_prior_needed = bool(
            categorical_schedule is None
            and named_relationship_spec is None
            and model_candidate_proposal is None
            and governed_candidate is None
            and not events
            and not raw.get("effect_proposal")
            and not raw.get("transformations")
            and preliminary_dossier.get("forecast_candidate") is None
            and numeric_interpretation_hypotheses
            and _has_material_numeric_context(context))
        if ((categorical_prior_needed or interpretation_prior_needed
             or qualitative_future_event_prior_needed)
                and self.forecaster.output_role in {
                    "publication_best_effort", "llm_candidate_shadow"}):
            model_candidate_status = (
                "requested_after_governed_rejection"
                if categorical_prior_needed
                else "requested_for_typed_future_event"
                if qualitative_future_event_prior_needed
                else "requested_for_typed_interpretation")
            context_prompt = build_sampled_context_prior_prompt(
                timestamps=self.timestamps,
                values=(self.companion_histories[
                    str(reference_power_spec["driver"])]
                    if reference_power_spec is not None else self.values),
                future_timestamps=future_timestamps,
                context=(
                    context + "\n\nForecast the future DRIVER series "
                    + str(reference_power_spec["driver"])
                    + ", not the target. Gnomon will apply the cited "
                      "reference power law to every returned driver value."
                    if reference_power_spec is not None else context))
            model_candidate_prompt_bytes = len(context_prompt.encode("utf-8"))
            def transform_sampled_path(path: list[float]) -> list[float]:
                if reference_power_spec is None:
                    return path
                input_reference = float(
                    reference_power_spec["input_reference"])
                output_reference = float(
                    reference_power_spec["output_reference"])
                exponent = int(reference_power_spec["exponent"])
                if input_reference <= 0 or output_reference <= 0:
                    raise ValueError("reference values must be positive")
                return [output_reference * (value / input_reference) ** exponent
                        for value in path]
            try:
                requested_paths = recommended_sample_count(
                    len(future_timestamps))
                initial_paths = (
                    requested_paths if (
                        deterministic_reference_power is not None
                        or deterministic_named_relationship is not None)
                    else recommended_initial_sample_count(
                        len(future_timestamps)))
                responses = complete_many(
                    context_prompt, "model_context_candidate_samples_initial",
                    n=initial_paths)
                transport_requested = initial_paths
                transport_failed = sum(not response.strip()
                                       for response in responses)
                proposed, model_candidate_sampling = candidate_from_sampled_paths(
                    [response for response in responses if response.strip()],
                    future_timestamps, history_values=self.values,
                    path_transform=transform_sampled_path)
                initial_sufficiency = sampled_prior_sufficiency(
                    model_candidate_sampling)
                expanded = False
                expansion_skipped_reason = None
                if (requested_paths > initial_paths
                        and not initial_sufficiency[
                            "eligible_for_human_recommendation"]
                        and compilation_deadline - time.monotonic() >= 5.0):
                    expanded_responses = complete_many(
                        context_prompt,
                        "model_context_candidate_samples_expansion",
                        n=requested_paths - initial_paths)
                    responses.extend(expanded_responses)
                    transport_requested += requested_paths - initial_paths
                    transport_failed += sum(
                        not response.strip() for response in expanded_responses)
                    proposed, model_candidate_sampling = (
                        candidate_from_sampled_paths(
                            [response for response in responses
                             if response.strip()], future_timestamps,
                            history_values=self.values,
                            path_transform=transform_sampled_path))
                    expanded = True
                elif (requested_paths > initial_paths
                      and not initial_sufficiency[
                          "eligible_for_human_recommendation"]):
                    expansion_skipped_reason = "workflow_deadline_exhausted"
                final_sufficiency = sampled_prior_sufficiency(
                    model_candidate_sampling)
                model_candidate_sampling["transport"] = {
                    "requested": transport_requested,
                    "failed": transport_failed,
                    "returned": transport_requested - transport_failed,
                    "interpretation": (
                        "transport availability is reported separately; "
                        "semantic validity is measured over returned paths"),
                }
                if reference_power_spec is not None:
                    model_candidate_sampling["governed_transformation"] = {
                        "kind": "reference_normalized_power_law",
                        **reference_power_spec,
                        "model_authored_quantity": "future_driver_path",
                        "engine_authored_quantity": "target_forecast_path",
                    }
                model_candidate_sampling["sufficiency"] = final_sufficiency
                model_candidate_sampling["adaptive_sampling"] = {
                    "initial_requested": initial_paths,
                    "maximum_requested": requested_paths,
                    "expanded": expanded,
                    "expansion_skipped_reason": expansion_skipped_reason,
                    "expansion_reason_codes": (
                        initial_sufficiency["reason_codes"] if expanded else []),
                    "stopped_early": not expanded,
                    "interpretation": (
                        "additional paths are requested only when the initial "
                        "elicitation is malformed or materially incoherent"),
                }
                if isinstance(proposed, dict):
                    model_candidate_sample_paths = proposed.pop(
                        "_validated_sample_paths", None)
                    model_candidate_proposal = proposed
                    model_candidate_status = (
                        "sampled_paths_proposed_after_governed_rejection"
                        if categorical_prior_needed
                        else "sampled_paths_proposed_for_typed_future_event"
                        if qualitative_future_event_prior_needed
                        else "sampled_paths_proposed_for_typed_interpretation")
                else:
                    model_candidate_status = (
                        "withheld_after_governed_rejection"
                        if categorical_prior_needed
                        else "withheld_for_typed_future_event"
                        if qualitative_future_event_prior_needed
                        else "withheld_for_typed_interpretation")
                    compile_rejections.append(
                        "model context candidate returned no valid sampled path")
            except Exception as error:
                model_candidate_status = (
                    "request_failed_after_governed_rejection"
                    if categorical_prior_needed
                    else "request_failed_for_typed_future_event"
                    if qualitative_future_event_prior_needed
                    else "request_failed_for_typed_interpretation")
                compile_rejections.append(
                    f"model context candidate failed: {error}")
        dossier, dossier_rejections = validate_temporal_dossier(
            raw, context_text=context, cutoff=self.timestamps[-1],
            future_timestamps=future_timestamps, history=self.values,
            history_timestamps=self.timestamps,
            compiler_model=self.forecaster.openrouter_model,
            validated_events=events,
            governed_candidate=governed_candidate,
            candidate_selection_eligible=not candidate_blocked_by_transform,
            candidate_selection_reason=(
                "Accompanying governed transformation failed preflight; the "
                "sealed model path remains visible as a scenario but cannot "
                "become the default recommendation."
                if candidate_blocked_by_transform else
                "The cited source omitted a constant required for deterministic "
                "execution. Best-effort may rank the separately sealed model "
                "prior for human review; it remains unsupported for automation."
                if prior_only_semantic_gap else None),
        )
        dossiers = [dossier]
        if model_candidate_proposal is not None:
            model_raw = {**raw,
                         "forecast_candidate": model_candidate_proposal}
            model_dossier, model_dossier_rejections = (
                validate_temporal_dossier(
                    model_raw, context_text=context,
                    cutoff=self.timestamps[-1],
                    future_timestamps=future_timestamps,
                    history=self.values,
                    history_timestamps=self.timestamps,
                    compiler_model=self.forecaster.openrouter_model,
                    validated_events=events))
            if model_dossier.get("forecast_candidate") is not None:
                if model_candidate_sampling is not None:
                    from gnomon.llm_dossier import (
                        attach_host_candidate_elicitation)
                    model_dossier = attach_host_candidate_elicitation(
                        model_dossier,
                        requested_paths=int(model_candidate_sampling[
                            "requested"]),
                        accepted_paths=int(model_candidate_sampling[
                            "accepted"]),
                        aggregation=str(model_candidate_sampling[
                            "aggregation"]), temperature=1.0,
                        stability=model_candidate_sampling.get("stability"),
                        request_mode=str(model_candidate_sampling[
                            "request_mode"]),
                        sample_paths=model_candidate_sample_paths)
                dossiers.append(model_dossier)
                model_candidate_status = "accepted"
            else:
                model_candidate_status = "rejected"
            dossier_rejections = [
                *dossier_rejections,
                *(f"model_candidate:{item}"
                  for item in model_dossier_rejections),
            ]
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
        deterministic_front_door = bool(
            deterministic_companion_tables or categorical_schedule
            or deterministic_ended_disruption is not None
            or deterministic_reference_power is not None
            or deterministic_named_relationship is not None
            or deterministic_calibration_claim is not None
            or deterministic_zero_window is not None
            or deterministic_multiplier is not None
            or deterministic_directional_event is not None)
        payload = {
            "schema_version": 1,
            "compiler": {
                "kind": ("deterministic_parse_plus_sealed_model_candidate"
                         if deterministic_front_door
                         and model_candidate_prompt_bytes else
                         "deterministic_structured_parse"
                         if deterministic_front_door
                         else "llm_proposes_gnomon_validates"),
                "deterministic_front_door": deterministic_front_door,
                "model": self.forecaster.openrouter_model,
                "contract": ("explicit_lag_relationship"
                             if relationship_contract else
                             "historical_observation_semantics"
                             if observation_contract else
                             "categorical_state_schedule"
                             if categorical_schedule else
                             "structured_companion_paths"
                             if companion_contract else
                             "ended_recurring_disruption_hypothesis"
                             if deterministic_ended_disruption is not None else
                             "explicit_reference_power_relationship"
                             if deterministic_reference_power is not None else
                             "named_driver_relationship_prior"
                             if deterministic_named_relationship is not None else
                             "explicit_additive_measurement_drift"
                             if deterministic_calibration_claim is not None else
                             "explicit_dated_zero_window"
                             if deterministic_zero_window is not None else
                             "explicit_dated_multiplier"
                             if deterministic_multiplier is not None else
                             "explicit_dated_directional_event"
                             if deterministic_directional_event is not None else
                             "universal_dossier"),
                "prompt_bytes": (model_candidate_prompt_bytes
                                 if deterministic_front_door
                                 else len(prompt.encode("utf-8"))),
                "workflow_budget_seconds": MAX_CONTEXT_COMPILATION_SECONDS,
                "elapsed_seconds": round(
                    time.monotonic() - compilation_started, 6),
                "calls": compiler_calls,
                "repair_decisions": repair_decisions,
                "representation_normalizations": ({
                    "covariate_duplicate_events_demoted":
                        duplicate_events_demoted,
                    "rule": "companion_covariate_precedes_target_override",
                } if duplicate_events_demoted else {}),
                "model_candidate_status": model_candidate_status,
                "model_candidate_sampling": model_candidate_sampling,
            },
            "source": {
                "kind": "benchmark_task_context",
                "sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            },
            "events": [event_to_dict(event) for event in events],
            "context_receipt_id": compilation["receipt_id"],
            "hypotheses": compilation["hypotheses"],
            "dossier": dossier,
            "dossiers": dossiers,
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
        seed = getattr(
            self.forecaster, "benchmark_seed", getattr(self.task, "seed", "x"))
        compiled_dossiers = (
            self.context_compilation.get("dossiers") or
            [self.context_compilation.get("dossier") or {}]
            if self.context_compilation is not None else [])
        hypothesis_rejections = sum(len(
            (item.get("hypothesis_critique") or {}).get("rejected") or [])
            for item in compiled_dossiers)
        candidate_rejections = sum(
            (item.get("candidate_critique") or {}).get("status") == "rejected"
            for item in compiled_dossiers)
        top_level_rejections = len(
            (self.context_compilation or {}).get("rejections") or [])
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
                    "candidate_available": any(
                        item.get("effect_proposal") or item.get(
                            "forecast_candidate") for item in compiled_dossiers),
                    "dossier_count": len(
                        self.context_compilation.get("dossiers") or
                        [self.context_compilation["dossier"]]),
                    "candidate_origins": [
                        str((item.get("candidate_critique") or {}).get(
                            "candidate_origin") or "none")
                        for item in (self.context_compilation.get("dossiers") or
                                     [self.context_compilation["dossier"]])
                        if item.get("forecast_candidate")],
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
                    "rejection_count": (
                        top_level_rejections + hypothesis_rejections
                        + candidate_rejections),
                    "top_level_rejection_count": top_level_rejections,
                    "hypothesis_rejection_count": hypothesis_rejections,
                    "candidate_rejection_count": candidate_rejections,
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
            wire_rejections, omitted_rejections = _bounded_context_rejections(
                receipt.get("rejections") or [])
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
                    "temporal_dossiers": (
                        receipt.get("dossiers") or
                        [receipt.get("dossier") or {}]),
                    "context_submission": {
                        "known_at": self.timestamps[-1],
                        "transformations": receipt.get("transformations") or [],
                        "rejections": wire_rejections,
                    },
                })
            entry["host_context_binding"] = {
                "receipt_sha256": (receipt.get("source") or {}).get("sha256"),
                "events": len(receipt.get("events") or []),
                "covariate_tables_proposed": len(
                    (receipt.get("covariates") or {}).get("tables") or []),
                "covariate_table_bound": bool(covariate_arguments),
                "rejections": len(receipt.get("rejections") or []),
                "wire_rejections": len(wire_rejections),
                "wire_rejections_omitted": omitted_rejections,
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
                        "context_summary": publication.get("context_summary"),
                        "context_dispositions": [{
                            "context_id": item.get("context_id"),
                            "disposition": item.get("disposition"),
                            "reason_code": item.get("reason_code"),
                            "reason": item.get("reason"),
                            "selection_reason_code": item.get(
                                "selection_reason_code"),
                            "selection_reason": item.get("selection_reason"),
                            "recovery_action": ({
                                key: (item.get("recovery_action") or {}).get(key)
                                for key in ("code", "message",
                                            "required_evidence",
                                            "automation_eligible", "scope",
                                            "required_for_current_recommendation")
                            } if item.get("recovery_action") else None),
                        } for item in publication.get(
                            "context_dispositions") or []],
                        "candidates": [{
                            "scenario_id": item.get("scenario_id"),
                            "role": item.get("role"),
                            "support": item.get("support"),
                            "selection_eligible": item.get("selection_eligible"),
                            "human_selection_eligible": item.get(
                                "human_selection_eligible"),
                            "elicitation_sufficiency": (
                                item.get("effect") or {}).get(
                                    "elicitation_sufficiency"),
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
            dossiers = [item for item in (
                self.context_compilation.get("dossiers") or [dossier])
                if isinstance(item, dict)]
            top_level_rejections = len(
                self.context_compilation.get("rejections") or [])
            hypothesis_rejections = sum(len(
                (item.get("hypothesis_critique") or {}).get("rejected") or [])
                for item in dossiers)
            candidate_rejections = sum(
                (item.get("candidate_critique") or {}).get("status") ==
                "rejected" for item in dossiers)
            model_dossier = next((
                item for item in reversed(dossiers)
                if (item.get("candidate_critique") or {}).get(
                    "candidate_origin") == "model_authored"
                and item.get("forecast_candidate")), None)
            extra_info["context_compilation"] = {
                "receipt_path": self.context_compilation["path"],
                "source_sha256": self.context_compilation["source"]["sha256"],
                "event_count": len(self.context_compilation["events"]),
                "claim_count": len(
                    self.context_compilation["dossier"]["claims"]),
                "hypothesis_count": len(dossier.get("hypotheses") or []),
                "hypothesis_status": (dossier.get("hypothesis_critique") or {}).get(
                    "status"),
                "candidate_available": any(
                    item.get("effect_proposal") or item.get(
                        "forecast_candidate") for item in dossiers),
                "dossier_count": len(dossiers),
                "candidate_origins": [
                    (item.get("candidate_critique") or {}).get(
                        "candidate_origin") for item in dossiers
                    if item.get("forecast_candidate")],
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
                "rejection_count": (
                    top_level_rejections + hypothesis_rejections
                    + candidate_rejections),
                "top_level_rejection_count": top_level_rejections,
                "hypothesis_rejection_count": hypothesis_rejections,
                "candidate_rejection_count": candidate_rejections,
                "future_observations_exposed": False,
            }
            shadow_dossier = model_dossier or dossier
            if (shadow_dossier.get("forecast_candidate")
                    or shadow_dossier.get("effect_proposal")):
                # Retained for matched shadow scoring against this exact
                # compiler generation. It is never sent back into the agent
                # conversation and never replaces the canonical submission.
                extra_info["llm_candidate_shadow"] = {
                    "support": shadow_dossier["candidate_support"],
                    "seal_sha256": shadow_dossier["seal_sha256"],
                    "forecast_candidate": shadow_dossier["forecast_candidate"],
                    "effect_proposal": shadow_dossier.get("effect_proposal"),
                    "candidate_origin": (
                        shadow_dossier.get("candidate_critique") or {}).get(
                            "candidate_origin"),
                    "automation_eligible": False,
                    "primary_forecast_unchanged": True,
                }
        if self.submission["route"] == "gnomon":
            extra_info["artifact_path"] = self.submission["artifact_path"]
            extra_info["support"] = getattr(self, "_submitted_support", None)
            extra_info["selected_model"] = getattr(self, "_submitted_model", None)
        return self.submission["rows"], extra_info
