# Agent Surface Redesign — Design Plan

Status: proposed (2026-08-11). Not yet implemented.

## Why

The 2026-08 MCP evaluation (`gnomon-mcp-assessment.md`) measured three failure
modes in the agent experience, all interaction economics rather than engine
quality:

- **Cost.** 250–500K tokens per task, 6 tool calls where 1 would do,
  round trips burned on jail-path errors between calls.
- **Yield.** ~486 of ~490 forecast channels abstained or degraded — the agent
  got nothing quotable to work with.
- **Fit.** Most of the benchmark's temporal questions were descriptive
  (what changed, is this normal, compare periods), and every question paid
  the full backtest toll.

The one bright spot: the evidence-injection arm (compute facts once, hand
them to the model) *beat* the raw-LLM control on T1. The harness helps when
it hands the agent compact computed facts; it hurts when the agent must
drive an 18-tool workflow.

Design principle for everything below: **a tool is useful to an agent when
calling it is cheaper and more reliable than not calling it.** Every removed
round trip is removed tokens, removed error surface, and one fewer chance
for the agent to wander.

## Part 1 — The redesigned surface

### Target default profile: 7 tools

| Tool | Change |
| --- | --- |
| `gnomon_describe` | **New.** Cheap descriptive temporal facts; always answers. See Part 2. |
| `gnomon_forecast` | One-shot (see below). |
| `gnomon_investigate_change` | One-shot. |
| `gnomon_detect_anomalies` | One-shot. |
| `gnomon_monitor` | One-shot. |
| `gnomon_get_artifact` | Kept; gains field selectors so detail is fetched, never inlined. |
| `gnomon_capabilities` | Kept; still the source of truth. |

Moved to the `full` profile (not deleted — demoted): `gnomon_decide`,
`gnomon_route`, `gnomon_inspect`, `gnomon_ingest`, `gnomon_list_datasets`,
the tracking suite (`submit_actuals`, `status`, `resolve_outcome`,
`list_open_forecasts`, `model_performance`), the covariate suite
(`validate_covariates`, `covariate_guide`, `propose_covariates`,
`proposer_skill`, `preflight_context`), `explain_run`, `get_run`,
`install_tsfm`. The planner and v0.2 compat tools stay behind their
existing env gates.

Rationale: `decide` asks agents for utility numbers they rarely have;
tracking is a longitudinal workflow no host has exhibited yet; ingestion
and inspection are plumbing the verbs can do internally. Demotion keeps
every workflow reachable for the operators who opt in, while the default
surface an agent sees is small enough to hold in one system-prompt
description.

### One-shot verb semantics

Each verb becomes self-contained: it accepts a data reference and the
question, and internally runs schema inference → regrid → repair →
execution, with every step disclosed in the response exactly as today.
No prior `ingest`/`inspect` call required.

- **Inputs:** `path` (file) *or* `data` (inline rows — small series pasted
  by the agent are first-class; today they force a tempfile dance).
  Optional: `time_column`/`target_column`/`series_column` (inference
  otherwise), `horizon`, `as_of`, `regrid`, `repair`.
- **Output ownership:** Gnomon owns artifact placement. No `output_dir`
  argument on the default surface — the class of jail-violation errors
  observed in the evaluation is removed by removing the parameter.
- **Failure posture:** an error response carries machine-readable repair
  options (unchanged) **and**, whenever one exists, a degraded answer at
  whatever tier the evidence supports. Abstention-with-nothing is reserved
  for unreadable data. The tier system already makes this honest; the
  change is defaults, not epistemics: the MCP default becomes
  `best_effort` publication with unstrippable tiers, and
  `minimum_support: supported` remains the opt-in strict mode.

### Response contract (all verbs)

Every response has the same shape, in this order:

1. `headline`: 2–5 deterministic sentences with numbers, tiers, and
   caveats inline — written to be quoted verbatim (Part 3).
2. `key_numbers`: a flat dict of the ~dozen values an agent most often
   needs (point at threshold date, interval at final step, improvement,
   tier counts). Exact values, not rounded.
3. `support` / `tier` summary and typed caveats.
4. `artifact_id` + `next_actions` (existing repair-options machinery).

Hard budget: **≤ ~1,200 tokens per verb response, ≤ ~600 for
`describe`.** Full per-step arrays never appear inline; they live in the
artifact and are fetched by field selector via `gnomon_get_artifact`
(e.g. `fields: ["forecast.q10", "forecast.q90"]`, `steps: "1..14"`).
`RESPONSE_BUDGET_BYTES` exists today; this makes the budget a designed
contract instead of a truncation backstop.

### Phases

Ordering note: Phase 0 is a prerequisite — measuring an improved surface
around a miscalibrated engine would launder the bugs into the baseline.

- **Phase 0 — engine honesty.** Fix the five confirmed bugs from the
  2026-08 code review (published-ensemble mismatch `pipeline.py:431`,
  fold desync `evaluation.py:795`, adaptation `known_time` stamp
  `tracking.py:819`, `voting_forecast` bias `ensemble.py:142`, NaN/Inf
  through the strict loader `data.py:355`). Add the seeded Monte Carlo
  interval-calibration harness (~50 synthetic series, assert empirical
  q10–q90 coverage within Monte Carlo bounds) and un-skip the two
  silently-skipping leakage-guard tests.
  *Done when:* calibration harness green; the two guards execute their
  target branches.
- **Phase 1 — response contract.** Headline generator (Part 3) +
  `key_numbers` + token budget on the five existing verbs; artifact
  field selectors on `gnomon_get_artifact`.
  *Done when:* every verb response ≤ budget on the evaluation corpus;
  golden headline tests pass; the numbers-round-trip property test
  (Part 3) passes.
- **Phase 2 — one-shot folding.** Inline `data` input, internal
  inference/repair, remove `output_dir` from the default surface,
  degraded-answer-on-error posture, MCP default `best_effort`.
  *Done when:* the evaluation's T2 workflow completes in ≤ 2 tool calls
  per question; supported-or-tiered yield ≥ 80% on the messy-CSV corpus.
- **Phase 3 — `gnomon_describe`.** Part 2.
  *Done when:* p95 latency < 1s on 25K-row series; response ≤ 600
  tokens; answers on 100% of readable inputs.
- **Phase 4 — profile re-cut.** Default profile to the 7 tools;
  everything else to `full`; capabilities-reachability contract test
  updated (it already pins tool-surface consistency).
- **Phase 5 — measure.** Re-run the TemporalBench MCP condition and the
  leaktrap suite. Metrics that decide success, per task: tokens, tool
  calls, yield, wall-clock, end-task accuracy vs both the raw control
  and the evidence-injection arm.
  *Done when:* Gnomon-MCP ≥ control on accuracy at < 50K tokens per
  task, with leaktrap still 0/40. Until then, no new tools.

## Part 2 — `gnomon_describe`

The verb an agent calls ten times a session. Descriptive temporal facts,
computed exactly, sub-second, no backtest folds, and it always answers on
readable data. Most agent questions over a series are not forecasts —
they are "what changed", "is this normal", "compare this month to last".
Today those questions pay the full evaluation toll or go unanswered.

### Contract

```
gnomon_describe(path | data,
                time_column?, target_column?, series_column?,
                window?,          # e.g. "last 30" | ISO range; default: full series
                compare?,         # "previous_period" | "same_period_last_cycle" | ISO range
                as_of?, regrid?, repair?)
```

### Output blocks (each present only when computable; caveats otherwise)

- **grid** — frequency, span, count, gaps (count, longest run, fraction),
  timezone/DST notes, repair summary if repair ran.
- **level** — latest value and timestamp; window median/IQR; latest value's
  robust z-score and percentile against the baseline window.
- **trend** — direction, robust slope per period, and since-when (anchored
  at the most recent changepoint, not the series start).
- **seasonality** — detected period(s), strength, peak/trough phase in
  calendar terms ("weekly; peaks Mondays, troughs Saturdays").
- **changepoints** — top-k with dates, magnitude, and kind (level/trend),
  from the existing changepoint operator.
- **outliers** — robust-z flags with dates; whether the latest point is one.
- **extremes & runs** — min/max with dates; current streak and whether it
  is unusual for the series ("7 consecutive rises; longest in the series").
- **comparison** — window vs the `compare` window: Δ level, Δ%, and
  whether the change clears the series' own noise floor (robust scale),
  stated as "beyond/within typical variation" — never a probability.

### Implementation notes

- Reuses what exists: the changepoint and robust-score operators
  (`operators.py`), season detection (`temporal.py`), repair and regrid,
  the snapshot layer for `as_of` (describe honours vintages like every
  other verb — "describe this series as it was known on June 1" falls
  out for free, and it is a capability nothing else in an agent's toolbox
  has).
- Two known engine fixes land with it, since describe makes them visible:
  season detection must take the *strongest* ACF peak, not the first
  above threshold (`temporal.py:147`), and the near-constant-series
  robust-scale floor must not manufacture ±1e12 z-scores
  (`operators.py:43`).
- **Epistemics are trivial by design.** Every number is an exact
  computation over served observations — descriptive claims in the
  existing verifier taxonomy. No selection, no calibration, no
  probability, and therefore nothing to abstain over. The only tiering
  is data-quality caveats (gappy grid, repaired values, short baseline).
  Causal language is structurally absent: `compare` reports differences,
  never explanations.
- Cost ceiling: one pass over the series plus the changepoint scan
  (linear after the prefix-sum fix). Cache by content fingerprint so
  repeated describes of the same file are free.
- Lightweight lineage: one evidence record (the computation over the
  snapshot), claims attached — so describe output is quotable under the
  same verifier contract as everything else, at ~none of the cost.

### What it is not

No forecast, no interval, no probability, no cause. The moment a question
needs "what happens next", the response's `next_actions` points at
`gnomon_forecast` — escalation is the agent's one-line decision, not five
exploratory calls.

## Part 3 — Headline sentences without an LLM

The requirement: every response leads with sentences an agent can quote
verbatim. Generating them must not involve a model — the whole point is
that prose containing numbers is produced by the same deterministic
process that produced the numbers.

This is template NLG — the oldest, most boring kind, and exactly right
here (it is how weather bulletins and earthquake reports have been
machine-written for decades). There is no generation in the ML sense, so
there is no hallucination surface.

### Mechanics

1. **Sentence specs.** A fixed library of parameterized templates, each:
   a guard predicate over artifact fields, a template string with typed
   slots, and slot bindings (artifact field + formatter). Per verb, specs
   are evaluated in a fixed order; the first whose guard matches emits.
   Selection is deterministic branching — the same artifact always yields
   byte-identical headlines (they join the golden-artifact discipline).

   ```python
   SentenceSpec(
     guard=lambda a: a.threshold and a.first_crossing_step is not None,
     template="{series} is forecast to cross {threshold} around {date}"
              " ({p:.0%} peak exceedance probability, {tier} tier).",
     bindings={"series": F("series"), "threshold": F("threshold.value", fmt=num),
               "date": F("first_crossing_date", fmt=day),
               "p": F("threshold.peak_probability"), "tier": F("support")},
   )
   ```

2. **Formatters, centralized.** One module owns number formatting
   (3 significant figures in prose; exact values stay adjacent in
   `key_numbers`), humanized dates, percent deltas, and unit handling.
   Rounding policy is stated once and tested once.

3. **Word choice from binned magnitudes.** Direction and intensity verbs
   come from one documented threshold table over effect size in robust
   units — e.g. |Δ|/scale < 0.25 → "held roughly flat", < 1 → "edged
   up/down", < 3 → "rose/fell", else "jumped/dropped". No synonym
   sampling, no free text. The table is a contract: changing a threshold
   is a reviewed diff, not a vibe.

4. **Grammar helpers only.** Pluralization, list joining, a/an. Nothing
   open-ended.

5. **Verified by construction, then checked anyway.** The generator emits
   `(sentence, claim)` pairs: each headline is registered as a claim in
   the existing lineage, citing the evidence for the fields it used — so
   the verifier's existing checks (tier labelling, calibration gating,
   no-causal-language) apply to Gnomon's own prose with zero new
   machinery. Two test layers on top:
   - **Golden headlines** per fixture (joins the byte-stability policy).
   - **A numbers-round-trip property test:** regex-extract every numeral
     from every generated sentence across the corpus and assert each one
     equals a cited artifact field under the stated rounding. This is the
     "no invented numbers" guarantee applied to our own prose — the
     failure mode where a template binds the wrong field becomes a CI
     failure, not a shipped lie.

6. **Tier and caveat are part of the sentence, not metadata.** A
   best-effort row's headline says so inline ("naive extrapolation
   beyond day 9, best_effort tier") because quoted metadata is metadata
   lost. This is the same rule the verifier already enforces on claims
   (`SUB_SUPPORTED_UNLABELLED`) — the generator satisfies it by
   construction instead of by rejection.

### Example (forecast, mixed tiers)

> requests is forecast to rise 12% over the next 14 days (evaluated tier
> through day 9; naive extrapolation beyond, best_effort tier). ETS beat
> the strongest baseline by 23% across 4 backtest folds. The q10–q90 band
> at day 14 spans 310–410; measured interval coverage on the held-out
> fold was 78%. Crossing 340 first becomes more likely than not around
> Aug 21.

Four sentences, ~90 tokens, every number traceable, tier inline, nothing
for the agent to compose. The agent's best move — and its laziest move —
is to copy. That alignment of laziness with honesty is the design.
