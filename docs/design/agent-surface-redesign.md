# Agent Surface Redesign — Design Plan

Status: proposed (2026-08-11), revision 2. Not yet implemented.
Revision log at the end.

## The goal, stated measurably

Make Gnomon a tool an agent is *better off calling* on temporal questions.
Operationally: on an agent evaluation with a raw-LLM control,

- **accuracy** at or above the control on tasks where the control cannot
  win by leaking (see Phase 5 on why this qualifier is load-bearing);
- **cost** under ~50K tokens per task, ≤ 2 tool calls per question;
- **yield**: a quotable, tiered answer on ≥ 80% of readable inputs;
- **safety**: leaktrap stays 0/40.

Everything below is judged against those four numbers.

## Why redesign

The 2026-08 MCP evaluation (`gnomon-mcp-assessment.md`) measured three
failure modes, all interaction economics rather than engine quality:

- **Cost.** 250–500K tokens per task, 6 tool calls where 1 would do,
  round trips burned on jail-path errors between calls.
- **Yield.** ~486 of ~490 forecast channels abstained or degraded — the
  agent got nothing quotable to work with.
- **Fit.** Most temporal questions agents ask are descriptive (what
  changed, is this normal, compare periods), and every question paid the
  full backtest toll.

The evidence-injection arm — compute facts once, hand them over — *beat*
the control on T1 (48.9% vs 43.4%). The harness helps when it hands the
agent compact computed facts; it hurts when the agent must drive an
18-tool workflow.

Design principle: **a tool is useful to an agent when calling it is
cheaper and more reliable than not calling it.** Every removed round trip
is removed tokens, removed error surface, and one fewer chance to wander.

## Part 1 — The redesigned surface

### Target default profile: 7 tools

| Tool | Change |
| --- | --- |
| `gnomon_describe` | **New.** Descriptive temporal facts; always answers; the designated first call. Part 2. |
| `gnomon_forecast` | One-shot (below). |
| `gnomon_investigate_change` | One-shot. |
| `gnomon_detect_anomalies` | One-shot. |
| `gnomon_monitor` | One-shot. |
| `gnomon_get_artifact` | Kept; gains field/series selectors so detail is fetched, never inlined. |
| `gnomon_capabilities` | Kept; still the source of truth. |

Moved to the `full` profile (demoted, not deleted): `gnomon_decide`,
`gnomon_route`, `gnomon_inspect`, `gnomon_ingest`, `gnomon_list_datasets`,
the tracking suite (`submit_actuals`, `status`, `resolve_outcome`,
`list_open_forecasts`, `model_performance`), the covariate suite
(`validate_covariates`, `covariate_guide`, `propose_covariates`,
`proposer_skill`, `preflight_context`), `explain_run`, `get_run`,
`install_tsfm`. Planner and v0.2 compat stay behind their env gates.

Rationale: `decide` asks agents for utility numbers they rarely have;
tracking is a longitudinal workflow no host has exhibited yet; ingestion
and inspection are plumbing the verbs can do internally. Demotion keeps
every workflow reachable for operators who opt in.

### The schema tax (why 7 is not just tidier — it is cheaper every turn)

MCP hosts inject every visible tool's JSON schema into the model's
context on every turn. The default surface is therefore a *per-turn*
token tax, paid whether or not Gnomon is called. This makes surface size
a first-order cost, and it sets a budget the toolspec must meet:

- **Total default-surface schema budget: ≤ ~2,500 tokens.** Measured in
  CI (a contract test serializes the default surface and counts tokens
  with a reference tokenizer; the number is pinned with slack, like the
  response budget).
- Descriptions are terse decision rules, not documentation. Prose,
  examples, and workflow guidance live in one place: the
  `gnomon_capabilities` response and the docs. Each tool description
  states *when to call it* in ≤ 2 sentences.
- `gnomon_describe`'s description carries the triage rule: "When unsure
  which Gnomon tool a question needs, call describe first; its
  `suggested_next` names the verb." This replaces `gnomon_route` on the
  default surface with a rule that costs one sentence.

### One-shot verb semantics

Each verb is self-contained: it accepts a data reference and the
question, and internally runs schema inference → regrid → repair →
execution, with every step disclosed in the response exactly as today.
No prior `ingest`/`inspect` call required.

- **Inputs:** `path` (file) *or* `data` (inline rows). Inline data is
  first-class for small series (cap ~2,000 rows; beyond that the
  response instructs writing a file and passing `path`) and is stamped
  `known_time_assumed` like any un-vintaged CSV. Hosts without a
  filesystem (remote MCP) work within the inline cap. Optional:
  `time_column` / `target_column` / `series_column` (inference
  otherwise), `horizon`, `as_of`, `regrid`, `repair`.
- **Output ownership:** Gnomon owns artifact placement. No `output_dir`
  argument on the default surface — the class of jail-violation errors
  observed in the evaluation is removed by removing the parameter.
- **Failure posture:** an error response carries machine-readable repair
  options (unchanged) **and**, whenever one exists, a degraded answer at
  whatever tier the evidence supports. Abstention-with-nothing is
  reserved for unreadable data. The MCP default becomes `best_effort`
  publication with unstrippable tiers; `minimum_support: supported`
  remains the opt-in strict mode. The change is defaults, not
  epistemics.

### Response contract (all verbs)

Every response, in order:

1. `headline`: 2–5 deterministic sentences with numbers, tiers, and
   caveats inline — written to be quoted verbatim (Part 3).
2. `key_numbers`: a flat dict of the ~dozen values an agent most often
   needs (point at threshold date, interval at final step, improvement,
   tier counts). Exact values, not rounded. Key names are a contract,
   pinned by test — agents write extraction code against them.
3. `support` / `tier` summary and typed caveats.
4. `artifact_id` + `next_actions` (existing repair-options machinery),
   plus `suggested_next` where escalation is obvious.

Hard budget: **≤ ~1,200 tokens per verb response, ≤ ~600 for
`describe`** — for any input, including multi-series (next section).
Full per-step arrays never appear inline; they live in the artifact and
are fetched by selector via `gnomon_get_artifact`
(e.g. `fields: ["forecast.q10","forecast.q90"]`, `series: "checkouts"`,
`steps: "1..14"`). `RESPONSE_BUDGET_BYTES` exists today; this makes the
budget a designed contract instead of a truncation backstop.

### Multi-series response shaping

The evaluation's hardest task had ~490 channels; no flat per-series
response fits any budget. For an input with more than a handful of
series, every verb responds with a **triage view**, not a list:

1. An aggregate headline: totals by tier and by outcome
   ("Forecast 487 of 490 series; 3 abstained (all: history shorter than
   one season). 41 series carry best_effort rows beyond day 9.").
2. **Top-k notable series** (default k=5), each with a one-line
   headline. Notability is deterministic and verb-appropriate: largest
   forecast change vs history, most recent changepoint, highest anomaly
   score, first threshold crossing — with the ranking rule named in the
   response so the agent knows what "notable" meant.
3. A grouped remainder: counts per tier / per abstention reason, never
   per-series prose.
4. The full per-series table lives in the artifact; `gnomon_get_artifact`
   accepts `series` selectors and simple filters
   (`where: {tier: "best_effort"}`, `order_by: "notability"`,
   `limit: 20`) so the agent pages through exactly what it needs.

This turns the 490-channel case from 490 abstention notices into one
readable paragraph plus targeted drill-down — the difference between the
evaluation's worst transcript and a usable one.

### Phases

Ordering note: Phase 0 is a prerequisite — measuring an improved surface
around a miscalibrated engine would launder the bugs into the baseline.

- **Phase 0 — engine honesty.** Fix the five confirmed bugs from the
  2026-08 code review (published-ensemble mismatch `pipeline.py:431`,
  fold desync `evaluation.py:795`, adaptation `known_time` stamp
  `tracking.py:819`, `voting_forecast` bias `ensemble.py:142`, NaN/Inf
  through the strict loader `data.py:355`). Add the seeded Monte Carlo
  interval-calibration harness (~50 synthetic series; assert empirical
  q10–q90 coverage within Monte Carlo bounds) and un-skip the two
  silently-skipping leakage-guard tests.
  *Done when:* calibration harness green; both guards execute their
  target branches.
- **Phase 1 — response contract.** Headline generator (Part 3) +
  `key_numbers` + token budget on the five existing verbs; artifact
  field/series selectors; multi-series triage shaping.
  *Done when:* every verb response ≤ budget on the evaluation corpus
  *including the 490-channel task*; golden headline tests pass; the
  numbers-round-trip and template-coverage tests (Part 3) pass;
  `key_numbers` keys pinned.
- **Phase 2 — one-shot folding.** Inline `data`, internal
  inference/repair, remove `output_dir` from the default surface,
  degraded-answer-on-error posture, MCP default `best_effort`.
  *Done when:* the evaluation's T2 workflow completes in ≤ 2 tool calls
  per question; supported-or-tiered yield ≥ 80% on the messy-CSV corpus.
- **Phase 3 — `gnomon_describe`.** Part 2.
  *Done when:* p95 latency < 1s on 25K-row series; response ≤ 600
  tokens; answers on 100% of readable inputs; season-peak and
  constant-series fixes landed.
- **Phase 4 — profile re-cut.** Default profile to the 7 tools; schema
  budget test in CI; capabilities-reachability contract test updated.
- **Phase 5 — measure.** Re-run the TemporalBench MCP condition and
  leaktrap. Per task: tokens (including the per-turn schema tax), tool
  calls, yield, wall-clock, accuracy.

  **The control comparison must be leakage-aware.** The control's
  0.46% forecast SMAPE and the leaktrap finding (leaking is worth ~78%
  of headroom) both say the same thing: on public benchmarks the
  control can win by reading answers from pretraining or the prompt,
  and "beat the control on accuracy" is then unwinnable *by design* —
  ours. So the accuracy criterion is scored on a leakage-controlled
  subset: tasks whose targets postdate the model's cutoff, are
  synthetic, or are perturbed vintages (the leaktrap generator already
  builds these). On the leakage-possible remainder we report accuracy
  alongside the control's leak-flag rate rather than treating it as a
  fair race.

  *Done when:* on the leakage-controlled subset, Gnomon-MCP ≥ control
  **and ≥ the evidence-injection arm** (the cheaper in-house competitor
  — if injection still wins, the tool surface is not paying for
  itself) at < 50K tokens per task, yield ≥ 80%, leaktrap 0/40.
  Until then, no new tools.

## Part 2 — `gnomon_describe`

The verb an agent calls ten times a session, and the designated triage
call: descriptive temporal facts, computed exactly, sub-second, no
backtest folds, always an answer on readable data. Most agent questions
over a series are not forecasts — they are "what changed", "is this
normal", "compare this month to last". Today those pay the full
evaluation toll or go unanswered.

### Contract

```
gnomon_describe(path | data,
                time_column?, target_column?, series_column?,
                relate_to?,       # second column: adds the associational block
                window?,          # "last 30" | ISO range; default full series
                compare?,         # "previous_period" | "same_period_last_cycle" | ISO range
                as_of?, regrid?, repair?)
```

### Output blocks (each present only when computable; caveats otherwise)

- **grid** — frequency, span, count, gaps (count, longest run, fraction),
  timezone/DST notes, repair summary if repair ran.
- **level** — latest value and timestamp; window median/IQR; the latest
  value's robust z-score and percentile against the baseline window.
- **trend** — direction, robust slope per period, since-when (anchored at
  the most recent changepoint, not the series start).
- **seasonality** — detected period(s), strength, peak/trough phase in
  calendar terms ("weekly; peaks Mondays, troughs Saturdays").
- **changepoints** — top-k with dates, magnitude, kind (level/trend).
- **outliers** — robust-z flags with dates; whether the latest point is one.
- **extremes & runs** — min/max with dates; current streak and whether it
  is unusual for this series ("7 consecutive rises; longest in the series").
- **comparison** — window vs `compare` window: Δ level, Δ%, and whether
  the change clears the series' own noise floor (robust scale), stated as
  "beyond/within typical variation" — never a probability.
- **relate** (only when `relate_to` given) — contemporaneous and best-lag
  cross-correlation between target and the named column, with the lag in
  calendar terms ("Y tracks X with a ~7-day delay") and a
  beyond-noise-floor label from a block-shuffle reference. Wording is
  structurally associational — templates for this block contain no causal
  verbs, and the claims register as associational, so the existing
  verifier rejects any causal restatement downstream. This answers the
  most common cross-series question agents ask without opening the
  covariate machinery.
- **suggested_next** — machine-readable escalation: the verb (and
  arguments) that answers what describe cannot ("what happens next" →
  `gnomon_forecast` with inferred schema pre-filled). This is the routing
  mechanism on the default surface.

Multi-series inputs get the same triage shaping as every verb: aggregate
grid/tier counts, top-k notable series, drill-down via `get_artifact`.

### Implementation notes

- Reuses what exists: changepoint and robust-score operators
  (`operators.py`), season detection (`temporal.py`), repair and regrid,
  the snapshot layer for `as_of` — "describe this series as it was known
  on June 1" falls out for free, and nothing else in an agent's toolbox
  can do it.
- Two engine fixes land with it, since describe makes them user-visible:
  season detection must take the *strongest* ACF peak, not the first
  above threshold (`temporal.py:147`), and the near-constant-series
  robust-scale floor must not manufacture ±1e12 z-scores
  (`operators.py:43`).
- **Epistemics are trivial by design.** Every number is an exact
  computation over served observations — descriptive claims in the
  existing verifier taxonomy. No selection, no calibration, no
  probability, nothing to abstain over; the only tiering is data-quality
  caveats (gappy grid, repaired values, short baseline).
- Cost ceiling: one pass plus the (linear) changepoint scan; block-shuffle
  reference for `relate` bounded to a fixed small number of shuffles.
  Cache by content fingerprint so repeated describes are free.
- Lightweight lineage: one evidence record (the computation over the
  snapshot) with claims attached — quotable under the same verifier
  contract as everything else at ~none of the cost.

### What it is not

No forecast, no interval, no probability, no cause. The moment a
question needs "what happens next", `suggested_next` points at
`gnomon_forecast` — escalation is a one-line decision, not five
exploratory calls.

## Part 3 — Headline sentences without an LLM

The requirement: every response leads with sentences an agent can quote
verbatim, produced by the same deterministic process that produced the
numbers. This is template NLG — the oldest, most boring kind (weather
bulletins and earthquake reports have been machine-written this way for
decades). No generation in the ML sense, so no hallucination surface.

### Mechanics

1. **Sentence specs.** A fixed library of parameterized templates, each:
   a guard predicate over artifact fields, a template string with typed
   slots, and slot bindings (artifact field + formatter). Per verb,
   specs are evaluated in fixed order; the first matching guard emits.
   Deterministic branching: the same artifact always yields
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

2. **Guaranteed coverage.** Each verb's spec list ends in a fallback
   whose guard always matches, stating the outcome, the tier, and the
   artifact id in neutral terms — no artifact can render zero sentences.
   A CI coverage test renders headlines for every artifact in the
   golden/evaluation corpus and asserts ≥ 2 sentences each and zero
   formatter errors; a fallback firing on a *supported* artifact fails
   the test (it means a situation went untemplated).

3. **Formatters, centralized.** One module owns rounding (3 significant
   figures in prose; exact values adjacent in `key_numbers`), humanized
   dates, percent deltas, units. The rounding policy is stated once and
   tested once.

4. **Word choice from binned magnitudes.** Direction and intensity verbs
   come from one documented threshold table over effect size in robust
   units — |Δ|/scale < 0.25 → "held roughly flat", < 1 → "edged
   up/down", < 3 → "rose/fell", else "jumped/dropped". No synonym
   sampling, no free text. Changing a threshold is a reviewed diff.

5. **Grammar helpers only.** Pluralization, list joining, a/an. Nothing
   open-ended.

6. **Verified by construction, then checked anyway.** The generator
   emits `(sentence, claim)` pairs: each headline registers as a lineage
   claim citing the evidence for the fields it used, so the existing
   verifier (tier labelling, calibration gating, no-causal-language)
   applies to Gnomon's own prose with zero new machinery. Two test
   layers on top: **golden headlines** per fixture, and a
   **numbers-round-trip property test** — regex-extract every numeral
   from every generated sentence across the corpus and assert each
   equals a cited artifact field under the stated rounding. "No invented
   numbers", enforced against ourselves in CI: a template bound to the
   wrong field becomes a build failure, not a shipped lie.

7. **Tier and caveat are part of the sentence, not metadata.** A
   best-effort row's headline says so inline ("naive extrapolation
   beyond day 9, best_effort tier") because quoted metadata is metadata
   lost. Same rule the verifier already enforces by rejection
   (`SUB_SUPPORTED_UNLABELLED`), satisfied by construction.

### Example (forecast, mixed tiers)

> requests is forecast to rise 12% over the next 14 days (evaluated tier
> through day 9; naive extrapolation beyond, best_effort tier). ETS beat
> the strongest baseline by 23% across 4 backtest folds. The q10–q90
> band at day 14 spans 310–410; measured interval coverage on the
> held-out fold was 78%. Crossing 340 first becomes more likely than not
> around Aug 21.

Four sentences, ~90 tokens, every number traceable, tier inline, nothing
for the agent to compose. The agent's best move — and its laziest move —
is to copy. That alignment of laziness with honesty is the design.

## Revision log

- **rev 2** (2026-08-11): added measurable goal statement; the schema-tax
  budget and CI test; multi-series triage shaping with notability
  ranking and artifact filters (the 490-channel case from the
  evaluation); inline-data caps and no-filesystem hosts; `key_numbers`
  key pinning; describe `relate_to` block (associational cross-series)
  and `suggested_next` routing (replacing `route` on the default
  surface); guaranteed-fallback templates with a corpus coverage test;
  Phase 5 rewritten to score accuracy on a leakage-controlled subset —
  "beat the control" is not a fair race where the control can leak —
  and to require beating the evidence-injection arm, not just the
  control.
- **rev 1** (2026-08-11): initial plan — 7-tool surface, one-shot verbs,
  response contract, describe, deterministic headlines, phases.
