# Agent Surface Redesign — Design Plan

Status: proposed (2026-08-11), revision 3. Not yet implemented.
Rev 3 incorporates two adversarial reviews (agent-behavior and
codebase-feasibility); revision log at the end.

## The goal, stated measurably

Make Gnomon a tool an agent is *better off calling* on temporal
questions. Operationally, on an agent evaluation with a raw-LLM control:

- **accuracy** at or above the control *and the evidence-injection arm*
  on tasks where the control cannot win by leaking (Phase 5 explains why
  both qualifiers are load-bearing);
- **cost**: under ~50K **cumulative conversation tokens** per task
  (input + output across all turns — response sizes alone are the
  smallest line item); ≤ 2 tool calls per question median, ≤ 4 p95;
- **yield**: a quotable, tiered answer on ≥ 80% of readable inputs,
  with accuracy **reported split by tier** so yield gains cannot
  silently launder naive extrapolation into the accuracy number;
- **safety**: leaktrap stays 0/40.

A note on baselines: several of the 2026-08 evaluation's frictions
(schema inference, inline data, long-series budgets, jail guidance) were
fixed in commits that postdate it (`861c3db`..`2401d2e`). Phase 5
therefore starts by re-measuring HEAD *before* this redesign lands, so
improvements are attributed honestly.

## Why redesign

The 2026-08 MCP evaluation (`gnomon-mcp-assessment.md`) measured three
failure modes, all interaction economics rather than engine quality:
**cost** (250–500K tokens/task, 6 calls where 1 would do), **yield**
(~486 of ~490 channels abstained or degraded), **fit** (most agent
temporal questions are descriptive, and every question paid the full
backtest toll). The evidence-injection arm — compute facts once, hand
them over — *beat* the control on T1 (48.9% vs 43.4%) while the tool
surface lost (32.7%). The harness helps when it hands the agent compact
computed facts; it hurts when the agent must drive an 18-tool workflow.

Design principle: **a tool is useful to an agent when calling it is
cheaper and more reliable than not calling it** — where "cheaper" is
measured over the whole conversation, including the tokens the agent
spends *making* calls and re-reading history, not just response sizes.

## Part 0 — Where the tokens actually go

Rev 2 budgeted the two smallest line items (response size, schema size)
and ignored the largest. Full accounting for a typical task:

| Line item | Cost | Who pays |
| --- | --- | --- |
| Tool schemas | every turn | host injects; see schema budget |
| Data transfer into the call | ~8–15 tokens/row inline | agent pays as *output* tokens, then re-pays as input every subsequent turn (call args live in history) |
| Verb responses | bounded below | re-paid as input every subsequent turn |
| Agent reasoning | free to us | — |

Two consequences the design must absorb:

1. **Inline data is the budget.** 2,000 inline rows ≈ 16–30K tokens,
   transcribed by the model (slow, expensive, and a transcription error
   silently corrupts the series). The inline cap is therefore ~500 rows,
   the tool description states the cost rule ("inline data costs ~10
   tokens/row *per turn*; prefer `path` or a `data_ref`"), and:
2. **Data crosses the wire once per session.** Every verb response
   returns a `data_ref` — the content fingerprint the cache layer
   already computes — carrying its *resolved schema*. Every verb accepts
   `data_ref` in place of `path`/`data`. Follow-up questions
   ("now just checkouts", "what changed in June?") cost a ~50-token
   call, cannot re-infer the schema differently, and skip re-ingestion
   entirely. This restores the session continuity that demoting
   `gnomon_ingest` would otherwise have removed.

## Part 1 — The redesigned surface

### Target default profile: 7 tools

| Tool | Change |
| --- | --- |
| `gnomon_describe` | **New.** Descriptive temporal facts; always answers. Part 2. |
| `gnomon_forecast` | One-shot; slimmed default schema (see schema budget). |
| `gnomon_investigate_change` | One-shot. |
| `gnomon_detect_anomalies` | One-shot. |
| `gnomon_monitor` | One-shot (computes the alert rule; the longitudinal loop lives in `full`). |
| `gnomon_get_artifact` | Gains field/series selectors (extension of the existing runner, `toolspec.py:1368`). |
| `gnomon_capabilities` | Kept; still the source of truth. |

Demoted to `full` (not deleted): `decide`, `route`, `inspect`, `ingest`,
`list_datasets`, the tracking suite, the covariate suite, `explain_run`,
`get_run`, `install_tsfm`. Planner and v0.2 compat stay behind their env
gates. **Explicit trade acknowledged:** the inline covariate/context
channels on `gnomon_forecast`'s schema move to the `full` profile too —
they are what make its schema alone ~2,850 tokens, larger than the
entire target budget. Demotion keeps every workflow reachable for
operators who opt in; the default surface is sized for agents.

### The schema tax

MCP hosts inject every visible tool's schema every turn — a per-turn
tax paid whether or not Gnomon is called. Measured today: the 18-tool
default is ~45KB (~11K tokens); even the existing 7-tool `core` profile
is ~21KB (~5.3K tokens). Budget: **≤ 12KB serialized (~3K tokens) for
the whole default surface**, measured in CI as *bytes* (bytes/4 tracks
tokens within ~10% and avoids adding a tokenizer dependency to a
zero-dependency project). Meeting it requires a schema diet, not just a
re-cut: terse decision-rule descriptions (≤ 2 sentences: *when to call
this*), no examples, no workflow prose, covariate/context channels off
the default forecast schema. Longer guidance lives in docs and the
`capabilities` response — and because agents rarely call `capabilities`
unprompted, the one rule that must be ambient ("descriptive or
'which tool?' questions → `gnomon_describe` first") is repeated as one
sentence in *every* tool description, paid seven times because that is
the only place agents actually read.

### One-shot verb semantics

Each verb is self-contained. Most of this is **already shipped** and the
plan claims no credit for it: inline `observations` exist
(`toolspec.py:98-111`), schema inference with disclosed assumptions
covers all five verbs (`toolspec.py:269-379`), `best_effort` is already
the MCP default (`support.py:33`, pinned by
`test_graduated_support.py:413`), and horizon defaulting landed. What
remains, and is genuinely new:

- **`data_ref` in and out** (Part 0), with pinned schema; inline cap
  ~500 rows with the cost rule in the description.
- **`focus`** — the parameter the multi-series design was missing:
  `focus: ["checkouts"]` returns full per-series headlines for the
  *named* series in the verb response. Agents ask about series they can
  name far more often than about "what's notable"; triage is only the
  no-focus default. Series and column names fuzzy-match with
  did-you-mean in the failure path.
- **Best-effort inference on ambiguity.** Today, more than one
  plausible target column refuses with `AMBIGUOUS_SCHEMA` — on messy
  operational CSVs (several numeric columns) that makes the *modal
  first call* a refusal, contradicting the yield goal. New posture:
  answer for the top-ranked candidate and put the assumption **in the
  headline** ("assumed target `requests` of [`requests`, `errors`]"),
  with corrected ready-to-issue calls in `next_actions`. Refusal is
  reserved for no-plausible-candidate.
- **Lenient optional arguments.** `window`/`compare`/date arguments
  parse the obvious natural forms ("last 30 days", "past month",
  "yoy"), always echo the resolved absolute dates in the response, and
  **never error on a bad optional argument** — ignore, default,
  disclose. Optional-arg errors are pure loop generators.
- **Degraded answers on error paths** — the hard part. Load/validation
  errors (`IRREGULAR_TIME_GRID`, `MIXED_SERIES_FREQUENCIES`, …) are
  raised before any computation exists to degrade to; attaching "here
  is what *was* computable" to those errors requires a partial-result
  envelope through the load/repair ladder. Scoped honestly as the bulk
  of Phase 2.
- **`output_dir` removed from the default surface.** Touches the shared
  registry schema (`registry.py:186`), capabilities text
  (`runtime.py:1370`), and the Hermes plugin's schema copies — listed
  so it isn't "one line".

### Response contract (all verbs)

1. `headline`: 2–5 deterministic sentences, numbers/tiers/caveats
   inline (Part 3).
2. `key_numbers`: the ~dozen values agents extract, **with tiers fused
   into the structure** — paired keys (`"day14_point": 372,
   "day14_tier": "best_effort"`) plus a top-level `tier_floor` — so a
   paraphrasing agent's extraction code cannot shed the tier even when
   its prose does. Key names are a contract, pinned by test.
3. **Temporal grounding, every response:** `series_end`,
   `wall_clock_now`, and a staleness sentence in the headline whenever
   the gap exceeds ~one grid step ("series ends 42 days before
   today"). A forecast tool that doesn't say what time it is lets
   agents report June's extrapolation as next week's outlook; this is
   the cheapest fix in the design and closes a whole class of
   confidently-wrong answers. All relative windows resolve against
   series-end, and the response says so.
4. `support`/tier summary and typed caveats.
5. `artifact_id`, `data_ref`, and `next_actions` — where `next_actions`
   entries are **literal ready-to-issue tool calls** (full argument
   dicts), never selector syntax the agent must compose.

Hard budget: ≤ ~1,200 tokens per verb response, ≤ ~600 for describe.
Honesty note: the *current minimum* brief forecast response measures
~1,300 tokens, so this budget is a restructuring mandate (per-step
forecast rows leave the response body for the artifact), not a trim.
The existing rule that protected epistemic subtrees may exceed the
budget (`toolspec.py:184-194`) survives; for multi-series inputs the
support assessments aggregate (counts per tier, exemplar reasons)
rather than enumerate.

### Multi-series triage

The evaluation's hardest task had ~490 channels. Today that response is
dominated by an *untrimmable* headline — `artifact_headline` joins
every series' sentence with `" | "` (`support.py:115-125`) and
`_trim_bulk` only trims lists — so triage shaping is a bug fix, not an
enhancement. For inputs beyond a handful of series (and no `focus`):

1. Aggregate headline: totals by tier and outcome ("Forecast 487 of
   490 series; 3 abstained (history < one season). 41 carry
   best_effort rows beyond day 9.").
2. Top-k notable series (k=5), one line each, ranking rule named in
   the response (largest change vs history / most recent changepoint /
   highest anomaly score / first crossing — verb-appropriate).
   Notability scores are persisted in the artifact (small schema
   addition) so `get_artifact` can sort by them.
3. Grouped remainder: counts per tier / abstention reason only.
4. Full table in the artifact via selectors: `series`, `fields`,
   `steps`, `where: {tier: best_effort}`, `order_by: notability`,
   `limit`. This is an extension of the existing `_run_get_artifact`
   over an in-memory dict, not a new subsystem.

`runtime.forecast_multi` (`runtime.py:946`) already computes per-series
results in one pass; triage is a response-layer change.

### Phases

- **Phase 0 — engine honesty.** The five bugs found by the 2026-08-11
  session review (all verified live at HEAD; the closed
  `codebase-review-2026-08.md` findings are separate and already
  fixed): (1) published ensemble ≠ evaluated ensemble — the pipeline
  publishes hardcoded `weighted_mean` over built-ins
  (`pipeline.py:434-440`) while selection/calibration honored
  `config.ensemble.strategy` and TSFM members (`evaluation.py:725`);
  bites whenever strategy ≠ weighted_mean or TSFMs compete. (2) Fold
  desync: a built-in model failing `predict` on a strict subset of
  folds compacts its `fold_forecasts`, misaligning fold-indexed reads
  in ensemble/meta-model scoring (`evaluation.py:798-810` vs
  `894-916`). (3) Adaptation outcomes stamped `known_time =
  cutoff_time` (forecast origin) against the code's own
  horizon-end comment (`tracking.py:816-821`). (4) NaN/Inf pass the
  strict loader (`data.py:355`) and poison selection
  (`error_score` returns NaN, not None). (5) `voting_forecast`'s
  up-biased ratio test (`ensemble.py:142-147`) — unreachable for
  published points today *because of* bug 1; fix together. Plus: the
  seeded Monte Carlo interval-calibration harness (~50 synthetic
  series, empirical q10–q90 coverage within Monte Carlo bounds) and
  forcing the two conditionally-skipping guard tests
  (`test_contract_holes.py:439`, `test_config_ensemble.py:369`) to
  exercise their branches.
  *Done when:* harness green; both guards execute their targets.
- **Phase 1 — response contract.** Extend the existing headline seam
  (`forecast_headline`/`artifact_headline`, `support.py:49-125`) into
  the Part 3 generator; `key_numbers` with fused tiers; temporal
  grounding; multi-series triage (replacing the pipe-joined headline);
  `data_ref` out/in; `get_artifact` selectors + persisted notability.
  *Done when:* every verb response ≤ budget on the evaluation corpus
  including the 490-channel task; golden headlines (per-interpreter
  policy), numbers-round-trip, template-coverage, and `key_numbers`
  pinning tests pass.
- **Phase 2 — one-shot gaps.** Degraded-answer-on-error envelope (the
  bulk); best-effort ambiguity posture; lenient optional args; inline
  cap; `focus` + fuzzy matching; `output_dir` removal.
  *Done when:* evaluation T2 workflow ≤ 2 calls/question median;
  supported-or-tiered yield ≥ 80% on the messy-CSV corpus; no
  optional-argument error appears in any evaluation transcript.
- **Phase 3 — `gnomon_describe`.** Part 2. Includes the measured perf
  work: one shared detrend/ACF pass (season detection and
  `seasonality_analysis` currently each recompute it, ~0.5s each on
  25K rows) and memoized per-phase stats in `anomaly_score` (currently
  O(n²/season), measured 2.9s) — naive composition is ~4s against the
  1s gate, so this is scheduled, not hoped.
  *Done when:* p95 < 1s on 25K rows; ≤ 600 tokens; answers on 100% of
  readable inputs; season-strongest-peak (`temporal.py:150`) and
  constant-series floor (`operators.py:43`) fixes landed.
- **Phase 4 — profile re-cut + schema diet.** The re-cut itself is
  small (`PROFILES` seam, `toolspec.py:1839`); the listed work is the
  diet (covariate/context channels off the default forecast schema,
  every description rewritten to the ≤ 2-sentence rule), the bytes
  budget test, the capabilities-prose rewrite the reachability contract
  test will otherwise fail on, and Hermes schema copies.
- **Phase 5 — measure, with a fair race.**
  First re-baseline HEAD (pre-redesign) so wins are attributed
  honestly. Then per task: cumulative tokens, calls
  (median/p95), yield, wall-clock, accuracy —
  **accuracy split by tier of evidence used**, and **quote-vs-
  paraphrase and caveat-survival rates measured** (the Part 3
  assumption that agents copy headlines is load-bearing and currently
  unevidenced; if caveat survival is poor, the fused-tier
  `key_numbers` are the guarantee and Part 3's prose investment gets
  cut, not expanded).
  **Leakage-aware comparison:** the control's 0.46% SMAPE and the
  leaktrap result (leaking ≈ 78% of headroom) say the control can win
  public benchmarks by reading answers from pretraining or the
  prompt. Accuracy is therefore *gated* on a leakage-controlled subset
  (targets past model cutoff, synthetic, or perturbed vintages — the
  leaktrap generator builds these); on the leakage-possible remainder
  we report accuracy alongside the control's leak-flag rate.
  *Done when:* on the leakage-controlled subset, Gnomon-MCP ≥ control
  **and ≥ the evidence-injection arm** at < 50K cumulative tokens,
  yield ≥ 80%, leaktrap 0/40.
  **Pre-committed fallback:** if the injection arm still wins, the
  product conclusion is that the winning architecture is *injection as
  a tool* — a single mega-describe/evidence-pack call — and the
  surface contracts around `describe` + `forecast` rather than growing.
  Writing this down now is the insurance against motivated reasoning
  later.

## Part 2 — `gnomon_describe`

The verb an agent calls ten times a session: descriptive temporal
facts, computed exactly, sub-second, no backtest folds, always an
answer on readable data. Most agent questions over a series are not
forecasts — "what changed", "is this normal", "compare this month to
last" — and today they pay the full evaluation toll or go unanswered.

### Contract

```
gnomon_describe(path | data | data_ref,
                time_column?, target_column?, series_column?,
                focus?,           # named series get full blocks
                relate_to?,       # second column: adds the associational block
                window?,          # lenient: "last 30 days" | ISO range; resolved dates echoed
                compare?,         # "previous_period" | "same_period_last_cycle" | ISO range
                as_of?, regrid?, repair?)
```

### Output blocks (each present only when computable; caveats otherwise)

- **grid** — frequency, span, count, gaps, timezone/DST notes, repair
  summary; plus `series_end` vs `wall_clock_now` staleness.
- **level** — latest value; window median/IQR; robust z and percentile
  vs the baseline window, **phase-conditioned when seasonality is
  detected** ("for a Tuesday, this is p94") — the season detector has
  already run, and "is this normal?" usually means "normal for a
  Tuesday", not "normal globally".
- **trend** — direction, robust (Theil–Sen-style) slope per period,
  since-when (anchored at the most recent changepoint).
- **seasonality** — period(s), strength, peak/trough in calendar terms
  ("weekly; peaks Mondays").
- **changepoints** — top-k with dates, magnitude, kind.
- **outliers** — robust-z flags with dates; whether the latest point is
  one (phase-conditioned as above).
- **extremes & runs** — min/max with dates; current streak and whether
  it is unusual for this series.
- **comparison** — window vs `compare` window: Δ, Δ%, and whether the
  change clears the series' own noise floor — "beyond/within typical
  variation", never a probability.
- **relate** (when `relate_to` given) — contemporaneous and best-lag
  cross-correlation in calendar terms ("Y tracks X with a ~7-day
  delay"), beyond-noise labeled from a block-shuffle reference.
  Templates for this block contain no causal verbs and the claims
  register as associational. Honest boundary: this constrains *our*
  prose; nothing can stop an agent restating association as cause in
  its own voice — which is why the wording is chosen so the quoted
  form is safe and the measurement in Phase 5 tracks what agents
  actually do with it.
- **suggested_next** — ready-to-issue escalation calls with arguments
  pre-filled (`gnomon_forecast` with the resolved `data_ref` and
  schema).

Multi-series inputs get the same triage shaping as every verb; `focus`
gets named series in full.

### Implementation notes

- Reuses: changepoint scan (linear, measured 0.06s on 25K rows), robust
  scores, season detection, repair/regrid, snapshot layer. New
  operators, named because "reuse" was oversold in rev 2: robust slope,
  percentile-vs-baseline, runs/streaks, calendar phase naming, the
  window/compare grammar, the block-shuffle reference
  (`cross_correlation` today uses a 2/√n bound and `max_lag=8`).
- Perf work as scheduled in Phase 3 (shared ACF pass; memoized phase
  stats).
- `as_of` honesty: on plain files, `as_of` replays assumed known-times
  (disclosed, as everywhere); *vintage-honest* replay requires a
  `store:` dataset, and the ingest tools live in the `full` profile —
  the response's caveat says so rather than implying the stronger
  guarantee.
- Epistemics are trivial by design: exact computations over served
  observations — descriptive claims, no selection, no calibration,
  nothing to abstain over; tiering is data-quality caveats only.
  Lightweight lineage (one evidence record, claims attached).
- Cost ceiling: one pass + linear changepoint scan + bounded shuffles;
  cached by content fingerprint.

### What it is not

No forecast, no interval, no probability, no cause. "What happens
next" escalates via `suggested_next` in one line.

## Part 3 — Headline sentences without an LLM

Every response leads with sentences an agent can quote verbatim,
produced by the same deterministic process that produced the numbers.
This is template NLG (weather bulletins, earthquake reports — decades
old, no ML generation, no hallucination surface). A seam already
exists — `forecast_headline`/`artifact_headline` (`support.py:49-125`)
ship in every forecast response today — so this is an extension, not a
green field.

Scope discipline (from the rev-3 review): the template library starts
with **the fallback plus the highest-traffic situations per verb**, not
exhaustive coverage. The safety guarantee agents can't shed rides in
`key_numbers`' fused tiers; prose breadth grows only if Phase 5 shows
agents actually quote it.

### Mechanics

1. **Sentence specs.** Guard predicate over artifact fields + template
   + typed slot bindings (field + formatter). Fixed evaluation order;
   first matching guard emits. Deterministic **per interpreter**: the
   repo's golden policy already documents that 3.11/3.12 differ by an
   ulp in float sums (`test_golden_artifacts.py:23-27`), and a digit at
   a rounding boundary can differ in prose — headline goldens are
   captured on the reference interpreter (3.12), matching the existing
   per-interpreter policy rather than contradicting it.
2. **Guaranteed coverage.** Each verb's spec list ends in an
   always-matching fallback (outcome, tier, artifact id, neutral
   terms). CI renders headlines for every artifact in the golden and
   evaluation corpora: ≥ 2 sentences each, zero formatter errors, and
   a fallback firing on a *supported* artifact fails the build (an
   untemplated situation).
3. **Formatters, centralized.** One module owns rounding (3 significant
   figures in prose; exact values in `key_numbers`), humanized dates,
   percent deltas, units.
4. **Word choice from binned magnitudes.** One documented threshold
   table over effect size in robust units ("held roughly flat" /
   "edged up" / "rose" / "jumped"). No synonym sampling, no free text;
   changing a threshold is a reviewed diff.
5. **Verification policy — prose bugs must not become refusals.**
   Each sentence registers as a lineage claim citing the fields it
   used, so the existing verifier's checks (tier labelling per
   `SUB_SUPPORTED_UNLABELLED`, calibration gating, no-causal-language)
   apply to Gnomon's own prose. But `verify_or_raise` is
   raise-or-nothing (`verifier.py:287`), and one bad template guard
   must not turn a whole valid response into
   `CLAIM_VERIFICATION_FAILED`. Policy: headline claims verify in a
   **pre-pass**; a failing sentence is dropped and the fallback emitted
   in its place, with the drop logged — artifact-level claims keep the
   hard-fail path. The numbers-round-trip property test (regex-extract
   every numeral from every generated sentence; assert it equals a
   cited artifact field under the stated rounding) is new machinery
   and is the real enforcement — rev 2's "zero new machinery" was
   oversold and is withdrawn.
6. **Tier inline in the sentence AND fused in `key_numbers`.** The
   sentence protects the quoting agent; the fused structure protects
   against the paraphrasing one. Phase 5 measures which agent we
   actually have.

### Example (forecast, mixed tiers)

> requests is forecast to rise 12% over the next 14 days (evaluated
> tier through day 9; naive extrapolation beyond, best_effort tier).
> The series ends 2026-08-09, two days before today. ETS beat the
> strongest baseline by 23% across 4 backtest folds. The q10–q90 band
> at day 14 spans 310–410 (measured coverage 78% on the held-out
> fold). Crossing 340 first becomes more likely than not around
> Aug 21.

~100 tokens, every number traceable, tier and staleness inline. The
agent's best move — and its laziest — is to copy; and when it
paraphrases instead, the tier survives in the structure it extracts
from. Belt and suspenders, because Phase 5 will tell us which one held.

## Revision log

- **rev 3** (2026-08-11), after two adversarial reviews:
  *Token accounting rebuilt* — inline data identified as the dominant
  cost (rev 2 budgeted the two smallest line items); inline cap cut to
  ~500 rows; `data_ref` in/out on every verb for session continuity
  and schema pinning; goal restated in cumulative conversation tokens
  and median/p95 calls (rev 2's canonical flows were 3 calls against
  its own ≤ 2 criterion). *Named-series reality* — `focus` parameter;
  fuzzy matching; `next_actions` must carry literal ready-to-issue
  calls. *Posture fixes* — best-effort inference on ambiguous schema
  (refusal was the modal first call on messy data); lenient
  never-erroring optional args; temporal grounding
  (`series_end`/`wall_clock_now`/staleness) in every response.
  *Paraphrase honesty* — tiers fused into `key_numbers` structure;
  quote-vs-paraphrase and caveat-survival measured in Phase 5; Part 3
  scope cut to fallback + high-traffic templates; withdrew the claim
  that the verifier constrains agent restatements. *Feasibility
  corrections* — Phase 2 re-scoped (inline data, inference, and the
  best_effort default are already shipped; the remaining work is the
  degraded-answer envelope); response budget acknowledged as a
  restructuring mandate (current minimum ~1,300 tokens); schema budget
  raised to a bytes-measured ~12KB and the covariate/context-channel
  demotion named as its price; describe perf work scheduled from
  measurements (~4s naive vs 1s gate); headline generator re-founded
  on the existing `support.py` seam with a drop-sentence (never
  refuse) verification policy and per-interpreter determinism; Phase 0
  bug list re-verified at HEAD with narrowed trigger conditions; Phase
  5 re-baselines HEAD first and pre-commits the injection-as-a-tool
  fallback if the evidence-injection arm keeps winning.
- **rev 2** (2026-08-11): multi-series triage; leakage-aware
  acceptance; schema-tax budget; relate/suggested_next; fallback
  templates.
- **rev 1** (2026-08-11): initial plan.
