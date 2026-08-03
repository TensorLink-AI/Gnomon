# Earning trust on short-history, news-driven series

Exploration, 2026-08-03, at commit `d2c503a`. This is a design document,
not a build: it audits what exists (Part 1), reports four pre-registered
experiments plus two labeled post-hoc diagnostics (Part 2 — registered in
`results/news-regime-explore/HYPOTHESIS.md` before running, outcomes in
`RESULTS.md` there), and takes build/don't-build positions with a
dependency-ordered roadmap (Part 3).

**Provenance constraints, stated once:** this environment has no LLM API
key and its egress policy denies huggingface.co, so the official MTBench
processed datasets and the untracked `results/deepseek-v4-flash/`
artifacts are unreachable. Experiments ran on a disclosed surrogate — 50
real S&P 500 daily-close windows, 30 in / 7 out, same asset class, bar
convention, window shape, and price-scale regime as MTBench
`finance_long` (`results/news-regime-explore/scripts/build_tasks.py`,
seed 20260803). Official numbers cited below (control MSE ≈ 6.5 /
MAPE ≈ 2.8%, Gnomon arm ≈ 10.9 / 3.8%; CiK: zero admissions across 141
event-proposing runs) are the recorded means from that prior run.
Cross-dataset comparisons are qualitative by construction. The
`future-context-prompt.md` file referenced by the task does not exist in
the repo; the merged future-context lane itself
(`src/gnomon/future_context.py`, `results/future-context-ab/`) served as
the composition reference instead.

## The one-paragraph conclusion

At 30 daily points with horizon 7, Gnomon's evaluation skeleton yields
exactly two folds — one selection fold, one calibration fold, no test
fold — and the artifact *does* disclose this (loudly, in five places).
The trust problem is not silence; it is that **the defaults still act on
the under-powered evidence they disclose**: single-fold selection with a
2% margin picks a non-baseline model on 39 of 50 near-martingale tasks
and thereby runs 2.9× the MSE of its own `last_value` baseline, and the
constant-width intervals it publishes cover 63.7% instead of the nominal
80%, decaying to 44% by step 7. Both are self-inflicted and fixable
without any news mechanism, and fixing them closes most of the official
gap to the LLM control, whose 2.8% MAPE sits just above the 1.6%
martingale floor. The news edge worth chasing after that is thinner
than it first looks: a direction-only oracle capped at 1σ is worth
24.7% MSE on this regime, but wrong directions cost more than right
ones gain, so the exact break-even hit rate is 0.65 at k=0.5σ and 0.81
at k=1σ — at a plausible 0.6–0.7 proposer, even optimally-sized tilts
are worth only 1–3%, and a 2σ cap destroys value for the *oracle*. So
the build order is: honesty-preserving selection guardrails first
(simulation-validated: MSE 2.98 vs today's 7.42), the proposer
calibration ledger second (it is the measurement substrate every news
mechanism needs, and the only thing that can ever justify a tilt),
directional tilts *not built* until the ledger measures sustained skill
above break-even, population-level event studies only after a
corpus-backed measurement study — they are now the only mechanism with
headroom above single digits, precisely because magnitude information
is what a sign cannot carry.

---

## Part 1 — Audit

### 1.1 Support degradation at short history

**Fold arithmetic** (`src/gnomon/evaluation.py`): `minimum_train =
max(2·season, 2·horizon, 8)` (`evaluation.py:562`) and non-overlapping
origins step by the horizon (`_origins`, `evaluation.py:412-423`). At 30
daily points, horizon 7, season 7: origins = [14, 21] — two folds. With
fewer than three origins the split is `selection_origins=[14],
calibration_origin=21, test_origin=None` (`evaluation.py:595-599`):
**one selection fold, no held-out test**. `degraded = len(origins) < 4`
(`evaluation.py:593`). Dense selection origins (`evaluation.py:426-445`)
cannot buy anything here — the stride range collapses to the same single
origin. Interval calibration pools the selection fold with the
calibration fold (default `pool_residuals`, `evaluation.py:604-615`), so
q10–q90 rest on 14 residuals, of which 7 are selection-optimistic;
coverage is unmeasured (`evaluation.py:1196-1200`) and unmeasured
coverage is deliberately not disqualifying
(`contracts.py:71-80`).

**What the artifact actually says** (verbatim, from a real 30-point run,
`gnomon forecast … --horizon 7 --frequency D`):

- `support: "degraded"`, public status `"conditionally_supported"` with
  reasons: *"Model selection ran without separated calibration and test
  folds."*; *"Limited evaluation: only 2 rolling folds were available;
  42 observations enable fully separated selection, calibration, and
  test windows."*; *"Limited evaluation: no held-out test fold remained,
  so interval coverage is unmeasured."* (mapping in
  `support.py:156-168`).
- Four candid disclosures: `point_is_not_the_median`,
  `constant_interval_width` (*"Uncertainty is not modelled as growing
  with distance here"*), `conformal_residuals_pooled_across_selection`
  (*"the interval is narrower than strict split conformal would
  give"*), `quantile_levels_collapsed` (*"at 7 of 7 lead times, adjacent
  levels share an order statistic"*).
- A recovery action (`provide_more_history`) and, on abstention, the
  reachable horizon (`supportable_horizon`, `evaluation.py:448-455`).

So the disclosure layer is genuinely loud — closer to exemplary than to
a bug. The two things it does **not** say:

1. `sensitivity.baseline_improvement: 0.1266` is published with no
   sample size. That 12.7% is a **single-fold** measurement; nothing in
   the artifact says selection ran on n=1 folds, and a reader will read
   a two-digit improvement as evidence. (The fold count exists internally
   — `residual_fold_count`, `evaluation.py:61` — but selection fold
   count is not surfaced.)
2. Nothing says "on this evidence the selection *decision itself* is
   unreliable". Experiment E1 measures exactly that unreliability: the
   2% default margin (`config.py:124`) on one fold selected theta /
   window_average / drift / ets on 39 of 50 near-martingale tasks and
   lost to `last_value` by 2.9× MSE. Disclosed under-power plus
   confident action on it is the trust bug (position (a)).

The MTBench adapter (`benchmarks/mtbench/gnomon_forecaster.py:224-238`)
abstains only on `unsupported`; `degraded` runs are scored, so the
official Gnomon arm carried this behavior into its 10.9 / 3.8% numbers.

### 1.2 The tracking loop vs a per-proposer calibration ledger

The registry (`~/.local/share/gnomon/registry.db`, schema v4,
`tracking.py:260-393`) stores per tracked forecast: identity, cutoff,
horizon, `selected_model`, `support`, `naive_error`, a series
fingerprint, and — after `submit_actuals` — MASE / WAPE / MAPE / bias /
q10–q90 coverage / threshold accuracy (`score_forecast`,
`tracking.py:661-741`), scored one-shot when every horizon period has an
actual (`tracking.py:821-826`). Aggregation (`leaderboard`,
`tracking.py:1019-1060`) is `AVG` grouped **by model only** — no
grouping by horizon/frequency/fingerprint though all are stored, no
count-weighting, no shrinkage. Decisions and richer decision artifacts
exist (`tracking.py:1100-1129,1153-1225`); episodes (`episodes.py`) are
a synthetic offline grading harness with **zero** references to the
tracking store.

Distance to a per-proposer calibration ledger — the loop is open at one
seam, and it is exactly the seam mechanisms (b)–(d) need:

- **No context-event concept in the registry at all** (grep of
  `tracking.py`: no event/proposer/enrichment fields). The only trace of
  an admitted event is the `selected_model` string becoming
  `context`/`combined` (`pipeline.py:583`, `adjudication.py:35-36`).
- **The admission record exists but is write-only**: `context_ablation`
  / `context_gate` evidence (with per-check `gate_checks`, exclusion
  reasons, `decided_by` — `context_eval.py:102-160`,
  `pipeline.py:560-581`) lands in `evidence.jsonl` and no reader ever
  joins it back; `register_artifact` walks `artifact.results` only
  (`tracking.py:1568-1608`).
- **Proposer identity is two-valued and dropped early**:
  `ContextEvent.created_by ∈ {"user","llm"}` (`context.py:61`,
  `workflows.py:201`) — no model name, version, or run id — and
  `ContextAssessment.events_used` is a bare list of event-id strings
  (`context_eval.py:107`), so even the artifact loses the proposer.
  Event ids are positional (`event_llm_{index:02d}`, `workflows.py:191`)
  and not stable across runs.
- **The counterfactual is computed and discarded**: adjudication has the
  enrichment-free base path per fold (`adjudication.py:149-150`) but
  returns only the winner (`adjudication.py:249-258`); nothing persists
  the base path, so realised lift per admitted event is uncomputable
  after the fact.
- `confidence` (`context.py:58`) is validated and never calibrated
  against anything; no proper scoring rule for probabilistic claims
  exists in `tracking.py`.

### 1.3 The context event model

- **Effect shapes**: `level` / `decay` (fixed 0.6 rate, deliberately
  unfitted) / `ramp` (`context_model.py:59-94`), area-normalized so
  shapes differ in timing not magnitude (`context_model.py:97-116`).
  Magnitude is purely measured from history: detrended
  active-vs-inactive mean difference (`event_effect`,
  `context_model.py:37-54`). **The LLM never writes a magnitude**,
  enforced three ways: the proposal schema has no magnitude field
  (`workflows.py:39-66`), the prompt forbids computing numbers
  (`workflows.py:116-119`), and `parse_context_response` strips
  model-supplied `source_span`/`claim` attributes
  (`workflows.py:167,176`). Residual LLM influence on numbers is
  structural only (window placement, which drives what the effect is
  measured on).
- **Admission gate** (`assess_context`, `context_eval.py:212-486`):
  identical-fold ablation — base vs event-adjusted candidate on the same
  selection origins with cutoff-gated flags (leakage guard,
  `context_eval.py:181-209`), symmetric relative improvement, then five
  recorded checks: mean improvement ≥ margin, majority of folds improve,
  gain survives dropping the best fold, coverage not degraded (Wilson
  upper bound), optional shrinkage. **Hard precondition:
  `separated_folds_available` requires ≥ 4 origins**
  (`context_eval.py:247-262`) — at 30/7 there are 2, so on
  MTBench-shaped tasks **every proposal is structurally rejected before
  any evidence is weighed**. This is the same shape as the CiK zero-of-141
  gap but sharper: on short history the gate cannot even convene.
- **The future-context lane** (`future_context.py`) is the one
  non-fold admission path: constraints and overrides whose numbers are
  re-parsed deterministically from quoted spans, restricted to
  future-only windows (`future_context.py:504-514` — deliberately not a
  fallback for gate-rejected historical events), applied as clamps /
  stated values, disclosed via the `context_trusted` support state with
  a history-only counterfactual (`pipeline.py:778,788-805`;
  `support.py:71-108`). This lane does not help MTBench-style soft news
  ("earnings beat expectations" states no bound and no value) — its own
  hypothesis file excludes soft directional effects as out of scope
  (`results/future-context-ab/HYPOTHESIS.md`).
- **Attachment seams for new paths** (for Part 3): a population-level or
  tilt path attaches as a sibling of `_future_context_stage`
  (`pipeline.py:757-806`) or as a new adjudication rung
  (`adjudication.py:173-184`); `event_adjusted` already takes arbitrary
  `base_points` and a shape, needing only the scalar effect factored out
  (`context_model.py:119-152`); the codebase σ is
  `operators._robust_scale` (1.4826·MAD, `operators.py:43-46`), with
  `fingerprint["noise_ratio"]` (`fingerprint.py:36-37`) as its
  scale-free form.
- **ID-salting pattern**: note the task brief's `model_backends` symbol
  does not exist; the real exemplars are `repair`, `model_weights`, and
  `future_context` in `runtime.py:523-539` — the payload key is added
  **only when the path is active**, and `content_id`
  (`ids.py:45-60`) hashes the exact key set, so every flag-off ID is
  byte-identical. Any new lane must add exactly one guarded key there;
  even `{"enabled": False}` would break every existing artifact ID.

### 1.4 TSFM tier at history = 30

Seven adapters registered (`chronos_bolt_mini/small`, `toto2_22m`,
`flowstate`, `ttm`, `moirai2_small`, `moment_small` —
`tsfm.py:1094-1104`), executed by preference in per-adapter `uv`
sandboxes (`tsfm_sandbox.py`), competing as ordinary candidates on the
same folds, same loss, same margin as classical models
(`evaluation.py:704-756,1012-1016`) — that part is clean. But:

- **Eligibility says nothing at 30 points** (verified by execution:
  `eligible_tsfms(30, 7, "D")` admits all seven, zero exclusions).
  `min_context_length` defaults to 1 and no adapter overrides it;
  `max_horizon` is never set; `max_context_length` is declared but
  unread by the gate (`tsfm.py:124-126,137-168,193-202`). The only real
  backstop is the single selection fold with its 14-point training
  window.
- **Audit findings that gate any E2 build-out** (full table in the
  session audit): `SANDBOX_ROOT` resolves to the *cwd* — `Path("")` is
  truthy — so sandboxes are per-directory and `gnomon capabilities`
  answers differently per cwd (`tsfm_sandbox.py:61-63`); sandbox
  workers load weights **unpinned** while the artifact records the
  pinned revision in `model_weights` (`tsfm_sandbox.py:289-291` etc. vs
  `tsfm.py:56-83`, `runtime.py:527-530`) — the ID attests a commit the
  worker never guaranteed; `models.tsfm.candidates` never reaches
  `evaluate()` (`pipeline.py:300-307` drops `tsfm_names`); and
  `capabilities()["models"]["tsfm"]` reports `[]` for a working sandbox
  install because `installed_tsfms()` requires in-process torch
  (`runtime.py:913`, `tsfm.py:327-353`) — a `gnomon capabilities`
  truthfulness violation today, independent of anything this design
  adds.
- Weights come from huggingface.co at inference, so the tier is
  unusable in Hub-denied environments; capabilities/notes do not
  currently express that failure mode (the "no sandbox installed" note
  at `evaluation.py:694-702` is the nearest disclosure).

---

## Part 2 — Measured experiments

Pre-registered in `results/news-regime-explore/HYPOTHESIS.md` (commit
`d2c503a`, before any run); outcomes in `RESULTS.md`; summaries
versioned in `summary.json` / `margin_sweep.json`; 50-task surrogate as
described under provenance. No abstentions, no official-filter failures,
so every mean is over all 50 tasks.

| # | Question | Prediction | Outcome |
|---|---|---|---|
| E1 | Martingale floor | floor MAPE ∈ [1.5, 3.5]%; gnomon within ±20% of floor | floor **1.61%** / MSE 2.56 (holds); gnomon-pure **2.37% / 7.42 — +47% rel., falsifier fired** |
| E2a | TSFM eligibility at 30/7/D | all seven admitted, zero exclusions | **confirmed by execution** |
| E2b | Chronos-Bolt closes gap | wins ≥40% of tasks, moves mean MAPE < 0.5 pts | **blocked** (Hub denied); registered for a Hub-enabled machine |
| E3 | Oracle-direction tilt ceiling, k·σ cap | best k ∈ {1,2}, MSE −20…−50% | k=0.5: −22.3%; **k=1: −24.7%** (holds); **k=2: +29.7% (harm)** |
| E4 | q10–q90 coverage at 30 pts | [60, 76)% pooled, below nominal 80% | **63.7%** pooled; per-step 82→44% (confirmed) |

Post-hoc diagnostics (labeled, not pre-registered; first pass plus the
second-pass `iterate_analysis.json`):

- **Margins can't fix a one-fold contest**: raising
  `minimum_baseline_improvement` 0.02→0.50 shrinks gnomon-pure MSE
  7.42→4.32, never reaching the 2.56 floor; on tasks where `last_value`
  won selection, gnomon ≈ floor (MSE ratio 1.018), so the damage is
  model choice, not the `point_bias_correction` recentring.
- **Exact tilt break-even**: wrong-direction tilts are asymmetrically
  costly (k=1σ: MSE 5.21 wrong vs 1.93 right, floor 2.56). Break-even
  hit rate **0.654 at k=0.5σ, 0.808 at k=1σ**; with k re-optimized per
  p, expected reduction is +1.0% (p=0.6), +1.9% (p=0.65), +3.0%
  (p=0.7) — all far below the 10% build bar. The first-pass "(2p−1) of
  the oracle" discount was wrong and is corrected here.
- **Guardrail simulation (H-G1 preview)**: baselines-only selection on
  the single fold → MSE **2.98** / MAPE **1.69%**, inside the H-G1
  targets. Even the two-baseline contest is noisy (`seasonal_naive`'s 9
  fold-wins lose to the raw floor 6-of-9; unconditional `last_value`
  gives 2.56), so "skip selection entirely below 2 selection folds" is
  a live design option.
- **Widening-rule search (H-G3 preview)**: √(step/4) anchored at the
  mean lead **fails** (narrows early steps; pooled 62.6%); √step
  anchored at step 1 meets every target (pooled 89.7%, step-7 88%) with
  disclosed mid-horizon over-coverage (96% at step 4) — the
  conservative direction.
- **Bootstrap 95% CIs**: paired gnomon−floor MSE difference
  [+1.67, +9.40], MAPE difference [+0.36, +1.20] pts — the
  selection-harm finding is not sampling noise. Pooled coverage CI
  [53.4%, 73.4%] excludes nominal 80%. Oracle k=1σ absolute reduction
  CI [0.05, 1.25] — the ceiling exists, its size is uncertain.

Interpretation against the official numbers (qualitative): the control's
2.8% MAPE sits just above a ~1.6–2% martingale floor — the LLM's "win"
is mostly *being* a decent martingale; the Gnomon arm's 3.8% is mostly
self-inflicted selection noise, and the genuine news edge on this regime
is bounded by roughly the floor-to-control gap, i.e. small in absolute
terms and consistent with E3's thin oracle ceiling.

---

## Part 3 — Positions

Shared constraints for every mechanism, restated as acceptance criteria:
the LLM never writes a forecast number; any un-fold-tested influence
carries its own support state (weaker than or equal to
`context_trusted`) plus the history-only counterfactual in evidence;
artifact IDs salt via one guarded payload key per lane, absent when
inactive, following `runtime.py:523-539`; flag-off runs byte-identical;
`gnomon capabilities` states exactly what each lane can and cannot do.

### (a) Louder support degradation at short history — **BUILD, first**

The audit shows disclosure is already loud; the measured trust bugs are
behavioral, so "louder" alone is the wrong spec. Three changes:

1. **Selection guardrail at fold-starved history.** When selection has
   < 2 disjoint selection folds (i.e. total origins < 3, today's
   30/7 case), a non-baseline candidate must not be selected on the
   default margin: either require a fold-count-scaled margin, or (the
   E1-supported default) select the strongest baseline and report every
   candidate's single-fold score as *unranked evidence*, with a new
   typed reason (`selection_underpowered`) naming the fold count.
   E1 sizes the prize and the simulation validates the target:
   baselines-only selection on the single fold measures MSE 2.98 /
   MAPE 1.69% against 7.42 / 2.37% today (floor 2.56 / 1.61%). The
   simulation also shows the two-baseline fold contest is itself noisy
   (`seasonal_naive`'s nine single-fold wins lose to the raw floor six
   times), so the implementation should also evaluate skipping the fold
   contest entirely below 2 selection folds and taking `last_value`
   unless the series is measurably seasonal. Long-history behavior
   (origins ≥ 4) is untouched, so existing goldens there stay
   byte-identical.
2. **Disclose n beside every selection statistic.**
   `sensitivity.baseline_improvement` and `selection_scores` gain
   `selection_fold_count` (the internal `residual_fold_count` pattern,
   `evaluation.py:55-61`, extended to selection). Metadata-only for
   unaffected series.
3. **Lead-time-honest intervals when per-lead residuals are absent.**
   E4: constant-width pooled intervals cover 82% at step 1 and 44% at
   step 7. When `MIN_RESIDUALS_PER_LEAD` (`evaluation.py:99`) is unmet,
   widen the pooled spread deterministically with lead time — same
   family as the existing `interval_from_spread` scale factor
   (`evaluation.py:130-144`). The schedule is measured, not assumed:
   the second-pass rule search shows anchoring √-growth at the *mean*
   lead fails (it narrows early steps; pooled coverage drops to 62.6%),
   while **√step anchored at step 1** (monotone, never narrows) meets
   every target on the 50-task set — pooled 89.7%, step-7 88% — with
   mid-horizon over-coverage (96% at step 4) as the disclosed,
   conservative cost. Build with that schedule and an over-coverage
   falsifier (H-G3).

### (b) Population-level event studies — **NEEDS MORE EVIDENCE** (spec below, don't build yet)

This is the only mechanism that could beat the martingale by more than
the thin directional ceiling — a measured magnitude distribution is
strictly more informative than a sign — but nothing measured in this
session establishes that typed-event effect distributions on news-driven
finance series are stable enough to admit from. The tilt break-even
result sharpens the case both ways: a sign alone cannot be worth more
than single digits at plausible skill, so magnitude distributions are
where the remaining headroom lives — and the same asymmetric-penalty
geometry will apply to a *mis-signed* population effect, so the corpus
study must measure sign stability per event type, not just median
magnitude. Only a corpus study can say whether typed, high-salience
events (earnings, guidance, M&A) clear that bar.

**Prerequisite experiment** (pre-register before building anything): on
a Hub-enabled machine with the MTBench raw corpus (20k labeled finance
news + aligned series), measure per-type post-event effect
distributions (abnormal return over horizon steps 1–7, normalized by
each series' σ). Build only the types whose distribution is (i) sign-
stable across ≥ 100 episodes, (ii) with median |effect| ≥ 0.5σ.

**Data model spec** (attaches to the tracking store, §1.2): an
`event_studies` table — `(event_type, series_class, lead, n_episodes,
effect_q10/q50/q90 in σ units, corpus_id, measured_at)` — written by an
offline `gnomon events study <corpus>` command, read at admission time
by a third lane beside `_future_context_stage` (`pipeline.py:757-806`):
classify the proposal (LLM proposes `event_type` only — it already
does), look up the measured distribution, apply the *distribution's*
quantiles through `event_adjusted(base_points, shape, effect)` with the
effect scalar factored out (`context_model.py:146`), widen intervals by
the distribution spread via the `conditional._effect_standard_error`
hook pattern (`conditional.py:69-91`). New support state
(`population_informed`), weaker than `context_trusted` (the magnitude
was measured, but on *other* series); counterfactual and one guarded ID
key as per the constraints. Corpus provenance (id + hash) must appear in
the artifact, or the number is unauditable.

### (c) Volatility-capped directional tilts — **DON'T BUILD NOW; re-evaluate from ledger data**

E3's oracle ceiling (24.7% MSE at k=1σ) is not the decision-relevant
number, and the second-pass exact computation retires the first-pass
"(2p−1) of the oracle" shortcut: because a wrong-direction tilt costs
more than a right-direction tilt gains (k=1σ: MSE 5.21 wrong vs 1.93
right, floor 2.56), the **break-even hit rate is 0.654 at k=0.5σ and
0.808 at k=1σ**, and even with the cap re-optimized per skill level the
expected gain is **+1% at p=0.6, +2% at p=0.65, +3% at p=0.7** — all
far below the pre-registered 10% build bar, and *negative* below
p ≈ 0.65 at any useful k. Published evidence on LLM directional
accuracy for multi-day equity moves from news clusters well under 0.7.

Position: do not build the tilt lane on today's evidence. What survives
into the roadmap:

- **The measurement comes first.** Mechanism (d)'s ledger records
  `direction_hit` per resolved proposal at zero risk (variance-only
  admission — a proposal may widen intervals, never move the point). If
  a proposer × event-type cell sustains **shrunk p̂ ≥ 0.70 over ≥ 50
  resolved calls** (frozen threshold, chosen above the k=0.5σ
  break-even with margin), the tilt lane becomes worth building —
  as a flag-off lane whose k is mapped from measured p̂
  (∝ (2p̂−1), hard-capped at 1σ; the 2σ oracle harm makes the cap a
  contract), never from stated confidence.
- **The design sketch is retained** for that contingency (sign + type
  as enum fields in `CONTEXT_RESPONSE_SCHEMA` per the `attributes.pop`
  discipline, `workflows.py:39-66,167,176`; σ = `_robust_scale` of the
  detrended history, `operators.py:43-46`; `directional_tilt` support
  state, counterfactual, one guarded ID key), but no code is written
  until the ledger produces a cell above threshold.
- **Corollary**: magnitude-bearing mechanisms — (b), and TSFMs if E2b
  surprises — are now the only news paths with headroom above single
  digits on this regime, because a sign is worth at most the break-even
  geometry allows.

### (d) Per-proposer calibration ledger — **BUILD, second** (it is the substrate for (b) and (c))

The audit (§1.2) shows both halves exist and are never joined. Spec, on
top of the existing store (new tables in `_TABLE_DEFINITIONS`,
`tracking.py:260-389`, schema v5):

- `event_proposals(proposal_id PK, project, forecast_id, series,
  event_id, event_type, proposer_id, proposer_kind, model_name,
  model_version, run_id, direction, confidence, known_at,
  effective_start, effective_end, created_at)` — requires widening
  `ContextEvent.created_by` into a structured proposer
  (`context.py:61`) and carrying it through
  `ContextAssessment.events_used` (`context_eval.py:107`), plus
  content-addressed event ids (today's `event_llm_{index:02d}` is
  positional, `workflows.py:191`).
- `event_admissions(proposal_id, admitted, lane, decided_by,
  mean_improvement, shrinkage, effect_shape)` — populated at register
  time from the `context_gate` / `future_context_gate` evidence that
  already exists (`pipeline.py:568-581,782-787`); today that JSONL is
  write-only.
- `event_outcomes(proposal_id, resolved_at, direction_hit,
  realised_lift, brier)` — written by `score_forecast` when actuals
  arrive. `direction_hit` needs only the actuals and the proposal;
  `realised_lift` needs the **persisted history-only counterfactual
  path** — the single hard gap: persist `forecast_base.csv` beside
  `forecast.csv` (the base points already exist at
  `adjudication.py:149-150` and `pipeline.py:778` and are discarded).
  `brier` scores the stated `confidence` against event-window
  occurrence, giving `confidence` a meaning for the first time.
- `proposer_skill(project, proposer_id, event_type)` view: shrunk hit
  rate `(n·p̂ + k·0.5)/(n + k)` (empirical-Bayes toward coin-flip,
  k ≈ 10) and shrunk lift — fixing, for this table, the no-shrinkage
  defect the leaderboard already has (`tracking.py:1019-1060`).
- Surfacing: `gnomon track proposers` CLI + MCP tool; **every artifact
  whose numbers a proposer influenced must carry that proposer's current
  shrunk skill and n in its evidence** — skill disclosed where it acts,
  not only in a separate report.
- Cold-start policy (shared with (c)): no resolved outcomes → no point
  influence, variance-only admission.

### (e) Disclosed mixtures — **DON'T BUILD standalone; it is the disclosure format of any influence lane**

The future-context lane already established the pattern a mixture needs:
the history-only counterfactual persisted in evidence
(`pipeline.py:788-805`) plus a distinct support state. If (c) is ever
built, a news-tilted forecast ships with its history-only counterfactual
and the tilt parameters (sign, k, σ, proposer skill) — that *is* the
disclosed mixture, with the mixing weight computed by Gnomon from
measured skill, never by the LLM; a (b) lane disclosing the population
distribution and its counterfactual is the same shape. A separate
mixture-of-forecasts object would add an artifact concept without adding
information; skip it unless a customer asks for tunable blending, and
revisit only with a measured case that blending beats the parameterized
influence it would blend.

### TSFM tier (cross-cutting from §1.4) — fix truthfulness before capability

Before any E2b-motivated promotion of the TSFM tier: fix `SANDBOX_ROOT`
(cwd bug), make sandbox workers load the pinned revisions the artifact
attests, and make `capabilities()["models"]["tsfm"]` reflect sandbox
installs. These are pre-existing trust bugs in exactly the "artifact
says X, runtime does Y" class this design is meant to eliminate. E2b
itself (does Chronos-Bolt at 30 points move anything?) stays registered
and cheap to run on a Hub-enabled machine.

---

## Roadmap (dependency-ordered)

1. **(a) Short-history honesty**: selection guardrail below 3 origins +
   `selection_fold_count` disclosure + lead-time interval widening.
   No dependencies; measured targets exist (E1/E4); closes most of the
   official MTBench gap without touching news.
2. **TSFM truthfulness fixes** (parallel with 1; small, independent):
   sandbox root, worker revision pinning, capabilities reporting. Then
   run E2b on a Hub-enabled machine (recipe: `gnomon tsfm install
   chronos_bolt_mini`, run `scripts/run_experiments.py` with the sandbox
   present, compare per-task pairs on matched folds).
3. **(d) Calibration ledger**: schema v5, structured proposer identity,
   persisted counterfactual path, `direction_hit`/`realised_lift`/
   `brier`, shrunk skill view, skill-in-artifact disclosure. Depends on
   nothing above; prerequisite for 4 and 5.
4. **(b) Population event studies**: first the corpus measurement study
   (pre-registered, MTBench raw corpus), build the `event_studies`
   lane only for event types passing the stability bar. After the tilt
   break-even result, this is the only news mechanism with measured
   headroom above single digits.
5. **(c) Volatility-capped directional tilts** — *contingent, not
   scheduled*: build only when the ledger shows a proposer ×
   event-type cell with shrunk p̂ ≥ 0.70 over ≥ 50 resolved calls
   (break-even at k=0.5σ is 0.654). If built: flag-off lane, k ∝
   (2p̂−1) capped at 1σ, `directional_tilt` support state,
   counterfactual, guarded ID key. Includes (e) as its disclosure
   format.

## Pre-registered hypotheses for build item 1 (ready to hand off)

To be re-registered verbatim in `results/short-history-guardrail/
HYPOTHESIS.md` at the implementing commit, before any run; the 50-task
surrogate and scripts in `results/news-regime-explore/` are the fixed
benchmark. Both mechanisms were simulated in this session
(`iterate_analysis.json`: guardrail 2.98 / 1.69%; √step-anchored-1
widening 89.7% pooled), so these are predictions that the *integrated
implementation* reproduces the simulation — an implementation that
misses them differs from the simulated mechanism in a way that must be
explained, not accepted. Thresholds frozen now:

> **H-G1 (guardrail closes the self-inflicted gap).** With the
> fold-starved selection guardrail on, gnomon-pure on the 50-task
> surrogate scores filtered mean MSE ≤ 3.0 and mean MAPE ≤ 1.8%
> (from 7.42 / 2.37%; floor is 2.56 / 1.61%), with abstention count
> unchanged (0) and wins-vs-floor not worse than 16/50.
> *Falsifier:* MSE > 3.0, or any new abstention, or a single task where
> the guarded forecast is > 10% worse MSE than the raw floor without a
> disclosed reason.
>
> **H-G2 (long history untouched).** On every series in the existing
> golden set with ≥ 4 rolling origins, artifacts are byte-identical
> (IDs and bodies).
> *Falsifier:* any diff.
>
> **H-G3 (intervals honest with lead).** With √step widening anchored
> at step 1 (the schedule selected by the second-pass rule search,
> which measured 89.7% pooled / 88% step-7 / 82% step-1 in simulation)
> active at short history, the implemented lane reproduces the
> simulation within noise: pooled q10–q90 coverage on the 50-task
> surrogate in **[74%, 93%]**, step-7 ≥ 60%, step-1 ≤ 88%.
> *Falsifier:* pooled < 74% (under-covers), pooled > 93% (blanket
> over-widening beyond the simulated schedule), step-7 < 60%, or
> step-1 > 88%.
>
> **H-G4 (disclosure names the power).** Every degraded-run artifact
> carries `selection_fold_count` and a `selection_underpowered` typed
> reason when the guardrail acted; `gnomon capabilities` states the
> guardrail's existence and default. *Falsifier:* any 30/7 artifact
> whose selection statistics appear without their n.

Prediction of the follow-on (not a commitment): item 1 alone should put
a re-run of the official MTBench Gnomon arm within ~0.5 MAPE points of
the LLM control (the guardrail simulation's 1.69% MAPE sits inside the
floor-to-control band), at which point the remaining gap *is* the news
edge — bounded, on this evidence, at a few tenths of a MAPE point —
and items 3–5 compete for it with measured, disclosed influence rather
than asserted confidence.

## Build outcome (added after implementation; see `results/short-history-guardrail/`)

Build item 1 was implemented in this branch over three registered
iterations, each falsification explained before the next fix was
registered. The mechanism that survived measurement differs from the
spec above in two instructive ways:

1. **The hard baseline lock became a fold-count-scaled margin**
   (`SINGLE_FOLD_SELECTION_MARGIN = 0.75`): the lock regressed the
   repo's own perfect-trend fixture (`daily_requests` → forced
   `last_value`, test coverage 0.43), and a 75% single-fold bar was
   measured to admit zero spurious wins on the 50 near-martingales
   while a deterministic trend clears it easily. This was the design's
   own registered alternative; the fixture picked between them.
2. **The √step widening was removed, not tuned.** H-G1's falsification
   isolated a *fourth* instance of the unevidenced-influence class this
   document is about: the median-residual recentring
   (`point_bias_correction`) is a ~1σ coin-flip tilt at ≤ 2 folds.
   Suppressing it alone restored near-nominal coverage (79.1% vs 80%),
   revealing that the 82→44% per-step decay that motivated the widening
   was mostly recentring wobble — and that whole-horizon pooled
   residuals already contain multi-step dispersion, so any lead-growth
   multiplier double-counts.

Final benchmark state: gnomon-pure MSE 7.42 → 2.995 (floor 2.56), MAPE
2.37% → 1.71% (floor 1.61%), pooled coverage 63.7% → 79.1% (nominal
80%); ≥ 4-origin runs byte-identical throughout. H-G1 and H-G5c were
falsified and are reported as such; H-G2/G4/G5a/G5b/G6/G7 hold. The
follow-on prediction above (official MTBench re-run within ~0.5 MAPE of
control) is now testable on any machine with the official data and an
LLM key.

## Roadmap status (as implemented on this branch)

1. **(a) Short-history honesty — DONE** (see Build outcome above and
   `results/short-history-guardrail/`).
2. **TSFM truthfulness — DONE**: sandbox root cwd bug, worker revision
   pinning, sandbox-aware `capabilities()["models"]["tsfm"]`, and
   `models.tsfm.candidates` threading (§1.4's findings 1–4). Still
   known-dead and deliberately untouched: `resolve_tsfm_backend`,
   `backends.sandbox.venv_root` / `auto_install` (finding 5) — wiring
   them is config plumbing, not truthfulness. **E2b remains blocked**
   here (Hub egress denied) and registered for a Hub-enabled machine.
3. **(d) Calibration ledger — DONE** (tracking schema 5):
   `event_proposals` / `event_admissions` / `event_outcomes`,
   content-addressed version-independent proposal keys, the persisted
   `enrichment_counterfactual`, realised-lift resolution on
   `submit_actuals`, shrunk `proposer_skill` (k = 10) via `gnomon track
   proposers` + MCP. Deviations from the §(d) spec, with reasons:
   proposer identity travels in `attributes.proposer` (caller-set,
   model-forgery discarded) rather than as a new `ContextEvent` field,
   because a dataclass field would enter `event.__dict__` and change
   the forecast-id payload of every event-carrying run;
   `direction_hit`/`brier` columns exist but stay NULL until proposals
   carry directions (mechanism c) or resolvable occurrence claims; and
   skill-in-artifact disclosure is deferred to the first lane that
   grants influence from skill — today no lane does, so there is no
   influenced artifact to disclose in, and disclosure lives on the
   tracking surfaces with an explicit "no proposal earns influence"
   warning.
4. **(b) Population event studies — NOT BUILT, gate unfireable here**:
   the prerequisite corpus study needs the HF-hosted MTBench raw
   corpus; egress policy denies the host.
5. **(c) Directional tilts — NOT BUILT, by design**: activates only on
   ledger evidence (shrunk p̂ ≥ 0.70 over ≥ 50 resolved directional
   calls). The ledger now exists to accumulate exactly that evidence;
   its `direction_hit` column is the gate's future input.
