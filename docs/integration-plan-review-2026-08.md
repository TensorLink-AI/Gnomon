# Integration plan — external design review, August 2026

Design only. Nothing here is implemented; Phase 1 begins after approval.

This plan reorders the review's suggested phasing. The reordering is not
a matter of taste: the suite was run end to end against GLM-5.2 on
2026-08-01/02, and the measurements contradict one of the review's
premises. The evidence is stated first so the ordering can be argued
with.

## 1. The measured starting point

Reproduce with `python -m benchmarks.report --root results/glm52`.

| Benchmark | Control | Aion | Paired result |
| --- | --- | --- | --- |
| TimeSage-MT (127 turns) | 68.5% | **73.2%** | 21 fixed / 15 broke, **p = 0.41** |
| MTBench (50 tasks) | **43.18 mean MSE** | 92.25 | Aion wins 12/50, **p = 0.0003** |
| TemporalBench (50 rows) | scores 50, worst MASE **10,753,522** | scores **2**, worst MASE 2.4 | not comparable at n = 2 |
| CiK (71 tasks) | not run (~$140) | **0 scored**, 36 abstained, 35 cache-failed | — |
| AnomLLM (400 series) | not run (~$18 / 22 h) | flagged nothing on 397, reported `supported` | — |

Four findings that bear directly on this plan:

1. **Aion mostly does not answer.** TemporalBench scores 2 of 50 rows;
   CiK scores 0 of 71. The causes are minimum-history rules and a
   row-level all-channels-or-nothing convention — not gate width, not
   interval width. Any benchmark delta from proposals A or B would be
   measured on ~4% of the data.
2. **The premise behind proposal A did not survive testing.** A assumes
   the admission gate is what suppresses context value. An `aion-tools`
   condition was built for MTBench: the model sees the prices *and* the
   article, calls Aion 6.9 times per sample, may propose events and
   compare runs. It submitted the plain history-only forecast in **48 of
   50 cases** and scored identically to the one-shot pipeline (11.06
   MSE). The model is not being blocked by a narrow gate; it is not
   producing admissible claims at all.
3. **What survived contact is not accuracy.** It is: no catastrophic
   outputs (control's worst TemporalBench error was seven orders of
   magnitude; Aion's worst was 2.4), 2.3× cheaper, 2.9× faster, and
   structural leakage safety. Only proposal C2 measures the last one,
   and the review buries it in Phase 3.
4. **A proxy can be confidently wrong.** Adding a trend detector
   (`local_slope`, F1 0.755 against AnomLLM's real labels where the best
   existing detector reaches 0.096) did not change the benchmark,
   because the synthetic grader that selects detectors scores
   `rolling_median_residual` at 1.00 on *planted* trend shifts and 0.096
   on real ones. Every phase below therefore carries a falsification
   criterion.

## 2. Invariants (unchanged from the review)

- Core stays stdlib-only. Heavy dependencies live in benchmark adapters
  and optional TSFM extras.
- Every number is computed deterministically or is absent. No LLM output
  becomes a forecast value, interval, metric, or selection decision.
- Every adapter or modelling decision is disclosed in artifacts and
  docstrings, in the existing style.
- Evaluation partitions stay disjoint: selection folds / calibration
  fold / report-only test fold.
- Abstention stays first-class, with typed reasons and repair options.
- The frozen v0.2 surface (`COMPATIBILITY.md`) gains fields and tools,
  never changed semantics.

## 3. Per-proposal design

### A. Context admission gate

Current shape, for reference: `context_model.event_effect` estimates one
additive shift from detrended history and **raises** when the event never
occurred in training (`context_model.py:20`). `context_eval.assess_context`
then stacks five conditions — mean fold improvement ≥ margin, a majority
of folds improved, the gain not confined to one fold, coverage not
degraded past `COVERAGE_DEGRADATION_LIMIT`, and at least four rolling
origins — and admits only if none fires.

**A7 — Gate instrumentation (moved to Phase 1). Effort: S.**
Modules: `context_eval.py`, `pipeline.py`, `runtime.py` (evidence
record). Every run emits a `context_gate` evidence record: events
supplied, eligible, admitted, and for each rejection the condition that
fired, with its measured value and threshold. `ContextAssessment`
already accumulates `reasons`; this makes them structured rather than
prose. Benchmark adapters gain an `oracle` condition where ground truth
exists (admit with true effects) to bound headroom.
*Why first:* it answers whether A1–A5 have anything to work on. If
admission is 0% because nothing arrives, widening the gate is wasted
work — which finding 2 already suggests.
Tests: gate decisions round-trip into the artifact; a rejection names
exactly one condition; oracle condition changes admission without
changing the recorded reasons for the non-oracle run.

**A1 + A2 — merged: typed numeric claims from context. Effort: M.**
The review separates constraint events (bounds, pins, monotonicity) from
known-future covariate magnitudes. They are one pipeline — extract a
typed numeric claim, validate it against history, apply it
deterministically — differing only in where the value lands. Designing
them apart yields two extraction paths and two validation stories.
Contract: **no new dataclass.** `ContextEvent.attributes` is already a
free-form dict (`context.py:55`); constraint claims live there under a
reserved `claim` key with a typed schema, and `event_type` gains a
reserved `constraint:` / `magnitude:` namespace. This keeps the frozen
tool surface untouched — a v0.2 client sending events without claims
behaves identically.
Application points: a constraint is a **projection of the emitted
quantiles** applied in `pipeline.py` after interval construction, never
an adjustment to the point path; a magnitude enters through
`covariates.py`'s existing fold-safe path as a future regressor level,
with the coefficient still estimated by Aion.
Rejection rule: a claim violated by history within its own scope is
rejected before use (e.g. a `min` bound that the training window already
breaches), disclosed with the violating timestamps.
Disclosure: each applied clamp is one `constraint_applied` evidence
record naming the claim, the steps it bound, and the pre-clamp value.
Risk: a clamp can make intervals non-monotone in the horizon; the
projection must preserve quantile ordering and be idempotent.
Tests: clamp is idempotent; violated-in-history claims are rejected;
clean runs byte-identical when no claims are supplied (goldens).

**A3 — Effect shapes. Effort: M.**
`context_model.py` grows from one additive shift to a small
deterministic family: level shift (current), transient with geometric
decay, and lead/lag ramp. Selection is the *same* identical-fold
ablation, so the shape is chosen by measurement, not by the caller. Each
shape is a pure function of history and flags.
Risk: three shapes on the same folds is three times the fitting, and
raises the multiple-comparisons problem the single-fold guard already
worries about — the "gain confined to one fold" condition must apply per
shape, and the winner must beat base, not merely beat the other shapes.

**A5 — Shrinkage admission. Effort: M.**
Replace binary admit/reject with a deterministic shrinkage factor
λ ∈ [0, 1] applied to the estimated effect, derived empirical-Bayes
style from the dispersion of fold improvements. The five conditions
become inputs: they scale λ instead of vetoing. λ ships in the artifact
and in the `context_gate` record.
*Conflict with an invariant, and its resolution:* shrinkage makes
admission continuous, so "admitted" stops being a clean boolean in the
frozen result shape. Resolution: keep `context.admitted` as
`λ > 0` for v0.2 readers, and add `context.shrinkage` alongside. No
existing field changes meaning.
Risk: λ derived from ≤ 4 folds is itself noisy; needs a floor below
which λ is pinned to 0 rather than a small non-zero effect that no
evidence supports.

**A6 — Conditional (future-only) path (moved to Phase 2). Effort: M.**
Events that fail backtest admissibility can still produce a clearly
separated conditional forecast alongside the unconditional one.
Contract: a new `conditional_forecasts` list on the result, each entry
carrying its own `support: conditional_on_event`, the assumptions, and
the events it is conditioned on. The unconditional forecast remains the
value of every existing field — a v0.2 client that ignores the new key
sees exactly what it sees today, which is what makes this safe under the
freeze.
*Why Phase 2, not 4:* it is the only proposal that converts an
abstention into an answer, and TemporalBench T4 is event-conditioned by
construction.

**A4 — Analog pooling (Phase 4, as a spike with a kill criterion).
Effort: L.**
Storage: `tracking.py` schema v4 adds an `event_effects` table
(event_type, series fingerprint, estimated effect, fold count, outcome
known_time). Leakage rule, which must be a **test and not a docstring**:
an analog is usable at a fold cutoff only if its own outcome was known
by that cutoff — enforced through the same `Snapshot` access path that
already makes post-`as_of` reads structurally impossible.
Kill criterion: if Phase 1 instrumentation shows event supply below ~20%
of tasks, pooling has nothing to pool and the spike stops.

### B. Uncertainty

**B1 — Per-lead-time residuals + split conformal, and a double-widening
audit (Phase 1). Effort: M.**
`evaluation.interval_bounds` (`evaluation.py:51`) shifts by the median
residual and scales the spread by `step ** 0.5`. The residual quantiles
it receives are pooled across *all* lead times of a horizon-h
calibration fold (`context_eval.py:203` builds them exactly this way),
so they already contain lead-time growth — and are then widened by
√step again. **This looks like a genuine double-widening bug, not a
design choice**, and it is worth confirming before any conformal work:
if intervals are systematically too wide, measured coverage overstates
calibration quality and the gate's coverage veto is mis-tuned.
Design: collect fold residuals indexed by lead time h, take conformal
quantiles per h, and fit an isotonic (monotone non-decreasing) spread-
vs-h curve where folds are sparse. Monotonicity is the honest prior;
it also prevents a sparse-fold artefact from producing intervals that
narrow with distance.
Tests: coverage on held-out folds within tolerance of nominal; spread is
non-decreasing in h; a synthetic series with constant noise reproduces
constant per-h spread (i.e. no √h inflation).

**B2 — Distributional selection loss. Effort: S.**
Pinball / weighted quantile loss added to fold scoring, and made the
selection criterion whenever the task requests quantiles. The current
point loss is reported alongside, never removed.
Risk: selection can now differ from today's for the same data, so
goldens shift. That is a real change and belongs in COMPATIBILITY as
such — a run that requests no quantiles must be unchanged.

**B3 — More quantile levels. Effort: S, but wide.**
Emit 9 levels (0.05…0.95). **Frozen-surface note the review omits:**
`q10/q50/q90` keep their exact meaning and remain present in
`forecast_rows` and `forecast.csv`; new levels are additional keys and
columns. Threads through `contracts.py`, CLI, MCP, and the CiK
sample-conversion adapter, whose clamped q10/q90 tails currently
understate spread.

**B4 — Adaptive conformal via tracking (Phase 4). Effort: L.**
ACI-style width updates per project from realized coverage in
`tracking.py`. The hard part is not the update rule but determinism and
replay: the same inputs at the same `as_of` must produce the same
interval forever. Design: the adaptation state is itself bitemporal —
each update carries the `known_time` of the outcome that caused it, and
a run at `--as-of T` reads only updates known by T. Anything else breaks
replay.

**B5 — Interval-aware coverage guard. Effort: S.**
The gate compares point estimates of coverage measured on a single test
fold of h points. Put a binomial interval around it and trigger the veto
on the interval, not the estimate. Small change, directly reduces
false vetoes on short horizons. Fold into whichever phase touches
`context_eval.py`.

### C. Benchmark methodology

**C1 — Abstention-robust summaries (Phase 1, reduced scope). Effort: S.**
The review asks where cross-condition matching should live. **That half
is done:** `benchmarks/report.py` (added 2026-08-02) joins arms on task
id, reports matched-subset means, runs paired McNemar / sign tests, and
refuses comparisons whose manifests disagree on benchmark or target.
What remains is the **penalized mean**: each abstention imputed at a
mandated fallback's score (seasonal-naive on the same task). That is a
policy decision — it makes abstention cost something, which is the only
way the TemporalBench result can be read as a number rather than a
footnote.

**C2 — Leakage-trap eval family (moved to Phase 2 — the flagship).
Effort: L.**
`docs/agent-evaluation.md` already promises this family. Tasks built on
revision-heavy series (`examples/messy_requests_revisions.csv`) where
peeking past the cutoff measurably improves the score; control LLM vs
Aion treatment; report the leakage differential.
The assertion that makes it more than a benchmark: "provably could not
leak" is an assertion over the snapshot access log — the run's
`snapshot_access` evidence records the maximum `known_time` touched, so
the grader asserts that maximum ≤ the fold cutoff. The control has no
such structure, and the trap is designed so that leaking *helps*: a
control that scores above the no-leak ceiling has demonstrably peeked.
*Why it is the flagship:* it is the only proposal that measures the one
claim that survived this session's benchmarking.

**C3 — TSFM-enabled headline configs. Effort: S.** Config work plus the
sandbox install path; stdlib pool stays as the floor condition.

**C4 — AnomLLM `aion-agent`. Recommendation: drop for now.** It costs
LLM budget to test a path whose non-LLM version currently selects a
detector that cannot see the dataset's anomaly class. Fix selection
first (see §4 new item N3) or the condition measures nothing.

### D. Housekeeping

- `pytest -q` collection under pytest 9 (root `__init__.py` Hermes
  shim): guard the relative import; verify the CI matrix. **S**
- Stale docstring in `src/aion/context.py` ("No v0.1 pipeline consumes
  events yet"). **S**
- README section on the relation to AION (Zhan et al., arXiv:2605.25045)
  and TimeClaw (arXiv:2606.05404): theirs is agent-side scaffolding with
  LLM review; this is the deterministic execution actor with structural
  leakage safety. **Do not rename** — produce a rename impact inventory
  (code, package metadata, CLI strings, MCP tool prefixes, docs) for a
  human decision. **S**
- **New:** the benchmark suite is absent from CI. Its dependencies were
  undeclared until `benchmarks/SETUP.md` and
  `benchmarks/requirements/*.txt` (2026-08-02); CI should at least run
  `benchmarks/tests/` and validate every config with `--dry-run`. **S**

## 4. New items the review does not contain

**N1 — Abstention-policy review. Effort: M. Phase 1. MEASURED 2026-08-02;
conclusion below reverses the hypothesis.**
The highest-leverage item in this document, and it is not a proposal in
the review. Aion scores 2 of 50 TemporalBench rows and 0 of 71 CiK
tasks. Two specific questions:
(a) *Row-level all-or-nothing.* A TemporalBench row scores only if every
target channel forecasts; one short channel voids the row. Should a
partially-forecast result emit the channels it can, marking the rest
abstained? The artifact already supports per-series support states.
(b) *Is the minimum calibrated or merely conservative?* The floor comes
from `supportable_horizon` requiring separated folds
(`evaluation.py:68`). CiK tasks carry 6 observations; the refusal is
defensible, but nothing measures what a forecast at that length would
actually have scored. Measure it once — run the harness floor with the
minimum lifted, off the record, and compare to the mandated
seasonal-naive fallback. If a refused forecast would have beaten the
fallback, the floor is too high and is costing real answers.
This does not weaken abstention; it calibrates it with evidence, which
is the same standard every other decision in the system is held to.

**Result of the measurement — the floor is not the problem.**

(b) was tested two ways. On 200 synthetic series per cell, forecasting
below the floor beats a seasonal-naive fallback only 15-51% of the time
at 16-20 observations: refusing there is well founded, not timid. Then
the real distribution: across TemporalBench's 300 target channels, the
median channel carries **1.67 observations per requested horizon step**,
21% carry *less history than horizon*, and **not one** of the 300 has the
4x horizon that separated rolling folds need. These rows do not ask for a
forecast a lower floor would license; they ask for a 69-step forecast
from 50 observations. Lowering the floor would produce unvalidated
numbers, which is the one thing this system exists not to do. **Do not
lower the floor.**

(a) is where the loss actually is, and it is now quantified. Degraded
mode needs only `horizon + 2` observations, which **76% of channels**
satisfy — yet only 4% of rows scored, because a TemporalBench row scores
only when *every* target channel forecasts. The all-or-nothing row
convention, not the floor, is what turns 76% into 4%. Aion already emits
per-channel results with per-series support states; the benchmark's
metric is what demands completeness. So the fix is reporting, not
policy — which is what C1's penalized mean now delivers: charged at the
baseline's own score, TemporalBench reads **11.39 -> 11.41 sMAPE**
(roughly neutral) rather than "2 of 50".

**N2 — Falsification criterion per phase.** Stated in §5.

**N3 — Detector-selection proxy validity. Effort: M. Phase 3 or explicit
non-goal.** Not in the review, now a known gap: the synthetic grader
selects `rolling_median_residual` (1.00 on planted trend shifts, 0.096
on real ones) over `local_slope` (0.755 on real ones). Either injection
fidelity improves until the proxy predicts real performance, or
selection admits it cannot rank detectors for anomaly classes it does
not plant, and exposes `--detector` for callers who know. The disclosure
half already shipped (`graded_families`).

**N4 — Cost and wall-clock budget per phase.** The review authorises
benchmark deltas without authorising spend. Rates measured this session
with GLM-5.2: ~$0.04 and ~143 s per reasoning-model row; 50-row arm ≈
2 h and ~$3; CiK control ≈ **$140**; AnomLLM control ≈ **$18 / 22 h**.

## 5. Phase plan

Each phase states what must pass, what must be reported, and **what
result would mean the phase was the wrong bet**.

### Phase 1 — Measure before widening

A7 gate instrumentation · N1 abstention-policy review · B1 conformal +
double-widening audit · C1 penalized-mean summaries · B5 · D
housekeeping.

*Acceptance:* every run emits a `context_gate` record whose rejection
reasons are structured and round-trip through `lineage.json`; per-lead-h
conformal coverage within tolerance on held-out folds, with the √h audit
resolved either way and written down; `summary.json` carries scored-only,
matched-subset, and penalized means; goldens refreshed only for changes
recorded in COMPATIBILITY; full suite green on 3.11–3.13.
*Report:* admission rate and rejection histogram across all five
benchmarks; TemporalBench and CiK answer rates before/after N1.
*Falsified if:* admission rate is already high and rejections are spread
evenly across conditions — then the gate is not the bottleneck in the way
A assumes, and Phase 3 should be rescoped or dropped.

### Phase 2 — Prove the differentiator

C2 leakage-trap eval · A6 conditional path.

*Acceptance:* trap tasks where leaking measurably helps; the grader
asserts max touched `known_time` ≤ fold cutoff from the snapshot access
log; control vs treatment leakage differential reported with a paired
test from `benchmarks/report.py`; conditional forecasts appear as a new
key with v0.2 readers unaffected (round-trip test).
*Report:* leakage differential; how many previously-abstaining
event-conditioned tasks now return a conditional answer.
*Falsified if:* the control does not leak even when leaking would help —
then structural safety is protecting against a failure that frontier
models no longer commit, and the claim needs restating as a guarantee
rather than an advantage.

### Phase 3 — Widen, conditioned on Phase 1

A1+A2 merged · A3 effect shapes · A5 shrinkage · B2 pinball · B3
quantiles · N3.
**Branch:** if Phase 1 shows events rarely *arrive* (finding 2), the
bottleneck is extraction, not admission — in that case do A1+A2 (which
give the model something concrete and checkable to emit) and defer
A3/A5.

*Acceptance:* clean runs byte-identical with no claims supplied;
constraint clamps idempotent and quantile-order preserving; λ present in
artifacts; pinball selection changes nothing for quantile-free requests.
*Report:* admission rate before/after; MTBench and CiK deltas on the
matched subset.
*Falsified if:* admission rises substantially and benchmark scores do
not move — then admitted context is not carrying signal, and the
remaining work is extraction quality, not gate machinery.

### Phase 4 — Speculative

A4 analog pooling (spike, kill criterion above) · B4 adaptive conformal ·
C3 TSFM headline configs.

*Acceptance:* analog leakage rule enforced by test at fold cutoffs;
adaptive state bitemporal and `--as-of` replay byte-identical.
*Falsified if:* replay determinism cannot be preserved under adaptation —
then B4 is incompatible with the replay guarantee and should be dropped
rather than weakened.

## 6. Open questions for a human

1. **Does abstention get a price?** The penalized mean (C1) makes
   refusing cost something. It is the difference between "Aion scored 2
   of 50" reading as caution or as failure. This is a positioning call,
   not a technical one.
2. **May a partially-supported row answer?** (N1a) Emitting 4 of 6
   channels with 2 marked abstained is more useful and less clean.
3. **Is the history floor negotiable at all**, if measurement shows a
   refused forecast would have beaten the mandated fallback? (N1b)
4. **Budget for Phase 2.** The leakage eval needs a control arm; at
   measured rates a 50-task control is ~$3 and ~2 h, but the trap family
   may need more tasks to separate.
5. **Is B2 worth a golden refresh?** Pinball selection changes model
   choice for quantile-requesting runs — a real behaviour change under
   the freeze, defensible but it must be a decision, not a side effect.
6. **N3: fix the proxy or admit the limit?** Improving injection
   fidelity is research; exposing `--detector` is an afternoon.
