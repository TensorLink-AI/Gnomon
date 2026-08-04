# Pre-registered hypothesis: a proposer-trust warrant for directional claims

Registered 2026-08-04, at commit `e442593` (claude/gnomon-harness-issues),
**before** any implementation or run of the treatment. This file states
what we predict and what would falsify it; `RESULTS.md` will be written
against these predictions, not adjusted to fit whatever comes out.
`results/future-context-ab/HYPOTHESIS.md` deliberately excluded this
design as "riskier"; this registration is the separate experiment that
exclusion pointed to, now that its prerequisite — the per-proposer
calibration ledger (`gnomon_proposer_skill`, tracking schema 5) — exists.

## Background

The 2026-08 benchmark runs measured a warrant gap in three layers:

- Fold ablation admitted **0 context events across 141 proposing runs**
  (it requires historical precedent, which future-dated claims lack by
  construction).
- The textual-verifiability lane (PR #42) reached **12 of 355 runs**; the
  span parser rejected 176 of 220 proposals.
- A residual claim class has **no possible warrant in either lane**:
  directional or magnitude-unknown statements — "demand will be higher
  than usual next week", "a promotion runs Friday". No number to parse,
  no precedent to ablate. Stated *closures* are **not** in this class:
  zero-state words already parse as an override of 0
  (`benchmarks/tests/test_classify_rejections.py` pins this).

The MTBench news regime hit the same gap independently
(`docs/design/news-regime.md`).

## Measurement precondition (the experiment does not start without it)

Run `benchmarks/cik/classify_rejections.py` over the CiK proposing dumps.
Dumps written before rejected spans were recorded (commit `e442593`) are
unclassifiable; if they dominate, one flag-on proposing rerun is made
first to generate classifiable gate records — proposals and admission
only, no scoring needed.

Proceed only if **`non_numeric_claim` accounts for ≥ 25% of span-parse
rejections, or ≥ 40 instances** across the dumps. Below that, the
observed pain is parser narrowness or malformed proposals — the roadmap
is a grammar or prompt fix, and this experiment is deferred, not run
anyway because the design is already written.

## Intervention

A third typed event class, `direction:<label>`, behind
`context.directional_events` (default off), admitted by **proposer
trust** and applied to **uncertainty only**:

1. **Effect: quantiles, never the point.** An admitted directional event
   reshapes the emitted quantiles asymmetrically inside its window: the
   quantiles on the claimed side move away from the median by at most a
   fixed fraction (**25%**) of that step's existing interval half-width;
   quantiles on the other side are unchanged. The point path and q50 are
   **byte-identical** to the uninfluenced run. The say-so of a model can
   widen doubt in a stated direction; it can never move a number.
2. **Warrant: the calibration ledger.** Admission requires the proposing
   identity's ledger to show **≥ 10 resolved prior proposals** with
   calibration at or above the ledger's neutral line. No ledger, or a
   ledger below the line → rejected, with the ledger quoted in the
   rejection. The warrant is earned from realised outcomes, not asserted.
3. **Disclosure.** Support drops to `context_trusted`; the ledger
   snapshot that justified admission enters the evidence beside a
   history-only counterfactual, and admitted events enter the artifact ID
   payload. Flag-off artifacts stay byte-identical.

Directional claims quote a `source_span` like the existing lane; the
parser verifies the span states a *direction* (a curated pattern list:
higher/lower/increase/decrease/above usual/below normal), never a number.

## Primary hypothesis

On the benchmark arm the precondition was measured on (CiK matched
task-seeds, or the news-regime set if the census points there), with the
flag on and the flag-off arm as the before:

> **H1.** On matched task-seeds where at least one directional event is
> admitted, mean RCRPS improves by **≥ 5%** relative to flag-off, with
> - the **point path unchanged on every run** (audited from artifacts —
>   this is an invariant, not a hope),
> - **no new constraint violations**, and
> - **abstention and error counts not worse** than flag-off.

## Secondary predictions

- **H2 (the lane fires).** At least 5 directional admissions across the
  run; zero admissions means the experiment failed regardless of means,
  and `RESULTS.md` must say so.
- **H3 (the ledger gate does work).** At least one proposal is rejected
  *solely* on ledger grounds. If the gate never fires, trust was not
  tested and the warrant claim is unsupported even if H1 holds.
- **H4 (no harm where silent).** On task-seeds with no directional
  admission, per-run RCRPS is bit-identical to flag-off.

## Falsifiers

- Improvement on admitted-event runs < 5%, or mean RCRPS worse → **H1
  falsified**; the uncertainty-only effect is not worth its complexity.
- Any run where the point path or q50 differs from flag-off → the core
  invariant is violated; **the result is void regardless of score**.
- Any admitted event from a proposer below the ledger floor (audited) →
  void.
- More abstentions or errors than flag-off → H1 falsified.
- Precondition census < 25% and < 40 instances → the experiment is
  **not run**; recording that outcome in `RESULTS.md` closes this
  registration honestly.

## Analysis plan

- Matched task-seeds only; abstentions and errors reported beside every
  mean, never separately.
- Report: proposals, ledger rejections, span rejections, admissions;
  harm cases (matched run worse by > 0.01) listed individually.
- No post-hoc filtering of tasks, seeds, proposers, or directions. If
  the run cannot execute in full, `RESULTS.md` states exactly what ran.
- Dev-set discipline: CiK has absorbed three days of iteration. The
  effect must be confirmed on a split not used while building the lane
  (held-out CiK task classes or the news-regime set) before any claim
  leaves this directory.

## Out of scope (deliberately, still)

Magnitude or location effects from unverified say-so — even for a
well-calibrated proposer — cross-series analog transfer, and any effect
on the point path. Each is a separate registration if this one survives.
