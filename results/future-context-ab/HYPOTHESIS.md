# Pre-registered hypothesis: verifiable future-context events on CiK

Registered 2026-08-03, at commit `483f28b` (main), **before** any
implementation or run of the treatment. This file states what we predict
and what would falsify it; `RESULTS.md` will be written against these
predictions, not adjusted to fit whatever comes out.

## Background

A full Context-is-Key run (`results/deepseek-v4-flash/`, regenerable;
summary numbers restated here because that directory is not versioned)
measured a structural gap, not a tuning problem:

- The `gnomon-agent` arm scored **bit-identical to the no-LLM
  `gnomon-pure` arm**: mean RCRPS **0.391**. The admission gate rejected
  **100% of proposed events across 141 runs**. Events that survived to
  the final margin check ablated **30–90% worse** on the held-out folds.
- The ungated LLM `control` arm scored mean RCRPS **≈ 0.064** and won
  **84 of 85** matched task-seed runs.

The gate is behaving as designed: fold ablation is the only warrant it
understands, and CiK's essential context is mostly future-dated
information with no historical precedent (planned maintenance windows,
stated bounds, announced interventions). Such events cannot prove
themselves on historical folds by construction.

## Intervention

Two typed context-event classes admitted by **textual verifiability**
instead of fold ablation, behind `context.future_events` (default off):

1. **Constraint events** — bounds on future values, parsed
   deterministically from a source span quoting the context text, checked
   against recent history, applied as a projection (clamp) of Gnomon's
   own quantile paths. A constraint never invents a value.
2. **Deterministic override events** — a stated future state for a window
   ("offline Tue–Thu" → 0), value taken from the text and never
   estimated, applied by setting the affected horizon steps to the stated
   value with widened intervals at the window boundaries.

The LLM still never writes a forecast number: it selects and quotes
spans; a deterministic parser extracts every number that is applied.
Fold-testable events keep going through the existing ablation gate, and a
fold-tested failure stays rejected. Influenced forecasts are disclosed
with a distinct `context_trusted` support state and a history-only
counterfactual in the evidence.

## Primary hypothesis

With `context.future_events: on`, the `gnomon-agent` arm on CiK (same
model, same seeds, matched task-seeds against the existing
`results/deepseek-v4-flash/cik-gnomon-agent` run as the before):

> **H1.** Mean RCRPS on matched task-seeds closes **at least 50% of the
> gap** between the current 0.391 and the control's matched mean
> (≈ 0.064): treatment mean RCRPS **≤ 0.2275** (0.391 − 0.5 × 0.327),
> with
> - **zero new constraint violations** (the RCRPS constraint-violation
>   penalty component does not fire on any run where it did not fire in
>   the flag-off arm), and
> - **abstention and error counts not worse** than the flag-off arm.

## Secondary predictions

- **H2 (the lane fires).** At least one event of each class is admitted
  somewhere in the run. If per-class admission counts are zero, H1 is
  not vacuously saved by inaction — the experiment failed and RESULTS.md
  must say so.
- **H3 (constraints are the cheap win).** Constraint events account for
  the larger share of admissions, because CiK's constraint-satisfaction
  tasks state bounds verbatim, while overrides require a stated window
  *and* a stated value.
- **H4 (no harm where the lane is silent).** On task-seeds where no
  future-context event is admitted, per-run RCRPS is unchanged from the
  flag-off arm (the lane must be a strict no-op when it admits nothing).

## Falsifiers

- Treatment mean RCRPS on matched task-seeds > 0.2275 → **H1 falsified.**
- Any run where an admitted constraint or override *increases* that
  run's RCRPS constraint-violation penalty relative to flag-off → the
  "zero new violations" clause is falsified, and each such run is
  reported as a harm case.
- More abstentions or errors than the flag-off arm → H1 falsified.
- Any admitted event whose applied number did not come from a quoted
  span (audited from the artifacts) → the design's core invariant is
  violated; the result is void regardless of score.

## Analysis plan

- Compare on **matched task-seeds only** (both arms scored), as
  `gnomon eval compare` does; report abstentions/errors alongside means,
  never separately.
- Report per-class counts: proposed, admitted, rejected (by reason).
- Report harm cases individually (any matched run where treatment RCRPS
  is worse by > 0.01), whatever the means say.
- No post-hoc filtering of tasks, seeds, or event classes. If the run
  cannot be executed in full, RESULTS.md states exactly what ran.

## Amendment 1 (2026-08-03, before the next treatment run)

Gate diagnostics from the first treatment run's `future_context_gate`
evidence motivated two changes to the **intervention** (not to any H1
threshold, which stand as registered):

1. **Forward-scoped bounds bypass the history check.** CiK's
   constraint tasks state bounds in the voice "Suppose that in the
   forecast, the values are bounded above by 5.45" — a claim about the
   prediction window only, whose premise is that history sits elsewhere.
   `recent_history_respects_bound` was rejecting these (observed on
   13–17 runs, concentrated in the worst-scoring task class), which is a
   misreading of the claim's scope, not a suspect claim. A span that
   scopes its bound to the forecast window now skips the consistency
   check, with the skip recorded in the gate evidence.
2. **Defensive projection of rejected-but-verbatim bounds.** RCRPS
   penalises emitting values that violate a bound stated in the context
   text regardless of whether we believed it. A constraint whose claim
   fails admission but whose span states the bound verbatim is now
   recorded as `defensive` and the emitted quantiles are projected onto
   it: no admission, no support change, its own
   `future_context_defensive` evidence and `defensive_bound_applied`
   disclosure. Rejection still settles belief; it no longer forces the
   published rows to contradict the context text.

**H4 is restated accordingly** (same intent, new letter): on task-seeds
where the lane neither admits an event *nor applies a defensive
projection*, per-run RCRPS is unchanged from the flag-off arm. A
defensive projection that fires is expected to *reduce* the RCRPS
constraint-violation term; any run where it increases that term relative
to flag-off is a harm case and is reported individually, exactly as the
H1 violation clause already requires for admitted events.

Rejection records now carry the rejected `source_span`, so the next
run's span rejections are triageable
(`benchmarks/cik/triage_future_context.py`) into prompt-side vs
parser-side causes without re-running the proposer.

## Out of scope (future work, deliberately not built)

Soft directional effects ("demand will increase"), cross-series analog
transfer, and proposer-trust calibration are riskier — each puts an
unverifiable LLM judgement closer to the numbers — and are excluded from
this experiment by design.
