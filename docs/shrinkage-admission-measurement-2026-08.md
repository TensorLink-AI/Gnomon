# Should context admission be continuous rather than binary?

**Verdict: no — and unusually, the evidence is significant against it, not
merely absent.** `assess_context(..., shrink=True)` exists and λ is always
computed and disclosed; applying it stays off.

Third in the series with
[`fold-stride-measurement-2026-08.md`](fold-stride-measurement-2026-08.md)
and [`selection-loss-measurement-2026-08.md`](selection-loss-measurement-2026-08.md).

## The argument for changing it

A binary gate at a fixed margin is high variance when there are four folds to
decide on: a candidate just under the line gets nothing, one just over gets
everything, on a difference well inside the noise. The empirical-Bayes answer
is to keep the effect in proportion to how much of the observed mean could be
sampling noise:

    λ = max(0, 1 − (standard error / mean)²)

pinned to zero below 0.1, because a small non-zero effect that no evidence
supports is worse than none — it moves the answer while being indefensible if
asked why.

## What stays binary regardless

λ governs **strength**, never **validity**. Whether events are eligible,
whether separated folds exist, whether the candidate fits every fold, and
whether interval coverage degraded are correctness conditions and remain hard
vetoes. No amount of measured improvement makes a leaking event admissible,
and shrinkage must never become a route to partially admitting one.

## Measurement

120 synthetic series with planted, recurring events of known size (2, 6, 15)
against two noise levels, drift-like so that the context candidate can
actually win — on seasonal series it never gets admitted at all, for the
reason recorded in `context_model.event_adjusted`. Each arm scored on a
genuinely held-out horizon.

| | binary gate | shrinkage |
|---|---|---|
| admitted | 28 / 120 | 21 / 120 |
| mean held-out WAPE | **0.0334** | 0.0374 |

Of the 28 series where the two arms differed, shrinkage produced the better
forecast on **5**. Sign test **p = 0.0009**. λ where admitted: median 0.64,
range 0.16–0.87.

## Reading

This is not a null result — it is significant evidence that shrinkage makes
the forecast worse here, and the reason is structural rather than incidental.

Shrinkage is the right instrument when admission is otherwise unconditional,
because then the admitted set contains candidates whose apparent gain is
noise. Aion's gate is not unconditional: a candidate only reaches λ after
already clearing a mean-improvement margin, a majority-of-folds condition,
and survival of the single best fold's removal. Those three conditions have
*already* removed the candidates whose effects are noise. Applying shrinkage
on top discards a median 36% of effects that the gate has independently
established are real, and the 7 candidates it additionally rejected via the
λ floor were ones the strength conditions had passed.

In other words the two mechanisms answer the same question, and stacking them
double-penalises. If shrinkage were ever adopted it should **replace** the
three strength conditions rather than compose with them — which is a
different and larger change, and would need its own measurement.

## What would change the verdict

Running shrinkage as a replacement for the strength conditions rather than in
addition to them: admit on validity alone, then let λ decide how much effect
survives. That is the version the proposal was really arguing for, and this
measurement does not test it. It should be measured on the same corpus before
being preferred, and it would need care that λ ≈ 0 and "rejected" stay
distinguishable in the artifact.
