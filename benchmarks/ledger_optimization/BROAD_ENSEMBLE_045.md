# Solver refinement 045; failed attempt 044 preserved

Attempt 044 stopped after 407 completed cases. SLSQP reported success on the
ledger fit for pedestrian sensor 61, round 24, but the independently calculated
convex gap bound was 1.1271439576e-5, above the frozen 1e-5 threshold. Retain
that failure and its 816 started fits. No aggregate forecasting comparison was
computed or used to choose this correction.

Freeze a separate rerun with **ftol=1e-12** instead of 1e-10. Keep maxiter=500,
the exact objective, analytic gradient, initializer, weights, cohort rules,
all 416 tasks and **the same 1e-5 acceptance threshold**. Recompute every fit
under this one setting; do not selectively replace the failed fit. Preserve
both sets of computations and their costs. If any fit still fails, stop again.

This is a numerical implementation correction, not an improved ledger policy.
All scoring, controls, diagnostics, source hashes, temporal assumptions and the
20% promotion bar remain exactly those of 044. Do not infer product reliability
or statistical superiority from a solver-success flag. Freeze implementation
before executing the rerun.
