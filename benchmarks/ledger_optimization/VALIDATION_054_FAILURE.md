# Validation054 stopped on an exact-fit convergence certificate

All533 raw cases completed:117 warm-up and416 scored tasks,12,792 forecast
computations and6,396 estimator fits. During subsequent mixture fitting the
runner stopped at electricity:T298, round20, global CV anchor. It had completed
165 scored mixtures/495 weight fits; the496th fit started and failed its
certificate. No comparative aggregate or interval was computed.

SLSQP reported success after12 iterations, almost all weight on weekly. The
training objective was0.05001253880616671 versus0.14305213663563335 initially,
but the conservative convex bound was0.05497901247592551, above1e-5. Near-zero
residuals contributed only5.94e-13 to the existing omitted-norm term. Solver
success alone does not meet the frozen rule. Preserve this as a failed run.
Elapsed1090.37 seconds;0 API calls. Full receipt/archive:
[evidence/broad-validation-054-failed.json](evidence/broad-validation-054-failed.json).

The failure is in development numerical mixture fitting, not a Gnomon provider
failure. Investigate a numerical correction on synthetic exact-fit examples.
Any correction must retain the exact training objective, independent<=1e-5
suboptimality certificate, both control arms, task set and bootstrap protocol.
Freeze it separately, then uniformly recompute all mixture fits from the saved
raw forecasts. Do not regenerate forecasts, drop this case, substitute a
fallback, loosen the acceptance threshold or optimize after looking at held-out
aggregate scores. Label a corrected run as a disclosed numerical amendment,
not the uninterrupted preregistered run. Final-reserved data remains unopened.
