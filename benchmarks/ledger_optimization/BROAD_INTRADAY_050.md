# Shared intraday model mixtures 050

Freeze this protocol, implementation and synthetic tests before scoring. Prior
development findings are known. This is a new shared action, not a relabelled
success of an earlier policy or a release change.

Question: do historical errors help choose different model mixtures for recurring
parts of the forecast day? Use the same 416 tasks, six fixed original forecasts,
source/recording cutoffs and sixteen same-domain neighbors frozen in 047. Keep
all current task actuals out of training. Do not read 046 hindsight weights or
reserved future observations. All lead-hour checks from 049 apply.

Both arms can use four sets of six simplex weights, one for each consecutive
six-hour block (leads 0–5, 6–11, 12–17, 18–23). Combine in log1p space as before.
The anchor and initializer are the unchanged 045 current-CV ensemble weights,
repeated across all four blocks. Optimize weighted mean case RMSLE plus
`0.01 / 4 * sum_block sum_model((weight - anchor)^2)`.
The penalty shrinks towards the common strong control. The 0.01 penalty and
four-block partition are fixed, with no outcome-driven parameter sweep.

Control training: three current CV folds with masses 1/3. Ledger training:
current CV total mass 1/2 plus the sixteen earlier production episodes total
mass 1/2. Both arms use the same optimizer, regularizer, initializer and action.
Training uses the mean smooth norm `sqrt(mean_squared_log_error + 1e-12)`.
Its deviation above RMSLE is bounded by 1e-6 per case. This avoids an unstable
subgradient at exact fits. Final reported scores use exact mean case RMSLE,
without smoothing or the penalty. The correction was motivated by a synthetic
exact-fit test before the implementation was frozen or development data scored.

SLSQP, exact analytic gradients, ftol=1e-14, maxiter=500. Each block has weights
in [0,1] summing to one. Require finite feasible output, successful termination,
objective no worse than initializer by more than1e-8, and a convex gap bound
<=1e-5. The bound sums blockwise `w·gradient - min(gradient)` for the smooth
convex objective. No near-zero case gradient is omitted. The corresponding
unsmoothed objective is within an additional 1e-6 of this bound.
Before acceptance, allow at most32 convex line refinements towards the minimum-
gradient simplex vertex in each block. Solve the directional-derivative root
with Brent's method (xtol1e-15, rtol1e-14), or choose the endpoint if still
descending. Stop once the same1e-5 gap passes. Record every step and initial gap.
This bounded numerical correction follows the synthetic exact-fit failure;
it does not change any training data, objective, or acceptance threshold.
Normalize only tolerated floating-point violations (minimum weight >=-1e-8,
block sum within1e-8). Any failure stops the cohort and is preserved; do not
silently loosen tolerances or drop a failed task.

Primary comparator: CV-only block mixture. Also compare to the unchanged 045
global CV ensemble. Promotion requires >=20% lower mean RMSLE than each control
overall, and positive reduction against each within electricity and pedestrian
domains. This guards against apparent ledger wins due to a more flexible control
overfitting three CV days. Report every arm and case regardless of outcome.

Preserve all weights, exact training hashes, neighbor identities, objective
certificates, forecasts, scores, fit/iteration counts and measured runtime.
Expected832 weight fits, zero new provider fits/API calls; original common cost
12,984 forecast computations remains recorded. Historical availability still
assumes period-end recording. No positive development result alone satisfies
the held-out matched-agent20%/95% interval goal. No paid confirmation or final
access on a negative gate; main/PyPI unchanged.
