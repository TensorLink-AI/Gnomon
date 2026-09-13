# Frozen common-ensemble evidence screen 044

The six-recipe hard-selection rules, including warm-start context, failed.
Test a different evidence use: combine those same six forecasts using their
joint errors. This expands the common action space for **both** methods;
it must not be compared only with the weaker original hard-CV selector.
All 416 scored cases, original predictions and 125 usable warm-up cohorts stay
fixed. This is numerical development work, not a shipped Gnomon feature or
held-out agent evidence. No reserved observations or new provider fits.

For a weight vector w on the six-recipe probability simplex, return
`expm1(sum_m w_m * log1p(point_m))` at each horizon step. Nonnegative weights
sum to one. Preserve series, unit and timestamps; no clipping of observations,
date changes or outcome-dependent recipe exclusions. Ensemble forecasts are
new derived predictions and must be labelled as such, never substituted into
old execution records.

## Primary control and ledger method

Control weights minimize mean case RMSLE over the three current backtest folds.
Ledger weights minimize an equal mixture of that current-CV objective and mean
production RMSLE over the latest four complete, temporally visible same-series
origins (including warm-up), requiring at least three. Otherwise use control
weights. Past source/recording availability and all cutoffs remain the explicit
period-end replay assumptions from 042/043. Never use current production actuals
when choosing weights. Do not pool domains or differently identified series.

Both objectives add the same small strict convexity term
`1e-6 * sum((w_m - 1/6)^2)`. Solve with SLSQP, uniform initialization, analytic
gradient, six [0,1] bounds, sum(w)=1, maxiter=500, ftol=1e-10. Validate feasible
weights, finite predictions and objective not worse than initialization within
1e-8. Require a convex optimality gap bound <=1e-5, and retain that bound,
iterations, objective and solver status. Failure stops the experiment and is
retained; no quiet fallback or changed solver settings after seeing accuracy.

For certificate stability, a case norm <=1e-8 contributes its full current
weighted loss to the bound and contributes zero gradient; its global lower
bound is zero. Other norms and the regularizer use their analytic gradients.
The resulting bound is `w dot gradient - min(gradient) + omitted_current_loss`.
This remains a convex suboptimality bound, including exact-fit nonsmooth cases.
The solver itself uses the original objective and gradient, not a smoothed score.

Selection inputs are only current fold predictions/actuals and eligible earlier
production pairs. Apply the selected weights to current predictions afterward,
then score. Count all weight optimizations and their iterations/runtime in
addition to inherited forecast computations. Record exact fit input hashes and
retrieved origins. Both integrations would get the same combination primitive,
raw observations and earlier records; numerical helpers are not privileged data.

## Reporting and decision gate

Primary comparison: historical-evidence ensemble versus current-CV ensemble.
Also show both versus original hard-CV selection and a uniform ensemble, as
secondary diagnostics. No selecting among these as a replacement primary.
Report each domain, overall and later/mature development slices (already used,
not holdouts). Gate: at least 20% lower mean per-case RMSLE against the ensemble
control, and positive improvement in each domain. A small gain in combining
forecasts alone is not a ledger benefit.

Save both weight vectors, all derived forecasts, source hashes, retrieved
cohorts, objective certificates, all outcomes and failures. Original common
construction cost remains 12,984 computations and 6,492 forecast-estimator fits.
No additional original model fits or API calls. Passing only justifies further
integration work; it cannot establish the final objective or statistical
significance. Main/PyPI unchanged; future agent integration uses Gnomon 1.2.0.
