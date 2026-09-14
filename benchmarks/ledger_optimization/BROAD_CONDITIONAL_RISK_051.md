# Learned conditional historical error relationships 051

Freeze before scoring. Previous development outcomes are known. This tests a
different evidence estimator, not a final evaluation or a release claim. Keep
all416 development tasks, original six forecasts, twelve predecision features
from043, current origins and actuals. No future-aware046 weights or reserved
observations may be read.

For an earlier completed case, form the six-model log-error matrix E (24 rows,
six models) and its positive-semidefinite Gram matrix G=E^T E/24. This contains
both each model's error magnitude and relationships between their errors. It is
not a centered covariance matrix. Averaging such matrices remains PSD.

At each domain/origin, qualify all earlier same-domain complete episodes with
both last target and recorded availability <=current origin and origin strictly
earlier. Unlike047's recent-eight-origin nearest-neighbor rule, use all available
history. Require at least32 episodes across three distinct origins; otherwise
fall back to the current-CV matrix. Preserve every eligible ID and effective
cutoff. Do not expose same-origin outcomes during batch fitting.

Fit one ExtraTreesRegressor per domain/origin using the twelve original features
and the36 flattened Gram entries. Fixed settings:64 trees, max_depth4,
min_samples_leaf8, max_features1.0, bootstrap=False, random_state17, n_jobs1,
squared_error criterion. No parameter search. Trees can learn nonlinear context
similarity; each output is an average of earlier PSD matrices. Save tree split
structures and each query's effective nonnegative weights over training records.
Those weights must independently reconstruct the predicted matrix. Training
inputs contain only past outcomes; current features contain current CV/history.

Both arms share a quadratic combination primitive over six simplex weights,
minimizing w^T G w +0.001*sum((w-anchor)^2). Anchor is the045 global CV ensemble.
Control G: mean of the three current CV Gram matrices. Ledger G: equal blend of
that current matrix and the forest-predicted historical matrix. The same solver,
action, anchor and penalty are used in both. The training surrogate is expected
MSLE, not mean-case RMSLE; final scoring remains the original mean case RMSLE.
SLSQP ftol1e-12/maxiter500; feasible weights; successful termination; objective
<=initializer+1e-8; convex first-order gap<=1e-5. Reject materially non-PSD
matrices (minimum eigenvalue<-1e-9); do not replace them with outcome-driven fixes.
Stop and preserve any failure, retaining partial work and costs.

Primary comparison: ledger versus current-CV quadratic combination. Also retain
both stronger controls:045 global RMSLE ensemble and050 intraday CV ensemble,
byte-identical. Promotion requires >=20% improvement versus all three overall
and positive improvement versus each within both domains. Do not obtain success
by weakening the comparator or dropping cases. This screen cannot satisfy the
matched-agent objective by itself.

Expected52 evidence-forest fits (3,328 trees),832 quadratic weight fits, zero
new forecasting-provider fits or API calls. Separate evidence-model computation
from provider execution. Preserve original12,984 common forecast computations
and all prior experiment costs. Report current costs, matrices, weights, model
structures, provenance, scores and uncertainty limitations. Availability retains
the previously disclosed synthetic period-end assumption. No final-set access
or paid confirmation on a failed development gate. Main/PyPI remain unchanged.
