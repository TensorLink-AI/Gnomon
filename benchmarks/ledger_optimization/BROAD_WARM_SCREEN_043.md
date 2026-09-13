# Frozen warm-start contextual evidence screen 043

Use exactly the 125 usable earlier cohorts prepared in 042 and the unchanged
416 scored cases from 038. Preserve all three unavailable warm-up origins as
exclusions. Fit six unchanged hourly recipes at three CV folds plus production
on usable warm-up cases only: 3,000 additional forecast computations, including
1,500 estimator fits. Reuse the scored predictions byte-for-byte. Total common
cohort construction cost is 12,984 computations/6,492 estimator fits. No fitting
or imputation for incomplete warm-up cohorts.

This is a numerical development mechanism test, not an agent comparison or
published 1.2.0 product claim. All arms must have the same arrived raw data,
historical forecasts/actuals and common fitting budget. A no-ledger integration
could calculate these same features; test convenience/agent performance later.
Gnomon integration, if justified, uses the pinned 1.2.0 wheel and explicitly
separate development RMSLE helper. Main/PyPI and reserved outcomes stay unchanged.

## Primary new proposal rule

For each current task, retrieve completed earlier origins from **the same
domain**, across the eight fixed development series. Require origin < current
origin, horizon close <= current origin and recorded availability <= current
origin. Never use same-origin outcomes from another series. Select the latest
eight distinct available origin instants, retaining all complete six-recipe
records at those instants. Require at least sixteen records and three distinct
origins; otherwise use current CV. No cross-domain pooling.

Use twelve predecision features: log1p of each of the six current CV RMSLEs;
mean and standard deviation of log1p historical observations; zero fraction;
mean log1p level in the last 168 hours minus the preceding 168; mean absolute
log1p differences at lags 24 and 168. All observation features use only the
task's last 730 observed hours, never production targets. No sensor identity,
future date outcomes or eventual winner is a feature.

Standardize features using retrieved history only, dividing by
max(training population standard deviation, 0.1). Fit a multioutput ridge with
alpha=10, centered targets, predicting each recipe's
`production_RMSLE - contemporaneous_CV_RMSLE`. The intercept is mean training
residual; solve `(X'X + 10I) B = X' (Y - mean(Y))`. Estimate current loss as
`max(0, current_CV + 0.5 * predicted_residual)`. Select minimum estimate; ties
break by current CV and the original recipe order. This single rule and its
parameters are fixed before fitting warm-up forecasts. Related earlier retail
calibration methods failed; this is not a novelty or superiority claim.

Count each contextual fit separately from forecast fits. Save retrieved task
identities, effective cutoffs, feature schema/hash, support counts, coefficients,
training standardization and predicted loss estimates so decisions can be
independently reproduced. Report computational overhead rather than claiming
retrieval is free.

## Controls and diagnostics

Primary control remains current CV. Also replay the previously fixed recent
history, equal CV/history blend, and same-series CV-bias correction using warm-up
evidence. Report them as prespecified diagnostics, not replacement primaries.
Select every task before exposing its production scores. Historical warm-up and
scored records are added only through explicit availability filtering; iteration
order must not grant access to outcomes from later origins or other domains.

Report overall, each domain, rounds >=10, and earlier/later development slices
(0–17 / 18–25). These slices have already been examined and are not holdouts.
Promotion requires the primary rule to reduce mean case RMSLE by at least 20%
against current CV with positive improvement in both domains. Passing permits
planning an agent comparison, not a final superiority claim. No independent-case
confidence interval or reserved outcome access. Preserve every negative result.
