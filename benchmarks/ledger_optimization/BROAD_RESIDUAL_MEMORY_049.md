# Shared residual-correction action 049

Protocol amendment, frozen before computing results. Prior development findings
are known, including the limited headroom of filtering 047 proposals. This
experiment expands the forecast action for **both** arms to test whether stored
signed errors support useful correction. It does not change the target metric,
remove difficult cases, or supply private information or actions to the ledger.

Use all 416 fixed development tasks and the six archived forecasts from 038.
Both arms start with the **same 045 CV ensemble weights** for each current task;
do not use the 047 ledger weights. Reuse exactly the sixteen temporally visible
same-domain context neighbors from 047, with recording/closure checks repeated.
All CV and historical targets must have the same 24-hour lead alignment as the
current task: CV ends differ from the origin by whole days; historical origins
have the same nominal hour/minute/second. Reject any mismatch, never realign
using current outcomes. Published hourly labels retain their earlier timezone
and recording assumptions.

Common correction primitive: apply the current six ensemble weights to every
training forecast. At each lead h compute its signed log error
`r[h] = log1p(actual[h]) - sum(w[m] * log1p(point[m,h]))`.
For nonnegative training case masses summing to one, fit a 24-vector correction
by minimizing `sum_i mass_i * mean_h((r_i[h]-b[h])^2) + mean_h(b[h]^2)`.
The unique solution is `b = 0.5 * weighted_mean(r)`. The fixed unit ridge penalty
shrinks towards no correction and is identical for both arms. There is no
parameter sweep. This squared-log training surrogate is **not** mean-case RMSLE;
the primary evaluation remains the original mean of case RMSLE.

Control: train on the current three CV folds, equal masses 1/3.
Ledger: current CV total mass 1/2, retrieved sixteen production cases total
mass 1/2, equally within groups. Preserve raw residual vectors, case references,
masses, weights, input hashes, correction, objective and stationarity diagnostic.

For each arm the new forecast is
`expm1(max(0, sum(w[m]*log1p(current_point[m,h])) + b[h]))`.
The zero floor enforces nonnegative forecasts and applies identically to both.
Count all clipped leads. Current actuals enter scoring only after both corrected
forecasts exist. Both derived forecasts retain task series, timestamps and unit
by reference to the unchanged execution cohort; do not claim a new provider
execution or silently rewrite any original forecast.

Primary comparator is CV-only correction with the exact same primitive. Also
retain the uncorrected 045 CV ensemble as a stronger-control guard. Promotion
requires >=20% lower mean case RMSLE than **each** control overall and positive
improvement against each within both domains. This guard prevents a weak new
control from manufacturing success. Report all three arms, all cases, and
per-domain means regardless of outcome. Do not substitute an alternative primary.

This is a local numerical prototype, not a shipped ledger capability or a new
agent experiment. Preserve zero API calls/provider refits, 832 vector fits and
runtime. Original common cost remains 12,984 forecast computations; previously
incurred 047 weight fits remain in that experiment's cost record even though
only its context retrieval is needed here. No 046 hindsight artifacts or reserved
future observations are read. On a failed development gate, do not dispatch paid
confirmation. The final matched-agent >=20%/95% interval objective remains unmet
until verified on a properly frozen untouched evaluation.
