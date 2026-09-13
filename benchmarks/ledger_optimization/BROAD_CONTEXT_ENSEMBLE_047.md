# Context-matched ensemble evidence 047

Freeze this protocol and implementation before computing any 047 result. Prior
development results, including failed policies and hindsight bound 046, are
known. This is another development experiment, not held-out confirmation.

Keep the same 416 tasks, 16 development series, original forecasts, origins,
actuals and common warm-up cost from 038/043. Keep the stronger CV ensemble
control from 045 byte-for-byte. Both arms have the same raw information and
combination tool. The difference is use of retrieved prior production evidence.
Do not access reserved future observations or 046 hindsight weights.

Primary rule: use the twelve predecision features already frozen and verified
in 043 (six log1p CV errors and six observed-history summaries). For each current
task, qualify same-domain episodes with origin strictly earlier and both target
closure and recording time no later than the current origin. Keep the last eight
distinct visible origin instants. Fit feature mean and population standard
deviation on this candidate pool only, with a scale floor of 0.1 per coordinate.
Rank candidates by squared standardized Euclidean distance to the current
features, tie by origin then series ID. Retrieve the nearest sixteen. Require
at least sixteen candidates across three distinct origins; otherwise use the
unchanged control. Do not require nearest episodes to be from the same series.

Use the unchanged 045 convex log-space combination objective and certificate:
six nonnegative weights summing to one, regularizer 1e-6 sum((w-1/6)^2), SLSQP
uniform initializer, ftol=1e-12, maxiter=500, gap bound <=1e-5. Give the three
current CV folds half the objective mass and the sixteen retrieved production
episodes the other half, equally within each group. All training inputs must
precede the current task. Never retrieve by realized future score or fit to
current actuals. Score only after fitting. Preserve candidate IDs, distances,
scaling, chosen IDs, pair hashes, weights, certificates and per-case scores.

Stop and retain evidence on any optimizer or integrity failure; do not relax
requirements or report an incomplete cohort as a completed comparison. Report
fit counts, iterations, elapsed time, zero API calls, zero provider refits and
the inherited 12,984 forecast computations. Feature calculations and retrieval
overhead are additional. Timestamp availability remains the disclosed synthetic
period-end assumption of 042/043, not real historical recording evidence.

Primary metric: mean per-case RMSLE reduction versus 045 CV ensemble, all 416
tasks. Report electricity and pedestrian domains separately. Development
promotion requires >=20% overall and positive improvement in both domains.
No paid confirmation or final-set access on a negative gate. Even a positive
development gate would still require a frozen matched agent experiment on the
untouched final set, 95% interval excluding zero, and comparison to the prior
ledger. This prototype alone cannot satisfy the user's goal or justify release.
