# Preparation080: forecast-step conditional model-error evidence

079recent-error corrections did not beat068. Prior051predicted one whole-day
Gram matrix from12case-level features and produced global weights.070used
24independent lead weights but no shared forecast-step context learner. This
preparation defines a different evidence model: learn six-model error relationships
conditional on the forecast-step context, sharing evidence across leads/cases.
Synthetic preparation only; freeze a separate source driver before source fits.

Use the fixed six038/066configurations, exclude the search-selected seventh slot.
For each24-step pair, compute E_hm=log1p(point_mh)-log1p(actual_h) and the rank-one
positive-semidefinite matrix G_h=E_h E_h^T. These are error second moments, not
centered covariances. Predict all36entries jointly using the fixed077forest:
32trees,max_depth4,min_samples_leaf24,max_features1.0,bootstrapfalse,
random_state17,n_jobs1,squared_error. Inputs are077's ten forecast-only features:
six logs centered/scaled within the forecast's baseline window, baseline step
log, window scale and relative-lead sin/cos. The anchor is frozen045six-model
weights, applied to both current and historical training forecasts for this query.
No actuals enter feature construction. Tree leaves average earlier observed
error matrices with nonnegative weights, so every prediction remains PSD.

Control trains on three current CV pairs, equal case masses1/3. Ledger adds all
same-domain production pairs with origin strictly earlier and target/source/
recording times<=current origin; current and historical groups each total.5.
Without historical evidence ledger uses control masses. Every case's24leads
share its mass equally. Metadata filtering precedes historical payload access.
Retain all541original tasks/125available warmups, score416cases, exact fixed
configuration/phase identities and nominal period-end availability assumptions.
No protected validation/final, corrected079targets or hindsight075choices.

At each current forecast step, predict a6x6Gram. Use unchanged051fit_weights:
minimize w^T G w+.001||w-anchor||^2 over six nonnegative weights summing to one,
SLSQPftol1e-12/maxiter500; require finite PSD input, success, feasible weights,
objective no worse than anchor and convex gap<=1e-5. Failure is preserved and
stops source execution; no post-result tolerance change. Predict expm1 of the
weighted current model logs. Thus outputs stay in the raw forecast range; no
signed correction, negative weights or new base-provider execution.

Both arms use the same forest/24quadratic solves and upper budget. Record all
forest structures/node values/sample support, input hashes/masses, historical
refs, query matrices,24weight certificates and outputs. Source-run maximum:
832forests/26,624trees and19,968quadratic solves. Count actual costs/failures,
including inherited raw/model/search/anchor computation. Mean-case RMSLE remains
the primary score; per-step expected squared-log error is a disclosed surrogate.

Gate remains20%over matched control ANDstrong050, positive over061AND068,
overall and all four positive within both domains. Retain045uncorrected baseline.
No paid/final confirmation after a failed development gate. A synthetic pass is
not forecasting evidence. Actual final goal still requires matched1.2.0/
DeepSeekv4.1-flash agents, untouched final cases and95%uncertainty excluding zero.
Main/PyPI unchanged. Distinguish this common numerical primitive from shipped
ledger infrastructure or an established agent advantage.
