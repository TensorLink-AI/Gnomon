# Development056: retrieve evidence by backtest error pattern

Validation055 supports a modest5.17% numerical effect but misses the20% gate.
Do not tune that new-series panel. Return only to the original038/043 development
cases for a separately frozen retrieval hypothesis.

Current047 retrieval describes each case with six aggregate CV losses and six
history statistics. It loses which models over/under-predicted at different
hours. Test whether an explicit observed backtest error profile retrieves more
relevant prior experience. This changes the evidence index, not the forecasting
recipes or common mixture primitive. It is not a new released feature or agent
experiment.

## Fixed representation and retrieval

For each of six models in the frozen order, and each of four six-hour lead
blocks, compute two values across the three current CV folds: mean signed log
error and root mean squared log error (18 errors per model/block). Order model,
block, then signed mean/RMS. The48 values use only CV predictions and targets
inside the observed history. Never include production actuals or production
scores in current or historical query features.

Keep047's same-domain visibility rules, latest eight distinct origins, minimum
16 contexts/three origins and sixteen selected neighbors. Population standardize
each feature with floor0.1 using the visible candidate pool. Distance equals
mean squared standardized difference over the original12 features **plus** mean
squared standardized difference over the48 error features. These two families
get equal weight, fixed before scoring. No feature/neighbor/weight grid search.
Tie break by origin and series ID as before. Return component distances, feature
means/scales, candidate identities and chosen references for verification.

## Unchanged comparison and scoring

Reuse all416 scored038 raw cases and125 warm-up043 cases. Same050 intraday
mixture objective, original045 CV anchors, half current CV/half history mass,
solver and certificate rules. No new original model fits. Preserve byte-identical
050 matched-CV/ledger predictions and045 global-CV guard as comparators. Only
new profile-retrieved ledger mixtures are fitted. If insufficient contexts,
use the same matched-CV fallback, with task retained. Any numerical failure
stops and is preserved; no dropped case or changed threshold.

Report new ledger against matched CV, global guard and old050 ledger, overall
and each domain. The development gate requires>=20% reduction versus both
controls, positive improvement versus old ledger, and positive reductions versus
both controls and old ledger in each domain. Development selection on reused
cases is not confirmatory confidence evidence. Retain negative results unchanged.
Do not access052 validation outcomes, final-reserved data or paid APIs.

Freeze code and this protocol before computing the new mixtures. Retain all
profiles, retrieved IDs, fit inputs, weights, forecasts, scores, costs and
independent audit. This is development, not permission to promote a failed gate.
