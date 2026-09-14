# Learned historical error relationships 051: gate failed

The conditional risk estimator improved mean RMSLE **1.72%** versus its matched
current-CV quadratic control, but only **0.95%** versus the stronger intraday CV
control. Against that stronger control it worsened electricity by **2.32%**.
It does not improve on the previously tested intraday ledger method, and fails
the required 20% development gate. All 416 tasks completed; no paid agent run or
final-data access followed.

| Domain | Cases | Global CV | Intraday CV | Quadratic CV | Learned-history quadratic |
| --- | ---: | ---: | ---: | ---: | ---: |
| Electricity | 208 | 0.111226402 | 0.107486486 | 0.111307225 | 0.109983490 |
| Pedestrians | 208 | 0.410036795 | 0.409935646 | 0.410187001 | 0.402522352 |
| Overall | 416 | 0.260631598 | 0.258711066 | 0.260747113 | 0.256252921 |

This prototype learned context-dependent relationships between the six models'
errors from all temporally available same-domain episodes. It used the twelve
predecision features frozen in 043 and a fixed 64-tree ExtraTrees regressor per
domain/origin. Every forecasted error matrix is independently reconstructible
as a nonnegative weighted average of past completed cases. It is an uncentered
error Gram matrix, not a covariance matrix. Its labels were computed only after
source and recording eligibility checks; current outcomes were not training data.

Both new arms used the same quadratic combination primitive and current-CV
anchor. One used only the three current CV error matrices; the other blended
that evidence equally with the learned historical matrix. Training minimizes
expected MSLE plus an anchor penalty; final evaluation uses the original mean
case RMSLE. The two stronger controls were preserved byte-for-byte, preventing
improvement against a slightly weaker surrogate control from passing the gate.

Protocol/code were frozen at `5edde93`. Four synthetic tests passed before scoring.
The independent audit reproduced all 416 cases, 52 evidence models and 832 weight
fits. It traversed the saved tree structures independently, verified training
leaf support and effective record weights, recomputed historical matrices and
quadratic certificates, and checked temporal visibility, forecasts and scores.
It passed 6,721,431 assertions, mostly repeated tree-traversal integrity checks;
that count is not a sample size or a measure of statistical confidence. The
audit does not independently implement or prove optimality of tree-split training.

Additional local computation: 52 forest fits/3,328 trees and 832 quadratic weight
fits/11,980 optimizer iterations. Measured processing took 7.1360 seconds wall and
6.8995 CPU seconds, excluding source loading and audit. Zero API calls or new
forecasting-provider fits. The inherited common 12,984 forecast computations and
all prior experiment costs remain recorded. Historical availability still uses
the disclosed period-end assumption.

Raw cases, tree structures, training matrices, query record weights, manifests,
audit and rerun instructions are retained in `results/broad-conditional-risk-051-001/`.
[The receipt](evidence/broad-conditional-risk-051.json) contains verified archive
and file hashes. No future-aware 046 weights or reserved observations were read.
This is a negative development result, not a new agent comparison or a shipped
ledger change. Main/PyPI remain unchanged; the held-out 20% goal remains unmet.
