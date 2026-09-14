# Learned relevance062: negative against recent and lifetime incumbents

The outcome-supervised retrieval rule completed all416 original development
tasks, but did not improve the best existing ledger rule. Its RMSLE was2.3749%
below matched CV and0.2308% above lifetime061. The20% gate failed.

| Method | Mean case RMSLE |
|---|---:|
| Global CV guard |0.260631598421|
| Matched block CV |0.258711065759|
| Original ledger050 |0.253424725902|
| Expanded recent ledger060 |0.252258171733|
| Expanded lifetime ledger061 |0.251985214755|
| Learned relevance062 |0.252566824850|

062 improves original050 by0.3385%, but worsens recent060 by0.1224% and
lifetime061 by0.2308%. Electricity is0.0186% worse than matched CV and0.1501%
worse than lifetime061. Pedestrian improves matched CV by3.0025%, but worsens
lifetime061 by0.2526%. No confirmatory interval is claimed on reused development.

Protocol/code frozen at e5f6d01 before scoring. Same1066 episodes,416 tasks,
twelve predecision features, six forecast models, block RMSLE objective, anchor,
penalty, solver and numerical certificate. All five comparators are exact saved
forecasts. Only historical relevance changed:52 forests learned centered
production-minus-CV error contrasts from matured past outcomes, and their leaf
memberships supplied nonnegative historical weights. No current outcomes trained
the query model; recording and source maturity filters apply before label lookup.

Added-series memory contributed to every task, averaging43.70% of historical
mass. Queries retained275.43 positive-weight records on average; their effective
sample size (1/sum(weight squared)) averaged166.74. These counts describe actual
retrieval, not independent observations or proof that retrieved contexts helped.
The negative result does not support replacing the lifetime rule or simply
assuming a learned index will yield a large improvement.

Four synthetic tests passed. Independent audit completed14,185,041 assertions
with zero failures, mostly repeated tree-structure/traversal checks. They verify
maturity filtering, independently calculated centered labels, leaf memberships,
case weights, objective certificates, forecasts, metrics, comparators and costs.
They do not independently prove optimal tree split training or provide millions
of independent statistical samples.

Additional costs:52 evidence forests/3,328 trees;416 mixture fits/23,638 optimizer
iterations;20.50s wall and19.52s CPU. Zero new original forecasting/API calls.
Inherited historical cost remains25,584 computations/12,792 model estimator fits.
All416 tasks retained, no numerical failure or cold-start exclusion.

[Evidence receipt and archive hashes](evidence/broad-learned-retrieval-062.json).
This is a numerical development prototype. No paid confirmation, final-reserve
access, main merge or PyPI release. Separate055 validation and the actual Hermes
comparison remain unchanged. The final matched-agent20% objective is unmet.
