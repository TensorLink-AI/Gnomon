# Lifetime retention061: another small gain, target unmet

Removing the eight-origin age filter improved the expanded-memory rule by only
**0.1082%** on the original416 development tasks. The resulting ledger mixture
is **2.5998% better than matched CV**. The20% gate remains unmet.

| Method | Overall mean case RMSLE |
|---|---:|
| Global CV guard |0.260631598421|
| Matched intraday CV |0.258711065759|
| Original ledger050 |0.253424725902|
| Expanded recent ledger060 |0.252258171733|
| Expanded lifetime ledger061 |0.251985214755|

Overall reductions:3.3175% versus global guard,2.5998% versus matched CV,
0.5680% versus original ledger and0.1082% versus expanded recent ledger.
Electricity improves0.1314% versus matched CV but is0.00126% worse than the
original ledger (essentially unchanged at this scale). Pedestrian improves
3.2470% versus matched CV and0.7210% versus the original ledger. No confirmatory
interval is claimed on repeatedly used development cases.

Older-than-eight-origin records were selected in396/416 cases, averaging7.935
of16 neighbors. The index used the retained history; its marginal value remained
small. Sixteen initial cases had no older candidates: training hashes and
forecasts exactly match060 in those cases, as expected.

Frozen1807fb5 preceded scoring. Same1066 contexts, twelve features, sixteen
neighbors, maturity rules, source data,045 anchors and050 mixture objective.
Only the age filter and resulting eligible-pool standardization changed. All
four comparator forecasts remain exact saved artifacts. No new forecasting
models, source values, validation outcomes or final-reserved targets were used.

Three new/four inherited tests pass. Independent audit177,489 checks, no failures,
plus the16-case unchanged-pool equivalence check. The audit independently checks
scalar distances, eligible/selected identities, training hashes, certificates,
forecasts, scores, all comparators and costs. Checks are not statistical samples.

Additional cost:416 weight fits,24,065 optimizer iterations,5.173s.0 original
forecast calls/API calls. Shared inherited historical cost remains25,584
computations/12,792 estimator fits, including the prior memory expansion. No
cold-start/failure task was silently removed.

[Full receipt and archive hashes](evidence/broad-lifetime-memory-061.json).
This development result does not replace055's separate5.17% new-series estimate
or the earlier1.14% inconclusive Hermes comparison. No promotion, paid follow-up,
main merge or PyPI release. The20% final matched-agent objective is unachieved.
