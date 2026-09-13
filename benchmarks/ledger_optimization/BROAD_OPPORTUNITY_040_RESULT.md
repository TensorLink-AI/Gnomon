# Opportunity audit 040: filters and cold starts cannot reach the bar here

The existing proposals cannot deliver 20% merely by rejecting harmful overrides.
Even a perfect hindsight choice among all three proposals and current CV yields
only **10.09%** improvement on these 416 development tasks.

| Nondeployable hindsight diagnostic | Maximum reduction versus current CV |
| --- | ---: |
| Perfect filter of recent-history proposals | 7.59% |
| Perfect filter of blended proposals | 4.81% |
| Perfect filter of calibrated proposals | 4.32% |
| Perfect choice among all three proposals and CV | 10.09% |
| Perfect choice among all six recipes from origin zero | 21.69% |
| Perfect six-recipe choice after three matured origins; CV before | 19.71% |
| Same, after four origins | 19.11% |
| Same, after eight origins | 14.97% |
| Same, after ten origins | 12.08% |

These deliberately use future realized errors. They are not observed ledger
performance, deployable rules or an upper bound over unexecuted recipes. They
do not establish what can happen on the still-unopened final set.

With no cold-start restriction, reaching 20% would require capturing **92.20%
of all available six-recipe oracle improvement**. Target RMSLE is 0.225266,
against oracle 0.220505: only 0.004762 average excess loss is allowed. If an
algorithm retains current CV until at least three same-series outcomes mature,
even perfect subsequent selections fall short on this development portfolio.

The proper next direction is not another filter on the same proposals. Any
continuation toward 20% needs a stronger source of proposals, useful evidence
available during cold start, or a broader common forecast portfolio. These
would require a new prospective development protocol and equal access/costs
for the control. Do not silently drop early origins, change the primary metric,
or treat hindsight as a proposed algorithm. A modest real advantage would still
be informative, but is not achievement of the requested objective.

## Verification and preservation

Diagnostic list frozen at `cbd7d15`; code/tests at `85983f9`, before calculation.
All 416 inputs verified against the 038/039 receipts. Three synthetic tests
passed. An independent enumeration of the permitted execution sets passed
**6,346 checks, zero failures**, covering each bound, cold-start availability,
aggregates and feasibility. No source observations, fitted forecasts or original
choices were changed. No extra forecasts, API calls or reserved future reads.

Computation took 0.0137s excluding loading and verification. Original shared
cohort costs remain 9,984 forecast computations. Full per-case bounds, both
domain results, source hashes and verification are retained in
`results/broad-opportunity-040-001/`. The [receipt](evidence/broad-opportunity-040.json)
records archive and individual hashes; the archive contents were verified.

```sh
python3 -m benchmarks.ledger_optimization.broad_opportunity results/broad-screen-038-001 results/broad-calibration-039-001/report.json results/broad-opportunity-040-rerun
python3 -m benchmarks.ledger_optimization.broad_opportunity_verify results/broad-screen-038-001 results/broad-calibration-039-001/report.json results/broad-opportunity-040-rerun
```

No paid confirmation is justified by this audit. Main/PyPI unchanged; actual
20% held-out improvement and a positive 95% interval remain unestablished.
