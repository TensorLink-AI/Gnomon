# CV error calibration 039: do not promote

The frozen correction rule was **0.26% worse overall** than current CV across
the same 416 already-seen hourly development tasks. It reduced the harm of the
earlier historical rule but did not beat the control or pass the 20% gate.

| Selection method | Overall RMSLE | Electricity | Pedestrian |
| --- | ---: | ---: | ---: |
| Current CV | 0.281583 | 0.119683 | 0.443482 |
| Original recent historical selection (038) | 0.295602 | 0.131091 | 0.460113 |
| Shrunk correction of current CV error (039) | 0.282303 | 0.125700 | 0.438905 |

The new rule changed 120/416 current-CV choices. It was 5.03% worse in electricity
and 1.03% better in pedestrians. The later development slice improved 1.04%, but
it was already used development data and cannot substitute for the failed
overall result. On rounds >=10 the rule remained 0.55% worse.

## What went wrong with the original historical overrides

| Domain | Helpful | Harmful | Sum of RMSLE saved | Sum of RMSLE added |
| --- | ---: | ---: | ---: | ---: |
| Electricity | 77 | 74 | 1.63743 | 4.01031 |
| Pedestrian | 52 | 83 | 7.25430 | 10.71344 |
| Total | 129 | 157 | 8.89173 | 14.72376 |

These are descriptive paired differences on the 286 cases where the original
historical rule changed the current-CV choice; there were no ties. They are not
confidence intervals or counts of independent events. Electricity demonstrates
why a majority of helpful overrides is insufficient: fewer harmful choices cost
more than twice what the helpful choices saved. This supports tracking downside
magnitude alongside historical ranks. It does not establish that a particular
new guard can identify those losses before they occur.

## Fidelity, verification and cost

Protocol, implementation and tests frozen at `0c0f9a9`, before this rule's
performance calculation. There was exactly one new rule, with the last eight
matured matched origins, minimum four, and shrinkage n/(n+4). No variant was
selected on the results. Related methods failed on earlier retail data; this
does not supersede those failures.

Three focused tests passed. The separate verifier passed **9,248 checks with
zero failures**, covering all original evidence hashes, retrieved origins,
corrections, selections, override accounting and aggregates. Forecast metrics
come from the independently audited 038 evidence; forecasts were not refitted.
No new model computations, API calls or reserved future reads. Additional
analysis runtime was 0.0474s wall and 0.0447s CPU, excluding verification and
input loading. The shared cohort still cost 9,984 computations/4,992 estimator
fits in 038 and is not treated as free evidence.

All 416 per-case corrections and diagnostic recipe error means are retained in
`results/broad-calibration-039-001/`. The [receipt](evidence/broad-calibration-039.json)
contains archive/file hashes and complete aggregates. Both archive and contents
were hash-verified. To rerun, use a fresh output path:

```sh
python3 -m benchmarks.ledger_optimization.broad_calibration results/broad-screen-038-001 results/broad-calibration-039-rerun
python3 -m benchmarks.ledger_optimization.broad_calibration_verify results/broad-screen-038-001 results/broad-calibration-039-rerun/report.json results/broad-calibration-039-rerun/verification.json
```

No paid agent confirmation or final evaluation is justified by this result.
Further work must remain developmental and preserve these negative findings.
The held-out 20% objective is still unmet; main and PyPI remain unchanged.
