# Hourly development result 038: historical rule rejected

All 416 development cases completed. **The primary historical selector was
4.98% worse than current-CV selection.** It failed the prospectively frozen
20% improvement gate. Do not promote it to paid agent confirmation or open the
reserved future outcomes.

| Selector | Overall RMSLE | Electricity | Pedestrian |
| --- | ---: | ---: | ---: |
| Current CV control | 0.281583 | 0.119683 | 0.443482 |
| Primary: latest four matched origins | 0.295602 | 0.131091 | 0.460113 |
| Secondary: equal current-CV/history blend | 0.280284 | 0.126138 | 0.434430 |
| Nondeployable hindsight among six recipes | 0.220505 | 0.094501 | 0.346509 |

Primary historical selection worsened error by 9.53% in electricity and 3.75%
in pedestrians, changing the current-CV choice in 286/416 cases. On rounds >=10
it was still 2.62% worse. The secondary blend was 0.46% better overall, but 5.39%
worse in electricity. It is not a replacement primary result or a passing rule.

Hindsight could improve by 21.69% overall within these six recipes, versus
21.04% in electricity and 21.87% in pedestrians. That is only an oracle diagnostic
for the executed portfolio. It is neither deployable nor an upper bound over
unexecuted models. The tested historical rule does not capture that headroom;
the result does not establish that a practical selector can capture nearly all
of it to reach 20%.

## Evidence and cost

Rules, recipes and selector code frozen at `7c6431d` before performance execution.
All 16 series and 26 origins retained; no source/date/recipe changes during the
run. Equal cohorts cost 24 computations per task for every selector: **9,984
forecast computations**, including **4,992 estimator fits**. Wall time 838.17s;
CPU time 824.50s. API calls and LLM tokens: zero. No paid service costs incurred;
local CPU time is retained, not treated as free computation.

The independent saved-evidence verifier passed **176,846 checks, zero failures**:
all scored pairs, fold/task slices, three deterministic baseline predictions,
historical visibility, selector choices and aggregates. It did not independently
refit Ridge or random forest. Four synthetic tests passed, including constant
levels at every fold length, exclusion of future outcomes and current-target
mutation invariance. The interpreter and dependency versions are in the manifest.

This is a development mechanism screen, not a Gnomon-versus-no-Gnomon agent
treatment, not held-out evidence and not an independent-case significance test.
No Gnomon version was used to compute these forecasts. A separate installed
runtime preflight confirmed Gnomon **1.2.0**, build
`1.2.0+ga38cd0cad353.s9723394ccb6d`, for future integration. Its public ledger
comparison is MAE-only; the already frozen development RMSLE adapter calculates
the objective from referenced executions and actuals. Do not mislabel that helper
as part of the published 1.2.0 wheel.

The 32 reserved series' later values remain unparsed. Nominal source-hour labels
and assumed period-end availability retain the limitations in protocols 035/037.
Existing retail and M5 negative results remain unchanged.

## Reproduce and next decision

All per-case predictions, actuals, fold errors and decisions are in
`results/broad-screen-038-001/`, archived beside it. The tracked
[receipt](evidence/broad-screen-038.json) inventories every file and the archive
SHA-256. Source spans are pinned by SHA-256 in the manifest and protocol 038.

With the same dependency versions and pinned development spans, run to a new
output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m benchmarks.ledger_optimization.broad_screen results/broad-panel-037-001/development-spans.json results/broad-screen-038-rerun
python3 -m benchmarks.ledger_optimization.broad_screen_verify results/broad-panel-037-001/development-spans.json results/broad-screen-038-rerun results/broad-screen-038-rerun/verification.json
```

Next work should diagnose when historical overrides help or hurt using this
development evidence, then freeze any new rule before another evaluation. Do
not assume that accumulating more history improves decisions: the simple rule
tested here made them worse. The actual held-out 20% objective remains unmet.
