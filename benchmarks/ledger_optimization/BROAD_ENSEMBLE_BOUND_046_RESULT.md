# Ensemble opportunity 046: target not ruled out, practical gain unproven

Hindsight combination of the same six forecasts can reduce the strong
CV-ensemble control's error by approximately **24.114%** on these development
cases. This does **not** establish a practical ledger improvement. It only
shows that the common ensemble action space does not itself rule out 20%.

| Quantity | Overall | Electricity | Pedestrian |
| --- | ---: | ---: | ---: |
| Frozen CV-ensemble RMSLE | 0.260632 | 0.111226 | 0.410037 |
| Hindsight minimum RMSLE, lower bound | 0.19778307 | 0.08441037 | 0.31115578 |
| Hindsight minimum RMSLE, feasible upper bound | 0.19778363 | 0.08441090 | 0.31115636 |
| Hindsight relative improvement bounds | 24.1137–24.1139% | 24.1089–24.1094% | 24.1150–24.1152% |

These are numerical optimization bounds, **not statistical confidence
intervals**. They deliberately use each current future outcome when fitting
weights. The deployable historical weighting rule in 045 achieved only 0.96%,
and worsened electricity. Reaching 20% would require capturing about 83% of the
hindsight gain; this audit does not show that such information is predictable.

The remaining obstacle within this action space is learning useful weights
before observing the outcome. A further developmental retrieval experiment can
test matching earlier error patterns to the current observed context rather
than treating all recent cases equally. It must retain the strong ensemble
control, common information and computation access, and untouched final set.
No hindsight weights may be supplied to that learner or to an agent.

## Evidence and limitations

Protocol/code frozen at `9ec79fb` before calculation. All 416 cases retained;
source/control predictions verified against their existing receipts. The exact
same refined solver/objective was used, with the same convergence threshold.
Each feasible forecast bounds achievable error from above. The regularized
objective minus its convex gap and maximum simplex regularizer bounds it from
below. Floating-point tolerances apply; this is not interval-arithmetic proof.

Two synthetic bound tests passed. Independent scalar verification passed
**15,838 checks, zero failures** for objectives, gradients/gaps, feasible
predictions, the regularizer correction, hashes and all reported aggregates.
Cost: 416 diagnostic weight fits, 6,213 optimizer iterations, 0.4710s compute
wall time excluding loading/verification. No new provider fits, API calls or
reserved-data access. Original computation costs remain in earlier receipts.

Artifacts: `results/broad-ensemble-bound-046-001/`.
[Tracked receipt](evidence/broad-ensemble-bound-046.json) inventories the verified
archive and files. Reproduce to a fresh directory:

```sh
OPENBLAS_NUM_THREADS=1 python3 -m benchmarks.ledger_optimization.ensemble_opportunity results/broad-screen-038-001 results/broad-ensemble-045-001 results/broad-ensemble-bound-046-rerun
python3 -m benchmarks.ledger_optimization.ensemble_opportunity_verify results/broad-screen-038-001 results/broad-ensemble-045-001 results/broad-ensemble-bound-046-rerun
```

The 20% held-out objective and positive 95% uncertainty interval remain
unestablished. No paid confirmation, release, main merge or PyPI change.
