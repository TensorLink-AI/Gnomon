# Generate525 frozen historical episodes for memory-breadth development

Source058 fixes400 main plus125 usable warm-up cases for the named057 training
series. This stage generates historical evidence only; the416 original scored
forecasts and model recipes remain unchanged. No new scored tasks or validation
observations are used. Final reserves stay untouched.

Use the exact038 compute_case/hourly_numerical implementation, including three
CV folds and six recipes. Preserve12 predecision context features from043,
all forecasts, targets, history hashes and period-end availability assumptions.
New total:12,600 forecast computations and6,300 estimator fits. These are shared
historical evidence-generation costs, not free/private ledger computation.
Existing original evidence cost remains12,984 computations/6,492 estimator fits.

Use two independent local worker processes with spawned interpreters; each
numerical library uses one thread. Every case/model uses the frozen deterministic
seed/settings. Parallelism changes execution scheduling only. Workers save
per-case started-call counters before each computation, outcomes and CPU/wall
usage. Parent retains source/code/runtime hashes and aggregate costs. On failure,
stop queued work, let in-flight cases finish and retain their counters/artifacts;
do not silently omit the failure or continue to comparative scoring.

Cases are generated from permitted observed histories. They become eligible
for later retrieval only when origin, target closure and recording availability
meet057's rules. A completed retrospective computation does not grant a later
outcome early visibility. No production actuals enter the12 context features.

Freeze this code/protocol before forecasting; no API calls or main/PyPI changes.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m benchmarks.ledger_optimization.memory_forecasts results/broad-memory-source-058-001 results/broad-memory-forecasts-059-001
```
