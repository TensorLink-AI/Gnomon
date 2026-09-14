# Execution adapter054 for locked validation052

Implement the already frozen052 protocol, with source053 (117 warm-up plus416
scored cases). Source extraction is complete; no validation forecasts have been
computed at this freeze. Numerical modules locked in052 are unchanged.

Each of533 cases computes six recipes, three CV calls plus one production call:
12,792 common forecast computations,6,396 estimator fits. Neither treatment gets
private additional forecasts. Raw cases retain all requests' history hashes,
fold predictions, actuals and costs. Warm-up exclusions remain explicit.

Use045's global CV mixture as anchor/guard and050's matched four-block mixtures.
Ledger contexts come only from visible earlier validation/warm-up outcomes;
current actuals and realized model scores are excluded from the fitting API.
The ledger uses the matched block CV prediction when the frozen retrieval rule
has insufficient evidence. Every case remains scored. No variants are selected
using these validation results.

The paired bootstrap uses the exact052 specification,10,000 replicates and seed
20260914. Generate series indices first, then origin-block starts, with NumPy's
default_rng. Retain resampling arrays and replicate reductions. All arms use the
same indices. No paid agent confirmation unless052's gates pass.

Run with single-threaded numerical libraries:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m benchmarks.ledger_optimization.validation_run results/broad-validation-source-053-001 results/broad-validation-identity-052-001 results/broad-validation-054-001
```

Failure stops execution and preserves raw cases, fit certificate and attempted
costs. Corrections require separately retained uniform reruns, never replacing a
hard case. Current main/PyPI unchanged; no final-reserved data access.
