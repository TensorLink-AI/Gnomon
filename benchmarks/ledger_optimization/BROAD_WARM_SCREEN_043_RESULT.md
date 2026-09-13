# Warm-start contextual evidence 043: rejected

Giving the selector earlier outcomes did not establish a useful accuracy gain.
The primary contextual rule was **1.84% worse** than current CV on all 416
unchanged development tasks. It was worse in both electricity (4.92%) and
pedestrians (1.01%). The frozen 20% gate failed.

| Prespecified rule | Overall RMSLE | Relative change versus CV |
| --- | ---: | ---: |
| Current CV control | 0.281583 | baseline |
| **Primary: same-domain contextual CV residual model** | **0.286768** | **1.84% worse** |
| Warm recent-history selector | 0.295967 | 5.11% worse |
| Warm equal CV/history blend | 0.279898 | 0.60% better |
| Warm same-series CV-bias correction | 0.282109 | 0.19% worse |

The primary rule changed 128 choices. It remained 2.04% worse on rounds >=10
and 1.60% worse on the later development slice. The secondary blend's 0.60%
overall improvement is not a replacement primary result and is far short of
the target; it was 5.48% worse in electricity. These are reused development
tasks, not held-out findings or a paid agent treatment.

Warm-up supplied five earlier complete origins for sensor 1 and eight for
every other series. All three unavailable warm-up origins remain explicitly
excluded; all sixteen series and 416 scored cases remain. The original scored
forecasts, CV evidence, actuals and task boundaries were reused byte-for-byte.
No recipe or parameter was changed during this run.

## What this changes

The tested failure cannot be attributed solely to starting without prior
same-series outcomes: those outcomes were supplied and the gate still failed.
This does not rule out every possible use of earlier evidence, but it provides
no justification for spending on an agent trial of these rules or opening the
reserved set. More stored records did not by themselves make these selections
more reliable. The actual held-out 20% objective remains unmet.

## Verification and cost

Protocol, code and primary rule frozen at `e0aa538` before warm-up forecasts.
Three focused context tests passed. The separate verifier passed **73,138
checks, zero failures**: source hashes, all warm-up scored pairs, three
deterministic baseline predictions, observed-history-only features, exact
temporal cohorts, primary and secondary choices, and all reported aggregates.
It reconstructed the contextual ridge fit with augmented least squares rather
than the implementation's normal-equation solve. Warm-up Ridge/random-forest
forecasts themselves were not independently refitted.

Additional costs: **3,000 forecast computations**, including **1,500 estimator
fits**, and **416 contextual estimator fits**. Warm-up plus selection took
241.14s wall and 235.34s CPU (excluding source loading and the independent audit).
The inherited 9,984 computations still count: common forecast construction
totals **12,984**, including 6,492 forecast-estimator fits. Contextual fitting is
additional overhead. API calls, LLM tokens and reserved-count access: zero.

This numerical prototype does not use the Gnomon runtime to produce forecasts;
it is not a claim about functionality shipped in 1.2.0. Any subsequent agent
integration remains pinned to the installed 1.2.0 wheel, with development
helpers separately identified. Nominal-hour and assumed-availability limits
from 037/042 remain in effect.

Full episodes, warm-up predictions, fitted coefficients, training hashes,
retrieved identities and all decisions are in `results/broad-warm-screen-043-001/`.
The [receipt](evidence/broad-warm-screen-043.json) records all file/archive hashes,
runtime versions and aggregates. Archive contents were individually verified.
Reproduce to a new output directory using the same pinned inputs/dependencies:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m benchmarks.ledger_optimization.broad_warm_screen results/broad-warmup-042-001 results/broad-panel-037-001 results/broad-screen-038-001 results/broad-warm-screen-043-rerun
OPENBLAS_NUM_THREADS=1 python3 -m benchmarks.ledger_optimization.broad_warm_screen_verify results/broad-warmup-042-001 results/broad-panel-037-001 results/broad-screen-038-001 results/broad-warm-screen-043-rerun
```

No paid confirmation, final-outcome evaluation, main merge or PyPI release.
