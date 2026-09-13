# Shared-ensemble comparison 044/045: small ledger gain, gate failed

The ledger ensemble improved mean RMSLE by **0.96%** against the equally capable
current-CV ensemble control. It failed the 20% gate and worsened electricity.
No paid confirmation or final-outcome evaluation is justified by this result.

| Rule | Overall RMSLE | Electricity | Pedestrian |
| --- | ---: | ---: | ---: |
| Original single-model CV choice | 0.281583 | 0.119683 | 0.443482 |
| **Primary control: CV-fitted ensemble** | **0.260632** | **0.111226** | **0.410037** |
| **Ledger-fitted ensemble** | **0.258137** | **0.113018** | **0.403256** |
| Uniform ensemble diagnostic | 0.263883 | 0.126299 | 0.401467 |

The control ensemble itself improved 7.44% versus single-model CV selection.
The ledger ensemble's 8.33% improvement against that old control therefore must
**not** be presented as ledger value. The matched primary gain is only 0.96%:
electricity worsened 1.61%, while pedestrians improved 1.65%. Mature/later
development gains were 1.07%/1.22%; those are already-used slices, not holdouts.
The uniform ensemble outperformed the ledger ensemble in pedestrians; it is
retained as a diagnostic, not substituted as the primary result.

## Failed solver attempt and explicit refinement

Attempt 044 was frozen at `08d6ab0` and stopped after 407 complete cases. One
ledger fit reported solver success but had convex gap 1.12714e-5, narrowly above
the frozen 1e-5 certificate threshold. Keep the failed attempt and all 816
started fits. No aggregate accuracy comparison was computed from that attempt.

Refinement 045 was frozen at `acf05e4` before rerunning. It changed only SLSQP
termination ftol from 1e-10 to 1e-12, kept the acceptance threshold and all
data/objectives/rules unchanged, and recomputed **every** fit. The wrapper loads
isolated copies of the original modules, so the frozen 044 source files and
artifacts are preserved. Both implementation identities are recorded.

All 416 cases and 832 weight fits completed in 045. Four synthetic tests passed,
including analytic gradients and an exact-fit nonsmooth case. An independent
scalar-arithmetic verifier passed **43,441 checks with zero failures**: source
hashes, exact training-pair hashes, objectives, feasible weights, convex gap
bounds, visible historical cohorts, derived predictions and all aggregates.
Original providers were not refitted. The log-space combined forecasts are
explicitly new derived outputs; old execution records are unchanged.

## Cost and reproducibility

Attempt 044: 816 started weight fits, 12,855 completed-fit iterations plus the
failed fit's 14 iterations, 1.8969s compute wall time. Refinement 045: 832 fits,
13,830 iterations, 2.1377s compute wall and 2.0808s CPU, excluding source loading
and audit. Both attempts count. The inherited common forecast construction
still cost 12,984 computations/6,492 estimator fits. No additional original
provider fits, API calls or reserved data access. Prior failed experiments and
their costs remain separate, not erased by reuse.

Exact inputs, certificates, weights, derived forecasts and exceptions are under
`results/broad-ensemble-044-001/` and `results/broad-ensemble-045-001/`. Both
archives and their contents were hash-verified. Tracked receipts:
[failed 044](evidence/broad-ensemble-044.json),
[completed 045](evidence/broad-ensemble-045.json).

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m benchmarks.ledger_optimization.ensemble_refinement results/broad-screen-038-001 results/broad-warm-screen-043-001 results/broad-ensemble-045-rerun
python3 -m benchmarks.ledger_optimization.broad_ensemble_verify results/broad-screen-038-001 results/broad-warm-screen-043-001 results/broad-ensemble-045-rerun
```

This is a numerical development prototype with a stronger common action space,
not a Gnomon release feature or an agent causal comparison. It does not establish
the 20% held-out objective or a positive 95% uncertainty interval. Main/PyPI and
reserved outcomes are unchanged. Any subsequent integration remains on 1.2.0.
