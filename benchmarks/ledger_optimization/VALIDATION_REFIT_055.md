# Numerical amendment055: uniformly refit locked validation mixtures

054 completed all raw forecasts but rejected an exact-fit global-CV certificate.
Preserve the failed run and all costs. No aggregate validation result was read.
This amendment changes the global mixture's numerical search/certificate only;
it does not change its exact training objective, penalty, inputs or acceptance
threshold. Every task/control/ledger mixture is recomputed under one rule.

Search the simplex using sqrt(mean(error²)+1e-12), keeping the original1e-6
uniform-anchor regularizer. The search objective exceeds the exact objective by
at most1e-6. SLSQP: uniform start,ftol1e-14,maxiter500; at most32 convex line
refinements, with the same brentq tolerances as050. Report both objectives.

Certify the **exact unsmoothed** objective. For each case, let n be its RMSLE,
s=sqrt(n²+1e-12), and use norm support vector e/(sqrt(24)*s), whose norm is<=1.
The exact objective admits that affine lower support plus the regularizer's
tangent. With its aggregate gradient g, the simplex suboptimality bound is:

`w·g − min(g) + sum(case_mass * (n − n²/s))`.

This bound is valid even when a case has zero residual. Require solver success,
feasible weights, exact objective no worse than initialization within1e-8, and
exact suboptimality bound<=1e-5, unchanged. Never accept on solver success alone.
Scalar independent arithmetic verifies this certificate. Synthetic tests cover
an exact-fit median cusp, arbitrary feasible-point lower bounds, analytic search
gradients and the finite support at zero. Freeze before fitting validation data.

Reuse054's533 byte-identical raw cases/contexts. No provider is rerun. All416
scored mixtures and the predeclared10,000 paired bootstrap replicates are
recomputed; no hard case is dropped. Block objectives, anchors' role, retrieval,
budgets, denominators and gates retain052/050 semantics. Record all inherited
forecast costs,054's failed weight fits, and055's additional fits separately.
Label results as disjoint-series validation with a disclosed numerical amendment,
not an uninterrupted preregistered run or proof of agent benefit.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m benchmarks.ledger_optimization.validation_refit results/broad-validation-054-001 results/broad-validation-055-001
```

No API calls, final-reserved reads, main merge or PyPI publication.
