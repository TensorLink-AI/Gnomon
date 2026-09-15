# Prospective final analysis with a prior-ledger reference

This closes a numerical-contract gap in the earlier three-arm helper. The goal
also requires improvement over the prior ledger; plain Hermes is not that
reference. No final data were opened to make this amendment, and no final run
has started. Existing development experiments and three-arm results are unchanged.

`m5_ml_reference_analysis.analyze(panel, seeds, rows, reference_identity=...)`
requires four arms: `plain`, `gnomon`, `ledger`, and `ledger_reference`. It retains
the same 24-series panel, eight stores, three items per store, 26 origins and two
frozen agent seeds. This makes 4,992 decisions, 1,248 per arm. The existing plain
Hermes comparison is retained rather than replaced. Missing, duplicate, unexpected
or malformed reference decisions are rejected just like other arms. Failed
executions remain in the complete-grid score under the predeclared fallback.

The unchanged cluster/block resampling calculation uses the same store and origin
coordinates for all four arms: 5,000 draws, seed 20260912, eight store draws,
shared four-origin circular blocks, all items and both agent seeds retained
within each store-origin. A zero control mean makes its relative comparison
undefined; draws are never selectively discarded.

The numerical objective requires both:

- At least 20% lower mean per-case RMSLE than Gnomon without ledger, with its
  paired 95% interval's lower bound strictly above zero.
- Strictly lower mean per-case RMSLE than the frozen prior-ledger reference.
  Its paired interval is reported as well. The original goal's 20% threshold and
  explicit interval requirement apply to the no-ledger contrast; no second 20%
  threshold is silently imposed on the prior-ledger contrast.

The new `numerical_all_objective_criteria_met` distinguishes these combined
requirements from `numerical_primary_criteria_met`. `target_established` remains
false regardless of numerical results: numerical scores cannot prove untouched
status, correct executions, matched information/budgets, costs or temporal safety.

## Reference identity is still an execution requirement

A reference label, 64-character policy SHA-256 and the user-requested 1.2.0 runtime
must be supplied explicitly. The declared identity is included in the analysis
hash alongside the canonical numerical-input hash (panel, seeds and all rows).
A well-formed hash is not proof that the reference code actually ran.

Before final dispatch, the prior-ledger behavior, prompts, integration, source
hashes and runtime/budget parity must be frozen and independently checked. The
original objective names 1.1.9 ledger behavior; later user direction requires
1.2.0 for new runs. This module does not invent a mapping between those versions,
claim an existing candidate is that baseline, or accept historical development
scores as its replacement. That implementation and final freeze remain required.
The extra arm does not authorize opening the final dataset now.

## Synthetic verification

Eleven tests pass across the original and extended analysis. New cases include
passing the no-ledger criterion while losing to the prior ledger (combined
criterion false), improvement over both controls, absent/duplicate/malformed
reference cases, missing/invalid identity, and an unbeatable zero-error reference.
Changing only the reference policy identity changes the analysis hash.

The original three-arm implementation was retained and executed alongside the
refactored helper on a heterogeneous synthetic 3,744-decision fixture. Complete
result dictionaries matched, including hashes and bootstrap intervals. The
shared internal routine changes neither the original public three-arm contract
nor any saved results. No provider, runtime, network or reserved-data access was
used by these numerical tests.

Evidence: `results/m5-four-arm-analysis-synthetic-001/`, including original source,
original and extended test outputs, and the explicit compatibility check.
