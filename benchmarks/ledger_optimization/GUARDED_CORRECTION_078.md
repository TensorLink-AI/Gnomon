# Source experiment078: chronological correction guard

Execute077without changing its recipe, cutoffs, feature definitions or gate.
All416original development tasks are scored; all125available warm-up raw
forecast records may enter mature historical training. No new warm-up forest
executions are required because this method learns raw residuals rather than
prior candidate trial scores. No protected-data or API access.

Reuse hash-verified066first-six fixed forecasts and their three CV folds; the
seventh search-selected output is not used. Both arms receive identical first-six
forecasts. Validate every CV end is658/682/706and actuals align across models.
Guard baseline: isolated045solver, fit only first two CV pairs at equal masses.
Guard forests: first two CV pairs, plus ledger-only all same-domain historical
production pairs mature by t-24h. Forecast third fold in both arms before its
actuals enter the correction decision. Training references retain logical stage
cutoff, fixed config IDs, source and recording times. Source timing is nominal
period-end as in066; this is not a claim forests existed at historical wall time.

Production baseline: unchanged saved045full-three-fold weights and point. If
its own guard enables correction, an arm fits on three CV pairs plus ledger's
historical pairs mature by t. Otherwise it returns the uncorrected045point
without a forest fit. Retain both validation scores, selected branch, fit/model
identity and all tree structures. Do not train on this task's production actuals.
Both production predictions exist before their actuals enter scoring.

Count416new guard-baseline solves,832guard forests if complete, actual enabled
production forests and32trees per forest; charge all attempted fits and preserve
failures. Log stage training sizes, temporal cohorts, clipping, source file hashes,
algorithm/package identities, wall/CPU and original costs. Historical metadata is
filtered before corresponding outcome payloads are read. Missing or inconsistent
nominal timing/config identity fails explicitly, never shortens the cohort.

Compare guarded ledger/control, unchanged045baseline, strong050, lifetime061,
and incumbent068. Use077gate unchanged. New control degradation cannot establish
success against a weak comparator. Report all domains/phases and branch counts.
Freeze this source driver before computing new source fits; independently audit
base solver, fixed forecasts, tree leaf support/weighted residual means, stage
visibility, guard decisions, predictions, scoring and costs. No paid confirmation
or final-data access after a failed development gate. Main/PyPI unchanged.
