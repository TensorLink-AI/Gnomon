# Small retrospective matched cohort

`matched-retrospective.jsonl` contains 11 tool-neutral tasks for the existing
ordinary/lean/full shared driver. Its manifest pins source bytes, cutoffs,
transformations, episode input and the resulting corpus hash. Rebuild in memory
and print the cases/manifest with `python -m benchmarks.workflow.cohort`. Changing
a source CSV without updating its reviewed pin fails before generation.

The cohort contains:

- Four free-choice point-forecast tasks: one window each from the existing web
  traffic, temperature, pedestrian and retail CSVs. Each exposes 64 observations and withholds
  the next four. Values are positively affine-transformed with recorded constants.
  The model receives indexed values/cadence, not original source names, physical
  units or fabricated calendar timestamps. `history.csv` and inline history agree.
- Four synthetic utility tasks: time-weighted energy, expected inventory cost
  under a supplied distribution and approval constraint, elapsed time across
  explicit UTC offsets, and abstention when the target is not selected.
- The three existing committed episodes: forecast revision, denied approval and
  bitemporal replay. They retain their separate smoke/protocol interpretation.

The four CSVs were compared value-for-value with Prophet's example datasets at
commit `79ef5ecbe85179a3a2afa3b62f0d2a6b223cc7db`; exact upstream URLs and hashes
are recorded in the manifest. Three matched directly. Temperature matched after
excluding 12 nonfinite values; its selected final 68 rows contain no missing value
or five-minute gap. The selected daily and monthly windows also have their stated
cadence. Original timestamp labels have no declared timezone and remain provenance
only. This verifies extraction, not original collection or supply-chain integrity.
Rebuilding uses already tracked CSVs without network or new production dependencies.

The initial Statsmodels-derived candidates were rejected: the ordinary image
bundles those complete datasets, making future lookup directly possible. The final
software checks require Prophet to be absent and check for the known source CSV
filenames in installed packages. This is a specific bundle-leakage check, not proof
that no transformed/renamed copy or trained model could ever recognize the series.

These are retrospective holdouts at the task-input boundary, **not certified
unseen model-training data**. The underlying series are famous; affine transforms
do not prove decontamination. Four windows are a small scoped experiment, not a
SOTA forecast leaderboard, an independent-task guarantee or evidence of agent
improvement before an actual model run. No data or task was selected based on the
new arms' results; no paid agent run has been made during cohort preparation.

## Forecast scoring

The optional private `oracle.forecast` object names numeric answer keys, a positive
error scale, and a nonnegative maximum MAE. All keys must be unique and present in
the numeric oracle; limits must be finite. Those forecast keys are graded as one
complete-horizon MAE check, not individual exact-value matches. Any other numeric
or choice requirements retain their usual checks.

For this cohort, success requires all four point forecasts and MAE no worse than
repeating the last observed value. That baseline's holdout MAE is fixed before
agent evaluation and stays private. Any model/library may produce the forecast;
the agent is not told to use the baseline. This is an explicit minimum-competence
criterion, not a claim that the last-value baseline is SOTA.

Every forecast row also reports MAE, RMSE and MASE. Here MASE divides MAE by mean
absolute **one-step training changes**, not a seasonal denominator or TemporalBench's
official metric. Missing horizons, abstention, task errors and nonfinite derived
losses produce incomplete forecasts with null loss, never zero error. Forecast
numbers come only from the submitted answer, not recovered tool output.

Matched comparisons retain all-task success, errors, cost and budget results. The
additional forecast-error report gives planned/complete/missing counts and mean
MASE conditional on complete forecasts, both per arm and on the common three-arm
complete subset. Conditional loss cannot establish all-task improvement when
completion differs. Raw MAE/RMSE remain per case; differently scaled series are not
silently pooled in physical units. Existing cases without `oracle.forecast` retain
their prior scoring and corpus hashes.

Use the [operator-filled experiment templates](../experiment/README.md), declare
the cohort and execution order before results, and preserve the failed rows.
The lean/full arms have additional service compute, and full enables optional
ledger and temporal tools. Both use explicit providers. This is a declared feature
comparison, not an isolated causal test of tool count.
