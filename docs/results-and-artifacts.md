# Understanding results and artifacts

## Support status

Every series receives one of three statuses:

| Status | Meaning in v0.1 |
| --- | --- |
| `supported` | Evaluation completed, a baseline or candidate was selected, and no warning was triggered. |
| `weakly_supported` | A forecast was produced with at least one warning: final-test 80% interval coverage below 70%, or a limited (degraded) evaluation with fewer than four rolling folds. Read every warning. |
| `unsupported` | There was insufficient evaluation history (the warning states the exact required and available counts) or no baseline completed every selection fold. No forecast rows are emitted. |

Unsupported is a valid analytical result, unlike malformed input, which produces
a structured error. Do not relabel unsupported as merely “low confidence.”

## Selected model and baseline

`strongest_baseline` is the baseline with the lowest average selection-window
error. `selected_model` is either that baseline or a candidate model. A
candidate is selected only when its selection score beats the strongest
baseline by at least `minimum_baseline_improvement`.

The current baselines are:

- `last_value`: repeats the latest observation; and
- `seasonal_naive`: repeats values from the default seasonal period.

The current candidate models are:

- `drift`: extends the average change from the first to latest training value;
- `linear_trend`: ordinary-least-squares line fitted to all training values;
- `window_average`: mean of the most recent seasonal window;
- `theta`: the classic Theta method — exponential smoothing plus half the
  OLS trend slope; and
- `ets`: additive Holt-Winters exponential smoothing (level, trend, and —
  when history covers two seasonal cycles — seasonality), with smoothing
  parameters chosen by one-step-ahead error on the training window.

All models are dependency-free pure Python.

Selecting a baseline is intentional. A more complex method has not earned its
place unless it shows enough historical improvement.

## Scores

`selection_scores` contain average error across the model-selection folds.
`test_scores` report the selected model and strongest baseline on the untouched
final fold. The final test does not change model selection.

For nonzero actuals, the score is:

```text
sum(abs(actual - predicted)) / sum(abs(actual))
```

Lower is better. For an all-zero evaluation window, Aion falls back to mean
absolute error to avoid division by zero. Consequently, scores from those two
scales should not be compared as though they were identical units.

## Forecast columns

`forecast.csv` contains:

| Column | Meaning |
| --- | --- |
| `series` | Series identifier, or `__default__` without `--series`. |
| `timestamp` | Future timestamp generated from the validated grid. |
| `point` | Raw selected-model forecast. |
| `q10` | Lower interval bound (10th residual percentile, widened by step). |
| `q50` | Point plus the pooled residual median. |
| `q90` | Upper interval bound (90th residual percentile, widened by step). |

`q10`–`q90` form a nominal central 80% residual interval. Residuals are pooled
from every selection fold plus the calibration fold, and the spread around the
median widens with the square root of the forecast step, so uncertainty grows
over the horizon instead of staying constant. The interval is empirical, not a
probability guarantee. Check `interval_coverage` and warnings before using it
for decisions. The median `q50` may differ from `point` when residuals show
systematic bias.

## Threshold analysis

When `--threshold VALUE` is supplied, each supported series carries a
`threshold` block in `artifact.json` (and a section in `summary.md`) with:

- `probability_above`: per-step empirical probability of exceeding the value,
  computed from the pooled backtest residuals with the same square-root
  horizon widening as the intervals;
- `first_timestamp_point_above` / `first_timestamp_point_below`: earliest
  forecast timestamp whose point crosses the value; and
- `first_timestamp_interval_above` / `first_timestamp_interval_below`:
  earliest timestamp where the 80% interval reaches across the value.

These probabilities inherit every caveat of the intervals; treat them as
calibrated only as far as `interval_coverage` supports.

## Artifact files

### `artifact.json`

The canonical complete run record. It embeds:

- schema and task settings;
- absolute source path and SHA-256 content fingerprint;
- selected model, strongest baseline, scores, improvement, and support;
- final forecast rows and warnings; and
- evidence records for evaluation partitioning and support assessment.

JSON is written with NaN and Infinity disabled. A run directory is exposed only
after artifact assembly completes.

### `forecast.csv`

Flattened future forecast rows for plotting, spreadsheets, and downstream code.
Unsupported series have no rows.

### `evidence.jsonl`

Append-friendly, one-JSON-record-per-line evidence. v0.1 emits evaluation and
support records for each series. Enrichment runs add `context_ablation` and
`covariate_ablation` records; a run carrying both enrichment types also emits
one `enrichment_adjudication` record per series — the shared fold origins and
cutoffs, every candidate with its per-fold scores, each rung's comparison, and
the winner.

### `summary.md`

A compact human-readable overview. It is deliberately less detailed than
`artifact.json`; automation should consume JSON or CSV.

## Reproducibility limits

The input fingerprint proves whether source bytes changed, and the artifact
preserves the resolved task. v0.1 does not yet record a source-control commit,
wheel hash, operating-system details, or a lockfile inside each artifact. Retain
the installed package version and original input when exact reproduction matters.

