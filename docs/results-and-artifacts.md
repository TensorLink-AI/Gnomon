# Understanding results and artifacts

## Support status

Every series receives one of three statuses:

| Status | Meaning in v0.1 |
| --- | --- |
| `supported` | Evaluation completed, a baseline or candidate was selected, and final-test interval coverage did not trigger a warning. |
| `weakly_supported` | A forecast was produced, but final-test 80% interval coverage was below 70%. Read every warning. |
| `unsupported` | There was insufficient separated evaluation history or no baseline completed every selection fold. No forecast rows are emitted. |

Unsupported is a valid analytical result, unlike malformed input, which produces
a structured error. Do not relabel unsupported as merely “low confidence.”

## Selected model and baseline

`strongest_baseline` is the baseline with the lowest average selection-window
error. `selected_model` is either that baseline or the drift candidate. Drift is
selected only when its selection score beats the strongest baseline by at least
`minimum_baseline_improvement`.

The current models are:

- `last_value`: repeats the latest observation;
- `seasonal_naive`: repeats values from the default seasonal period; and
- `drift`: extends the average change from the first to latest training value.

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
| `q10` | Point plus the calibration residual's 10th percentile. |
| `q50` | Point plus the calibration residual's median. |
| `q90` | Point plus the calibration residual's 90th percentile. |

`q10`–`q90` form a nominal central 80% residual interval. It is empirical and
uses one calibration fold in v0.1; it is not a probability guarantee. Check
`interval_coverage` and warnings before using it for decisions. The median `q50`
may differ from `point` when calibration residuals show systematic bias.

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
support records for each series.

### `summary.md`

A compact human-readable overview. It is deliberately less detailed than
`artifact.json`; automation should consume JSON or CSV.

## Reproducibility limits

The input fingerprint proves whether source bytes changed, and the artifact
preserves the resolved task. v0.1 does not yet record a source-control commit,
wheel hash, operating-system details, or a lockfile inside each artifact. Retain
the installed package version and original input when exact reproduction matters.

