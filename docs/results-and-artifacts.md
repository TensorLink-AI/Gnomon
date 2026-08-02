# Understanding results and artifacts

## Support status

Every series receives one of five statuses:

| Status | Meaning |
| --- | --- |
| `supported` | Evaluation completed, a baseline or candidate was selected, and no warning was triggered. |
| `weakly_supported` | A forecast was produced with at least one warning — interval-coverage shortfalls, or disclosed assumptive data repairs. Read every warning. |
| `degraded` | A forecast was produced from a limited evaluation (single trailing holdout instead of separated selection, calibration, and test windows). |
| `supported_ensemble` | An inverse-error-weighted ensemble of eligible models beat the strongest baseline. |
| `unsupported` | There was insufficient evaluation history (the warning states the exact required and available counts) or no baseline completed every selection fold. No forecast rows are emitted. |

Each result additionally carries a `support_assessment` — a five-state
harness-wide status (`supported` / `conditionally_supported` /
`inconclusive` / `unsupported` / `invalid`) with typed reasons,
assumptions, sensitivity, and recovery actions.

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

Lower is better. For an all-zero evaluation window, Gnomon falls back to mean
absolute error to avoid division by zero. Consequently, scores from those two
scales should not be compared as though they were identical units.

## Forecast columns

`forecast.csv` contains:

| Column | Meaning |
| --- | --- |
| `series` | Series identifier, or `__default__` without `--series`. |
| `timestamp` | Future timestamp generated from the validated grid. |
| `point` | Raw selected-model forecast. **Not** the centre of the interval. |
| `q10` | 10th residual percentile at this lead time. |
| `q50` | Point plus the median residual — the bias-corrected centre. |
| `q90` | 90th residual percentile at this lead time. |
| `q05`, `q20`, `q30`, `q70`, `q80`, `q95` | The remaining emitted levels. |
| `point_bias_correction` | Exactly `q50 - point`. |

### `point` is not the median

`point` is what the selected model output. Every `q*` level is recentred on
the median backtest residual, so if the model carries systematic bias the
two differ — sometimes substantially. In the `quickstart-mcp` vintage
example `point` is 333.86 while `q50` is 346.80, which puts the headline
number near the 30th percentile of Gnomon's own distribution.

**Read `q50` when you want the centre.** `point_bias_correction` ships on
every row so the gap is visible without recomputing it, and any result
where it is non-zero carries a `point_is_not_the_median` disclosure in its
support assessment naming the size.

### What the interval is

`q10`–`q90` form a nominal central 80% split-conformal interval, built from
the residual order statistics at each lead time. Two properties are
enforced: half-widths are fitted monotone across lead times, and levels
within a lead cannot cross.

Residuals are **not** widened by `sqrt(step)` — the spread at a lead time is
the spread measured at that lead time. A lead with fewer than 8 residuals
borrows the pooled set rather than trusting a two-sample quantile, and with
the usual three or four folds *every* lead borrows, which makes the interval
the same width across the whole horizon. That is honest, and it is stated:
such a run carries a `constant_interval_width` disclosure.

By default residuals are pooled across the selection folds and the
calibration fold. The selected model was chosen to minimise error on the
selection folds, so those residuals are optimistically small and the
interval is narrower than strict split conformal would give; every run says
so via `conformal_residuals_pooled_across_selection`. Set
`evaluation.pool_residuals: false` in `gnomon.yaml` to calibrate on the
held-out fold alone — genuine split conformal, noisier and wider.

The interval is empirical, not a probability guarantee. Check
`interval_coverage`, the disclosures, and the warnings before using it for
decisions — and note that measured coverage comes from a single test fold
of `horizon` points, which the `coverage_sample_size` disclosure states.

## Threshold analysis

When `--threshold VALUE` is supplied, each supported series carries a
`threshold` block in `artifact.json` (and a section in `summary.md`) with:

- `probability_above`: per-step empirical probability of exceeding the value,
  computed from the pooled backtest residuals rescaled to the same per-lead
  conformal spread the published intervals use;
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
- final forecast rows, warnings, and notes (informational disclosures —
  e.g. an eligible-but-uninstalled TSFM tier — that, unlike warnings,
  never downgrade support); and
- evidence records for evaluation partitioning and support assessment.

JSON is written with NaN and Infinity disabled. A run directory is exposed only
after artifact assembly completes.

### `forecast.csv`

Flattened future forecast rows for plotting, spreadsheets, and downstream code.
Unsupported series have no rows.

### `evidence.jsonl`

Append-friendly, one-JSON-record-per-line evidence: evaluation and support
records per series, snapshot-access proof, any `data_repair` disclosure, and
context/covariate ablations when supplied. Artifact directories also contain
`lineage.json` — typed artifacts, evidence, and verified claims.

### `summary.md`

A compact human-readable overview. It is deliberately less detailed than
`artifact.json`; automation should consume JSON or CSV.

## Reproducibility limits

The input fingerprint proves whether source bytes changed, and the artifact
preserves the resolved task. Gnomon does not yet record a source-control commit,
wheel hash, operating-system details, or a lockfile inside each artifact. Retain
the installed package version and original input when exact reproduction matters.

