---
name: aion
description: "Produce evidence-backed time series forecasts that abstain when the data cannot support a forecast. Use for any forecasting, projection, or 'what will X be next week/month' request involving a time series data file."
version: 0.1.0
author: TensorLink AI
license: Apache-2.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Forecasting, TimeSeries, Data, Statistics, Evidence]
---

# Aion — evidence-backed forecasting

Aion forecasts time series from a CSV or Parquet file. It differs from asking a
model to guess at numbers in one specific way: **it refuses to forecast when the
evidence is inadequate, and that refusal is the correct answer.**

The runtime ships today as the `headwater` command (the rename to `aion` is
pending).

## The one rule

Aion returns a `support` level per series. It is the most important field in the
output and it governs what you are allowed to say to the user.

| `support` | Meaning | What you do |
|---|---|---|
| `supported` | Selected model beat the strongest baseline on held-out folds; interval coverage held up | Report the forecast and its intervals |
| `weakly_supported` | Forecast produced, but a check failed (usually interval coverage below 70%) | Report the forecast **and** state the warning verbatim |
| `unsupported` | Aion abstained. No forecast exists | Tell the user Aion abstained and why. **Stop.** |

`unsupported` is a **successful run**. It exits `0` and reports
`"status": "complete"`. It is not an error, not a retryable condition, and not a
signal to try different parameters.

When `support` is `unsupported`:

- `selected_model` is `null`
- `warnings` explains what was missing
- `forecast.csv` contains a header row and **no data rows**

## Commands

Check what the installed runtime can actually do. Treat this as the source of
truth over anything written here:

```bash
headwater capabilities
```

Validate a file and see its series, date range, and inferred frequency before
forecasting:

```bash
headwater inspect data.csv --time timestamp --target requests
```

Forecast:

```bash
headwater forecast data.csv \
  --time timestamp --target requests \
  --horizon 3 --frequency D \
  --output ./forecast-runs
```

Flags: `--series COLUMN` for multi-series (panel) files. `--frequency` one of
`h`, `D`, `W`, `MS` — omit it and Aion infers it. `--minimum-baseline-improvement`
(default `0.02`) is how much a candidate model must beat the best baseline by
before it gets selected.

## Reading the result

`forecast` prints a small JSON summary to stdout and writes a run directory:

```json
{
  "status": "complete",
  "forecast_id": "forecast_1c0d1cf5...",
  "artifact_path": "/abs/path/forecast-runs/forecast_1c0d1cf5...",
  "results": [
    {"series": "__default__", "support": "supported",
     "selected_model": "drift", "warnings": []}
  ]
}
```

The run directory holds `artifact.json`, `forecast.csv`, `evidence.jsonl`, and
`summary.md`.

**Read `summary.md` first** — it is written for exactly this purpose. Only open
`forecast.csv` when the user wants the actual numbers, and only open
`artifact.json` when they ask how a model was chosen. Do not read the whole run
directory into context by default; `artifact.json` embeds full evaluation
scores and evidence and is large for multi-series files.

## Errors

Errors exit `2` and print JSON to **stderr** with a `code`. These are genuine
failures and usually mean the data or the flags need fixing:

`INPUT_NOT_FOUND`, `UNSUPPORTED_INPUT`, `MISSING_COLUMNS`, `INVALID_TIMESTAMP`,
`INVALID_TARGET`, `EMPTY_DATASET`, `MIXED_TIMEZONES`, `DUPLICATE_TIMESTAMPS`,
`IRREGULAR_TIME_GRID`, `AMBIGUOUS_FREQUENCY`, `FREQUENCY_MISMATCH`,
`UNSUPPORTED_FREQUENCY`, `INVALID_HORIZON`, `MISSING_OPTIONAL_DEPENDENCY`.

`MISSING_COLUMNS` lists the available columns in `details` — use that to correct
`--time` / `--target` rather than guessing again.

`IRREGULAR_TIME_GRID` means the series has a gap. Aion rejects gapped data
rather than silently imputing it. Tell the user which timestamp the gap follows
(it is in `details`) and ask how they want it handled. Do not fill gaps yourself.

`MISSING_OPTIONAL_DEPENDENCY` on a Parquet file means the extra is not
installed: `pip install 'headwater-forecast[parquet]'`.

## How much data is needed

Aion partitions the series into disjoint windows — earlier folds select the
model, the second-to-last calibrates prediction intervals, and the last is a
report-only test. That requires roughly `max(2 × season, 2 × horizon, 8) +
4 × horizon` observations, where season is 24 (`h`), 7 (`D`), 52 (`W`), or 12
(`MS`).

Concretely: daily data at horizon 3 needs 26 observations. Weekly data needs
well over 100. If a series is shorter, Aion abstains and the warning states the
exact number required.

If the user's data is too short, say so and state the number needed. Shortening
the horizon does change the threshold, but do not do it silently to manufacture
a forecast — that trades a real answer for a worse one the user did not ask for.

## What not to do

- **Do not treat `unsupported` as a failure to work around.** No re-running with
  a shorter horizon, a different frequency, or a trimmed file to get a number
  out. If Aion abstained, the answer to the user is that it abstained.
- **Do not supply the missing forecast yourself.** Aion abstaining and you then
  estimating the numbers from the data or from general knowledge defeats the
  entire purpose of running it.
- **Do not present a `weakly_supported` forecast as if it were solid.** State
  the warning.
- **Do not describe intervals as confidence intervals.** They are empirical
  residual quantiles from the calibration fold. On short or perfectly
  regular series they can collapse to the point estimate.
- **Do not pass context you learned elsewhere into a forecast.** Aion's
  guarantee is that it only used information present in the file. There is
  currently no supported way to inject external context, and narrating it
  alongside the result implies a rigor the run does not have.

## Reporting to the user

Lead with the support level, not the number. A good report:

> Aion forecasts 205, 208, 211 requests for Feb 5–7 (`supported`, selected
> model: drift, beat the seasonal-naive baseline). Intervals collapsed to the
> point estimate because the calibration residuals were zero — this series is
> almost perfectly linear, so treat the precision as an artifact of synthetic-
> looking data rather than genuine certainty.

And for an abstention:

> Aion abstained — the series has 12 observations and needs at least 26 to run
> separated selection, calibration, and test windows. There is no forecast. To
> get one you would need more history, not different settings.
