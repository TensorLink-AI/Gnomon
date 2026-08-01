# Forecasting and evaluation concepts

## Why Aion runs baselines

A forecast is useful only relative to a credible simple alternative. Aion
always tries last-value and seasonal-naive forecasts before considering the
candidate models (drift, linear trend, window average, Theta, ETS). A
candidate must beat the strongest successful baseline by a configured margin.

## Temporal evaluation

Random train/test splitting leaks future structure into time-series evaluation.
Aion instead uses ordered rolling origins. At each origin, the model sees
only earlier observations and predicts the next complete horizon.

The available origins are divided chronologically:

```text
earlier origin(s)       penultimate origin       final origin
model selection    →    interval calibration  →  report-only test
```

The selection origins choose the model. Signed residuals from the calibration
origin construct forecast quantiles. The final origin measures score and
interval coverage without changing either choice. The selected method is then
fit against all observations to forecast beyond the dataset.

This separation reduces selection and coverage optimism, although a single
calibration and test horizon still provide limited evidence. That limitation is
why warnings and support status matter.

## Per-series selection

Panel series can behave differently. Aion evaluates and selects each one
independently rather than forcing one model across an entire panel. One series
may retain seasonal-naive while another selects a candidate model or abstains.

## Improvement threshold

If baseline error is `B` and candidate error is `C`, candidate improvement is:

```text
(B - C) / B
```

With the default threshold of `0.02`, a candidate must reduce selection error
by at least 2%. When baseline error is exactly zero, Aion retains the baseline.

## Abstention

Aion distinguishes two failure classes:

- invalid data or task: a structured error and exit code `2`;
- valid data but inadequate forecasting evidence: a complete artifact with
  `unsupported` support and no future values.

This prevents a plausible-looking forecast from being returned when the
evaluation contract cannot be satisfied.

An abstention is never a dead end. Alongside `provide_more_history`, the
support assessment computes the largest horizon the supplied observations
*can* support and, when one exists, names it as a `reduce_horizon`
recovery action (and in the warning text: "retry with `--horizon N`") —
an immediate trade of forecast reach for an honest result, instead of
waiting for more data. When no shorter horizon would succeed either, the
recovery is absent rather than aspirational.

## Current methodological limits

Aion is a correct, deliberately narrow foundation—not a general forecasting
suite. Its built-in candidates are deterministic classical models (drift,
window average, linear trend, theta, ETS), extendable by plugin model
backends (the `aion.model_backends` entry-point group; the first-party
statsforecast backend ships as the `statistical` extra) and optional
sandboxed TSFM adapters — every candidate, whatever its origin, enters
the same rolling folds and is selected only by beating the mandatory
baselines. Seasonal periods are detected or overridden, not learned per
model. Covariates and context events are admitted only through
identical-fold ablation; when both are supplied, a deterministic
adjudication ladder compares the base model against every admitted
challenger on identical folds and records the comparison as evidence.
There are no transformations and no dedicated intermittent-demand
methods. Use `aion capabilities` as the machine-readable
source of truth.

