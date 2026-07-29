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

## Current methodological limits

v0.1 is a correct, deliberately narrow foundation—not a general forecasting
suite. It has one candidate statistical model, fixed seasonal periods, no
covariates, no transformations, no intermittent-demand methods, no TSFM, and no
context-event evaluation. Use `aion capabilities` to detect future changes.

