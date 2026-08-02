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

## Context events

A context event is something you know about the world that the series
alone cannot show: a promotion window, a capacity cap, a migration. Aion
never takes your word for its *effect* — it measures one, on the same folds
everything else competes on, and excludes the event when it cannot.

`examples/context_events.json` is a worked file with one of each kind:

```bash
aion forecast examples/messy_requests.csv \
  --time timestamp --target requests --horizon 14 \
  --context examples/context_events.json
```

**An event with a `claim`** states a bound on what is possible.
`capacity-cap-2026-06` says throughput cannot exceed 360, and that bound is
projected onto every emitted quantile after the model has said what it
believes. Bounds are admissible; pinned values are not — an event that
supplied a *value* would be supplying the answer. A bound the training
window already breaches is rejected, with the violating timestamps named.

**An event without one** is a window whose effect Aion estimates.
`marketing-push-2026-06` marks a campaign; the context ablation measures
its effect from detrended history and admits it only if it beats the
history-only baseline on identical folds. The effect shape (level, decay,
ramp) is chosen by the same measurement, never by the caller — a caller who
could name the shape could fit a story to the data.

Every event needs a `known_at`. It is what makes the backtest honest: a
fold cutting at T may only use events knowable by T, so an event recorded
after the fact cannot improve a historical fold.

Events must carry an explicit timezone offset. When the dataset's own
timestamps are naive — as every example here is — the windows are matched
on wall-clock time and the result carries a `context_timezone_aligned`
disclosure saying so.

## Current methodological limits

Aion is a correct, deliberately narrow foundation—not a general forecasting
suite. Its built-in candidates are deterministic classical models (drift,
window average, linear trend, theta, ETS) plus optional sandboxed TSFM
adapters; seasonal periods are detected or overridden, not learned per
model. Covariates and context events are admitted only through
identical-fold ablation; when both are supplied, a deterministic
adjudication ladder compares the base model against every admitted
challenger on identical folds and records the comparison as evidence.
There are no transformations and no dedicated intermittent-demand
methods. Use `aion capabilities` as the machine-readable
source of truth.

