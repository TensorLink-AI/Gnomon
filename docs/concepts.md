# Concepts and boundaries

## Inference is not evaluation

Inference executes a selected provider. Evaluation compares explicit providers
and a baseline on matched historical folds, with bounded calls/time. A successful
request says nothing by itself about accuracy. Quantiles are uncertainty outputs,
not proof of calibration.

## Three different times

- **Valid time:** when the observation applies in the world.
- **Source availability:** when its publisher made it available.
- **Recorded time:** when this store received that revision.

Source-time and recorded-time replay answer different questions. Late ingestion
must not be presented as locally known in the past. Unknown recording times cannot
be reconstructed from valid times.

## Forecast history is immutable

An execution ID identifies a run. A content fingerprint identifies its inputs and
provider configuration; it is not execution identity. The ledger appends actual
revisions and evaluations instead of overwriting forecasts or scores. Decisions
retain their evidence cutoff. See [ledger operations](production/OPERATIONS.md).

## User-owned models

Callables and fresh factories implement the forecast boundary. Requests/results
carry grids, identity, units and capabilities. Libraries stay outside Gnomon.
Ephemeris uses the same boundary for remote inference. Tasks such as imputation
need their own output semantics: not every returned array is a forecast.

## Calculations are not authority

Temporal calculations make arithmetic/order explicit; forecasts and backtests
provide evidence. Neither grants permission to trade, order, deploy or otherwise
act. Recommendations remain recommendations. Unsupported requests are rejected,
not silently reinterpreted.

Retained advanced workflows have additional support/publication rules described in
[results](results-and-artifacts.md) and [publication modes](publication-modes.md).

For advanced context-event enrichment, the shipped
[examples/context_events.json](../examples/context_events.json) uses `known_at`
to distinguish source availability from the event's effective time. That context
must pass the evaluated workflow's admission rules; it is not automatically added
to a direct provider request. See [covariates](covariates.md).
