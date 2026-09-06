# Compatibility and migration

## 0.9 development: one execution model

This is a deliberate breaking simplification. The new goal is provider-neutral
time-series execution, explicit data/time semantics, optional durable evidence
and budgeted evaluation. Features that only served the older opinionated runtime
are removed, not hidden behind flags or moved into another shipped package.

Removed:

- The evaluated forecasting/macros runtime and its top-level Python `forecast`,
  `investigate_change`, `detect_anomalies`, `decide` and `monitor` exports.
- Legacy MCP profiles core, data, decision, evidence and full; their tool registry
  and context/publication/sensitivity/effect-learning machinery.
- Built-in TSFM catalogues, sandbox installers, per-library StatsForecast adapters,
  generic legacy HTTP adapters and automatic model-admission/routing policies.
- TrackingStore writers, feedback/supervision commands, report generation and
  Prometheus/webhook monitoring. Host applications own these workflows.
- CLI forecast/investigate/detect/decide/monitor/track/tsfm/context/covariates/report/eval.

Retained:

- `ForecastRequest → ForecastResult`, `InferenceEngine`, callables and fresh factories.
- `GnomonSession` shared by Python, CLI and MCP; optional Ephemeris connector.
- Three reference baselines: last_value, historical_mean and seasonal_naive.
- Explicit snapshot-backed file/store inspection, exact descriptive statistics,
  revision-aware `TemporalStore`, optional `TemporalLedger`.
- Budgeted backtesting and cutoff-bound study routing.
- Optional explicit date/instant/interval/event calculations.
- Read-only import of old sealed/unsealed artifacts and tracking registries.

Use `gnomon infer` or `session.forecast` to run a named provider.
Use `gnomon evaluate` or `session.evaluate` for a separate comparison.
Choose and load model libraries in your own callable; use a factory when fitting
must be fresh per fold. Unsupported request features fail instead of disappearing.

There is only one MCP profile: execution. Enable ledger/temporal tools with
operator configuration, not agent arguments. Legacy commands and profiles fail;
they do not silently reinterpret old arguments.

CLI inspect/describe now use the execution contract: explicit column mappings,
repairs off by default, one selected panel series, exact observed statistic.
Full CLI results contain provenance but their temporary references expire at exit.
MCP/Python sessions can reuse references and page large retained results.

## Existing data and recovery

No user database, saved forecast or input file is deleted. Keep old artifacts
and registries read-only; import into a separate ledger when needed. Missing
historical inputs, recording times and overwritten scores remain unknown.

Git commit `333ed2c` preserves the pre-cull implementation, tests and docs.
The published `0.8.0rc3` artifacts and tag are immutable and unchanged.
This cleanup is `0.9.0.dev0` in source only; it has not been published to PyPI.

[Provider/session contract](docs/production/INFERENCE.md) ·
[Historical imports](docs/production/OPERATIONS.md)
