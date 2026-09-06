---
name: use-gnomon
description: Use Gnomon to inspect time series, forecast with chosen models, compare historical evidence, retrieve forecast history, or perform exposed date and time calculations.
---

# Use Gnomon

Follow the current session's schemas. Discover capabilities
when provider names, supported inputs or storage are unknown. Respect the user's
chosen model. Ephemeris is one optional connector, not the default forecasting path.

## Default execution session

- Inspect file/store data with `gnomon_inspect`, using known column names.
  Reuse the frozen `data_ref`; select a returned series explicitly for panels.
  Disclose knowledge-time assumptions, declared units and repairs.
- Use `gnomon_describe` for an exact observed statistic: mean, median, latest,
  minimum, maximum or sum. Start/end timestamps select an inclusive window.
  Do not substitute latest for average.
- Forecast with an explicit registered `provider` and either a typed `request`
  or `data_ref` plus positive `horizon`. If model choice is unspecified, clarify
  when it matters or label a baseline as a reference, not the user's chosen model.
- Horizon counts grid steps: seven days means seven steps only on daily data.
  Clarify missing frequency/timezone semantics or disclose assumptions; do not
  invent dates. Covariates must meet provider capabilities and time cutoffs.
- URLs, imports, credentials and ledger paths belong in operator startup
  configuration, not tool arguments. Availability is not spending approval:
  remote inference and evaluation may incur charges.

For supplied numeric history, inspection is unnecessary. This offline example
uses the built-in `last_value` baseline:

```json
{"name":"gnomon_forecast","arguments":{"provider":"last_value","request":{"history":[10,12,11],"horizon":2}}}
```

Preserve provider/revision, uncertainty, snapshot and execution identifiers.
Unknown weights or training cutoffs stay unknown. Do not silently drop unsupported
inputs. Inference alone proves neither accuracy, calibration nor action authority.

## Evaluation and retained evidence

Use `gnomon_evaluate` for requested comparisons with explicit candidates, baseline,
horizon and fold/budget settings. Respect operator ceilings and task scope; count
failed and incomplete folds. Retrieve a saved study by calling the same tool with
only `{"study_id":"<returned study_id>"}`; this does not rerun models. Without a
ledger, study retention is bounded and session-local.

When `partial: true` accompanies `result_ref`, use `gnomon_read` for needed evidence.
A JSON pointer selects a field. Otherwise concatenate `text` pages at `next_offset`
until null before parsing JSON. References expire on session closure or eviction.
A compact summary is not the full result. `RESULT_RETENTION_LIMIT` may mean execution
already completed: check its receipt and ledger IDs before retrying billable work.

With a ledger, `gnomon_route` requires one original study, matching providers and
explicit source-availability and local-recording cutoffs. A baseline fallback means
insufficient evidence, not a demonstrated win. Rescoring appends a new study;
original predictions stay unchanged.

Use `gnomon_ledger` only when exposed. Reads, score creation and outcome/import
writes have different permissions; writes require operator authorization and user
task scope. Keep valid time, source availability and recording time distinct.
Recording a decision neither executes nor authorizes an action.

## Optional temporal calculations

When exposed, `gnomon_temporal` handles date shifts, instant normalization, elapsed
duration, half-open intervals and event ordering. Supply explicit facts, not an
invented current time. Calendar days differ from elapsed 24-hour periods at clock
changes. Local timestamps need a named timezone and ambiguous times an explicit
fold; nonexistent times are rejected. Dates are not midnight instants. Timestamp
ties establish neither causality nor source availability. Calculations do not
verify the supplied facts.

For other analyses, use the user's chosen software or host tools. Do not invent
extra Gnomon operations, reinterpret an exact statistic, or imply that a recorded
decision was executed. Legacy context/publication profiles and model installers
are retired; local models are loaded and owned by the operator's provider callable.
