---
name: use-gnomon
description: Inspect, forecast, compare recorded evidence, and calculate dates with Gnomon.
---

# Use Gnomon

Follow current schemas. Respect the user's
model choice. Ephemeris is optional; local models need no account.

## Optional Ephemeris signup

Read `onboarding.ephemeris` from `gnomon_capabilities` (or its retained summary).
When not configured and relevant, offer signup once per conversation. Respect a
decline or saved preference; never block local forecasting. If accepted, show
`signup_url`. In Hermes install/load `connect-ephemeris` via `hermes_setup`;
otherwise use `connect_command` in a hidden terminal prompt. Never put keys in
chat/MCP arguments. Connect discovers models. Reload MCP after setup.
Explicit TOML overrides
the saved profile. `configured_unverified` is local setup, not verified credentials
or permission to spend. `check_command` checks balance without forecasting.

## Execution

- Inspect file/store data with `gnomon_inspect`, using known column names.
  Reuse the frozen `data_ref`; select a returned series explicitly for panels.
  Disclose known-time assumptions, units and repairs.
- Use `gnomon_describe` for an exact observed statistic: mean, median, latest,
  minimum, maximum or sum. Start/end timestamps select an inclusive window.
  Do not substitute latest for average.
- Forecast with an explicit registered `provider` and either a typed `request`
  or `data_ref` plus positive `horizon`. Clarify unspecified model choices or
  label a baseline explicitly.
- Horizon counts grid steps: seven days means seven steps only on daily data.
  Clarify missing frequency/timezone semantics or disclose assumptions; do not
  invent dates. Covariates must meet provider capabilities and time cutoffs.
- URLs, provider entrypoints, credentials and ledger paths belong in operator startup
  configuration, not tool arguments. Availability is not spending approval:
  remote inference and evaluation may incur charges.

Numeric history needs no inspection. Baseline:

```json
{"name":"gnomon_forecast","arguments":{"provider":"last_value","request":{"history":[10,12,11],"horizon":2}}}
```

Preserve provider/revision, uncertainty, snapshot and execution identifiers.
Unknown weights or training cutoffs stay unknown. Do not silently drop unsupported
inputs. Inference alone proves neither accuracy, calibration nor action authority.

## Evaluation and evidence

Use `gnomon_evaluate` with explicit candidates, baseline, horizon and fold/budget
settings. Respect ceilings; count failed/incomplete folds. The same tool with only
`{"study_id":"<returned study_id>"}` retrieves without rerunning models. Without
a ledger, retention is bounded and session-local.

When `partial: true` accompanies `result_ref`, use `gnomon_read` for needed evidence.
A JSON pointer selects a field. Otherwise concatenate `text` pages at `next_offset`
until null before parsing JSON. References expire on session closure or eviction.
A compact summary is not the full result. `RESULT_RETENTION_LIMIT` may mean execution
already completed: check its receipt and ledger IDs before retrying billable work.

With a ledger, `gnomon_route` requires one original study, matching providers and
explicit source-availability and local-recording cutoffs. A baseline fallback means
insufficient evidence, not a demonstrated win. Rescoring appends a new study;
original predictions stay unchanged.

Use `gnomon_ledger` only when exposed. Reads, scores and outcome writes have
different permissions; writes require operator authorization and user task scope.
Keep valid, source-availability and recording times distinct. Recording a
decision neither executes nor authorizes it.

Use ledger `search` across sessions; follow `next_cursor` with unchanged filters,
even after empty pages. `ready` needs scoring, `stale` rescoring, `waiting` actuals.
Authorized `append_actual` accepts up to 1,000 `actuals`; ledger `evaluate` accepts
100 `execution_ids`. Batches are atomic; exact scoring retries reuse scores.

Ledger `compare_history` needs a series, unit, horizon, provider revisions and both
cutoffs. Its date window selects origins; search dates select recording times.
Report matched counts and exclusions. Overlapping horizons are not independent;
never cherry-pick a window. Missing models need separately budgeted evaluation.
Reads make no model calls. `next_step` grants no data-fetch, spending or action authority.

## Optional temporal calculations

Use `gnomon_temporal` only when exposed. Supply explicit facts, not invented
current times. Calendar days differ from elapsed 24 hours at clock changes.
Local timestamps need a timezone and ambiguous times a fold; nonexistent times
are rejected. Dates are not midnight instants. Ties prove neither causality nor
source availability. Calculations do not verify supplied facts.

Use the user's software for other analyses; never invent Gnomon operations.
The operator owns and loads local model callables.
