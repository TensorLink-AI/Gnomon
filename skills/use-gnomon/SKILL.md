---
name: use-gnomon
description: Use Gnomon's MCP/CLI tools to inspect series, forecast with chosen models, backtest and compare them, read saved results and do date calculations. Use when gnomon_* tools are connected; for Python forecasting use forecast-with-gnomon, for past-outcome review use-gnomon-ledger.
---

# Use Gnomon

Follow the session's schemas and the user's chosen model. Ephemeris is optional;
local models need no account. Usual order: `gnomon_capabilities` when names or
inputs are unknown; `gnomon_inspect`/`gnomon_describe` for file or store data;
`gnomon_forecast` with an explicit provider; `gnomon_evaluate` to compare models;
`gnomon_read` for partial results; `gnomon_ledger`/`gnomon_memory` when exposed.

## Task discovery and results

Use capabilities `tasks`, then `{"task":"forecast"}` for
setup and a schema/template. Older builds: `schema_tool`.
Read `agent_summary` or `agent_summary_read`; pointers
refer to the full response (`agent_summary_pointer_root` when paged).
Follow-ups are suggestions: fill `requires` with facts and
cutoffs, check `effect` and task authority. The host supplies actuals and scores; pending outcomes cannot establish accuracy.

## Optional Ephemeris signup

Read `onboarding.ephemeris` from `gnomon_capabilities`. When not configured and
relevant, offer signup once per conversation; respect a decline and never block
local forecasting. If accepted, show `signup_url`. In Hermes use the
`connect-ephemeris` skill; otherwise `connect_command` in a hidden terminal prompt.
Never put keys in chat or MCP arguments. Reload MCP after setup.
`configured_unverified` is not verified credentials or permission to spend.

## Execution

- Inspect data with `gnomon_inspect` using known columns; reuse the frozen
  `data_ref`; select panel series explicitly; disclose known-time assumptions,
  units and repairs.
- `gnomon_describe` gives an exact observed statistic (mean, median, latest,
  minimum, maximum, sum) over an optional inclusive window; never substitute one
  statistic for another.
- Forecast with an explicit registered `provider` and a typed `request` or
  `data_ref` plus `horizon` (grid steps, not days). Label baselines. Do not
  invent dates, frequencies or timezones; covariates must meet capabilities.
- URLs, credentials and ledger paths are operator configuration, not arguments.
  Availability is not spending approval: remote calls may incur charges.

Numeric history needs no inspection. Baseline:

```json
{"name":"gnomon_forecast","arguments":{"provider":"last_value","request":{"history":[10,12,11],"horizon":2}}}
```

Routers under `ledger.routers` are used like a provider name; they need
`series_id`, `timestamps` and `future_timestamps`. Report the `routing` block's
served provider, reason and `evidence_level`. With `memory`, cite `effective_n` and
the top `memory_neighbours`; a low `effective_n` or unrelated neighbours is weak
evidence. A baseline fallback is not a finding that the baseline is best. Setting
up and testing routers: [route-with-gnomon](../route-with-gnomon/SKILL.md).

Preserve provider/revision, uncertainty and execution IDs. Unknown weights or
training cutoffs stay unknown. Inference alone proves neither accuracy nor
action authority.

## Evaluation and evidence

Use `gnomon_evaluate` with explicit candidates, baseline, horizon and budget;
count failed folds. `{"study_id": ...}` alone retrieves without rerunning.
With `partial: true` and `result_ref`, use `gnomon_read` (a JSON pointer selects a
field; otherwise concatenate pages until `next_offset` is null). A compact summary
is not the full result. `RESULT_RETENTION_LIMIT` may follow completed billable
work: check receipts and ledger IDs before retrying.

With a ledger, `gnomon_route` needs one original study, matching providers and
explicit source-availability and local-recording cutoffs. A baseline fallback
means insufficient evidence. Rescoring appends a study; predictions stay unchanged.

Use `gnomon_ledger` only when exposed. Reads, scores and outcome writes have
different permissions; writes need operator authorization. Recording a decision
neither executes nor authorizes it. For comparisons, reviews and lessons follow the
[ledger skill](../use-gnomon-ledger/SKILL.md) (packaged with Gnomon; install it if
absent). `next_step` grants no authority.

## Optional temporal calculations

Use `gnomon_temporal` only when exposed, with explicit facts, not invented current
times. Calendar days differ from elapsed hours at clock changes; local times need a
timezone. Calculations do not verify supplied facts.

For other analyses use the user's software; never invent Gnomon operations.

## Bounded recovery

- Unknown provider: use a name from `gnomon_capabilities`; never substitute models.
- `INVALID_ARGUMENTS`: apply the returned repair options once.
- Missing tool: it needs operator configuration; report it, do not emulate it.

Report which tools ran, provider/revision and IDs, and what was not done.
