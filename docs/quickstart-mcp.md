# Quickstart: data to a provider forecast

## Start one session

Install the reviewed wheel or use the package launcher:

```bash
gnomon mcp serve
# Or: uvx --from 'gnomon-forecast==0.8.0rc3' gnomon mcp serve
```

For an offline installation, follow [the wheel procedure](offline-installation.md).
To use this checkout, run `uvx --from . gnomon mcp serve`.
For a host that accepts MCP server JSON:

```json
{"mcpServers":{"gnomon":{"command":"gnomon","args":["mcp","serve"]}}}
```

The connection owns one session. Its frozen references survive between calls and
are released when it closes. No configuration file is searched for implicitly.

For explicit date, interval or event calculations, operator configuration can add
`enable_temporal=true`. This exposes one optional `gnomon_temporal` tool; it is
not needed for forecasting. See [temporal semantics and examples](production/TEMPORAL.md).

## Inspect, describe, forecast

Call `gnomon_inspect` with:

```json
{"input":"examples/messy_requests.csv","time_column":"timestamp","target_column":"requests"}
```

Use the returned `data_ref` in subsequent calls. Inspection discloses repairs,
frequency, series identifiers, snapshot identity and knowledge-time assumptions.
For a panel, select a returned series explicitly.

Call `gnomon_describe` with `{"data_ref":"RETURNED_REF","statistic":"mean"}`.
The result is the exact observed mean, not a forecast or a latest-value substitute.
Optional start/end timestamps select an inclusive window.

Call `gnomon_forecast` with:

```json
{"provider":"last_value","data_ref":"RETURNED_REF","horizon":7}
```

This built-in reference forecast is inference only: no implicit backtest, calibrated
probability claim or permission to act. `seasonal_naive` and `historical_mean` are
also available. Forecasts from your preferred library use the same typed contract.

## Add your provider and optional ledger

```bash
gnomon mcp serve --providers-config providers.toml
gnomon capabilities --providers-config providers.toml
```

See [provider configuration and executable Python examples](production/INFERENCE.md)
for callables, fresh fitting factories, Ephemeris, capabilities and transport limits.
URLs, token environment-variable names, imports and ledger paths are operator
startup settings, not model-authored tool arguments. Do not put tokens in prompts.

The optional `gnomon_evaluate` tool requires explicit candidates and a baseline.
It runs matched historical folds within operator budgets and returns a compact
study summary; the study ID retrieves full evidence. With a ledger,
`gnomon_route` uses an original immutable study and explicit source/recording
cutoffs. Insufficient evidence retains the baseline; no new inference is hidden.

The ledger preserves original forecasts and actual revisions. New scores append
instead of overwriting old evaluations. Writes of outcomes or historical imports
require operator authorization; read/scoring operations do not silently authorize them.

## Default and advanced profiles

The default `execution` profile exposes six tools: capabilities, inspect,
describe, forecast, evaluate and read. A configured ledger adds route and ledger,
for eight tools. Discover exact schemas and configured providers with `tools/list`
and `gnomon_capabilities`.

Large results have `status: result_available`, `partial: true` and `result_ref`.
Use `gnomon_read` with that reference; a JSON pointer such as `/result/point/0`
selects an exact field. Read returns JSON text pages: concatenate `text` using
`next_offset` until null before parsing a complete value. Offsets count Unicode
codepoints, not bytes. These references expire with session closure or LRU eviction;
the ledger provides independent durable execution/study records. CLI JSON output
and explicit Python full-result methods remain unabridged.

Advanced evaluated/context workflows remain explicitly available through
`gnomon mcp serve --profile core` (ten tools), or `--profile full`.
The retained evidence/decision/data profiles narrow or expand that legacy workflow.
They use different forecast inputs and artifact semantics; never send their
arguments to the default session. The duplicate describe and experimental mega
profiles are retired, not aliases.

For advanced analysis, see the [CLI reference](cli-reference.md),
[data and vintage formats](data-format.md) and [results guide](results-and-artifacts.md).
General inference, model selection, calibration and action authorization are
separate claims. Historical benchmark results do not prove the current default
improves an agent's accuracy. Remaining release gates are in the
[production plan](production/PLAN.md).
