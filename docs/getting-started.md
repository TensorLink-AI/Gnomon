# First run

Install the [package or source checkout](installation.md), then run a reference forecast:

```bash
gnomon infer --provider last_value --request '{"history":[10,12,11],"horizon":2}'
```

The result repeats 11 twice. This is an offline baseline, not an accuracy claim.

With a file whose columns are timestamp and value:

```bash
gnomon inspect examples/daily_requests.csv --target-column requests
gnomon describe examples/daily_requests.csv --target-column requests --statistic mean
gnomon infer --input examples/daily_requests.csv --target-column requests --provider last_value --horizon 7
```

Choose your own [provider](production/INFERENCE.md), then explicitly evaluate it
against a baseline if you need historical evidence. Add a ledger when records
should survive the session. Start [MCP](quickstart-mcp.md) to expose the same tools
to an agent.

## Record and score a forecast

Create an operator-owned `providers.toml`:

```toml
schema_version = 1
ledger_path = "gnomon-ledger.db"
allow_outcome_writes = true
```

Make a forecast whose identity and horizon can be matched later:

```bash
gnomon infer --providers-config providers.toml --provider last_value \
  --request '{"history":[10,12,11],"horizon":1,"series_id":"sales","unit":"requests","timestamps":["2025-01-01T00:00:00+00:00","2025-01-02T00:00:00+00:00","2025-01-03T00:00:00+00:00"],"future_timestamps":["2025-01-04T00:00:00+00:00"]}'
```

Copy the returned `execution_id`. Append an actual with the same `series_id`,
unit and forecast timestamp, then score that execution:

```bash
gnomon ledger --providers-config providers.toml \
  --arguments '{"operation":"append_actual","series_id":"sales","unit":"requests","valid_time":"2025-01-04T00:00:00+00:00","source_available_at":"2025-01-05T00:00:00+00:00","value":13}'
gnomon ledger --providers-config providers.toml \
  --arguments '{"operation":"evaluate","execution_id":"PASTE_EXECUTION_ID_HERE"}'
```

All ledger timestamps require an explicit timezone. An actual only scores a
forecast step when its `series_id`, unit and `valid_time` exactly match the
forecast request's identity, unit and one of its `future_timestamps`.
