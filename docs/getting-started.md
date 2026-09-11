# First run

Install the [package or source checkout](installation.md). From a checkout, the
bundled example has `timestamp` and `requests` columns of daily data; forecast
the next week with the seasonal baseline, which repeats the previous seven days:

```bash
gnomon forecast examples/messy_requests.csv --time timestamp --target requests \
  --horizon 7 --provider seasonal_naive --season 7
```

On a terminal this prints one line per day, then the provider and revision, the
execution ID and the frozen snapshot ID with its `as_of` cutoff. Pipes and agents
receive the JSON envelope (`--json` forces it). It is an offline baseline, not an
accuracy claim; a configured Ephemeris or registered model is chosen the same
way with `--provider`.

Inspect and summarise the same file first when you want the frozen snapshot and
an exact observed statistic:

```bash
gnomon inspect examples/messy_requests.csv --time timestamp --target requests
gnomon describe examples/messy_requests.csv --time timestamp --target requests --statistic mean
```

For your own CSV, use its path and column names. A two-column CSV needs no
column flags: `gnomon forecast data.csv --horizon 7` infers them, runs the
`last_value` reference baseline and discloses both choices in `assumptions`.

## Inspect, compare and route

This complete workflow uses the checkout example, declares its dates as UTC,
and saves frozen data so later commands can reuse it. For your own data, declare
its actual timezone and use its last visible timestamp as the source cutoff.

```bash
gnomon inspect --input examples/daily_requests.csv --target-column requests --timezone UTC --for route --save-snapshot requests.gnomon
gnomon evaluate --input requests.gnomon --candidates historical_mean seasonal_naive --baseline last_value --season 7 --horizon 2 --ledger-path evidence.db --save-result study.json
gnomon route --input requests.gnomon --study @study.json --ledger-path evidence.db --source-as-of 2026-02-04T00:00:00Z --recorded-as-of 2099-01-01T00:00:00Z
```

The example's 2099 recording cutoff includes all evidence recorded so far;
replace it with your analysis cutoff, at or after the study's recording time.
Routing reads the recorded comparison without new model calls. Evaluation exits
0 for a complete comparison, 3 for a partial one and 2 if none of the folds could
be scored. Inspect `issues` for history and budget requirements.

If data has gaps, `inspect` explains which operations are possible. To work only
with observed values, `--window latest_contiguous --frequency D` selects the most
recent uninterrupted daily segment and discloses excluded rows. More options
and the equivalent JSON interface are in the [CLI reference](cli-reference.md).

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
