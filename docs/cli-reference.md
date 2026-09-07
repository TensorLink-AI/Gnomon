# CLI reference

Use `-` as the input for piped CSV (up to 8 MiB); larger inputs need a file.
This works with inspect/describe and `infer --input -`. The CLI freezes input
before computing results and removes its temporary file when inspection finishes.

One CLI uses the same execution session as Python and MCP. Success is JSON on
stdout with exit code 0. Errors are structured JSON on stderr with exit code 2;
interruption exits 130. Run `gnomon --help` or a command's `--help`.

| Command | Purpose |
| --- | --- |
| `gnomon capabilities` | Registered providers, limits and enabled tools |
| `gnomon infer` | Explicit provider inference; no implicit backtest |
| `gnomon inspect` | Validate a file/store input and disclose snapshot semantics |
| `gnomon describe` | Exact mean, median, latest, minimum, maximum or sum |
| `gnomon evaluate` | Budgeted rolling-origin comparison |
| `gnomon route` | Select from a recorded study at explicit cutoffs |
| `gnomon ledger` | Read evidence, append scores or authorized outcomes |
| `gnomon temporal` | Explicit date, instant, interval and ordering calculations |
| `gnomon self-check leakage` | Offline check of snapshot cutoffs |
| `gnomon mcp serve` | Long-lived stdio session for an agent |

```bash
gnomon infer --provider last_value --request '{"history":[1,2,3],"horizon":2}'
gnomon infer --provider last_value --input data.csv --horizon 7
gnomon inspect data.csv --time-column date --target-column sales
gnomon describe data.csv --statistic mean --start 2025-01-01 --end 2025-01-31
gnomon evaluate --arguments '{"data":{"input":"data.csv"},"candidates":["historical_mean"],"baseline":"last_value","horizon":2,"folds":4,"budget":{"max_calls":8}}'
gnomon ledger --providers-config providers.toml --arguments '{"operation":"pending"}'
gnomon mcp serve --providers-config providers.toml
```

`--request` and `--arguments` accept a JSON object or `@file.json`.
`infer` takes either a typed request or `--input` plus `--horizon`.
File options are `--time-column` (default timestamp), `--target-column`
(default value), `--series-column`, `--frequency`, `--as-of`,
`--recorded-as-of`, `--store-path`, `--unit`, `--repair` and `--regrid`.
They cannot be mixed with a typed request. Panel forecasts require `--series-id`
unless only one series is present. Repairs default to off.

`describe` accepts `--series-id`, inclusive `--start`/`--end` and the
same file options. It computes observed statistics, not forecasts or causal claims.
A CLI data reference ends with the process: use a Python or MCP session to reuse it.

Provider-backed commands accept explicit operator `--providers-config` TOML.
Ledger and route require it. Evaluation/routing JSON may contain a `data`
inspection object, replaced with a frozen reference before the operation.
They use the [same contracts and budgets](production/INFERENCE.md) as MCP.

`gnomon temporal --arguments` uses the [temporal contract](production/TEMPORAL.md).
`gnomon self-check leakage --cases 8 --seed 7` tests installed cutoff behavior,
not LLM reasoning or forecast accuracy.

Optional ledger/temporal tools are enabled in operator configuration.
