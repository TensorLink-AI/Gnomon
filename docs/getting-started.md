# First run

Install this [development checkout](installation.md), then run a reference forecast:

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
