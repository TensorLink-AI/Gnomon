# First run

Install Python 3.11–3.13 and Gnomon:

```bash
python -m pip install --pre 'gnomon-forecast==0.8.0rc3'
gnomon infer --provider last_value --request '{"history":[10,12,11],"horizon":2}'
```

Before publication, install the checkout with `python -m pip install -e '.[dev]'`.
The isolated Bash installer also supports the checkout: `bash install.sh --local`.
The distribution is `gnomon-forecast`; the import and command are `gnomon`.

The command returns two last-value predictions. It demonstrates execution, not
accuracy. Built-in `last_value`, `seasonal_naive` and `historical_mean` providers
run offline without credentials, downloads or implicit model selection.

## Python

```python
from gnomon import GnomonSession

with GnomonSession.from_config() as session:
    result = session.call("gnomon_forecast", {
        "provider": "last_value",
        "request": {"history": [10, 12, 11], "horizon": 2},
    })
    print(result["result"]["point"])
```

For timestamped files, use `gnomon_inspect` in the session. Its frozen data
reference is usable by describe, forecast and evaluate operations. Result references
are paged with `gnomon_read`; they are session-local, not persistent ledger entries.
See [the session contract](production/INFERENCE.md).

## Connect a model or agent

Register a callable returning `ForecastResult`, or a factory that creates a fresh
forecaster per fit. Your application owns library-specific setup. Follow the
[Python API](python-api.md) or [installable provider walkthrough](../examples/provider_plugin/README.md).
For remote TSFMs, configure [Ephemeris](production/INFERENCE.md#ephemeris).
For an agent host, follow [the MCP quickstart](quickstart-mcp.md).

## Save and evaluate

Set `ledger_path = "ledger.db"` in an explicit provider configuration and pass
`--providers-config` at startup. The ledger retains forecasts, actual revisions and
score versions. Actual ingestion requires separate operator permission: an agent
cannot enable it through a forecast argument. See [operations](production/OPERATIONS.md).

Use `gnomon_evaluate` for bounded backtests, not to place real-world orders.

## Existing evaluated workflows

`gnomon forecast`, `investigate`, `detect`, `decide` and `monitor` retain the
advanced evaluated-artifact workflow. They differ from direct `gnomon infer`.
See [CLI reference](cli-reference.md) and [compatibility](../COMPATIBILITY.md).
