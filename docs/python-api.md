# Python API

Install the **gnomon-forecast** distribution in the Python you will use, then
import **gnomon**:

```bash
python -m pip install gnomon-forecast
python -c 'from gnomon import GnomonSession; print(GnomonSession)'
```

The standalone `install.sh` creates its own environment. On version 1.1.0
or newer, use `gnomon python your_script.py` or
`gnomon python -c 'from gnomon import GnomonSession; print(GnomonSession)'`.
`gnomon environment` shows the exact interpreter and package paths.
For notebooks or existing applications, install into their interpreter instead
of injecting the standalone environment's site-packages into `sys.path`.

Start with `GnomonSession.from_config()` for built-in providers, inspection and
evaluation; no configuration file is required. The CLI uses this same entry point.
In 1.1.1 and newer, Python forecasts accept either a `ForecastRequest` or the same
dictionary accepted by the CLI's `--request`:

```python
from gnomon import GnomonSession

with GnomonSession.from_config() as session:
    forecast = session.forecast("seasonal_naive", {
        "history": [10, 20, 30, 40, 50, 60, 70], "horizon": 2, "season": 7,
    })
    assert forecast["result"]["point"] == (10, 20)
    schema = session.capabilities()["providers"]["seasonal_naive"]["request_schema"]
```

`season` is the JSON/Python field; `--season` is its CLI flag. There is no
`season_length` alias. `capabilities` includes a request schema for each provider,
with defaults, supported features and declared history/horizon/frequency limits.
Value-dependent requirements, such as history length being at least `season`
for seasonal naive, are described and checked at execution.

`InferenceEngine()` and a bare `GnomonSession()` start with an empty registry.
Use an engine when your application explicitly registers its own providers.
Engine forecasts also accept dictionaries, including mixed typed/dict batches.
Use a callable for a loaded model, or a factory for a fresh fit per evaluation fold.
Gnomon does not install model libraries or manage GPU memory.

To enable caching for built-ins, create an operator TOML file and keep one session
open for both calls:

```python
from pathlib import Path
from gnomon import GnomonSession

Path("providers.toml").write_text("cache_size = 8\n")
with GnomonSession.from_config("providers.toml") as session:
    request = {"history": [1, 2, 3], "horizon": 2, "series_id": "sales", "unit": "widgets"}
    assert not session.forecast("last_value", request)["cache_hit"]
    assert session.forecast("last_value", request)["cache_hit"]
```

`cache_size` is a TOML key, not a `from_config` keyword. The low-level custom-provider
engine accepts `InferenceEngine(cache_size=8)`. Caching requires a deterministic
provider with an explicit revision. CLI configuration uses
`--providers-config providers.toml`; separate CLI invocations cannot share the
session cache. `gnomon capabilities --config-schema` lists the configuration keys,
and `help(GnomonSession.from_config)` includes the two-call example offline.

```python
from gnomon import ForecastRequest, ForecastResult, InferenceEngine

def predict(request):
    return ForecastResult((request.history[-1],) * request.horizon,
                          series_id=request.series_id, unit=request.unit,
                          timestamps=request.future_timestamps)

with InferenceEngine() as engine:
    engine.register("my-model", predict, revision="my-config-v1")
    run = engine.forecast("my-model", ForecastRequest((1, 2, 3), 2, series_id="sales", unit="widgets",
                                                        future_timestamps=("2026-01-21T00:00:00Z", "2026-01-22T00:00:00Z")))
    assert run.result.point == (3, 3)
```

Custom predictors receive `ForecastRequest` and must return `ForecastResult`.
Results must echo `request.series_id`, `request.unit`, and `request.future_timestamps`
(including the empty tuple when no forecast timestamps are supplied). The engine
rejects mismatches instead of silently assigning another identity. A dictionary
is accepted as a forecast request, but not as a predictor's result.
`help(InferenceEngine.register)` includes a runnable typed-return example.

For data inspection, tools and evaluation, use `GnomonSession.from_config()`.
It registers three reference baselines without optional dependencies.
`session.call(name, arguments)` is the MCP tool contract; `compact=False`
returns the full value. `session.forecast`, `session.evaluate` and
`session.route` expose the same underlying operations.

`TemporalStore` holds observation vintages. `TemporalLedger` holds immutable
executions, actual revisions, scores and decisions. They serve different purposes;
neither turns a forecast or recorded decision into permission to act.

Full contracts and runnable examples:
[providers/session](production/INFERENCE.md), [ledger](production/OPERATIONS.md),
[temporal calculations](production/TEMPORAL.md),
[separate provider package](../examples/provider_plugin/README.md).
