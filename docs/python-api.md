# Python API

## Execute a user-owned model

```python
from gnomon import ForecastRequest, ForecastResult, InferenceEngine

def forecast(request):
    return ForecastResult(
        (request.history[-1],) * request.horizon,
        timestamps=request.future_timestamps,
        series_id=request.series_id,
        unit=request.unit,
    )

engine = InferenceEngine()
engine.register("preferred", forecast)
run = engine.forecast("preferred", ForecastRequest((10, 12, 11), 2))
assert run.result.point == (11.0, 11.0)
```

`ForecastRequest` carries history, horizon, timestamps/frequency, identity, units,
covariates and uncertainty requirements. `ForecastResult` carries aligned points
and uncertainty. Gnomon validates both. `ForecastExecution` records execution
identity, provider, revision, evidence and whether a ledger recorded the result.

Register a function, an object implementing `forecast(request)`, or a fresh
per-request factory with `engine.register_factory(...)`. Factories own fitting
and prevent state reuse between evaluation folds. Optional `forecast_batch`
supports native batches. Declare `AdapterCapabilities` honestly; support is not
inferred from a library name. See [the full protocol](production/INFERENCE.md).

StatsForecast, NeuralForecast and Darts objects stay in user code. Gnomon does not
install them when you register a callable. The [provider example](../examples/provider_plugin/README.md)
demonstrates a separate package, CLI/MCP setup, actual revisions and backup.

## Ephemeris

```python
import os
from gnomon import EphemerisProvider, ForecastRequest, InferenceEngine

provider = EphemerisProvider(
    os.environ["EPHEMERIS_BASE_URL"], token_env="EPHEMERIS_API_TOKEN",
)
engine = InferenceEngine()
engine.register("remote", provider, lifecycle="pretrained")
# This is a remote request and may incur service charges:
run = engine.forecast("remote", ForecastRequest((10, 12, 11), 2))
```

Ephemeris is one optional remote inference connector. The URL is configurable.
Unknown served model revisions remain unknown.
Forecast POSTs are not automatically retried.
See [transport/service limits](production/INFERENCE.md#ephemeris).

## Share the agent contract

```python
from gnomon import GnomonSession

with GnomonSession.from_config() as session:
    print(session.call("gnomon_capabilities", {}))
```

`from_config("providers.toml")` adds operator-configured providers and an optional
ledger. `session.tools()` returns actual schemas; `session.call` uses the same
dispatcher as CLI/MCP. Start with [the first-run guide](getting-started.md).

## Temporal and persistent evidence

`TemporalLedger` preserves executions, revised actuals, scores and decisions.
It is a local append-only store, not a distributed or tamper-proof database.
See [operations](production/OPERATIONS.md) for cutoffs, backup and migration.
`temporal_operation` exposes date/interval/event calculations independently of
forecasting; see [the temporal contract](production/TEMPORAL.md).

## Advanced compatibility API

`gnomon.forecast`, `detect_anomalies`, `investigate_change`, `decide` and
`monitor` remain lazily imported advanced evaluated workflows, not aliases for
direct inference. Existing artifact, tracking and temporal-store APIs remain.
See [CLI reference](cli-reference.md), [results and artifacts](results-and-artifacts.md)
and [compatibility](../COMPATIBILITY.md).
