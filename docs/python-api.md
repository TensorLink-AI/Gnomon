# Python API

Use a callable for a loaded model, or a factory for a fresh fit per evaluation fold.
Gnomon does not install model libraries or manage GPU memory.

```python
from gnomon import ForecastRequest, ForecastResult, InferenceEngine

def predict(request):
    return ForecastResult((request.history[-1],) * request.horizon)

with InferenceEngine() as engine:
    engine.register("my-model", predict, revision="my-config-v1")
    run = engine.forecast("my-model", ForecastRequest((1, 2, 3), 2))
    assert run.result.point == (3, 3)
```

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

The old top-level forecast/macros and TrackingStore writer are removed.
Historical imports are read-only ledger operations; see [migration](../COMPATIBILITY.md).
