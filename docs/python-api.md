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
