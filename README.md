<p align="center">
  <img src="https://raw.githubusercontent.com/TensorLink-AI/Gnomon/main/docs/assets/gnomon-logo.png" alt="Gnomon logo" width="180">
</p>

# Gnomon

## Know more than the prediction.

Gnomon helps people and AI agents inspect time-series data, run their chosen models,
challenge forecasts against honest baselines, and preserve the evidence behind every
result.

Use your own forecasting software. Add a remote service or a persistent ledger
when you need one.

## Quick start

Python 3.11–3.13. No required third-party dependencies.

```bash
python -m pip install 'gnomon-forecast==1.1.2'
gnomon infer --provider last_value --request '{"history":[10,12,11],"horizon":2}'
```

From a checkout, use `python -m pip install .`; this also works before the
versioned package is published.
The forecast command runs an offline baseline.
To use your own local model, install Gnomon in the **same Python environment** as
the model and its dependencies, then run that environment's `gnomon` or
`python -m gnomon`. An isolated Gnomon environment cannot import PyTorch or
another model library installed elsewhere.

Register your model as a callable:

```python
from gnomon import ForecastRequest, ForecastResult, InferenceEngine

def my_forecaster(request):
    # Replace this baseline with your preferred model.
    return ForecastResult((request.history[-1],) * request.horizon)

engine = InferenceEngine()
engine.register("my-model", my_forecaster)
execution = engine.forecast("my-model", ForecastRequest((10, 12, 11), 2))
print(execution.result.point)  # (11.0, 11.0)
```

StatsForecast, NeuralForecast, Darts or your own code: wrap the call and return a
`ForecastResult`. Use `register_factory` for a fresh model on each evaluation fold.
Gnomon checks inputs and outputs; you choose and install the model software.
See [provider integration](docs/production/INFERENCE.md).

For a scoreable ledger record, give the forecast a nonempty `series_id` and
explicit `future_timestamps`. Every ledger timestamp needs an explicit timezone
such as `+00:00`; each actual must use the forecast's exact `series_id`, unit and
one of its future timestamps. The [first-run guide](docs/getting-started.md#record-and-score-a-forecast)
shows the complete CLI loop.

## Connect an agent

Run `gnomon mcp serve` in your agent host. The agent gets 6 tools by default:

- `gnomon_inspect`: check data and freeze a reusable snapshot.
- `gnomon_describe`: calculate an observed statistic.
- `gnomon_capabilities`: find available models and their limits.
- `gnomon_forecast`: run the selected model.
- `gnomon_evaluate`: compare models on past data with an explicit budget.
- `gnomon_read`: retrieve saved results without running the model again.

Python, CLI and MCP share the same interface.
Start with the [MCP quickstart](docs/quickstart-mcp.md)
and [agent skill](skills/use-gnomon/SKILL.md).

## Optional connectors

Ephemeris is one connector for remote time-series inference. Set its deployment
URL and credentials in operator configuration; they are never agent tool arguments.
Local models work without it.
See [connector setup](docs/production/INFERENCE.md#ephemeris).

## Keep the history straight

The optional SQLite ledger saves forecasts, revised actuals, scores and decisions.
Later corrections do not overwrite earlier predictions. You can ask what was known
at a particular time, find forecasts that need scoring, and compare models on
matched past results. No automatic retraining or model calls are involved.

Optional date and time tools handle timezones, calendar shifts, intervals and event
order. They calculate supplied facts; they do not claim to improve an LLM's reasoning.

## Status and guides

The provider-neutral execution API is stable. Live-service verification and a
real-agent comparison remain pending. A forecast is not permission to act; model quantiles are not proof
of calibrated uncertainty. See [validation and limits](docs/agent-evaluation.md).

- [First run](docs/getting-started.md) · [Python API](docs/python-api.md) · [CLI](docs/cli-reference.md)
- [Ledger](docs/production/OPERATIONS.md) · [Time calculations](docs/production/TEMPORAL.md) · [All docs](docs/README.md)
- [Changelog](CHANGELOG.md)

*A gnomon is the part of a sundial that casts the shadow.*
