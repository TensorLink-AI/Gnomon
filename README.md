# Gnomon

Time-series tools for people and AI agents: inspect data, run a chosen forecast
model, compare it against baselines, and preserve what was predicted and known.

Use your preferred forecasting software through a callable or fresh model factory,
or connect a remote inference service. Gnomon provides shared execution contracts,
explicit evaluation and an optional revision-aware ledger.

## Quick start

Python 3.11–3.13. The core has no required third-party dependencies.

```bash
python -m pip install --pre 'gnomon-forecast==0.8.0rc2'
gnomon infer --provider last_value --request '{"history":[10,12,11],"horizon":2}'
```

This release candidate introduces a new default agent interface. Read
[compatibility and migration](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/COMPATIBILITY.md) when upgrading. Before publication,
install the checkout with `python -m pip install -e '.[dev]'`.

```python
from gnomon import ForecastRequest, ForecastResult, InferenceEngine

def my_forecaster(request):
    # Call your preferred library or fitted model here.
    return ForecastResult((request.history[-1],) * request.horizon)

engine = InferenceEngine()
engine.register("my-model", my_forecaster)
execution = engine.forecast("my-model", ForecastRequest((10, 12, 11), 2))
print(execution.result.point)  # (11.0, 11.0)
```

For models fitted on each request, use `register_factory` to give each backtest
fold a fresh instance. StatsForecast, NeuralForecast and Darts remain user-owned
dependencies/configurations. Unsupported inputs are rejected, not silently dropped.

## Connect an agent

Run `gnomon mcp serve` as a stdio MCP server in your agent host.
The agent gets 6 tools by default:

- `gnomon_inspect`: inspect and freeze input data.
- `gnomon_describe`: compute an explicitly requested statistic.
- `gnomon_capabilities`: discover providers and limits.
- `gnomon_forecast`: execute the selected provider.
- `gnomon_evaluate`: compare providers on bounded historical folds.
- `gnomon_read`: retrieve bounded pages of saved results.

A configured ledger adds history/scoring and evidence-based routing tools.
Optional temporal calculations add one tool for dates, intervals and event order.
Python, CLI and MCP use the same session contracts.
See the [MCP quickstart](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/quickstart-mcp.md).

## Optional connectors

Ephemeris is one available connector for remote time-series inference. It is
optional: local callables and factories use the same execution interface.

```toml
# providers.toml
schema_version = 1

[providers.remote]
kind = "ephemeris"
base_url_env = "EPHEMERIS_BASE_URL"
token_env = "EPHEMERIS_API_TOKEN"
mode = "route"
```

Set these environment variables in the operator's environment, then pass
`--providers-config providers.toml` to `gnomon infer` or `gnomon mcp serve`.
The URL is deployment-specific; credentials are never agent tool arguments.
See [inference and providers](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/production/INFERENCE.md).

## What is kept separate

Inference does not secretly run a backtest. Evaluation is explicit and budgeted.
The optional SQLite ledger preserves original executions, actual-value revisions,
rescoring and decisions. It distinguishes valid time, source availability and local
recording time: later corrections do not rewrite earlier forecasts.

Neither a forecast nor a good backtest grants permission to act. Model quantiles
are not proof of calibrated uncertainty. Historical routing requires appropriate
cutoff-bound evidence and otherwise falls back to a baseline.

This release does **not** establish forecasting superiority or improved LLM
reasoning. Authenticated live Ephemeris verification and an actual matched
ordinary/lean/full agent comparison remain pending. Local tests are not substitutes.
See [validation and limitations](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/agent-evaluation.md).

## Documentation

- [First run](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/getting-started.md), [Python API](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/python-api.md), [CLI reference](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/cli-reference.md).
- [Provider integration](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/production/INFERENCE.md) and an [installable plugin example](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/examples/provider_plugin/README.md).
- [Ledger operations](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/production/OPERATIONS.md), [temporal calculations](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/production/TEMPORAL.md), [data format](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/data-format.md).
- [Documentation index](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/README.md), [development](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/docs/development.md), [changelog](https://github.com/TensorLink-AI/Gnomon/blob/v0.8.0rc2/CHANGELOG.md).

Advanced evaluated workflows and legacy MCP profiles remain explicitly available;
they are not the default provider session. Superseded designs and old benchmarks
are recoverable in Git history, not parallel specifications in the active tree.

## The name

A gnomon is the part of a sundial that casts the shadow used to tell time.
