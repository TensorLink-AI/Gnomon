<p align="center">
  <img src="https://raw.githubusercontent.com/TensorLink-AI/Gnomon/main/docs/assets/gnomon-logo.png" alt="Gnomon logo" width="180">
</p>

# Gnomon

**Give your agent forecasts it can verify.**

<!-- TODO(F2): docs/assets/demo.gif -->

Gnomon connects agents to Ephemeris, TensorLink's hosted forecasting API, or
your own models. Run forecasts, compare performance, and preserve the evidence
behind each decision. Keep predictions and outcomes across tasks, so agents can
check what worked before—even when observations are revised.

## Quick start

```bash
python -m pip install gnomon-forecast
gnomon forecast --provider seasonal_naive \
  --request '{"history":[120,131,125,140,152,161,118,122,134,128,142,155,163,121],"horizon":7,"season":7}'
```

```
+1  122
+2  134
...
+7  121
provider: seasonal_naive (gnomon/1.2.0+…/seasonal_naive)
execution: 9b051531-a0f4-482d-8312-c8a49b7b4452
snapshot: request supplied directly
```

A CSV works the same way: `gnomon forecast data.csv --horizon 7` freezes a
snapshot and labels every inferred choice in `assumptions`. Pipes and agents
receive the JSON envelope; `--json` forces it.

Hosted models: [set up Ephemeris](docs/production/INFERENCE.md#ephemeris-hosted-models)
(two environment variables and a `providers.toml`), then add
`--provider ephemeris --providers-config providers.toml --quantiles 0.1 0.5 0.9`.

## Connect an agent

```bash
claude mcp add gnomon -- gnomon mcp serve --providers-config /absolute/path/providers.toml
```

The agent gets 6 tools by default:

- `gnomon_inspect`: check data, freeze a snapshot.
- `gnomon_describe`: exact observed statistic.
- `gnomon_capabilities`: models, limits, Ephemeris status.
- `gnomon_forecast`: run the selected model.
- `gnomon_evaluate`: compare models on past data, budgeted.
- `gnomon_read`: page saved results, no rerun.

## What every forecast carries

- A frozen snapshot with its `as_of` cutoff; the same file always freezes to the same snapshot.
- An execution ID and request fingerprint.
- The provider and revision.
- Every data repair itemised; nothing interpolated unless you chose `aggressive`.
- An optional ledger row that later actuals never overwrite.
- Rescoring after revisions, originals intact.

## Keep experience across tasks

The optional ledger records each forecast, the actuals that arrive later and the
scores. Revised observations refresh the scores; original predictions never
change. See [ledger operations](docs/production/OPERATIONS.md).

## Ephemeris

Ephemeris is TensorLink's hosted forecasting models: point values, requested
quantiles and `models_used` per forecast. Set `EPHEMERIS_BASE_URL` and
`EPHEMERIS_API_TOKEN` and register the provider in
[operator config](docs/production/INFERENCE.md#ephemeris-hosted-models);
credentials are never tool arguments.

## Your own models

Register any callable:

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

Install it in the model's environment (`python -m pip install .` from a checkout);
see [provider integration](docs/production/INFERENCE.md).

## Status and guides

The execution API is stable; a real-agent comparison and the authenticated
Ephemeris gate remain pending, and a forecast is not permission to act. See
[validation and limits](docs/agent-evaluation.md).

- [First run](docs/getting-started.md) · [Python API](docs/python-api.md) · [CLI](docs/cli-reference.md)
- [MCP quickstart](docs/quickstart-mcp.md) · [Agent skill](skills/use-gnomon/SKILL.md) · [Ledger](docs/production/OPERATIONS.md)
- [Decision memory](docs/decision-memory.md) · [All docs](docs/README.md) · [Changelog](CHANGELOG.md)

*A gnomon is the part of a sundial that casts the shadow.*
