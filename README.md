<p align="center">
  <img src="https://raw.githubusercontent.com/TensorLink-AI/Gnomon/main/docs/assets/gnomon-logo.png" alt="Gnomon logo" width="180">
</p>

# Gnomon

**Give your agent a forecast it can't fake.**

<!-- TODO(F2): link docs/assets/demo.gif (recording of examples/agent_cheating_demo.sh) -->

Gnomon runs time-series models, Ephemeris (hosted) or your own, against frozen
point-in-time data and returns every forecast with the snapshot, execution ID and
evaluation evidence behind it. Agents get numbers they can quote. You get a ledger
you can audit.

## Why

- Agents wired straight to data backtest on the future.
- Given too little history, they invent numbers.
- Without a snapshot and execution ID, nobody can say where a figure came from.

## Quick start

```bash
python -m pip install gnomon-forecast
```

With Ephemeris (credentials are environment variables named in
[operator config](gnomon.toml.example), never arguments):

```bash
EPHEMERIS_BASE_URL=https://... EPHEMERIS_API_TOKEN=... gnomon forecast requests.csv \
  --time timestamp --target requests --horizon 7 --provider ephemeris \
  --providers-config providers.toml --quantiles 0.1 0.5 0.9
```

Built-in baseline, no key needed:

```bash
gnomon forecast examples/messy_requests.csv --time timestamp --target requests \
  --horizon 7 --provider seasonal_naive --season 7
```

```
2026-06-10T00:00:00  334.1
...
provider: seasonal_naive (gnomon/1.2.0+g11c1331ec78e.s2fb3c6d977e0/seasonal_naive)
execution: a7873cde-6dce-4b6b-9b1e-803e34991ed1
snapshot: snapshot_be702c167b9001e3 as_of=latest
```

Pipes and agents receive the JSON envelope; `--json` forces it.

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

- A frozen snapshot with its `as_of` cutoff.
- An execution ID and request fingerprint.
- The provider and revision.
- An optional ledger row that later actuals never overwrite.
- Rescoring after revisions, originals intact.

## Ephemeris

Ephemeris is TensorLink's hosted forecasting models. Each forecast returns point
values, requested quantiles and `models_used`. Access is per deployment: set
`EPHEMERIS_BASE_URL` and `EPHEMERIS_API_TOKEN` and register the provider in
[operator config](docs/production/INFERENCE.md#ephemeris-hosted-models), never
as tool arguments.

## Your own models

Register any callable that maps a request to a result:

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

Install Gnomon in the model's Python environment (`python -m pip install .` from a
checkout). See [provider integration](docs/production/INFERENCE.md).

## Status and guides

The execution API is stable; a real-agent comparison and the authenticated
Ephemeris gate remain pending, and a forecast is not permission to act. See
[validation and limits](docs/agent-evaluation.md).

- [First run](docs/getting-started.md) · [Python API](docs/python-api.md) · [CLI](docs/cli-reference.md)
- [MCP quickstart](docs/quickstart-mcp.md) · [Agent skill](skills/use-gnomon/SKILL.md) · [Ledger](docs/production/OPERATIONS.md)
- [Decision memory](docs/decision-memory.md) · [All docs](docs/README.md) · [Changelog](CHANGELOG.md)

*A gnomon is the part of a sundial that casts the shadow.*
