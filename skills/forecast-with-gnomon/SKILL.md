---
name: forecast-with-gnomon
description: Execute and save time-series forecasts with Gnomon's Python API, including built-ins and your own registered models. Use when a forecasting task is done in Python with Gnomon installed (files in, forecast file out); for connected gnomon_* MCP tools use use-gnomon.
---

# Forecast with Gnomon

Use Gnomon to execute a forecast for the actual requested series and dates. Save a
valid result early, then improve it if time allows. The model choice is yours:
built-ins are baselines, not a claim of superiority, and Ephemeris or StatsForecast
are optional providers (see [setup](../setup-gnomon-ephemeris/SKILL.md)). Evaluate
candidates against the user's metric on earlier observed windows; never use target
outcomes. Inspect a small data summary rather than printing a long CSV.

## Correct API

`GnomonSession.from_config()` loads the built-ins without a TOML file.
Call `session.forecast(provider, request_dict)`. The result is a dictionary;
points are `reply["result"]["point"]`.

Do not pass a model name or dictionary to `from_config`: its positional argument
is a configuration **file path**. Do not invent `fit`, `predict`,
`add_observations`, or `list_providers` on a session.

## Execute and save a forecast

The example uses a simple file layout: `history.csv` has `timestamp,value`,
`future.csv` lists the requested `timestamp` values, and `task.json` holds
`series_id`, `unit` and `horizon`. Adapt paths and column mappings to the real
inputs; preserve their identity, units and dates. It uses last value
as a baseline; select any appropriate provider. For `seasonal_naive`, supply the
season in **observations** (e.g. 7 only when the intended daily season is weekly).
The other built-in is `historical_mean`.

```python
import csv
import json
from pathlib import Path
from gnomon import GnomonSession

meta = json.loads(Path("task.json").read_text())
with open("history.csv") as f:
    rows = list(csv.DictReader(f))
with open("future.csv") as f:
    future = [row["timestamp"] for row in csv.DictReader(f)]
assert len(future) == meta["horizon"]
request = {
    "history": [float(row["value"]) for row in rows],
    "timestamps": [row["timestamp"] for row in rows],
    "future_timestamps": future,
    "horizon": len(future),
    "cutoff": rows[-1]["timestamp"],
    "series_id": meta["series_id"],
    "unit": meta["unit"],
}
provider = "last_value"  # Baseline example; retain it unless evaluation supports another model.
with GnomonSession.from_config() as session:
    reply = session.forecast(provider, request)
assert reply["status"] == "ok", reply
Path("execution.json").write_text(json.dumps(reply, indent=2))
submission = {
    "point": reply["result"]["point"],
    "method": reply["provider"],
    "execution_id": reply["execution_id"],
}
Path("forecast.json").write_text(json.dumps(submission, indent=2))
```

If the user or task supplies an output check, run it. Keep the last valid forecast
until a replacement is validated. A saved inference is not evidence of accuracy.

Before finishing, confirm the saved file is the Gnomon execution you intend to
submit. Predictions computed outside Gnomon must go through `session.forecast` via
a registered provider (below), not be written straight to `forecast.json`:

```python
saved = json.loads(Path("forecast.json").read_text())
run = json.loads(Path("execution.json").read_text())
assert saved["execution_id"] == run["execution_id"] and saved["point"] == run["result"]["point"]
print("final method:", saved["method"])  # Say so if this is still the baseline.
```
Gnomon's built-ins do not consume promotion or other covariates; use your own
model when that is appropriate. Do not guess latent demand or causal effects.

## Custom predictions

You may fit your own model with the available libraries. After computing its
horizon-length `predictions` from data available before the forecast, register this return contract
and reuse `request` and the saving pattern above:

```python
from gnomon import ForecastResult, GnomonSession, InferenceEngine

engine = InferenceEngine()
engine.register("my_model", lambda r: ForecastResult(
    point=tuple(float(v) for v in predictions),
    series_id=r.series_id, unit=r.unit, timestamps=r.future_timestamps,
), revision="my_model-v1")
with GnomonSession(engine=engine) as session:
    reply = session.forecast("my_model", request)
```

Change the revision when the model changes. Return `ForecastResult`, not a dict,
and echo series, units and future timestamps. Registration validates the result
contract; it does not certify model accuracy or that the training data was valid.

## Bounded recovery

- Unexpected `from_config` argument: use `from_config()` and pass the provider and
  history/horizon to `forecast`, as above. Built-ins need no provider TOML.
- Empty registry: `InferenceEngine()` is deliberately empty; either register your
  custom provider or use `GnomonSession.from_config()` for built-ins.
- Insufficient seasonal history: retain the requested season and obtain enough
  observations, or explicitly choose a different method; don't invent history.
- Invalid timestamps/identity: correct the actual mapping or request facts, not
  illustrative dates. Help is available with `help(GnomonSession.forecast)`.

After a specific API failure, make one task-preserving correction. If it still
fails, retain any valid saved result and report the error; don't keep guessing
constructors or configuration schemas.

Report the final method, its execution ID and how it was chosen. Do not describe a
baseline fallback or an unsaved computation as the selected Gnomon forecast.
