# Gnomon forecasting for trading research

Connect a local model through `ForecastRequest -> ForecastResult`, or load an
Ephemeris provider from Gnomon's startup configuration. Install model libraries
in the same Python environment as Gnomon and inspect its installed schemas.


## Choosing providers

- Reuse saved evidence where appropriate. Unless the user names models, choose a
  target-appropriate baseline (zero return for returns, last value for volatility) and
  one versioned candidate (StatsForecast or your own registered model).
- Built-ins such as `historical_mean` are point-only: a quantile-based policy needs a
  quantile provider, and a point-only comparison does not validate a quantile gate.
- Use Ephemeris when requested or when its benefit justifies the cost; for ranking it,
  see the [ledger skill](../../use-gnomon-ledger/SKILL.md#bounded-recovery).

## StatsForecast: local callable

Install `gnomon-forecast`, `statsforecast` and `pandas` in a project environment
(install Gnomon with `pip install .` when working from a checkout). Record resolved
versions in the project's lockfile for reproducibility.

The example below uses AutoETS only to demonstrate the boundary, not as a default
financial model. For targets that can be zero or negative, such as returns or
spreads, choose a compatible model/configuration rather than automatically allowing
multiplicative ETS components. Select the target and model using validation.
The example models consecutive observations with an integer index; the caller
is responsible for validating the
trading calendar and supplying real future timestamps when needed. This avoids
inventing weekend/holiday bars. Do not use that index to hide missing observations.

```python
from importlib.metadata import version

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from gnomon import (
    AdapterCapabilities, ForecastResult, GnomonSession,
)


def statsforecast_point(request):
    # Build a fresh model on the supplied training history each invocation.
    model = AutoETS(season_length=request.season, model="ZZZ", alias="ets")
    frame = pd.DataFrame({
        "unique_id": [request.series_id or "series"] * len(request.history),
        "ds": list(range(len(request.history))),
        "y": list(request.history),
    })
    forecast = StatsForecast(models=[model], freq=1, n_jobs=1).forecast(
        df=frame, h=request.horizon,
    )
    return ForecastResult(
        point=tuple(float(x) for x in forecast["ets"]),
        timestamps=request.future_timestamps,
        series_id=request.series_id,
        unit=request.unit,
        metadata={"library": "statsforecast", "model": "AutoETS"},
    )


def register_statsforecast(session):
    session.engine.register(
        "statsforecast/autoets", statsforecast_point,
        capabilities=AdapterCapabilities(min_history=3),
        revision=f"statsforecast/{version('statsforecast')}/autoets-ZZZ-adapter-v1",
    )


if __name__ == "__main__":
    # Synthetic observations: API smoke example, not market evidence.
    request = {
        "history": [100 + 0.2 * i + (i % 3) * 0.1 for i in range(40)],
        "horizon": 3,
        "season": 1,
        "series_id": "synthetic-price",
        "unit": "USD",
    }
    with GnomonSession.from_config() as session:
        register_statsforecast(session)
        reply = session.forecast("statsforecast/autoets", request)
        assert reply["status"] == "ok", reply
        assert len(reply["result"]["point"]) == request["horizon"]
        print(reply["result"]["point"])
```

This callable supports univariate point forecasts. The engine rejects requested
quantiles/covariates because they are not declared supported. Do not silently drop
them when extending the adapter. For stateful fitting objects, use
`register_factory` for fresh state per invocation/fold. A minimum of three is an
adapter floor, not a guarantee every ETS configuration/history can be fitted.

StatsForecast's `level=[90]` requests a prediction interval, not a 0.9 quantile.
Before adding quantile support, map documented interval probabilities to Gnomon's
per-step marginals, preserve ordering and finite values, and verify the selected
model's point statistic instead of assuming it is a median. Validate empirical
coverage on held-out observations. See the official
[StatsForecast core API](https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html)
and [model reference](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html).

## Ephemeris, optionally alongside StatsForecast

Follow the [Ephemeris setup skill](../../setup-gnomon-ephemeris/SKILL.md), then use
that provider configuration explicitly:

```python
from gnomon import GnomonSession

with GnomonSession.from_config("/absolute/path/providers.toml") as session:
    providers = session.capabilities()["providers"]
    # Inspect available names; retain the user's chosen model or routing policy.
    provider = "ephemeris"
    assert provider in providers
    # Supply validated observations and their actual frequency in your program.
    reply = session.forecast(provider, {
        "history": observed_values,
        "horizon": horizon,
        "frequency": frequency,
        "timestamps": observed_timestamps,
        "future_timestamps": future_timestamps,
        "series_id": series_id,
        "unit": unit,
        "quantiles": [0.1, 0.5, 0.9],
    })
```

The second snippet is a template: supply its data variables from the validated
dataset and calendar. Use Gnomon's `frequency` hint, not a made-up seasonal period,
for Ephemeris. Discover explicit models rather than hard-coding a past catalog.
Remote forecasts/evaluations may be billable; reuse saved results where possible.
The connector uses the median for points and attests no model revision; see the
[ledger skill](../../use-gnomon-ledger/SKILL.md#bounded-recovery) for ranking it.

To compare both providers, call `register_statsforecast(session)` in this same
configured session, then use `session.evaluate` / `gnomon_evaluate` with explicit
candidates, baseline, horizon and budget. Inspect the installed evaluation schema
for input and fold parameters. Declare capabilities for each provider and preserve
timestamps, target identity and units when converting model outputs. Gnomon's
Ephemeris connector supports marginal quantiles but does not provide sample paths.

For evaluation timing, training provenance and execution assumptions, read
[backtest engines](backtesting.md). For decision recording, cutoff-bound comparisons
and lessons, run the [trade lifecycle](trade-lifecycle.md).
