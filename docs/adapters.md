# Package adapters

Gnomon ships thin adapters for nine actively maintained forecasting packages.
Each one is a provider `kind` in operator TOML with a matching pip extra. The
core package keeps zero dependencies: the third-party package is imported only
when a provider of that kind is configured, and a missing package fails at
startup with the exact install command.

An adapter converts `ForecastRequest` into the package's own call and the
package's output back into `ForecastResult`. It does not choose a model for
you, does not claim accuracy, and does not infer a training cutoff. `model`
is an explicit operator setting; every registration records the package
version as its revision.

```bash
python -m pip install 'gnomon-forecast[statsforecast]'
```

```toml
# providers.toml
schema_version = 1

[providers.arima]
kind = "statsforecast"
model = "AutoARIMA"
```

```bash
gnomon infer --providers-config providers.toml --provider arima \
  --request '{"history":[10,12,11,13,12,14,13],"horizon":2,"quantiles":[0.1,0.9]}'
```

The agent sees `arima` through `gnomon_capabilities` like any other provider.
If an agent asks for a provider named after a kind that is not configured, the
error says which TOML block and extra the operator needs.

## Kinds

| kind | extra | default `model` | quantiles | covariates | sample paths | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `statsforecast` | `statsforecast` | `AutoARIMA` | intervals | paired | no | Any class in `statsforecast.models`; `season_length` from the request |
| `statsmodels` | `statsmodels` | `ETS` | intervals | no | no | `ETS`, `SARIMAX` or `Theta`; ETS simulation is seeded by `seed` |
| `prophet` | `prophet` | `Prophet` | samples | paired regressors | yes | Requires request timestamps; never invents dates |
| `mlforecast` | `mlforecast` | `lightgbm` | conformal | paired | no | `lags`, `n_windows`; needs history for the conformal windows |
| `skforecast` | `skforecast` | `sklearn.linear_model:Ridge` | bootstrap | paired | no | Any scikit-learn regressor via `module:Class` |
| `sktime` | `sktime` | `sktime.forecasting.theta:ThetaForecaster` | `predict_quantiles` | paired | no | Rejects quantiles when the estimator lacks `capability:pred_int` |
| `darts` | `darts` | `ExponentialSmoothing` | samples | no | yes | Probabilistic models only for quantiles; deep models need `darts[torch]` |
| `gluonts` | `gluonts` | `gluonts.model.npts:NPTSPredictor` | samples | no | yes | Predictor used directly or estimator trained per request |
| `neuralforecast` | `neuralforecast` | `NHITS` | MQLoss | no | no | Trains on every call; keep `max_steps` small |

`pip install 'gnomon-forecast[adapters]'` installs every kind except
`neuralforecast`, which pulls PyTorch.

Discover the same table with installation state and no imports:

```bash
gnomon capabilities --config-schema   # see properties.providers.additionalProperties.adapter_kinds
```

## Provider fields

| field | meaning |
| --- | --- |
| `kind` | one of the kinds above |
| `model` | class name, `module:Class` path, or the adapter's named choice; see each kind |
| `options` | TOML table of keyword arguments passed to the model, plus adapter settings such as `lags` or `num_samples` |
| `seed` | nonnegative integer; seeds sampling and any `random_state` the model accepts |
| `revision` | overrides the recorded `kind/package=version` revision |
| `deterministic` | overrides the adapter's conservative default; needed for cache eligibility |

Every adapter registers as a factory: a fresh fitting object is created for
each request and each evaluation fold, and a fitting object refuses reuse.

Adapter validation errors reach the caller in full: a missing `timestamps`
field, one-sided covariates or too little history for the chosen lags name
the field and the required length. Exceptions raised by the package itself
stay redacted behind the usual execution failure, as for any user-owned
provider, because their text is not under Gnomon's control.

## Shared rules

**Timestamps are used, never invented.** When the request carries
`timestamps`, dated packages receive them and the horizon is placed from
`future_timestamps` or the frequency. Without timestamps most adapters model
an integer index and say so in `metadata.index`. Prophet refuses instead,
because its calendar effects would be fiction on an invented index. GluonTS
needs a calendar start, so undated requests get a synthetic daily index that
is disclosed in metadata and affects only calendar-aware models.

**Quantile rows are monotone.** Gnomon rejects crossing quantiles. Sample-based
adapters are monotone by construction. Interval-based adapters can cross at
the edges of a fit; the shared helper repairs the row and records the count in
`metadata.quantile_crossings_repaired`. The 0.5 quantile is the package's point
forecast for interval-based adapters and is flagged with
`median_quantile_is_point`.

**Covariates come in pairs.** Regression-style adapters need the same columns
over history and horizon. Supplying only `past_covariates` or only
`future_covariates` is rejected with the missing field named, never silently
dropped.

**Determinism is declared conservatively.** Statistical fits are deterministic.
Anything that samples is declared deterministic only when `seed` is set. Set
`deterministic = true` yourself when you know the model is, or the cache will
not apply.

**Revisions are package versions.** A registration such as
`statsforecast/statsforecast=2.1.1` names the software, not the data or the
fitted parameters. A model fitted per request has no stable weights to name.

**No accuracy claim.** Registering a package through an adapter establishes
neither forecasting lift nor calibration. Use `gnomon_evaluate` with an
explicit budget; training-per-request adapters make every fold pay the fit.

## Not included

Pretrained foundation models (Chronos, TimesFM, Moirai and similar) are
deliberately absent from this first set. They download weights at first use,
have unknown training cutoffs, and deserve their own review.
Clone-and-run research repositories with no PyPI release cannot be extras.
