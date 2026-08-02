# Python API

The Python API calls the same runtime used by the CLI.

## Inspect a dataset

```python
from aion import inspect_dataset

inspection = inspect_dataset(
    "observations.csv",
    time_column="timestamp",
    target_column="requests",
    series_column="service_id",
    frequency="D",
)

print(inspection["schema"])
print(inspection["series"])
```

`inspect_dataset` returns a JSON-compatible dictionary and does not write an
artifact directory.

## Create a forecast

```python
from aion import forecast

artifact, artifact_path = forecast(
    "observations.csv",
    time_column="timestamp",
    target_column="requests",
    series_column="service_id",
    frequency="D",
    horizon=7,
    output="aion-output",
    minimum_baseline_improvement=0.02,
)

for result in artifact.results:
    print(result.series, result.support, result.selected_model)
print(artifact_path)
```

### Forecast with covariates

```python
from aion import forecast, load_covariates, validate_covariate_file

validation = validate_covariate_file(
    "observations.csv",
    "covariates.csv",
    "is_holiday:binary:future_known",
    time_column="timestamp",
    target_column="requests",
    horizon=7,
    frequency="D",
)
if not validation["valid"]:
    raise ValueError(validation["validation"])

covariates = load_covariates(
    "covariates.csv", "is_holiday:binary:future_known"
)
artifact, artifact_path = forecast(
    "observations.csv",
    time_column="timestamp",
    target_column="requests",
    horizon=7,
    frequency="D",
    covariates=covariates,
)
```

The mapping requires explicit `future_known` availability. Aion uses
`known_at` to replay the value available at each historical fold cutoff.

The returned `ForecastArtifact` is a dataclass. Use `artifact.to_dict()` for a
JSON-compatible representation. Calling `forecast` also persists the standard
artifact files — see [Results and artifacts](results-and-artifacts.md) for the
canonical list and what each one carries.

## Handle structured errors

```python
from aion import inspect_dataset
from aion.contracts import AionError

try:
    inspect_dataset(
        "observations.csv",
        time_column="timestamp",
        target_column="requests",
    )
except AionError as error:
    print(error.code)
    print(error.message)
    print(error.details)
```

`AionError.to_dict()` returns the same structured error envelope emitted by
the CLI. An `unsupported` series is not an exception: inspect
`artifact.results[*].support` and its warnings.

## API stability

The artifact schema is versioned as `0.1`, but the Python API is still an MVP.
Pin the package version and consume persisted artifacts when long-term
compatibility is important.
