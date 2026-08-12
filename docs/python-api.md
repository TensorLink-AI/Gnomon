# Python API

The Python API calls the same runtime used by the CLI.

## Inspect a dataset

```python
from gnomon import inspect_dataset

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
from gnomon import forecast

artifact, artifact_path = forecast(
    "observations.csv",
    time_column="timestamp",
    target_column="requests",
    series_column="service_id",
    frequency="D",
    horizon=7,
    output="gnomon-output",
    minimum_baseline_improvement=0.02,
)

for result in artifact.results:
    print(result.series, result.support, result.selected_model)
print(artifact_path)
```

### Forecast with covariates

```python
from gnomon import forecast, load_covariates, validate_covariate_file

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

The mapping requires explicit `future_known` availability. Gnomon uses
`known_at` to replay the value available at each historical fold cutoff.

The returned `ForecastArtifact` is a dataclass. Use `artifact.to_dict()` for a
JSON-compatible representation. Calling `forecast` also persists the standard
artifact files — see [Results and artifacts](results-and-artifacts.md) for the
canonical list and what each one carries.

## Use the other governed views

The Python package exports the same five top-level views advertised by the CLI
and agent surfaces. The four non-forecast views return `(payload,
artifact_path)`; the payload is JSON-compatible and the artifact owns its
numbers and evidence.

```python
from gnomon import decide, detect_anomalies, investigate_change, monitor

common = {
    "time_column": "timestamp",
    "target_column": "requests",
    "frequency": "D",
    "output": "gnomon-output",
}

investigation, investigation_path = investigate_change(
    "observations.csv", **common
)

detection, detection_path = detect_anomalies(
    "observations.csv", threshold=3.5, **common
)

decision, decision_path = decide(
    "observations.csv",
    horizon=7,
    threshold=340,
    actions=[{"name": "scale_up"}, {"name": "wait"}],
    utilities={
        "scale_up": {"exceed": 8, "no_exceed": -2},
        "wait": {"exceed": -20, "no_exceed": 0},
    },
    **common,
)

monitoring, monitoring_path = monitor(
    "observations.csv",
    horizon=7,
    threshold=340,
    alert_cost=1,
    miss_cost=20,
    **common,
)
```

`investigate_change` ranks associational explanations, never causes.
`detect_anomalies` discloses how competing detectors were graded. `decide`
degrades to a feasible-action comparison when utilities are absent, and
`monitor` marks an uncosted default rule when alert and miss costs are absent.
Inspect each payload's support assessment and limitations before acting.

## Handle structured errors

```python
from gnomon import inspect_dataset
from gnomon.contracts import GnomonError

try:
    inspect_dataset(
        "observations.csv",
        time_column="timestamp",
        target_column="requests",
    )
except GnomonError as error:
    print(error.code)
    print(error.message)
    print(error.details)
```

`GnomonError.to_dict()` returns the same structured error envelope emitted by
the CLI. An `unsupported` series is not an exception: inspect
`artifact.results[*].support` and its warnings.

## API stability

The artifact schema is versioned as `0.1`, but the Python API is still an MVP.
Pin the package version and consume persisted artifacts when long-term
compatibility is important.
