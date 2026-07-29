# Aion

Aion is a local-first forecasting runtime that validates temporal data,
compares candidate forecasts with mandatory baselines, quantifies uncertainty,
and abstains when the evidence is inadequate.

## Quick start

```bash
bash install.sh
aion capabilities
aion inspect examples/daily_requests.csv --time timestamp --target requests
aion forecast examples/daily_requests.csv \
  --time timestamp --target requests --horizon 3 --frequency D
```

The forecast command writes an immutable run directory containing
`artifact.json`, `forecast.csv`, `evidence.jsonl`, and `summary.md`. Use
`--output DIR` to choose its parent directory.

CSV input is supported by the default installation. Parquet input is enabled
with `pip install 'aion-forecast[parquet]'`.

## Current scope

The v0.1 runtime implements inspection, strict temporal validation, last-value,
seasonal-naive and drift models, separated rolling selection/calibration/test
windows, per-series selection, residual uncertainty, support assessment, and
machine-readable artifacts. Run `aion capabilities` for the authoritative
feature list. Project lifecycle, actual submission, MCP, TSFM adapters, context
events, and sharing remain planned and are never presented as working features.

## Documentation

- [Documentation index](docs/README.md)
- [Installation options and one-command installer](docs/installation.md)
- [Installation and quick start](docs/getting-started.md)
- [Preparing data](docs/data-format.md)
- [CLI reference](docs/cli-reference.md)
- [Python API](docs/python-api.md)
- [Understanding results and artifacts](docs/results-and-artifacts.md)
- [Forecasting and evaluation concepts](docs/concepts.md)
- [Troubleshooting and error reference](docs/troubleshooting.md)
- [LLMs, API keys, OpenRouter, and planned integrations](docs/llm-integrations.md)
- [Development and testing](docs/development.md)
- [Containers](docs/containers.md)
- [CI/CD and release operations](docs/ci-cd.md)

The [product specification](Aion_MVP_Product_Specification.md) and
[system design](Aion_System_Design.md) describe the broader direction;
they include features that are not implemented yet. Use `aion capabilities`
as the source of truth for the installed runtime.
