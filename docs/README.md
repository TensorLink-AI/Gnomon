# Aion documentation

Aion turns regular temporal data into evaluated forecasts. It validates the
time grid, compares candidate methods with mandatory baselines, calibrates an
interval on a separate window, and either returns a supported result or abstains.

## Start here

1. [Install Aion and run the example](getting-started.md).
2. [Prepare your CSV or Parquet file](data-format.md).
3. [Understand support, scores, intervals, and artifacts](results-and-artifacts.md).

## Guides and reference

| Document | Use it when |
| --- | --- |
| [Installation](installation.md) | You want Bash, uv, or PyPI installation options. |
| [Getting started](getting-started.md) | You want a complete first run. |
| [Data format](data-format.md) | You need to prepare or validate input data. |
| [Covariate enrichment](covariates.md) | You want an agent to propose external data without temporal leakage. |
| [CLI reference](cli-reference.md) | You need exact commands and options. |
| [Python API](python-api.md) | You want to call Aion from Python. |
| [Results and artifacts](results-and-artifacts.md) | You need to interpret or automate outputs. |
| [Concepts](concepts.md) | You want to understand selection, evaluation, intervals, or abstention. |
| [Troubleshooting](troubleshooting.md) | A command failed or returned unsupported. |
| [LLM integrations](llm-integrations.md) | You are looking for API-key, OpenRouter, or Hermes support. |
| [Development](development.md) | You want to test or contribute to Aion. |
| [Containers](containers.md) | You want to build or run the Docker image. |
| [CI/CD](ci-cd.md) | You maintain validation, publishing, or releases. |

## Implemented

- Local CLI and Python API
- CSV input; Parquet with the optional `pyarrow` dependency
- Minute (1/5/15/30), hourly, daily, weekly, and month-start frequencies
- Independent series stored in one panel file
- Last-value and seasonal-naive baselines
- Drift, linear-trend, window-average, Theta, and ETS (Holt-Winters) candidate models
- Separated model-selection, interval-calibration, and final-test windows
- Horizon-widened uncertainty intervals and optional threshold-crossing analysis
- Per-series support assessment
- JSON, CSV, JSONL, and Markdown artifacts
- Point-in-time future-known covariates with leakage-safe ablation
- Optional MCP and Hermes tools, projects, actual scoring, and TSFM candidates

Hosted services, automatic model switching, and sharing remain roadmap features.
