# Aion

**Forecast anything you can measure—and test whether the forecast deserves to be trusted.**

Aion is a local forecasting engine for developers, operators, and AI agents. It
turns regular time-series data into a backtested forecast, compares every
candidate with simple baselines, measures uncertainty, and abstains when the
available evidence is inadequate.

It is designed to give agents such as Hermes a safe numerical capability. An
agent can discover data, formulate a question, and explain the result; Aion
remains authoritative for timestamps, backtests, model selection, intervals,
support status, and forecast values. The LLM never gets to invent or edit the
numbers.

> Aion is named for the Greek personification of enduring and cyclical time.

## Why Aion exists

Most forecasting tools answer, “What number comes next?” Aion also asks:

- Is the timestamp grid valid and reproducible?
- Does the chosen method beat a defensible naive baseline?
- Was evaluation performed without using future observations?
- Is the uncertainty estimate supported by historical residuals?
- Is there enough evidence to return a forecast at all?

Aion treats **unsupported** as a useful outcome. A structured abstention is
safer than a plausible-looking forecast built from insufficient or malformed
history.

## What it does today

| Capability | v0.1 status |
| --- | --- |
| CSV input | Available |
| Parquet input | Available with the `parquet` extra |
| Hourly, daily, weekly, and month-start data | Available |
| Multiple independent series in one file | Available |
| Last-value and seasonal-naive baselines | Available |
| Drift candidate model | Available |
| Separated selection, calibration, and final-test windows | Available |
| Residual quantile intervals and measured test coverage | Available |
| Per-series selection and abstention | Available |
| JSON, CSV, JSONL, and Markdown artifacts | Available |
| Local CLI and Python API | Available |
| Docker image and GitHub CI/CD | Available |
| Hermes plugin wrapping the CLI (tools + safe-use skill) | Available |
| Evidence-gated context events (`--context`, identical-fold ablation) | Available |
| Local MCP server (`aion mcp serve`) | Available |
| LLM workflow prompts (`aion context prompt` / `validate`) | Available |
| Standalone LLM providers, TSFMs, project mode, and sharing | Planned |

`aion capabilities` is the machine-readable source of truth. Roadmap features
are not exposed as mocked commands.

## See it work

Install Aion from this checkout and inspect the included dataset:

```bash
bash install.sh

aion inspect examples/daily_requests.csv \
  --time timestamp \
  --target requests \
  --frequency D
```

Then forecast the next three days:

```bash
aion forecast examples/daily_requests.csv \
  --time timestamp \
  --target requests \
  --horizon 3 \
  --frequency D \
  --output ./aion-output
```

The example produces a result like:

```text
Support: supported
Selected model: drift
Strongest baseline: last_value

2026-02-05  point=205  q10=205  q50=205  q90=205
2026-02-06  point=208  q10=208  q50=208  q90=208
2026-02-07  point=211  q10=211  q50=211  q90=211
```

The example is deliberately simple and perfectly linear, so its calibration
residuals—and therefore its interval widths—are zero. That demonstrates the
pipeline, not realistic certainty. Use noisy operational history to evaluate
forecast quality for a real decision.

## How Aion reaches a result

```text
CSV or Parquet
      │
      ▼
schema and temporal validation
      │
      ▼
rolling model-selection folds ── compare with mandatory baselines
      │
      ▼
separate calibration fold ────── estimate residual quantiles
      │
      ▼
untouched final test ─────────── measure error and interval coverage
      │
      ▼
select, retain a baseline, or abstain
      │
      ▼
forecast + evidence + reproducible artifacts
```

Earlier rolling folds select the method. The penultimate fold calibrates the
interval. The final fold reports performance without changing either choice.
The selected method is then run on all available observations to forecast the
future.

## Input

Aion needs a timestamp column and numeric target. A series column is optional:

```csv
timestamp,requests,service_id
2026-01-01T00:00:00+10:00,120,api
2026-01-02T00:00:00+10:00,128,api
2026-01-03T00:00:00+10:00,135,api
```

```bash
aion forecast observations.csv \
  --time timestamp \
  --target requests \
  --series service_id \
  --horizon 7 \
  --frequency D
```

The current policy rejects duplicate timestamps and missing periods instead of
silently aggregating or imputing them. See [preparing data](docs/data-format.md)
for supported timestamp forms, frequencies, panel rules, and history needs.

## Output

Every completed run receives an immutable directory:

```text
aion-output/forecast_<id>/
├── artifact.json    complete task, schema, scores, support, and forecast
├── evidence.jsonl   machine-readable evaluation and support evidence
├── forecast.csv     future timestamps, point values, and quantiles
└── summary.md       compact human-readable result
```

Start with `summary.md`. Use `forecast.csv` for charts and downstream systems.
Keep `artifact.json` when provenance, auditability, or reproducibility matters.

## Aion and Hermes

The intended relationship is:

- **Hermes** manages intent, permitted data discovery, orchestration, and explanation.
- **Aion** validates temporal data and owns every numerical result.

A packaged Hermes plugin lives in [`integrations/hermes`](integrations/hermes/README.md):
four tools wrapping the CLI — including LLM-assisted context-event proposal
run on the host's own model — plus an `aion:forecasting` safe-use skill. Any
MCP-capable agent can instead launch `aion mcp serve` and discover the same
tools over stdio. Either way the host must preserve Aion's support status and
warnings and must never manufacture values for an unsupported series. Aion
itself requires no LLM or API key.

## Installation

From a clone:

```bash
bash install.sh
aion capabilities
```

With uv:

```bash
uv tool install .
```

With Docker:

```bash
docker build -t aion .
docker run --rm aion capabilities
```

The direct GitHub installer, private-repository behavior, pinned releases,
Parquet extra, and future PyPI command are covered in the
[installation guide](docs/installation.md).

## Documentation

| Guide | Purpose |
| --- | --- |
| [Getting started](docs/getting-started.md) | Complete first run |
| [Installation](docs/installation.md) | Bash, uv, GitHub, Docker, and PyPI options |
| [Preparing data](docs/data-format.md) | Input schema and temporal requirements |
| [CLI reference](docs/cli-reference.md) | Commands, options, output, and exit codes |
| [Python API](docs/python-api.md) | Runtime integration from Python |
| [Results and artifacts](docs/results-and-artifacts.md) | Interpret support, scores, intervals, and files |
| [Forecasting concepts](docs/concepts.md) | Baselines, temporal evaluation, and abstention |
| [Troubleshooting](docs/troubleshooting.md) | Structured errors and remediation |
| [LLM integrations](docs/llm-integrations.md) | Current API-key status and intended agent boundary |
| [Hermes plugin](integrations/hermes/README.md) | Install and operate Aion inside Hermes Agent |
| [Containers](docs/containers.md) | Local Docker and GHCR operation |
| [CI/CD](docs/ci-cd.md) | Tests, PyPI trusted publishing, and releases |

The [product specification](Aion_MVP_Product_Specification.md) describes the
broader product direction. The [system design](Aion_System_Design.md) defines
the intended architecture. Both include roadmap features; the capability
response and this README distinguish those from working v0.1 behavior.

## Current limits

Aion v0.1 is a narrow foundation, not a general forecasting platform. It has
one statistical candidate, fixed seasonal periods, one calibration horizon, no
covariates, no transformations, no intermittent-demand model, no TSFM, and no
realized-actual scoring. A `supported` result means the current deterministic
checks passed; it is not a guarantee that the future will resemble history.

## Development

```bash
PYTHONPATH=src pytest -q
uv build
```

Contributions should preserve Aion's central boundary: agents may improve the
question and explanation, but only deterministic temporal tools may produce or
change forecast numbers.

Licensed under the [Apache License 2.0](LICENSE).
