# Installation and quick start

## Requirements

- Python 3.11 or newer
- `uv` for the recommended installation, or a Python environment with `pip`
- A regular time series stored in CSV; Parquet is optional

No account, network service, LLM, or API key is required to forecast.

## Install from this checkout

```bash
cd Gnomon
bash install.sh --local
gnomon --version
gnomon capabilities
```

`--local` installs the checkout you are standing in. Without it,
`install.sh` fetches the repository's default branch from GitHub — which is
right for a first install from a URL, and wrong when you are testing local
changes.

See [Installation options](installation.md) for uv, direct GitHub, pinned
release, and future PyPI installation methods.

To reinstall after changing the source:

```bash
uv tool install --force .
```

For development without a tool installation:

```bash
cd Gnomon
PYTHONPATH=src python3 -m gnomon capabilities
```

## Run the included example

Inspect the input before spending time on a forecast:

```bash
gnomon inspect examples/daily_requests.csv \
  --time timestamp \
  --target requests \
  --frequency D
```

A successful inspection reports the resolved schema, source fingerprint,
frequency, date range, and number of observations.

Run a three-day forecast:

```bash
gnomon forecast examples/daily_requests.csv \
  --time timestamp \
  --target requests \
  --horizon 3 \
  --frequency D \
  --output ./gnomon-output
```

The command prints JSON containing the forecast ID, support result, selected
model, warnings, and artifact directory. Each run receives a new directory:

```text
gnomon-output/forecast_<id>/
├── artifact.json
├── evidence.jsonl
├── forecast.csv
└── summary.md
```

Start with `summary.md`, use `forecast.csv` for charts or downstream work, and
retain `artifact.json` when reproducibility or auditability matters.

## Forecast multiple series

If one file contains several independent series, identify the grouping column:

```bash
gnomon forecast panel.csv \
  --time timestamp \
  --target requests \
  --series service_id \
  --horizon 7 \
  --frequency D
```

Every series is validated, evaluated, selected, and supported independently.
All series must currently share one regular frequency.

## Next steps

- Read [Preparing data](data-format.md) before using production data.
- Read [Understanding results](results-and-artifacts.md) before acting on a forecast.
- Use [Troubleshooting](troubleshooting.md) for structured errors or abstention.
