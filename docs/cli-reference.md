# CLI reference

The CLI writes successful machine-readable responses to standard output and
structured errors to standard error. Success exits with code `0`; Aion
input/runtime errors exit with code `2`.

## Global options

```bash
aion --help
aion --version
```

## `aion capabilities`

Reports only functionality available in the installed runtime:

```bash
aion capabilities
aion capabilities --output json
```

Use this response for feature detection instead of assuming that roadmap
features in the product specification are installed.

## `aion inspect`

Validates an input without forecasting:

```bash
aion inspect INPUT --time COLUMN --target COLUMN [OPTIONS]
```

| Option | Required | Meaning |
| --- | --- | --- |
| `INPUT` | Yes | CSV, `.parquet`, or `.pq` path. |
| `--time COLUMN` | Yes | Timestamp column. |
| `--target COLUMN` | Yes | Numeric target column. |
| `--series COLUMN` | No | Independent-series identifier. |
| `--frequency CODE` | No | `min`, `5min`, `15min`, `30min`, `h`, `D`, `W`, or `MS`; inferred when omitted. |

Inspection returns the source SHA-256 fingerprint, resolved schema, columns,
series names, observation counts, and date ranges.

## `aion forecast`

Runs validation, rolling evaluation, model selection, calibration, support
assessment, final forecasting, and artifact persistence:

```bash
aion forecast INPUT --time COLUMN --target COLUMN --horizon N [OPTIONS]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--series COLUMN` | None | Independent-series identifier. |
| `--frequency CODE` | Inferred | `min`, `5min`, `15min`, `30min`, `h`, `D`, `W`, or `MS`. |
| `--horizon N` | Required | Number of future periods; must be at least one. |
| `--output DIR` | `aion-output` | Parent directory for immutable run directories. |
| `--minimum-baseline-improvement FLOAT` | `0.02` | Fractional improvement required before selecting a candidate over the strongest baseline. |
| `--context FILE` | None | Validated context-events JSON (output of `aion context validate`). |
| `--threshold VALUE` | None | Decision threshold: the result reports when and how likely the forecast crosses this value. |
| `--project NAME` | None | Register each forecast series for later realised scoring. |

An improvement value of `0.02` means two percent, not two percentage points.
Baseline retention is a valid outcome and does not itself weaken support.

Context events are proposals. Each series' response carries a `context`
block recording the admission decision: events enter the forecast only when
they demonstrate stable improvement on identical backtest folds, and events
without a verifiable source never participate in backtests at all.

## `aion context`

The bring-your-own-brain workflow. Aion owns the prompt and the validation;
any LLM the host chooses runs in between.

```bash
aion context prompt --file launches.md --file holidays.md --series api-prod
# → {"instructions": ..., "response_schema": ..., "documents": [...]}
# run instructions on your model, save the JSON response, then:
aion context validate --response response.json --file launches.md --file holidays.md
# → {"events": [...], "rejected": [...]} — feed to `aion forecast --context`
```

`validate` grounds each event's source from the document metadata (never
from the model's claims), rejects non-verbatim evidence quotes, and marks
whether each event is admissible for backtesting.

## `aion mcp serve`

Serves `aion_capabilities`, `aion_inspect`, and `aion_forecast` as typed
tools over stdio MCP for any MCP-capable host. Logs go to stderr; the
protocol owns stdout.

## `aion track`

Persist forecasts in a local SQLite registry and score them after the complete
forecast horizon has been observed:

```bash
aion forecast data.csv --time timestamp --target value --horizon 7 \
  --project capacity
aion track actuals --project capacity --file actuals.csv
aion track list --project capacity
aion track performance --project capacity --model seasonal_naive
aion track leaderboard --project capacity
aion track compare --a FORECAST_ID --b FORECAST_ID
```

Single-series actuals require `timestamp,value` columns. For panel forecasts,
use `series,timestamp,value`; Aion rejects ambiguous panel actuals. Timestamps
are compared as instants when timezone offsets are present. A forecast remains
open until actuals cover its entire horizon, preventing a partial submission
from producing a misleading final score.

MASE uses the naive scaling error saved from the training series when the
forecast is registered. It is reported as unavailable for constant histories
whose scale is zero. The leaderboard is descriptive historical telemetry: it
does not prove that one model caused better outcomes, and it does not currently
change future model selection automatically.

The default registry is `~/.local/share/aion/registry.db`. Override it with
`AION_REGISTRY_PATH` for isolated projects, tests, or containers.

## Shell automation

Capture the successful response:

```bash
aion forecast data.csv --time timestamp --target value --horizon 7 > run.json
```

Do not infer success from the existence of output text; check the process exit
code and the response's `status` field.

## Not currently available

Commands described in future-facing design documents—such as `init`, `run`,
and `share`—are not implemented in v0.2.
