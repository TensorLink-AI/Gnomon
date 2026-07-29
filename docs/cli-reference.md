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
| `--frequency CODE` | No | `h`, `D`, `W`, or `MS`; inferred when omitted. |

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
| `--frequency CODE` | Inferred | `h`, `D`, `W`, or `MS`. |
| `--horizon N` | Required | Number of future periods; must be at least one. |
| `--output DIR` | `aion-output` | Parent directory for immutable run directories. |
| `--minimum-baseline-improvement FLOAT` | `0.02` | Fractional improvement required before selecting drift over the strongest baseline. |

An improvement value of `0.02` means two percent, not two percentage points.
Baseline retention is a valid outcome and does not itself weaken support.

## Shell automation

Capture the successful response:

```bash
aion forecast data.csv --time timestamp --target value --horizon 7 > run.json
```

Do not infer success from the existence of output text; check the process exit
code and the response's `status` field.

## Not currently available

Commands described in future-facing design documents—such as `init`, `run`,
`actuals`, `score`, `share`, and `mcp serve`—are not implemented in v0.1.

