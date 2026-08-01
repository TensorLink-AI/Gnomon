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

Messy files are diagnosed rather than rejected: `data_quality.status` reports
`clean`, `repaired_safe` (the default forecast repair level reads the file),
or `repaired_aggressive` (the file needs `--repair aggressive`), together
with the exact list of repairs each level would apply and a `suggested_next`
command including any required flag. `aion inspect` fails only when no repair
level can read the file.

## `aion forecast`

Short histories use a single trailing holdout and return `degraded` forecasts
by default. Pass `--strict-abstention` to retain empty-result abstention when
separated rolling evaluation is unavailable.

Forecast controls:

- `--seasonal-period N` overrides autocorrelation-based season detection.
- `--selection-strategy ensemble` (or `--ensemble`) averages eligible model
  forecasts using inverse-error weights and reports `supported_ensemble`.
- `--multivariate` tries a VAR(1) forecast for aligned, correlated series. It
  is used only when it beats an independent last-value forecast on a trailing
  holdout; otherwise Aion falls back to its normal per-series path.
- `--repair {off,safe,aggressive}` (default `safe`) controls messy-data
  handling. `safe` normalises cell text — mixed date formats, currency and
  thousands separators, percent signs, sentinel missing values, blank rows,
  identical duplicate rows — and never invents, moves, or drops a data
  point. `aggressive` additionally interpolates interior gaps, snaps
  jittered timestamps to the grid, resolves conflicting duplicates (last
  row wins), and drops unparseable rows, all capped: past roughly 30% of a
  series the run refuses with `EXCESSIVE_REPAIR`. Every repair is recorded
  in a `data_repair` evidence record; assumptive repairs also appear as
  `repaired_data:` warnings and downgrade support. `off` restores strict
  rejection. Repairs only ever fire where strict parsing would fail, so
  clean files produce byte-identical artifacts. Try it:
  `aion inspect examples/filthy_requests.csv --time timestamp --target requests`.

`aion inspect` reports the detected seasonal period for each series and
pairwise correlations for aligned multi-series inputs.

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
| `--covariates FILE` | None | Local point-in-time covariate CSV. |
| `--covariate-mapping MAP` | None | Required `name:type:future_known` entries. |
| `--covariate-time COLUMN` | `timestamp` | Covariate valid-at column. |
| `--covariate-known-at COLUMN` | `known_at` | Covariate availability column. |
| `--covariate-series COLUMN` | None | Optional panel-series key. |

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

## `aion covariates`

Ask Aion for the point-in-time format and exact fold cutoffs:

```bash
aion covariates guide INPUT --time COLUMN --target COLUMN --horizon N
```

Validate a local proposal before paying for a complete forecast run:

```bash
aion covariates validate INPUT --time COLUMN --target COLUMN --horizon N \
  --covariates covariates.csv \
  --covariate-mapping 'holiday:binary:future_known'
```

Validation rejects missing historical vintages and incomplete final-horizon
coverage. See [Covariate enrichment](covariates.md).

## `aion mcp serve`

Serves forecasting plus typed tracking, actual-submission, performance, and
decision-outcome tools over stdio MCP for any MCP-capable host. Discover the
installed list with `tools/list`; logs go to stderr and the protocol owns
stdout.

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
aion track leaderboard --project capacity --task forecast
aion track compare --a FORECAST_ID --b FORECAST_ID
aion track due --project capacity
aion track decision record --decision-id scale-001 --project capacity \
  --forecast-id FORECAST_ID --action "add two workers" \
  --expected-outcome "keep utilisation below 80%"
aion track decision resolve --decision-id scale-001 \
  --actual-outcome "peak utilisation was 74%" --correct true
aion track export --project capacity --output capacity-registry.json
aion track relocate --forecast-id FORECAST_ID --artifact-path /new/artifact/path
```

Single-series actuals require `timestamp,value` columns. For panel forecasts,
use `series,timestamp,value`; Aion rejects ambiguous panel actuals. Timestamps
are compared as instants when timezone offsets are present. A forecast remains
open until actuals cover its entire horizon, preventing a partial submission
from producing a misleading final score.

MASE uses the naive scaling error saved from the training series when the
forecast is registered. It is reported as unavailable for constant histories
whose scale is zero. The leaderboard is descriptive historical telemetry: it
does not prove that one model caused better outcomes, and it does not change
future model selection automatically — `aion route` consults it as a
disclosed, advisory prior, and evaluated runs still backtest every candidate.

Each registered run also records its task (`forecast` by default) and a
deterministic series fingerprint (trend, noise ratio, intermittency,
direction-change rate, season), which `--task` filtering and the router's
fingerprint-weighted prior are built on.

The default registry is `~/.local/share/aion/registry.db`. Override it with
`AION_REGISTRY_PATH` for isolated projects, tests, or containers.

## `aion eval compare`

Compare programmatically graded agent runs with and without Aion:

```bash
aion eval compare --baseline control.jsonl --treatment aion.jsonl
```

See [Agent evaluation](agent-evaluation.md) for the JSONL contract and fair
treatment/control protocol.

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

## `aion investigate`

What changed? Changepoint detection, regime-shift vs transient
classification, anomaly scores, and ranked associational explanations:

```bash
aion investigate data.csv --time timestamp --target value
aion investigate data.csv --time timestamp --target value \
  --context events.json --as-of 2026-06-01
```

## `aion detect`

What is abnormal? Candidate detectors — robust z-score, rolling-median
residual, forecast-interval exceedance, plus any installed multi-task TSFM
sandbox's reconstruction error — compete on a deterministic synthetic
anomaly-injection grader; the winner flags anomalies and every candidate's
precision/recall/F1 is disclosed in the artifact:

```bash
aion detect data.csv --time timestamp --target value
aion detect data.csv --time timestamp --target value \
  --threshold 3.0 --labels "2026-05-04,2026-06-11"
```

With `--labels` (known anomaly timestamps), detector selection uses label
F1 instead of the synthetic grader.

## `aion route`

Which method for this task on this data? A disclosed, advisory routing
decision: verified capability filter, then a fingerprint-weighted
realised-performance prior from the tracking store — claimed only when
enough scored history exists, never cold:

```bash
aion route data.csv --time timestamp --target value --task forecast \
  --horizon 14 --project ops
aion route data.csv --time timestamp --target value --task detect_anomalies
```

With `--project`, the prior is consulted and the decision recorded to the
tracking store for replay. Evaluated runs still backtest every candidate;
an explicit model choice always wins.

## `aion decide`

What should we do? Exceedance scenarios from an evaluated forecast plus
feasibility, constraints, and expected utility over candidate actions.
Without `--utilities` the result is the feasible-action comparison,
`conditionally_supported: missing utility inputs`:

```bash
aion decide data.csv --time timestamp --target value --horizon 14 \
  --threshold 340 \
  --actions '[{"name": "scale_up"}, {"name": "wait"}]' \
  --utilities '{"scale_up": {"exceed": 100, "no_exceed": -10},
                "wait": {"exceed": -400, "no_exceed": 5}}' \
  --project ops
```

`--actions` and `--utilities` accept inline JSON or `@path/to/file.json`.

## `aion monitor`

When should we intervene? Sequential exceedance risk per horizon step and
an alert rule — cost-optimal when `--alert-cost` and `--miss-cost` are
supplied, a flagged 0.5 default otherwise:

```bash
aion monitor data.csv --time timestamp --target value --horizon 14 \
  --threshold 340 --alert-cost 1 --miss-cost 20 --project ops
```

## `aion ingest` and `aion store`

Append observations to the bitemporal store; re-supplied corrected files
become new revision rows rather than overwrites:

```bash
aion ingest revisions.csv --dataset requests \
  --time timestamp --target value --known-at published
aion store list
```

Datasets are then addressable as `store:<dataset>` in any verb, and
`--as-of <instant>` replays a run using only data known at that moment.

## `aion status`

Pollable view of open forecasts, due horizons, unresolved decisions, and
realised-performance summaries (descriptive, never causal):

```bash
aion status --project ops
```

## `aion track outcome`

Resolve a recorded `DecisionArtifact` with what actually happened; returns
realised utility, regret versus the best feasible action in hindsight, and
ex-ante optimality:

```bash
aion track outcome --decision-id decision_abc123 \
  --realised-scenario no_exceed --note "traffic stayed under capacity"
```

## `aion eval episodes`

Run the built-in trap-family episode suite (leakage, abstention, regime
breaks) with the honest reference policy and emit rows for
`aion eval compare`:

```bash
aion eval episodes --workdir /tmp/aion-episodes --trials 2 --jsonl runs.jsonl
```

## `aion plan` (experimental)

Compile, validate, and execute `TemporalPlan`s. The agent-facing tools are
gated behind `AION_EXPERIMENTAL_PLANNER=1`; macros remain the default path:

```bash
aion plan compile --task-type forecast --params '{"input": "data.csv",
  "time_column": "timestamp", "target_column": "value", "horizon": 7}'
aion plan validate --plan @plan.json
aion plan execute --plan @plan.json --output aion-output
```
