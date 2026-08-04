# CLI reference

The CLI writes successful machine-readable responses to standard output and
structured errors to standard error. Success exits with code `0`; Gnomon
input/runtime errors exit with code `2`.

Every command below is implemented and reachable in this build. The one
exception is marked: `gnomon plan` is gated behind
`GNOMON_EXPERIMENTAL_PLANNER=1`. Commands that exist only in the design
documents are listed under [Not currently
available](#not-currently-available) at the end.

## The five verbs

| Command | Question | Section |
| --- | --- | --- |
| `gnomon forecast` | What happens next? | [↓](#gnomon-forecast) |
| `gnomon investigate` | What changed? | [↓](#gnomon-investigate) |
| `gnomon detect` | What is abnormal? | [↓](#gnomon-detect) |
| `gnomon decide` | What should we do? | [↓](#gnomon-decide) |
| `gnomon monitor` | When should we intervene? | [↓](#gnomon-monitor) |

## Everything else

`gnomon capabilities` · `gnomon inspect` · `gnomon route` · `gnomon ingest` ·
`gnomon store list` · `gnomon status` · `gnomon context prompt|validate` ·
`gnomon covariates guide|validate` · `gnomon mcp serve` ·
`gnomon tsfm list|install|install-all|remove` ·
`gnomon track actuals|compare|coverage|decision|due|export|leaderboard|list|outcome|performance|relocate|score` ·
`gnomon eval compare|episodes` · `gnomon plan compile|validate|execute`

## Global options

```bash
gnomon --help
gnomon --version
```

## `gnomon capabilities`

Reports only functionality available in the installed runtime:

```bash
gnomon capabilities
gnomon capabilities --output json
```

Use this response for feature detection instead of assuming that roadmap
features in the product specification are installed.

## `gnomon inspect`

Validates an input without forecasting:

```bash
gnomon inspect INPUT --time COLUMN --target COLUMN [OPTIONS]
```

| Option | Required | Meaning |
| --- | --- | --- |
| `INPUT` | Yes | CSV, `.parquet`, or `.pq` path. |
| `--time COLUMN` | Yes | Timestamp column. |
| `--target COLUMN` | Yes | Numeric target column. |
| `--series COLUMN` | No | Independent-series identifier. |
| `--frequency CODE` | No | Named codes (`s`, `min`, `5min`, `10min`, `15min`, `30min`, `h`, `D`, `W`, `MS`) or any whole-second sub-daily step (`10s`, `7min`, `2h`, …); inferred when omitted. |

Inspection returns the source SHA-256 fingerprint, resolved schema, columns,
series names, observation counts, and date ranges.

Messy files are diagnosed rather than rejected: `data_quality.status` reports
`clean`, `repaired_safe` (the default forecast repair level reads the file),
or `repaired_aggressive` (the file needs `--repair aggressive`), together
with the exact list of repairs each level would apply and a `suggested_next`
command including any required flag. `gnomon inspect` fails only when no repair
level can read the file.

## `gnomon forecast`

The minimal invocation is the whole invocation for most files:

```bash
gnomon forecast data.csv
```

Three flags are inferred, and every inference is disclosed in the
response's `assumptions` — an inference nobody is told about is a guess:

- `--time` is inferred when exactly one column parses as timestamps.
- `--target` is inferred when exactly one non-time column parses as
  numbers.
- `--horizon` defaults to one seasonal period of the inferred grid.
- `--frequency` is inferred from the observed step between timestamps.

Inference refuses — with an `AMBIGUOUS_SCHEMA` error naming the candidate
columns and the minimal working invocation — when more than one column
qualifies. It never guesses between two plausible readings.

To batch several columns of a wide file into one run, pass a comma list
or `auto`:

```bash
gnomon forecast vitals.csv --target hr,spo2,resp
gnomon forecast vitals.csv --target auto     # every numeric non-time column
```

One shared load pass; the channels evaluate concurrently and land in one
artifact, one result per column, each with its own support state. A
channel that abstains is disclosed in its own result and never blocks
the others. Per-channel numbers are identical to single-target runs.

`--brief` shrinks stdout to the q50 path with one q10–q90 interval per
step, plus — verbatim, never summarised — the support state and every
warning, abstention reason, recovery action, and disclosure. The full
artifact is written to disk unchanged; the default stdout format is also
unchanged.

Short histories use a single trailing holdout and return `degraded` forecasts
by default. Pass `--strict-abstention` to retain empty-result abstention when
separated rolling evaluation is unavailable.

When even the degraded path cannot run — the horizon exceeds what the
history can support at all — the run abstains with empty results. Pass
`--best-effort` to publish a clearly labelled naive fallback instead: the
last observed value carried flat, with random-walk intervals scaled from
the history's dispersion. The result reports support `best_effort`, carries
a verbatim `NO RELIABLE FORECAST` warning beside the abstention's original
reasons (including the horizon that *would* be supportable), and its
lineage claim is descriptive, never predictive — the numbers exist for
callers that must have numbers, and nothing about them claims measured
accuracy. Off by default; flag-off artifacts are byte-identical.

Forecast controls:

- `--seasonal-period N` overrides autocorrelation-based season detection.
- `--selection-strategy ensemble` (or `--ensemble`) averages eligible model
  forecasts using inverse-error weights and reports `supported_ensemble`.
- `--multivariate` tries a VAR(1) forecast for aligned, correlated series. It
  is used only when it beats an independent last-value forecast on a trailing
  holdout; otherwise Gnomon falls back to its normal per-series path.
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
  `gnomon inspect examples/filthy_requests.csv --time timestamp --target requests`.

`gnomon inspect` reports the detected seasonal period for each series and
pairwise correlations for aligned multi-series inputs.

Runs validation, rolling evaluation, model selection, calibration, support
assessment, final forecasting, and artifact persistence:

```bash
gnomon forecast INPUT [--time COLUMN] [--target COLUMN[,COLUMN…]] [--horizon N] [OPTIONS]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--time COLUMN` | Inferred | Timestamp column; inference refuses when ambiguous. |
| `--target COLUMN[,COLUMN…]` | Inferred | Numeric column(s) to model; a comma list or `auto` batches several columns into one run. |
| `--series COLUMN` | None | Independent-series identifier. |
| `--frequency CODE` | Inferred | Named codes (`s`, `min`, `5min`, `10min`, `15min`, `30min`, `h`, `D`, `W`, `MS`) or any whole-second sub-daily step (`10s`, `7min`, `2h`, …). |
| `--horizon N` | One seasonal period | Number of future periods; must be at least one. |
| `--brief` | Off | Compact stdout; disclosures verbatim; artifact unchanged. |
| `--output DIR` | `gnomon-output` | Parent directory for immutable run directories. |
| `--minimum-baseline-improvement FLOAT` | `0.02` | Fractional improvement required before selecting a candidate over the strongest baseline. |
| `--context FILE` | None | Validated context-events JSON (output of `gnomon context validate`). |
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

## `gnomon context`

The bring-your-own-brain workflow. Gnomon owns the prompt and the validation;
any LLM the host chooses runs in between.

```bash
gnomon context prompt --file launches.md --file holidays.md --series api-prod
# → {"instructions": ..., "response_schema": ..., "documents": [...]}
# run instructions on your model, save the JSON response, then:
gnomon context validate --response response.json --file launches.md --file holidays.md
# → {"events": [...], "rejected": [...]} — feed to `gnomon forecast --context`
gnomon context preflight data.csv --time timestamp --target value \
  --horizon 14 --events events.json
# → one verdict per event (would_influence / rejected / ablation_gated),
#   plus the span grammar the parser accepts — before spending a forecast
```

`validate` grounds each event's source from the document metadata (never
from the model's claims), rejects non-verbatim evidence quotes, and marks
whether each event is admissible for backtesting.

`preflight` dry-runs the deterministic admission checks — the
future-context lane and caller-supplied claims — against the actual data
and returns typed rejection reasons beside the accepted grammar, so a
rejected proposal is repaired and resubmitted instead of discovered in
the evidence after a spent run. Fold-ablation admission is a measurement
and is reported as `ablation_gated`, never predicted.

With `context.future_events: on` (or `prompt --future-events`), the prompt
also describes the two future-dated typed classes — `constraint:*` stated
bounds and `override:*` stated states — and `validate` copies each
verified evidence quote into the `source_span` the lane's deterministic
parser reads. A model-supplied `source_span` is always discarded: only a
quote verified verbatim against your documents can carry a number into
the forecast.

## `gnomon covariates`

Ask Gnomon for the point-in-time format and exact fold cutoffs:

```bash
gnomon covariates guide INPUT --time COLUMN --target COLUMN --horizon N
```

Validate a local proposal before paying for a complete forecast run:

```bash
gnomon covariates validate INPUT --time COLUMN --target COLUMN --horizon N \
  --covariates covariates.csv \
  --covariate-mapping 'holiday:binary:future_known'
```

Validation rejects missing historical vintages and incomplete final-horizon
coverage. See [Covariate enrichment](covariates.md).

## `gnomon mcp serve`

Serves forecasting plus typed tracking, actual-submission, performance, and
decision-outcome tools over stdio MCP for any MCP-capable host. Discover the
installed list with `tools/list`; logs go to stderr and the protocol owns
stdout.

## `gnomon tsfm`

Manage sandboxed time-series foundation models. The base install is
zero-dependency; each TSFM lives in its own isolated venv (created with
`uv`), so pulling one never touches your environment or another model's
pins. Model weights download from the Hugging Face Hub on first inference:

```bash
gnomon tsfm list                        # installable vs installed, with verified capabilities
gnomon tsfm install chronos_bolt_mini   # create the sandbox venv + dependencies
gnomon tsfm install moment_small        # multi-task: also unlocks a detect candidate
gnomon tsfm install-all                 # every registered adapter
gnomon tsfm remove chronos_bolt_mini    # delete the sandbox venv
```

Installing a sandbox is all the wiring there is: on the next `gnomon
forecast`, the model joins the backtest candidate pool and must beat the
statistical models to be selected. Installing `moment_small` additionally
adds a `moment_small_reconstruction` candidate to `gnomon detect`, graded
like every other detector. Nothing is ever selected on reputation —
uninstalled models are simply absent, and installed ones compete.

**For agents:** `gnomon_capabilities` reports the state machine-readably —
`models.tsfm_available` (installable), `models.tsfm_sandboxes` (installed),
and `models.tsfm_capabilities` (verified per-model limits and tasks) —
plus the install command template. Sandbox installation is a deliberate
human/shell step, not an MCP tool: an agent that wants a model should run
`gnomon tsfm install <name>` where it has shell access, or ask the operator
to.

## `gnomon track`

Persist forecasts in a local SQLite registry and score them after the complete
forecast horizon has been observed:

```bash
gnomon forecast data.csv --time timestamp --target value --horizon 7 \
  --project capacity
gnomon track actuals --project capacity --file actuals.csv
gnomon track list --project capacity
gnomon track performance --project capacity --model seasonal_naive
gnomon track leaderboard --project capacity
gnomon track leaderboard --project capacity --task forecast
gnomon track compare --a FORECAST_ID --b FORECAST_ID
gnomon track due --project capacity
gnomon track decision record --decision-id scale-001 --project capacity \
  --forecast-id FORECAST_ID --action "add two workers" \
  --expected-outcome "keep utilisation below 80%"
gnomon track decision resolve --decision-id scale-001 \
  --actual-outcome "peak utilisation was 74%" --correct true
gnomon track export --project capacity --output capacity-registry.json
gnomon track relocate --forecast-id FORECAST_ID --artifact-path /new/artifact/path
```

Single-series actuals require `timestamp,value` columns. For panel forecasts,
use `series,timestamp,value`; Gnomon rejects ambiguous panel actuals. Timestamps
are compared as instants when timezone offsets are present. A forecast remains
open until actuals cover its entire horizon, preventing a partial submission
from producing a misleading final score.

MASE uses the naive scaling error saved from the training series when the
forecast is registered. It is reported as unavailable for constant histories
whose scale is zero. The leaderboard is descriptive historical telemetry: it
does not prove that one model caused better outcomes, and it does not change
future model selection automatically — `gnomon route` consults it as a
disclosed, advisory prior, and evaluated runs still backtest every candidate.

Each registered run also records its task (`forecast` by default) and a
deterministic series fingerprint (trend, noise ratio, intermittency,
direction-change rate, season), which `--task` filtering and the router's
fingerprint-weighted prior are built on.

The default registry is `~/.local/share/gnomon/registry.db`. Override it with
`GNOMON_REGISTRY_PATH` for isolated projects, tests, or containers.

## `gnomon eval compare`

Compare programmatically graded agent runs with and without Gnomon:

```bash
gnomon eval compare --baseline control.jsonl --treatment gnomon.jsonl
```

See [Agent evaluation](agent-evaluation.md) for the JSONL contract and fair
treatment/control protocol.

## Shell automation

Capture the successful response:

```bash
gnomon forecast data.csv --time timestamp --target value --horizon 7 > run.json
```

Do not infer success from the existence of output text; check the process exit
code and the response's `status` field.

## `gnomon investigate`

What changed? Changepoint detection, regime-shift vs transient
classification, anomaly scores, and ranked associational explanations:

```bash
gnomon investigate data.csv --time timestamp --target value
gnomon investigate data.csv --time timestamp --target value \
  --context events.json --as-of 2026-06-01
```

## `gnomon detect`

What is abnormal? Candidate detectors — robust z-score, rolling-median
residual, local-slope deviation, forecast-interval exceedance, plus any
installed multi-task TSFM sandbox's reconstruction error — compete on a
deterministic synthetic anomaly-injection grader that plants spikes, level
shifts, dropouts, and trend shifts; the winner flags anomalies and every
candidate's precision/recall/F1 is disclosed in the artifact:

```bash
gnomon detect data.csv --time timestamp --target value
gnomon detect data.csv --time timestamp --target value \
  --threshold 3.0 --labels "2026-05-04,2026-06-11"
```

With `--labels` (known anomaly timestamps), detector selection uses label
F1 instead of the synthetic grader.

The grade covers the families the grader planted and nothing else. Each
result's support carries `graded_families` and, under synthetic selection,
an assumption naming them — a detector that recovers planted spikes has
not been tested on anomaly kinds outside that list.

## `gnomon route`

Which method for this task on this data? A disclosed, advisory routing
decision: verified capability filter, then a fingerprint-weighted
realised-performance prior from the tracking store — claimed only when
enough scored history exists, never cold:

```bash
gnomon route data.csv --time timestamp --target value --task forecast \
  --horizon 14 --project ops
gnomon route data.csv --time timestamp --target value --task detect_anomalies
```

With `--project`, the prior is consulted and the decision recorded to the
tracking store for replay. Evaluated runs still backtest every candidate;
an explicit model choice always wins.

## `gnomon decide`

What should we do? Exceedance scenarios from an evaluated forecast plus
feasibility, constraints, and expected utility over candidate actions.
Without `--utilities` the result is the feasible-action comparison,
`conditionally_supported: missing utility inputs`:

```bash
gnomon decide data.csv --time timestamp --target value --horizon 14 \
  --threshold 340 \
  --actions '[{"name": "scale_up"}, {"name": "wait"}]' \
  --utilities '{"scale_up": {"exceed": 100, "no_exceed": -10},
                "wait": {"exceed": -400, "no_exceed": 5}}' \
  --project ops
```

`--actions` and `--utilities` accept inline JSON or `@path/to/file.json`.

## `gnomon monitor`

When should we intervene? Sequential exceedance risk per horizon step and
an alert rule — cost-optimal when `--alert-cost` and `--miss-cost` are
supplied, a flagged 0.5 default otherwise:

```bash
gnomon monitor data.csv --time timestamp --target value --horizon 14 \
  --threshold 340 --alert-cost 1 --miss-cost 20 --project ops
```

## `gnomon ingest` and `gnomon store`

Append observations to the bitemporal store; re-supplied corrected files
become new revision rows rather than overwrites:

```bash
gnomon ingest revisions.csv --dataset requests \
  --time timestamp --target value --known-at published
gnomon store list
```

Datasets are then addressable as `store:<dataset>` in any verb, and
`--as-of <instant>` replays a run using only data known at that moment.

## `gnomon status`

Pollable view of open forecasts, due horizons, unresolved decisions, and
realised-performance summaries (descriptive, never causal):

```bash
gnomon status --project ops
```

## `gnomon track outcome`

Resolve a recorded `DecisionArtifact` with what actually happened; returns
realised utility, regret versus the best feasible action in hindsight, and
ex-ante optimality:

```bash
gnomon track outcome --decision-id decision_abc123 \
  --realised-scenario no_exceed --note "traffic stayed under capacity"
```

## `gnomon eval episodes`

Run the built-in trap-family episode suite (leakage, abstention, regime
breaks) with the honest reference policy and emit rows for
`gnomon eval compare`:

```bash
gnomon eval episodes --workdir /tmp/gnomon-episodes --trials 2 --jsonl runs.jsonl
```

## `gnomon plan` (experimental)

Compile, validate, and execute `TemporalPlan`s. The agent-facing tools are
gated behind `GNOMON_EXPERIMENTAL_PLANNER=1`; macros remain the default path:

```bash
gnomon plan compile --task-type forecast --params '{"input": "data.csv",
  "time_column": "timestamp", "target_column": "value", "horizon": 7}'
gnomon plan validate --plan @plan.json
gnomon plan execute --plan @plan.json --output gnomon-output
```

## Not currently available

`gnomon init`, `gnomon run`, and `gnomon share` appear in the product
specification and system design. They are **not implemented**, and no
mocked version of them is exposed. Ask `gnomon capabilities` rather than
either design document when you need to know what this build can do.
