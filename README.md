# Aion

**A temporal execution harness for agents — forecast, investigate, decide,
and monitor, with every number owned by Aion and every claim verified.**

Aion is a local temporal-reasoning engine for developers, operators, and AI
agents. It answers four questions over regular time-series data — *what
changed?* (`aion investigate`), *what happens next?* (`aion forecast`),
*what should we do?* (`aion decide`), and *when should we intervene?*
(`aion monitor`) — under one contract: the LLM proposes; Aion validates,
computes, and owns every number. A bitemporal store tracks when each value
became known, so any run can be replayed at any historical instant
(`--as-of`) and provably accessed nothing published after it. Results carry
typed support assessments (structured abstention included), evidence-linked
claims checked by a deterministic verifier, and machine-readable repair
options on every error. See `docs/quickstart-mcp.md` to go from install to
a grounded answer in a minute, and `COMPATIBILITY.md` for the frozen v0.2
surface (all v0.2 tools keep working unchanged).

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

**Four verbs.**

| Verb | Question | What you get |
| --- | --- | --- |
| `aion forecast` | What happens next? | Backtested model selection, residual-quantile intervals, threshold-crossing analysis — or a structured abstention |
| `aion investigate` | What changed? | Changepoints, regime shift vs transient, anomalies, ranked *associational* explanations (never a cause) |
| `aion decide` | What should we do? | Exceedance scenarios, feasibility and constraint checks, expected utility — degraded honestly when utilities are missing |
| `aion monitor` | When should we intervene? | Sequential exceedance risk and a cost-optimal alert rule |

**The harness around them.**

- Bitemporal store (`aion ingest`, `store:<dataset>` inputs): every value
  carries *when it became known*; `--as-of` replays any historical instant
  and the artifact proves nothing later was touched.
- Five-state support assessments with typed reasons and recovery actions;
  typed lineage and a deterministic claim verifier on every response.
- Messy-data repair (`--repair off|safe|aggressive`): mixed date formats,
  currency and thousands separators, sentinel `N/A` cells, duplicates,
  gaps, jittered timestamps — every fix disclosed as evidence, assumptive
  fixes downgrade support, excessive messiness is refused.
- Inputs: CSV (any common delimiter), TSV, JSON/JSONL, gzipped text,
  Parquet (`parquet` extra), and Excel (`excel` extra).
- Decision tracking with realised-outcome scoring: regret against the best
  feasible action in hindsight, never a bare "correct".
- Trap-family episode evaluation (`aion eval episodes`): temporal leakage,
  invented numbers, and silent-warning failures are caught mechanically.
- Surfaces: CLI, Python API, local MCP server (`aion mcp serve`), Hermes
  plugin, Docker. An experimental plan compiler/executor sits behind
  `AION_EXPERIMENTAL_PLANNER=1`.

`aion capabilities` is the machine-readable source of truth. Roadmap features
are not exposed as mocked commands.

Agents can enrich a forecast with externally fetched, future-known data without
being trusted to judge its value. Aion validates historical availability and
admits a covariate or context event only when it beats the univariate control
on identical folds. See [Covariate enrichment](docs/covariates.md).

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

Real exports are rarely that clean. Point `aion inspect` at the bundled
filthy dataset — conflicting duplicate rows, `$149`, an `N/A` outage day,
regional date formats, a trailing blank line — and it diagnoses instead of
rejecting:

```bash
aion inspect examples/filthy_requests.csv --time timestamp --target requests
# → data_quality.status: "repaired_aggressive", every needed fix listed,
#   and the exact follow-up command:
aion forecast examples/filthy_requests.csv --time timestamp --target requests \
  --frequency D --horizon 7 --repair aggressive
```

The forecast comes back `weakly_supported`, with both assumptive fixes named
in its warnings and every repair recorded in the artifact's evidence — the
cleaning is audited, not silent. And because `examples/messy_requests_revisions.csv`
carries a `published` column, `aion ingest` + `aion forecast store:… --as-of <instant>`
replays what was honestly knowable at any past moment.

## How Aion reaches a result

```text
CSV / TSV / JSON / Parquet / Excel
      │
      ▼
disclosed repair, then schema and temporal validation
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

Messy files are handled by the disclosed repair layer rather than silent
guessing or hard rejection: the default `--repair safe` normalises cell text
only (formats, currency, sentinels), `--repair aggressive` opts into
structural fixes (gap interpolation, timestamp snapping, conflict
resolution) — capped, recorded as evidence, and reflected in the support
status. `--repair off` restores strict rejection. See
[preparing data](docs/data-format.md) for supported formats, timestamp
forms, frequencies, panel rules, and history needs.

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
tools for forecasting, context, realised scoring, lifecycle, and decision
outcomes — including LLM-assisted context-event proposal run on the host's
own model — plus an `aion:forecasting` safe-use skill. Any
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
| [MCP quickstart](docs/quickstart-mcp.md) | Hook Aion to an agent and get a grounded answer in a minute |
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
| [Agent evaluation](docs/agent-evaluation.md) | Measure Hermes task uplift with and without Aion |

The [product specification](Aion_MVP_Product_Specification.md) describes the
broader product direction. The [system design](Aion_System_Design.md) defines
the intended architecture. Both include roadmap features; the capability
response and this README distinguish those from working behavior. See
`CHANGELOG.md` for what each release added and `COMPATIBILITY.md` for the
frozen surface and every amendment to it.

## Current limits

Aion remains a focused foundation, not a universal forecasting platform. Its
built-in models are deterministic classical methods (optional TSFM adapters
raise the ceiling but need their own sandbox dependencies). Context events
and covariates are evaluated separately, not jointly, in a single run. Causal
claims are never made — `investigate` stops at ranked associational
explanations by design. Realised leaderboards are observational telemetry and
never trigger automatic model switching. A `supported` result means the
current deterministic checks passed; it is not a guarantee that the future
will resemble history.

## Development

```bash
PYTHONPATH=src pytest -q
uv build
```

Contributions should preserve Aion's central boundary: agents may improve the
question and explanation, but only deterministic temporal tools may produce or
change forecast numbers.

Licensed under the [Apache License 2.0](LICENSE).
