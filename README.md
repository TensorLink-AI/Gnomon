# Aion

**A temporal execution harness for agents — forecast, investigate, detect,
decide, and monitor, under one contract: the LLM proposes; Aion validates,
computes, and owns every number.**

Aion is a local, deterministic temporal-reasoning engine for developers,
operators, and AI agents. It answers five questions over regular
time-series data:

| Verb | Question | What you get |
| --- | --- | --- |
| `aion forecast` | What happens next? | Backtested model selection, conformal intervals, threshold-crossing analysis — or a structured abstention |
| `aion investigate` | What changed? | Changepoints, regime shift vs transient, anomalies, ranked *associational* explanations (never a cause) |
| `aion detect` | What is abnormal? | Competing detectors graded on injected anomalies (or your labels); the winner flags, with every candidate's F1 disclosed |
| `aion decide` | What should we do? | Exceedance scenarios, feasibility and constraint checks, expected utility — degraded honestly when utilities are missing |
| `aion monitor` | When should we intervene? | Sequential exceedance risk and a cost-optimal alert rule |

An agent can discover data, frame the question, and explain the result.
Aion stays authoritative for timestamps, backtests, model selection,
intervals, support status, and every forecast value. The LLM never gets to
invent or edit a number.

> Aion is named for the Greek personification of enduring and cyclical time.

**`aion capabilities` is the machine-readable source of truth for what a
given build can do.** Roadmap features are never exposed as mocked
commands. If this README and `aion capabilities` disagree, the command is
right and this file is a bug.

## Hook it to your agent (60 seconds)

```bash
git clone https://github.com/TensorLink-AI/Aion && cd Aion

# Claude Code
claude mcp add aion -- uvx --from "$(pwd)" aion mcp serve

# any other MCP client: run this as a stdio server
uvx --from . aion mcp serve
```

Then ask your agent:

> Forecast `examples/messy_requests.csv` (column `requests`) 14 days ahead.
> What changed in it, and when should we alert if crossing 340 costs us 20x
> a false alarm?

The agent gets 22 tools — `aion_forecast`, `aion_investigate_change`,
`aion_detect_anomalies`, `aion_decide`, `aion_monitor`, `aion_route`, plus
ingestion, inspection, tracking, and artifact tools — and every number it
quotes comes from an evidence-linked, verified artifact. It cannot invent
values for an unsupported series; it can only report Aion's abstention and
its recovery options. See the [MCP quickstart](docs/quickstart-mcp.md) for
client configs, the vintage workflow, and the full tool surface.
(A `pip install aion-forecast` / `uvx aion-forecast` path arrives with the
PyPI release.)

## Why Aion exists

Wire an LLM agent to your operational data and it will, sooner or later:

- backtest on numbers that hadn't been published yet — and report the
  inflated accuracy as real;
- produce a confident 14-day forecast from nine data points;
- "clean" your messy CSV in its sandbox — silently, unaudited, differently
  every run;
- promote a correlation into a cause because the sentence flowed better
  that way.

Each of these reads perfectly plausibly in chat. You find out when the
capacity plan misses, the alert never fires, or the postmortem asks where a
number came from and nobody can answer.

A forecasting library doesn't fix this — the agent can misuse a library
fluently. Aion is a harness:

- **Leakage is structural, not behavioural.** Every read goes through a
  snapshot that cannot serve data published after its cutoff, and the
  artifact records exactly what was touched. `--as-of` replays any past
  moment as it was honestly knowable. Measured against an LLM control on
  40 trap tasks: the control leaked on 13 of 35 answered and transcribed
  the future verbatim on 4; Aion 0 of 40, McNemar *p* = 0.00024
  ([results](docs/leakage-trap-results-2026-08.md)).
- **Numbers are computed or absent.** Selection is backtested against
  baselines that must be beaten; every figure traces to evidence; a
  deterministic verifier rejects causal claims from associational evidence
  and uncalibrated probabilities before any response leaves the process.
- **Cleaning is disclosed, never silent.** Messy files are repaired
  deterministically, every fix listed as evidence, capped — and the support
  status downgrades to match.
- **Abstention is an answer.** When history can't support a forecast you
  get a typed refusal with recovery steps. The most dangerous forecast is
  the confident one that shouldn't exist — Aion won't produce it, and the
  agent can't fake it.
- **Predictions are remembered.** Forecasts and decisions are tracked and
  scored against realised outcomes — regret against the best feasible
  action in hindsight, not vibes.

## See it work

```bash
bash install.sh --local          # --local installs this checkout
aion tsfm install chronos_bolt_mini   # optional: adds a foundation-model candidate

aion inspect examples/daily_requests.csv \
  --time timestamp --target requests --frequency D

aion forecast examples/daily_requests.csv \
  --time timestamp --target requests \
  --horizon 3 --frequency D --output ./aion-output
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
residuals — and therefore its interval widths — are zero. That demonstrates
the pipeline, not realistic certainty. Use noisy operational history to
evaluate forecast quality for a real decision.

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

The forecast comes back `weakly_supported`, with both assumptive fixes
named in its warnings and every repair recorded in the artifact's evidence
— the cleaning is audited, not silent. And because
`examples/messy_requests_revisions.csv` carries a `published` column,
`aion ingest` + `aion forecast store:… --as-of <instant>` replays what was
honestly knowable at any past moment.

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

Earlier rolling folds select the method. The penultimate fold calibrates
the interval. The final fold reports performance without changing either
choice. The selected method is then run on all available observations to
forecast the future.

## The harness around the verbs

- **Bitemporal store** (`aion ingest`, `store:<dataset>` inputs): every
  value carries *when it became known*; `--as-of` replays any historical
  instant and the artifact proves nothing later was touched.
- **A foundation-model tier** (`aion tsfm install chronos_bolt_mini`):
  seven adapters — Chronos-Bolt (mini and small), Toto, Moment, Moirai,
  Granite TTM, and FlowState — each in its own sandboxed venv, at a pinned
  weight revision, so their conflicting dependencies never touch yours. Once
  installed they enter the same backtest folds against the same mandatory
  baselines — a TSFM wins only by out-forecasting them on your data. When
  an eligible tier is *not* installed the result says so in a note, so the
  stronger candidate is never silently absent.
- **Five-state support assessments** with typed reasons and recovery
  actions; typed lineage and a deterministic claim verifier on every
  response.
- **Messy-data repair** (`--repair off|safe|aggressive`): mixed date
  formats, currency and thousands separators, sentinel `N/A` cells,
  duplicates, gaps, jittered timestamps — every fix disclosed as evidence,
  assumptive fixes downgrade support, excessive messiness is refused.
- **Covariates and context events**, admitted only when they beat the
  univariate control on identical folds; when both are supplied a
  deterministic adjudication ladder picks the winner and records the
  comparison as evidence. See
  [Covariate enrichment](docs/covariates.md).
- **Decision tracking** with realised-outcome scoring: regret against the
  best feasible action in hindsight, never a bare "correct".
- **Trap-family episode evaluation** (`aion eval episodes`): temporal
  leakage, invented numbers, and silent-warning failures are caught
  mechanically.
- **Inputs:** CSV (any common delimiter), TSV, JSON/JSONL, gzipped text,
  Parquet (`parquet` extra), Excel (`excel` extra).
- **Surfaces:** CLI, Python API, local MCP server (`aion mcp serve`),
  Hermes plugin, Docker. An experimental plan compiler/executor sits behind
  `AION_EXPERIMENTAL_PLANNER=1`.

## Input

Aion needs a timestamp column and a numeric target. A series column is
optional:

```csv
timestamp,requests,service_id
2026-01-01T00:00:00+10:00,120,api
2026-01-02T00:00:00+10:00,128,api
2026-01-03T00:00:00+10:00,135,api
```

```bash
aion forecast observations.csv \
  --time timestamp --target requests --series service_id \
  --horizon 7 --frequency D
```

The default `--repair safe` normalises cell text only (formats, currency,
sentinels); `--repair aggressive` opts into structural fixes (gap
interpolation, timestamp snapping, conflict resolution) — capped, recorded
as evidence, and reflected in the support status. `--repair off` restores
strict rejection. See [preparing data](docs/data-format.md) for supported
formats, timestamp forms, frequencies, panel rules, and history needs.

## Output

Every completed run receives an immutable directory:

```text
aion-output/forecast_<id>/
├── artifact.json    complete task, schema, scores, support, and forecast
├── evidence.jsonl   machine-readable evaluation and support evidence
├── forecast.csv     future timestamps, point values, and quantiles
├── lineage.json     typed artifacts, evidence, and verified claims
└── summary.md       compact human-readable result
```

Start with `summary.md`. Use `forecast.csv` for charts and downstream
systems. Keep `artifact.json` when provenance, auditability, or
reproducibility matters.

## Installation

From a clone:

```bash
bash install.sh --local
aion capabilities
```

Without `--local`, `install.sh` fetches the repository's default branch
from GitHub — right for a first install from a URL, wrong when you are
testing local changes.

With uv:

```bash
uv tool install .
```

With Docker:

```bash
docker build -t aion .
docker run --rm aion capabilities
```

The direct GitHub installer, private-repository behaviour, pinned releases,
the Parquet extra, and the future PyPI command are covered in the
[installation guide](docs/installation.md).

## Aion and Hermes

- **Hermes** manages intent, permitted data discovery, orchestration, and
  explanation.
- **Aion** validates temporal data and owns every numerical result.

A packaged Hermes plugin lives in
[`integrations/hermes`](integrations/hermes/README.md): tools for
forecasting, context, realised scoring, lifecycle, and decision outcomes —
including LLM-assisted context-event proposal run on the host's own model —
plus an `aion:forecasting` safe-use skill. Any MCP-capable agent can
instead launch `aion mcp serve` and discover the same tools over stdio.
Either way the host must preserve Aion's support status and warnings and
must never manufacture values for an unsupported series. Aion itself
requires no LLM or API key.

## Documentation

| Guide | Purpose |
| --- | --- |
| [Documentation index](docs/README.md) | Everything below, plus what is and isn't built |
| [MCP quickstart](docs/quickstart-mcp.md) | Hook Aion to an agent and get a grounded answer in a minute |
| [Getting started](docs/getting-started.md) | Complete first run |
| [Installation](docs/installation.md) | Bash, uv, GitHub, Docker, and PyPI options |
| [Preparing data](docs/data-format.md) | Input schema and temporal requirements |
| [CLI reference](docs/cli-reference.md) | Commands, options, output, and exit codes |
| [Python API](docs/python-api.md) | Runtime integration from Python |
| [Results and artifacts](docs/results-and-artifacts.md) | Interpret support, scores, intervals, and files |
| [Forecasting concepts](docs/concepts.md) | Baselines, temporal evaluation, context events, and abstention |
| [Covariate enrichment](docs/covariates.md) | Let an agent propose external data without temporal leakage |
| [Troubleshooting](docs/troubleshooting.md) | Structured errors and remediation |
| [LLM integrations](docs/llm-integrations.md) | Current API-key status and the intended agent boundary |
| [Hermes plugin](integrations/hermes/README.md) | Install and operate Aion inside Hermes Agent |
| [Containers](docs/containers.md) | Local Docker and GHCR operation |
| [Development](docs/development.md) | Repository layout, tests, goldens, and contribution constraints |
| [CI/CD](docs/ci-cd.md) | Tests, PyPI trusted publishing, and releases |
| [Agent evaluation](docs/agent-evaluation.md) | Measure Hermes task uplift with and without Aion |
| [External benchmarks](benchmarks/README.md) | Runnable adapters for published time-series reasoning benchmarks (CiK, AnomLLM) with OpenRouter-served controls |

`CHANGELOG.md` records what each release added; `COMPATIBILITY.md` freezes
the v0.2 surface and every amendment to it (all v0.2 tools keep working
unchanged).

**Dated records, not descriptions of the current build:** the
[codebase review](docs/codebase-review-2026-08.md) (all 58 findings fixed),
the [integration plan review](docs/integration-plan-review-2026-08.md) (all
phases executed), the four measurement write-ups linked from the
[documentation index](docs/README.md#records-not-instructions), and the two
v0.1 direction documents —
[product specification](Aion_MVP_Product_Specification.md) and
[system design](Aion_System_Design.md) — which describe features that were
never built. Check `aion capabilities` before believing any of them.

## Relation to prior work

Two recent systems share vocabulary with this one, and one shares the name.
They solve a different half of the problem.

**AION** (Zhan et al., [arXiv:2605.25045](https://arxiv.org/abs/2605.25045))
and **TimeClaw** ([arXiv:2606.05404](https://arxiv.org/abs/2606.05404)) are
agent-side scaffolding: an LLM plans, critiques, and reviews its own
time-series reasoning, and the quality of the answer rests on the quality
of that review. The name collision with AION is coincidental and
unresolved; nothing here is derived from it.

This project is the other actor — the deterministic execution layer an
agent calls. Leakage safety is structural rather than reviewed: a snapshot
cannot serve rows published after its `as_of`, so a backtest cannot see the
future even if the agent asks it to. Model choice is decided by backtest
against mandated baselines, not by argument. The two are complementary:
better agent-side reasoning proposes better candidates; this decides which
of them survive contact with the data.

In the taxonomy of the TMLR survey [*A Survey of Reasoning and Agentic
Systems in Time Series with Large Language
Models*](https://arxiv.org/abs/2509.11575), an agent paired with Aion is a
branch-structured system spanning traditional analysis (`forecast`,
`detect`), explanation (`investigate`), and advisory decision support
(`decide`, `monitor`), with the full set of control-flow attributes. The
difference from the surveyed systems is where verification lives: there it
is typically LLM self-critique; here the verifier is deterministic code,
and the LLM is structurally unable to override it.

## Current limits

Aion remains a focused foundation, not a universal forecasting platform.
Its zero-dependency default models are deterministic classical methods; the
sandboxed TSFM tier raises the ceiling but is opt-in via
`aion tsfm install`, and a fresh install runs classical-only until then —
the result's notes disclose when that happened. There are no
transformations and no dedicated intermittent-demand methods. Seasonal
periods are detected or overridden, not learned per model. Causal claims
are never made — `investigate` stops at ranked associational explanations
by design. Realised leaderboards are observational telemetry and never
trigger automatic model switching. A `supported` result means the current
deterministic checks passed; it is not a guarantee that the future will
resemble history.

## Development

```bash
PYTHONPATH=src pytest -q
uv build
```

Contributions should preserve Aion's central boundary: agents may improve
the question and the explanation, but only deterministic temporal tools may
produce or change forecast numbers. See
[Development](docs/development.md) for the repository layout and the golden,
leakage-lint, and doc-drift suites.

Licensed under the [Apache License 2.0](LICENSE).
