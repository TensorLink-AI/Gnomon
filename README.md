# Gnomon

**Trusted time-series answers for people and AI agents.**

Give Gnomon timestamped data and a practical question. It validates the data,
tests competing methods against honest historical baselines, and returns an
answer with its uncertainty, strength of evidence, limitations, and a
reproducible artifact. If the data only supports a weak answer, Gnomon labels
it weak. If it cannot compute an honest answer, it abstains and says what is
needed next.

You do not need to choose a forecasting model or write a backtest. Gnomon can
run locally with its built-in models, optionally evaluate sandboxed time-series
foundation models, and expose the same deterministic runtime through the CLI,
Python, or MCP.

## What Gnomon does

Gnomon answers five kinds of question about a time series:

| Verb | Question | What you get |
| --- | --- | --- |
| `gnomon forecast` | What happens next? | Backtested model selection, conformal intervals, threshold-crossing analysis — or a structured abstention |
| `gnomon investigate` | What changed? | Changepoints, regime shift vs transient, anomalies, ranked *associational* explanations (never a cause) |
| `gnomon detect` | What is abnormal? | Competing detectors graded on injected anomalies (or your labels); the winner flags, with every candidate's F1 disclosed |
| `gnomon decide` | What should we do? | Exceedance scenarios, feasibility and constraint checks, expected utility — degraded honestly when utilities are missing |
| `gnomon monitor` | When should we intervene? | Sequential exceedance risk and a cost-optimal alert rule |

Its first product job is operational threshold risk:

> Which service metric may breach a meaningful limit, when, and does the
> evidence justify intervening?

The same runtime also works on demand, capacity, finance, health, sensor, and
other timestamped data. Gnomon is not an autonomous operator: it produces and
tracks evidence-backed temporal answers; a person or agent decides how those
answers are used.

## For humans and agents

| Caller | How it uses Gnomon | What Gnomon contributes |
| --- | --- | --- |
| Human operator or analyst | Runs CLI commands against local files or the bitemporal store | Data diagnosis, forecasts, change and anomaly analysis, decision support, readable summaries, and audit-ready artifacts |
| AI agent | Calls the local MCP server with the data and question | JSON-schema tools, computed numbers, support tiers, quotable headlines, provenance, and machine-readable recovery actions |
| Application | Calls the documented Python API | The same validated runtime and artifacts embedded in a larger workflow |

For a human, Gnomon replaces the fragile chain of cleaning a file, choosing a
model, inventing a backtest, and explaining the result by hand. For an agent,
it creates a hard boundary: the model may frame the question and explain the
answer, but Gnomon owns timestamps, evaluation, model selection, intervals,
support status, and every published value. The LLM cannot silently edit or
invent those numbers.

Depending on the question, a governed result includes:

- a deterministic headline and the key numbers;
- a support tier that says how much confidence the evidence earned;
- disclosed repairs, assumptions, limitations, and recovery actions;
- the evaluation evidence that justified publication; and
- an integrity-sealed artifact that can be inspected, replayed, and scored later.

## Where Gnomon sits

```text
agent / operator
      |  intent, permitted data access, explanation
      v
Gnomon
      |  point-in-time data, evaluation, executable candidate,
      |  support tier, deterministic headline, integrity-sealed artifact
      v
built-in models or explicitly configured model backends
```

Gnomon owns the middle boundary: what data was knowable, which executable
earned publication, every published number, and what the evidence permits the
caller to say. A hosted router, benchmark service, or model-training network
may supply better candidates later; none is required by this repository and
none may bypass that contract.

**`gnomon capabilities` is the machine-readable source of truth for what a
given build can do.** Roadmap features are never exposed as mocked
commands. If this README and `gnomon capabilities` disagree, the command is
right and this file is a bug.

## Use it directly

```bash
bash install.sh --local
gnomon forecast examples/daily_requests.csv --horizon 3
```

The CLI prints a readable answer and writes the complete evidence-linked
artifact to `gnomon-output/`. Start with `summary.md`; use `forecast.csv` in
downstream systems; keep `artifact.json` and `lineage.json` for replay and
audit. See [the full CLI workflow](#see-the-full-cli-workflow) or the
[getting-started guide](docs/getting-started.md).

## Hook it to your agent (60 seconds)

MCP is Gnomon's only agent-facing contract. The local server exposes the same
runtime as the CLI without shell quoting, with JSON Schemas the host can
validate against and structured errors carrying machine-readable repair
options.

```bash
git clone https://github.com/TensorLink-AI/Gnomon && cd Gnomon

# Claude Code
claude mcp add gnomon -- uvx --from "$(pwd)" gnomon mcp serve

# any other MCP client: run this as a stdio server
uvx --from . gnomon mcp serve
```

Then ask your agent one complete operational question:

> Forecast `examples/messy_requests.csv` (column `requests`) 14 days ahead.
> What changed in it, and when should we alert if crossing 340 costs us 20x
> a false alarm?

The agent gets 10 tools on the default `core` profile: capabilities and
inspection, `describe`, evaluated `forecast`, anomaly detection, change
investigation, monitoring, decision, routing, and run explanation. Tracking,
ingestion, scenario-selection, and administrative tools remain available
through explicit profiles, and every number it
quotes comes from an evidence-linked, verified artifact. It cannot invent
values for an unsupported series; it can only report Gnomon's abstention and
its recovery options. Data-reading calls return a session-scoped `data_ref`,
so follow-up verbs reuse the resolved data and schema without resending the
observations. Context-aware calls return a persistent, project-scoped
`context_ref`; later calls reuse the validated interpretation while Gnomon
rechecks knowledge timing and numerical admission. See the
[MCP quickstart](docs/quickstart-mcp.md) for
client configs, the vintage workflow, and the full tool surface.
(A `pip install gnomon-forecast` / `uvx gnomon-forecast` path arrives with the
PyPI release.)

Conversation cost is an engineering constraint, not a completed claim. Wide
data is handled in one batched call; brief responses keep disclosures while
moving bulk rows to integrity-sealed artifacts; repeated calls can use a session
`data_ref` instead of resending observations. The compact three-tool
`evidence` profile remains available for bounded evidence sessions; `full`
remains explicit opt-in for administration and deep audit. We do not publish
workflow accuracy or token claims until the
underlying complete run and its provenance are retained as citable evidence.

## Measured evidence

On the latest complete matched 80-case TemporalBench sample, the same DeepSeek
model answered **28.3%** of 240 temporal-choice fields correctly with Gnomon's
Evidence surface, versus **30.4%** directly. Gnomon fixed 52 fields and broke
57; the difference was not significant (two-sided exact McNemar *p* = 0.702).
The earlier positive choice results did not replicate and are retired as
product claims.

Forecast superiority was also not established. Mean sMAPE was 10.64 with
Gnomon and 11.83 directly (47 row wins, 32 losses, one tie, *p* = 0.115).
Across 478 mutually scoreable channels, Gnomon won 195, lost 179, and tied 104
on MASE (*p* = 0.438); heart-rate MASE improved significantly in isolation,
but the overall channel result did not. The median channel MASE was 1.089 for
Gnomon versus 1.106 directly. This remains primarily a safety result rather
than an estimator improvement: 476/480 published channels matched the robust
last-value path, four were worse, and none improved on it. All 80 tasks
completed; Evidence preserved 240/240 typed answers and 80/80 immutable
primary forecasts. The exact-head protocol, paired reports, and limitations
are in the [v0.6 external-validation release](results/benchmark-releases/2026-08-30-v06-external-validation/README.md).

On a fresh four-task, one-seed Context-is-Key sensor slice, Gnomon's mean RCRPS
was 0.1013 versus 0.1338 directly, but it won only one task and lost three;
the lower mean was driven by one large win and was not significant (*p* =
0.625). This narrow result is not a CiK superiority claim.

## Why this needs an execution layer

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
fluently. Gnomon is a harness:

- **Leakage is structural, not behavioural.** Every read goes through a
  snapshot that cannot serve data published after its cutoff, and the
  artifact records exactly what was touched. `--as-of` replays any past
  moment as it was honestly knowable. Measured against an LLM control on
  40 trap tasks: the control leaked on 13 of 35 answered and transcribed
  the future verbatim on 4; Gnomon 0 of 40, McNemar *p* = 0.00024
  ([results](docs/leakage-trap-results-2026-08.md)).
- **Numbers are computed or absent.** Selection is backtested against
  baselines that must be beaten; every figure traces to evidence; a
  deterministic verifier rejects causal claims from associational evidence
  and uncalibrated probabilities before any response leaves the process.
- **The evaluated executable publishes.** Evaluation returns an
  identity-carrying candidate specification—not merely a model name. It is
  independently fit at each permitted origin, and its final fit produces the
  published path. Strategy, member set, behavior-changing configuration,
  revisions, fallback policy, fitted weights, and visible-data fingerprint
  travel with composite and TSFM results.
- **Cleaning is disclosed, never silent.** Messy files are repaired
  deterministically, every fix listed as evidence, capped — and the support
  status downgrades to match.
- **Every answer is graded; abstention remains an answer.** The default
  publishes the most defensible result that exists, at whatever tier the
  evidence earned — a fully evaluated forecast, an evaluated prefix with
  a labelled naive remainder, or a naive extrapolation alone — with an
  unstrippable per-row tier, a typed reason for every rung walked down,
  and a one-sentence headline naming the weakest tier present. The
  dangerous forecast is the unlabelled one — Gnomon won't produce it,
  and the verifier rejects any response quoting a sub-supported number
  without its tier. Operators who want refusals set
  `minimum_support: supported`; a series where nothing is computable
  still gets the typed abstention with recovery steps.
- **Predictions are remembered.** Forecasts and decisions are tracked and
  scored against realised outcomes — regret against the best feasible
  action in hindsight, not vibes.

## See the full CLI workflow

```bash
bash install.sh --local          # --local installs this checkout
gnomon tsfm install toto2_4m            # optional: smallest foundation-model candidate

gnomon forecast examples/daily_requests.csv --horizon 3
```

That is the whole invocation: `--time`, `--target`, and `--frequency` are
inferred when the file leaves no choice (and each inference is disclosed
in the response's assumptions — omit `--horizon` too and it defaults to
one seasonal period). Inference refuses, naming the candidate columns,
whenever more than one column qualifies; the explicit flags always work:

```bash
gnomon inspect examples/daily_requests.csv \
  --time timestamp --target requests --frequency D

gnomon forecast examples/daily_requests.csv \
  --time timestamp --target requests \
  --horizon 3 --frequency D --output ./gnomon-output
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

Real exports are rarely that clean. Point `gnomon inspect` at the bundled
filthy dataset — conflicting duplicate rows, `$149`, an `N/A` outage day,
regional date formats, a trailing blank line — and it diagnoses instead of
rejecting:

```bash
gnomon inspect examples/filthy_requests.csv --time timestamp --target requests
# → data_quality.status: "repaired_aggressive", every needed fix listed,
#   and the exact follow-up command:
gnomon forecast examples/filthy_requests.csv --time timestamp --target requests \
  --frequency D --horizon 7 --repair aggressive
```

The forecast comes back `weakly_supported`, with both assumptive fixes
named in its warnings and every repair recorded in the artifact's evidence
— the cleaning is audited, not silent. And because
`examples/messy_requests_revisions.csv` carries a `published` column,
`gnomon ingest` + `gnomon forecast store:… --as-of <instant>` replays what was
honestly knowable at any past moment.

## How Gnomon reaches a result

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
calibration fold (+ selection residuals by default) ── estimate quantiles
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

Earlier rolling folds select the method. By default their residuals are pooled
with the penultimate calibration fold for interval sample size; this is
explicitly disclosed and is not strict split conformal. Set
`evaluation.uncertainty.pool_residuals = false` for held-out calibration only.
The final fold reports performance without changing the model choice or
calibration rule. The winning candidate specification is then fit on all visible
observations and that fitted executable forecasts the future. Publication
never rebuilds the winner from a model name.

## The harness around the verbs

- **Bitemporal store** (`gnomon ingest`, `store:<dataset>` inputs): every
  value carries *when it became known*; `--as-of` replays any historical
  instant and the artifact proves nothing later was touched.
- **A foundation-model tier** (`gnomon tsfm install toto2_4m`):
  eight adapters — Chronos-Bolt (mini and small), Toto 2.0 (4M and 22M),
  Moment, Moirai,
  Granite TTM, and FlowState — each in its own sandboxed venv, at a pinned
  weight revision, so their conflicting dependencies never touch yours. Once
  installed they enter the same backtest folds against the same mandatory
  baselines. That local contest is the default. For histories too short to
  rank candidates, an opt-in evidence-weighted policy can use a versioned
  held-out registry as a TSFM transfer prior, publishing either an explicitly
  externally validated model or an immutable shrinkage blend. External
  evidence is never called local validation, and without a qualifying pinned-
  revision registry the robust baseline still publishes. When
  an eligible tier is *not* installed the result says so in a note, so the
  stronger candidate is never silently absent.
  Local sandboxes are the default trust path. A CLI/Python project may opt
  into a configured TSFM API endpoint; that network path is off by default,
  explicit in `gnomon.toml`, and never inherited by MCP tool calls.
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
  comparison as evidence. Novel grounded events can additionally yield a
  separately labelled, non-probabilistic sensitivity path without changing
  the governed primary forecast. Every conditional path carries a typed
  effect distribution and provenance class; with `--project`, Gnomon freezes
  the primary/scenario pair and learns the realised onset, duration and
  magnitude when actuals arrive (`gnomon track effects --project NAME`). The
  registry does not infer occurrence from correlation: a dated event
  confirmation is required before an estimate becomes eligible for learning.
  If context changes the selected output, the artifact retains the immutable
  history-only path as `primary_forecast` and labels `forecast` as a
  context-conditioned projection.
  See [Covariate enrichment](docs/covariates.md).
- **Organizational effect learning**, gated by confirmed outcomes,
  leave-one-event-out validation and posterior-predictive calibration; robust
  decisions can compare the primary with credible context scenarios without
  inventing scenario probabilities. Agents can resolve evidence through
  `gnomon_status(section="effect_prior")`; the experimental unified surface
  accepts `question.kind="robust_decision"`. See
  [EffectBench](benchmarks/effectbench/README.md).
- **Canonical temporal profiles** separate trend persistence, seasonal
  stability, residual observation volatility, forecast-horizon marginal
  variability, expected-path movement, interval uncertainty, regimes,
  extremes, and dependence. A smooth point path is never presented as proof
  of low observation volatility.
- **Fitted temporal-property executables** answer typed questions about level,
  trend, seasonal continuation, residual volatility, regime shifts, extremes,
  and paired dependence. Every predictive executable is selected at rolling
  origins, carries its candidate identity and calibrated support, and returns
  a weak best estimate when categorical evidence misses the publication gate.
  Weak answers are explicitly ineligible for automation; abstention is reserved
  for cases with no meaningful computable estimate. These are additive receipts:
  they cannot rewrite the published forecast. `gnomon_forecast`,
  `gnomon_decide`, and `gnomon_monitor` share the same compact answer contract.
- **An integration-first temporal capability registry** routes each typed
  question and resolved dataset shape into one execution plan. It includes
  exact ADF(0)/KPSS-level stationarity tests, explicit-period additive
  decomposition, and ridge-linear exogenous regression with expanding-window
  validation. Unsupported methods terminate once; Gnomon never relabels a
  forecast, anomaly detector, or generic seasonal scan as the requested test.
- **Decision tracking** with realised-outcome scoring: regret against the
  best feasible action in hindsight, never a bare "correct".
- **Temporal-answer tracking** joins immutable question receipts to the
  realised horizon when actuals arrive; `gnomon_status` reports open and
  resolved answer counts alongside forecasts and decisions.
- **Trap-family episode evaluation** (`gnomon eval episodes`): temporal
  leakage, invented numbers, and silent-warning failures are caught
  mechanically.
- **Inputs:** CSV (any common delimiter), TSV, JSON/JSONL, gzipped text,
  Parquet (`parquet` extra), Excel (`excel` extra).
- **Surfaces:** CLI, Python API, and the local MCP server (`gnomon mcp
  serve`). Docker packages the CLI; it is not a separate contract surface.

## Input

Gnomon needs a timestamp column and a numeric target. A series column is
optional:

```csv
timestamp,requests,service_id
2026-01-01T00:00:00+10:00,120,api
2026-01-02T00:00:00+10:00,128,api
2026-01-03T00:00:00+10:00,135,api
```

```bash
gnomon forecast observations.csv \
  --time timestamp --target requests --series service_id \
  --horizon 7 --frequency D
```

Wide files with several numeric columns batch into one run — one shared
load pass, channels evaluated concurrently, one artifact with a result
(and its own support state) per column:

```bash
gnomon forecast vitals.csv --target hr,spo2,resp --horizon 14
gnomon forecast vitals.csv --target auto --horizon 14   # every numeric column
```

Add `--brief` for compact stdout: the q50 path with one q10–q90 interval,
plus every warning, abstention reason, and disclosure verbatim — the full
artifact on disk is unchanged.

The default `--repair safe` normalises cell text and bounded scheduler/scrape
jitter (1% of cadence, capped at 60 seconds). Alignment is phase-aware, never
merges observations, and reports its count and maximum displacement.
`--repair aggressive` additionally opts into value-changing structural fixes
(gap interpolation and conflict resolution) — capped, recorded as evidence,
and reflected in the support status. Bounded alignment does not consume the
invented-value ceiling. `--repair off` restores strict rejection. See
[preparing data](docs/data-format.md) for supported formats, timestamp forms,
frequencies, panel rules, and history needs.

## Output

Every completed run receives a content-addressed, integrity-sealed directory:

```text
gnomon-output/forecast_<id>/
├── artifact.json    complete task, schema, scores, support, and forecast
├── evidence.jsonl   machine-readable evaluation and support evidence
├── forecast.csv     future timestamps, point values, and quantiles
├── lineage.json     typed artifacts, evidence, and verified claims
├── report.html      self-contained offline visual and disclosures
├── summary.md       compact human-readable result
└── integrity.json   SHA-256 digest of every output above
```

Start with `report.html` or `summary.md`. Use `forecast.csv` for downstream
systems. Keep `artifact.json` when provenance, auditability, or
reproducibility matters.
Gnomon verifies `integrity.json` whenever it reads a sealed artifact. Older
unsealed artifacts remain readable for compatibility and are not represented
as tamper-evident.

## Installation

From a clone:

```bash
bash install.sh --local
gnomon capabilities
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
docker build -t gnomon .
docker run --rm gnomon capabilities
```

The direct GitHub installer, private-repository behaviour, pinned releases,
the Parquet extra, and the future PyPI command are covered in the
[installation guide](docs/installation.md).

## Configuration

The canonical project configuration is `gnomon.toml`, parsed by Python
3.11's standard library, so a base installation cannot silently ignore it.
Start from [`gnomon.toml.example`](gnomon.toml.example). Unknown and
recognized-but-inert keys fail before data execution.

`gnomon.yaml` remains a transitional compatibility format. If a YAML file is
present without PyYAML, Gnomon fails with a migration instruction; it never
falls back silently to defaults. If TOML and YAML are both present, Gnomon
refuses the ambiguity. MCP calls deliberately take behavior-changing options
as explicit tool arguments rather than reading ambient project config.

## Documentation

| Guide | Purpose |
| --- | --- |
| [Documentation index](docs/README.md) | Everything below, plus what is and isn't built |
| [Product position](docs/product-position.md) | Promise, first buyer, deployed boundary, and claims discipline |
| [MCP quickstart](docs/quickstart-mcp.md) | Hook Gnomon to an agent and get a grounded answer in a minute |
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
| [Containers](docs/containers.md) | Local Docker and GHCR operation |
| [Development](docs/development.md) | Repository layout, tests, goldens, and contribution constraints |
| [CI/CD](docs/ci-cd.md) | Tests, PyPI trusted publishing, and releases |
| [Agent evaluation](docs/agent-evaluation.md) | Measure agent task uplift with and without Gnomon |
| [External benchmarks](benchmarks/README.md) | Runnable adapters and citable releases for TemporalBench, CiK, AnomLLM, MTBench, and TimeSage-MT with matched-provider controls |

`CHANGELOG.md` records what each release added; `COMPATIBILITY.md` states the
current compatibility policy and retired entry points.

**Dated records, not descriptions of the current build:** the
[codebase review](docs/codebase-review-2026-08.md) (all 58 findings fixed),
the [integration plan review](docs/integration-plan-review-2026-08.md) (all
phases executed), the four measurement write-ups linked from the
[documentation index](docs/README.md#records-not-instructions), and the two
v0.1 direction documents —
[product specification](Gnomon_MVP_Product_Specification.md) and
[system design](Gnomon_System_Design.md) — which describe features that were
never built. Check `gnomon capabilities` before believing any of them.

## The name

A gnomon is the shadow-casting rod on a sundial. It computes nothing and
holds no model of the day — it casts a shadow, and the calibrated dial
turns that shadow into a time. Two properties carried the name over:
a sundial reads nothing at night rather than inventing a plausible hour,
and its honesty is structural rather than reviewed. Misalign a gnomon from
the celestial pole and it will tell you the wrong time confidently and
forever, with no internal signal that anything is wrong. Both are things
this runtime is built to get right: abstention is a complete answer, and
the `as_of` guarantee is enforced by the snapshot rather than by an agent
checking its own work.

## Relation to prior work

Two recent systems share vocabulary with this one. They solve a different
half of the problem.

**AION** (Zhan et al., [arXiv:2605.25045](https://arxiv.org/abs/2605.25045))
and **TimeClaw** ([arXiv:2606.05404](https://arxiv.org/abs/2606.05404)) are
agent-side scaffolding: an LLM plans, critiques, and reviews its own
time-series reasoning, and the quality of the answer rests on the quality
of that review. This project was called *Aion* through v0.4.0 and was
renamed in v0.5.0 to end that collision; nothing here is derived from
their work.

This project is the other actor — the deterministic execution layer an
agent calls. Leakage safety is structural rather than reviewed: a snapshot
cannot serve rows published after its `as_of`, so a backtest cannot see the
future even if the agent asks it to. Model choice is decided by backtest
against mandated baselines, not by argument. The two are complementary:
better agent-side reasoning proposes better candidates; this decides which
of them survive contact with the data.

In the taxonomy of the TMLR survey [*A Survey of Reasoning and Agentic
Systems in Time Series with Large Language
Models*](https://arxiv.org/abs/2509.11575), an agent paired with Gnomon is a
branch-structured system spanning traditional analysis (`forecast`,
`detect`), explanation (`investigate`), and advisory decision support
(`decide`, `monitor`), with the full set of control-flow attributes. The
difference from the surveyed systems is where verification lives: there it
is typically LLM self-critique; here the verifier is deterministic code,
and the LLM is structurally unable to override it.

The design tracks measured findings in that literature rather than
intuition. That general-purpose LLMs mishandle numerical temporal
reasoning is an empirical result, not an assumption
([*Language Models Still Struggle to Zero-shot Reason about Time
Series*](https://aclanthology.org/2024.findings-emnlp.201/), EMNLP 2024).
That naive methods are hard to beat — and therefore worth mandating as
baselines — is the point of [*Are Language Models Actually Useful for
Time Series Forecasting?*](https://openreview.net/forum?id=DV15UbHCY1)
(NeurIPS 2024) and the [context-parroting
baseline](https://arxiv.org/abs/2505.11349). That evaluation-time
future leakage silently inflates results is the warning of [*Time
Travel is Cheating*](https://arxiv.org/abs/2505.11065). Gnomon's answer
to each is structural rather than behavioural: baselines that cannot be
configured out, an `as_of` enforced by the snapshot, and a verifier the
model cannot argue with. And rather than inventing its own yardsticks,
`benchmarks/` ships runnable adapters for the field's published tasks —
[Context is Key](https://arxiv.org/abs/2410.18959) (ICML 2025) for
context-aware forecasting and
[AnomLLM](https://github.com/rose-stl-lab/AnomLLM) for anomaly
reasoning — so its claims can be scored on the literature's own terms.

## Current limits

Gnomon remains a focused foundation, not a universal forecasting platform.
Its zero-dependency default models are deterministic classical methods; the
sandboxed TSFM tier raises the ceiling but is opt-in via
`gnomon tsfm install`, and a fresh install runs classical-only until then —
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

Contributions should preserve Gnomon's central boundary: agents may improve
the question and the explanation, but only deterministic temporal tools may
produce or change forecast numbers. See
[Development](docs/development.md) for the repository layout and the golden,
leakage-lint, and doc-drift suites.

Licensed under the [Apache License 2.0](LICENSE).
