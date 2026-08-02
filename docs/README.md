# Aion documentation

Aion is a temporal execution harness. An agent frames the question; Aion
validates the data, computes every number, and refuses the questions the
evidence cannot carry. It answers five: *what happens next?*
(`forecast`), *what changed?* (`investigate`), *what is abnormal?*
(`detect`), *what should we do?* (`decide`), and *when should we
intervene?* (`monitor`).

**`aion capabilities` is the machine-readable source of truth for what this
build can do.** Prefer it to any prose here, including this file.

## Start here

1. [Hook Aion to an agent](quickstart-mcp.md) — install to first grounded
   answer in a minute.
2. [Install and run the example](getting-started.md) — the same path from a
   shell.
3. [Prepare your data](data-format.md).
4. [Understand support, scores, intervals, and artifacts](results-and-artifacts.md).

## Guides and reference

| Document | Use it when |
| --- | --- |
| [Installation](installation.md) | You want Bash, uv, Docker, or GitHub installation options. |
| [Getting started](getting-started.md) | You want a complete first run. |
| [MCP quickstart](quickstart-mcp.md) | You are wiring Aion to an agent over stdio MCP. |
| [Data format](data-format.md) | You need to prepare or validate input data. |
| [CLI reference](cli-reference.md) | You need exact commands and options. |
| [Python API](python-api.md) | You want to call Aion from Python. |
| [Results and artifacts](results-and-artifacts.md) | You need to interpret or automate outputs. |
| [Concepts](concepts.md) | You want to understand selection, evaluation, intervals, context events, or abstention. |
| [Covariate enrichment](covariates.md) | You want an agent to propose external data without temporal leakage. |
| [Troubleshooting](troubleshooting.md) | A command failed or returned unsupported. |
| [LLM integrations](llm-integrations.md) | You are looking for API-key, OpenRouter, or Hermes support. |
| [Agent evaluation](agent-evaluation.md) | You want to measure whether Aion improves an agent. |
| [Development](development.md) | You want to test or contribute to Aion. |
| [Containers](containers.md) | You want to build or run the Docker image. |
| [CI/CD](ci-cd.md) | You maintain validation, publishing, or releases. |

## Implemented

**Verbs and surfaces**

- Five verbs — `forecast`, `investigate`, `detect`, `decide`, `monitor` —
  plus `route`, `inspect`, `capabilities`, and `status`.
- Three front doors: CLI, Python API, and a local stdio MCP server
  (`aion mcp serve`, 22 tools). A packaged Hermes plugin and a Docker
  image wrap the same runtime.

**Temporal core**

- Bitemporal store (`aion ingest`, `store:<dataset>` inputs): every value
  records when it became *known*, and `--as-of` replays any past instant
  with proof that nothing published later was read.
- Separated model-selection, interval-calibration, and final-test windows
  over ordered rolling origins.
- Mandatory last-value and seasonal-naive baselines that a candidate must
  beat by a configured margin.
- Drift, linear-trend, window-average, Theta, and ETS candidates, plus an
  opt-in sandboxed TSFM tier of seven pinned adapters — Chronos-Bolt (mini
  and small), Toto, Moment, Moirai, Granite TTM, FlowState
  (`aion tsfm install`).
- Split-conformal intervals at nine quantile levels from a dedicated
  calibration fold.
- Point-in-time covariates and context events, admitted only by
  identical-fold ablation, adjudicated deterministically when both are
  supplied.

**Honesty machinery**

- Per-series support assessment with typed reasons and recovery actions;
  abstention is a complete artifact, not an error.
- Typed lineage and a deterministic claim verifier that rejects causal
  claims from associational evidence.
- Disclosed messy-data repair (`--repair off|safe|aggressive`): every fix
  recorded as evidence, assumptive fixes downgrade support.
- Persistent tracking with realised-outcome scoring and regret against the
  best feasible action in hindsight.
- Structured errors with machine-readable repair options on every failure.

**Inputs:** CSV, TSV, JSON, JSONL (each optionally gzipped), Parquet
(`parquet` extra), Excel (`excel` extra). Minute (1/5/15/30), hourly,
daily, weekly, and month-start frequencies. Independent series in one
panel file.

**Experimental:** the plan compiler and executor (`aion plan`), behind
`AION_EXPERIMENTAL_PLANNER=1`.

**Not built:** hosted services, sharing, and automatic model switching.
Realised leaderboards are observational telemetry and never switch a model
on their own.

## Records, not instructions

These files are dated records of decisions and measurements. They describe
a moment, not the current build, and several deliberately record things
that were *rejected*. Read them for the reasoning behind a design; do not
read them as a description of how Aion behaves today, and do not treat
their proposals as work outstanding.

| Record | What it is |
| --- | --- |
| [Codebase review, Aug 2026](codebase-review-2026-08.md) | 58 findings against v0.4.0. **All fixed** — see the status banner. |
| [Integration plan review, Aug 2026](integration-plan-review-2026-08.md) | A four-phase plan, all phases executed, with the outcomes recorded. |
| [Leakage-trap results](leakage-trap-results-2026-08.md) | The control leaked on 13/35; Aion 0/40. McNemar p = 0.00024. |
| [Fold stride](fold-stride-measurement-2026-08.md) | Measured; default unchanged. |
| [Selection loss](selection-loss-measurement-2026-08.md) | Measured; pinball selection lost on its own metric, default off. |
| [Shrinkage admission](shrinkage-admission-measurement-2026-08.md) | Measured; evidence significantly *against*, default off. |
| [Analog pooling](analog-pooling-kill-2026-08.md) | Killed by its own criterion before being built. |
| [Persistent tracking design](persistent-tracking-design.md) | The design behind `aion track`; parts marked future are still future. |
| [Rename impact inventory](rename-impact-inventory.md) | Why the name stayed after the AION collision was noted. |

The [product specification](../Aion_MVP_Product_Specification.md) and
[system design](../Aion_System_Design.md) are v0.1 documents describing
intended direction. **Both contain features that were never built.** Treat
them as history, and check `aion capabilities` before believing either.
