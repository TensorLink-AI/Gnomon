# Aion System Design

**A deployable forecasting capability by Cascade**

**Version:** 0.1  
**Date:** 28 July 2026  
**Status:** Historical design document — superseded in parts by the shipped runtime

> **Read this as intent, not as description.** It was written before
> implementation, at v0.1; the runtime is now 0.4.0 and its architecture
> has moved on — most visibly, the harness now has five verbs, a bitemporal
> store with `--as-of` replay, typed lineage with a deterministic claim
> verifier, and a sandboxed TSFM tier, none of which are specified here as
> built.
>
> Interfaces named here may not exist, and existing interfaces may differ.
> **`aion capabilities` is the source of truth**; the
> [CLI reference](docs/cli-reference.md) and
> [documentation index](docs/README.md) describe the current surface.

> **Architecture decision:** Aion uses a deterministic temporal runtime as the source of truth. Hermes Agent or an optional Aion orchestrator may formulate tasks, inspect permitted sources, propose context, select bounded experiments, and interpret results. No LLM is allowed to generate, edit, or override forecast values, evaluation metrics, uncertainty, selection decisions, or abstention.

## Purpose of this document

This document specifies the architecture, component boundaries, agent/runtime responsibilities, CLI and MCP interfaces, public contracts, forecast lifecycle, context-admission rules, evidence model, deployment modes, operational requirements, implementation phases, and release-blocking acceptance criteria.

The companion **Aion MVP Product Specification** defines the audience, positioning, product scope, launch strategy, validation plan, and success metrics.

## System goals

1. Deliver a correct and reproducible one-command forecasting path.
2. Allow Hermes and other agents to operate Aion through typed local tools.
3. Improve task formulation and interpretation through LLM reasoning without allowing the LLM to own numerical conclusions.
4. Preserve frequency, timezone, missing timestamps, cutoffs, context availability, and future timestamp geometry.
5. Compare all candidates against mandatory baselines using timestamp-aware rolling evaluation.
6. Select per series or abstain when evidence is inadequate.
7. Produce explicit uncertainty methods, measured coverage, evidence-linked claims, and realised scoring.
8. Keep the runtime independently deployable through CLI, MCP, Python, and later HTTP adapters.

## Non-functional goals

| Goal | Requirement |
| --- | --- |
| Local-first | A useful forecast must be possible without an account or hosted dependency. |
| Agent-safe | Typed schemas, stable errors, capability discovery, budgets, and immutable numerical artifacts. |
| Reproducible | Data fingerprints, cutoff snapshots, model versions, configuration, seeds where applicable, and reproducibility commands. |
| Honest failure | Unsupported tasks and insufficient evidence must return structured abstention or validation errors. |
| Extensible | New models, sources, and surfaces plug into stable protocols without modifying the temporal core. |
| Observable | Each stage records timings, failures, warnings, provenance, and evidence. |
| Privacy-preserving | No implicit uploads; source files are never silently modified or transferred. |

## Agentic reasoning design

> **Agent boundary:** The LLM may improve the question, inputs, experiments, and explanation. It may not improve a forecast by editing its numbers. Any claimed improvement must be demonstrated by the deterministic evaluation layer.

### Logical orchestrator

Implement one logical orchestrator with specialist steps rather than several agents chatting with each other. In Hermes mode, Hermes is the orchestrator and Aion exposes specialist tools. In standalone mode, a small Aion orchestrator invokes the same tool contracts.

| Reasoning step | Responsibility |
| --- | --- |
| Task formulation | Infer decision, target, horizon, frequency, quantiles, threshold, cost asymmetry, and missing material fields. |
| Workspace discovery | Search only explicitly allowed files and sources; rank candidate data and context. |
| Schema mapping | Propose column mappings and policies, with reasons and confidence. |
| Context proposal | Convert text, calendars, release notes, and records into typed events with effective time, known-at time, scope, and source. |
| Experiment proposal | Suggest bounded experiments with a hypothesis, operation, cost estimate, and stop condition. |
| Interpretation | Turn evidence into a user-facing conclusion without altering the artifact or hiding limitations. |

### Permission model

| Permission | Rule |
| --- | --- |
| Read | Dataset files, configuration, approved planning notes, calendar exports, and prior Aion artifacts. |
| Propose | Task fields, column mappings, missingness policy, candidate context, and bounded experiments. |
| Execute | Only typed Aion operations allowed by the host and budget. |
| Never | Modify source files, silently upload data, override validation, edit forecast values, or suppress abstention. |

### Agent budget

```json
{
  "maximum_tool_calls": 20,
  "maximum_context_candidates": 10,
  "maximum_context_bundles": 6,
  "maximum_backtest_experiments": 5,
  "maximum_follow_up_experiments": 3,
  "maximum_runtime_seconds": 300,
  "maximum_llm_tokens": 16000
}
```

The loop stops when the budget is exhausted, the decision is sufficiently supported, all remaining experiments have low expected information value, or the runtime must abstain.

## System architecture

### Architecture overview

```text
Human or external application
            │
            ▼
Hermes Agent / Aion optional orchestrator
  intent • file discovery • context proposals • interpretation
            │ typed MCP / CLI / Python calls
            ▼
Aion interface layer
  CLI • MCP server • Python API • future HTTP adapter
            │
            ▼
Temporal task compiler and policy gate
            │
            ▼
Deterministic snapshot and temporal normalisation
            │
     ┌──────┼───────────┐
     ▼      ▼           ▼
 diagnostics  context compiler  model eligibility
     │      │           │
     └──────┼───────────┘
            ▼
Forecast experiment runtime
  baselines • statistical model • TSFM adapter
            │
            ▼
Rolling evaluation • selection • uncertainty • support
            │
            ▼
Evidence ledger • forecast artifact • presentation outputs
```

### Component boundaries

| Module | Responsibility |
| --- | --- |
| aion.contracts | Versioned Pydantic models and generated JSON Schema for every public object. |
| aion.temporal | Frequency inference, timezone handling, regularisation, snapshots, and timestamp-aware folds. |
| aion.diagnostics | Missingness, duplicates, history sufficiency, seasonality, shifts, intermittency, leakage, and validity checks. |
| aion.models | Strict adapter protocol, model registry, baselines, statistical adapter, and TSFM adapter. |
| aion.evaluation | Rolling-origin evaluation, metrics, comparisons, context ablation, and per-series selection. |
| aion.calibration | Residual quantiles, coverage measurement, and later conformal methods. |
| aion.support | Deterministic support assessment and abstention reasons. |
| aion.context | Typed events, availability validation, covariate compilation, and context-gain evaluation. |
| aion.evidence | Append-only evidence records and evidence-linked claims. |
| aion.runtime | Typed state machine, budgets, idempotency, failure isolation, and artifact assembly. |
| aion.presentation | Human summary, table, CSV, JSON, JSONL, Markdown, and chart outputs. |
| aion.agent | Optional task compiler, investigator, planner, and interpreter using the public tool contracts. |
| aion.integrations.hermes | Hermes skill/plugin, MCP configuration, examples, and safe-use instructions. |

### Canonical layering rule

The runtime library is canonical. The CLI, MCP server, standalone agent, and future HTTP service are adapters. No business logic may exist only inside a command handler or LLM prompt.

## Interfaces and contracts

### Public CLI

| Command | Purpose |
| --- | --- |
| aion forecast INPUT | One-shot data-to-forecast workflow. |
| aion inspect INPUT | Diagnose mappings, frequency, quality, and eligible tasks without forecasting. |
| aion init | Create a persistent forecast project. |
| aion run | Execute the configured project. |
| aion explain FORECAST_ID | Render a deterministic or LLM-assisted evidence-linked explanation. |
| aion actuals submit FILE | Attach realised outcomes. |
| aion score FORECAST_ID | Calculate realised performance. |
| aion compare A B | Compare forecast versions or configurations. |
| aion share FORECAST_ID | Explicitly publish or export a shareable report. |
| aion mcp serve | Expose typed local tools to Hermes and other MCP clients. |
| aion capabilities | Return actual supported tasks, models, schemas, inputs, outputs, and limitations. |

### MCP tool surface

| Tool | Purpose |
| --- | --- |
| aion_inspect_dataset | Inspect a file or registered dataset and return mapping candidates, diagnostics, and blockers. |
| aion_compile_task | Compile a typed task from user intent and dataset metadata. |
| aion_validate_task | Validate task, capabilities, and execution blockers. |
| aion_create_forecast | Run the deterministic forecast workflow. |
| aion_evaluate_context | Compare approved context bundles against history-only variants. |
| aion_explain_forecast | Return evidence-linked interpretation payloads. |
| aion_compare_forecasts | Compare runs, model choices, metrics, and support changes. |
| aion_submit_actuals | Attach actual outcomes to an existing forecast. |
| aion_score_forecast | Calculate realised metrics and update the performance record. |

### Core task contract

```json
{
  "schema_version": "0.1",
  "task_id": "task_01K...",
  "dataset_id": "dataset_01K...",
  "target": "requests",
  "series_selection": {"type": "all"},
  "cutoff_time": "2026-07-28T00:00:00+10:00",
  "frequency": "h",
  "horizon": 168,
  "required_quantiles": [0.1, 0.5, 0.9],
  "decision": {
    "type": "capacity_threshold",
    "threshold": 4500000,
    "underforecasting_cost": "high"
  },
  "context_bundle_ids": [],
  "execution_policy": {
    "selection": "per_series",
    "minimum_valid_folds": 2,
    "minimum_baseline_improvement": 0.02
  }
}
```

### Forecast artifact

| Section | Contents |
| --- | --- |
| identity | forecast_id, task_id, schema_version, timestamps, status |
| snapshot | data version, cutoff, included/excluded rows, transformations |
| diagnostics | frequency, duplicates, missingness, shifts, seasonality, leakage findings |
| runs | per-series baseline and candidate model outputs, failures, latency, provenance |
| evaluation | per-fold/per-series metrics and aggregate summaries |
| selection | selected model per series, baseline improvement, policy and reasons |
| uncertainty | method, requested quantiles, residual population, target and measured coverage |
| support | supported, weakly_supported, or unsupported; reasons and checks |
| context | candidate bundles, temporal validation, ablation results, retained/rejected context |
| evidence | append-only evidence records and evidence-linked claims |
| outputs | forecast rows, threshold events, warnings, human summary, and reproducibility command |

### Structured errors

```json
{
  "schema_version": "0.1",
  "status": "error",
  "error": {
    "code": "INSUFFICIENT_HISTORY",
    "message": "Series api-prod has 31 periods but requires 56.",
    "retryable": false,
    "details": {"available_periods": 31, "required_periods": 56},
    "suggested_actions": [
      {"operation": "reduce_horizon", "horizon": 7},
      {"operation": "provide_more_history", "additional_periods": 25},
      {"operation": "use_unevaluated_baseline", "requires_confirmation": true}
    ]
  }
}
```

## Forecast execution workflow

| Stage | Behaviour |
| --- | --- |
| 1. Resolve request | Resolve input path/source, target question, horizon, threshold, and output requirements. |
| 2. Ingest and fingerprint | Read CSV/Parquet, generate dataset ID and content fingerprint, preserve source without mutation. |
| 3. Map and validate schema | Resolve timestamp, target, optional series and covariates; return ambiguity for agent/user resolution. |
| 4. Infer time grid | Infer or validate frequency and timezone; require panel consensus; reject ambiguity or unsupported frequency. |
| 5. Regularise | Sort, detect duplicates, construct canonical grids, apply configured missingness and aggregation policies, record transformations. |
| 6. Create cutoff snapshot | Exclude observations and context unavailable at the task cutoff. |
| 7. Run diagnostics | Measure timestamp consistency, history sufficiency, missingness, intermittency, seasonality, shifts, outliers, and leakage risks. |
| 8. Compile approved context | Convert typed events into horizon-aligned, series-scoped covariates; preserve availability metadata. |
| 9. Determine model eligibility | Filter adapters using frequency, context length, horizon, missing-data, panel, and covariate capabilities. |
| 10. Run baselines | Always run last-value and seasonal-naive where valid. |
| 11. Run candidates | Run one statistical model and one TSFM adapter; isolate failures. |
| 12. Backtest | Use timestamp-aware rolling folds and reconstruct context at each historical cutoff. |
| 13. Evaluate context | Compare history-only and approved context bundles on identical folds. |
| 14. Select per series | Require valid folds and minimum improvement over the strongest baseline; otherwise retain baseline or abstain. |
| 15. Calibrate uncertainty | Use native quantiles when trustworthy or rolling residual quantiles; measure held-out coverage. |
| 16. Assess support | Return supported, weakly_supported, or unsupported from deterministic checks. |
| 17. Assemble artifact | Persist forecast rows, evidence, claims, provenance, warnings, and reproducibility metadata atomically. |
| 18. Interpret | Render deterministic summary or allow an LLM to explain only evidence-backed material statements. |

### Initial frequency support

| Code | Frequency | Candidate seasonal periods |
| --- | --- | --- |
| h | Hourly | 24 and 168 where sufficient history exists |
| D | Daily | 7 and optionally 365 |
| W | Weekly | 52 |
| MS | Month-start | 12 |

### Selection and support policy

| Level | Definition |
| --- | --- |
| supported | Enough valid folds; selected forecast materially beats or appropriately retains a baseline; acceptable stability; uncertainty evidence available; no hard blockers. |
| weakly_supported | Forecast executes, but evidence is limited or material risks exist: few folds, high fold variance, recent shift, large missingness, or poor interval coverage. |
| unsupported | Invalid/ambiguous frequency, no valid folds, insufficient history, no successful baseline, incompatible horizon, invalid values/timestamps, or required context unavailable. |

## Context, interpretation, and evidence

### Context event contract

```json
{
  "event_id": "event_01K...",
  "event_type": "customer_launch",
  "entity_scope": ["api-prod"],
  "effective_start": "2026-08-14T00:00:00+10:00",
  "effective_end": "2026-08-20T23:59:59+10:00",
  "known_at": "2026-07-22T09:30:00+10:00",
  "attributes": {"expected_accounts": 120},
  "status": "confirmed",
  "confidence": 1.0,
  "source": {"type": "planning_file", "reference": "launches.md#enterprise-a"},
  "created_by": {"type": "hermes_agent"}
}
```

### Admission rule

- The event was known at each historical fold cutoff.
- The event is correctly scoped to the affected series and timestamps.
- The context model improves the primary metric by the configured minimum margin.
- More than half of valid folds improve.
- The gain is not confined to one anomalous fold.
- Calibration does not degrade beyond policy limits.
- Additional complexity and runtime are justified.

### Evidence-linked interpretation

```json
{
  "claim_id": "claim_threshold_risk_api_prod",
  "text": "The upper 80% interval exceeds reserved capacity from 20 to 25 August.",
  "evidence_ids": [
    "evidence_forecast_api_prod",
    "evidence_capacity_threshold",
    "evidence_coverage_80"
  ],
  "qualifiers": [
    "Historical 80% coverage was 76%, below target.",
    "A recent level shift increased fold variance."
  ]
}
```

### Explanation rules

| Rule | Behaviour |
| --- | --- |
| May explain | Direction, peaks/troughs, intervals, threshold risk, baseline comparisons, selected model reasons, context retained/rejected, disagreement, assumptions, and factors that could change the forecast. |
| Must qualify | Weak support, low coverage, insufficient folds, shifts, missingness, disagreement, and uncalibrated uncertainty. |
| Must not claim | Causality without evidence, fabricated confidence, unobserved context, hidden overrides, or recommendations outside an explicit decision policy. |
| Fallback | When no LLM is configured, deterministic templates render the same evidence and warnings. |

## Deployment and Hermes integration

> **Deployment strategy:** Hermes-first, agent-agnostic. Aion should be exceptionally easy for Hermes to install and operate, while preserving a stable open protocol for any agent or application.

### Deployment modes

| Mode | Use |
| --- | --- |
| Local CLI | Developer or agent invokes the executable. Lowest friction; full machine-readable output. |
| Local MCP server | Hermes starts aion mcp serve and discovers typed tools and schemas. |
| Hermes plugin/skill | Installation instructions, MCP configuration, safe-use policy, forecasting workflows, and examples bundled for Hermes. |
| Python library | Applications call the canonical runtime without subprocess overhead. |
| Container | Reproducible local/server deployment with mounted data and artifact volumes. |
| Future HTTP service | Thin authenticated adapter over the same runtime for hosted use; not required for MVP. |

### Hermes installation target

```bash
# Install Aion
pipx install aion-forecast

# Verify local runtime
aion capabilities --output json

# Start as a local MCP server
aion mcp serve

# Hermes configuration concept
hermes mcp add aion --command aion --args "mcp serve"
```

The exact Hermes command may evolve, so the integration package should also include a ready-to-copy MCP configuration and a health-check workflow.

### Hermes skill behaviour

- Prefer aion_inspect_dataset before creating a task when mappings are uncertain.
- Never infer a business threshold if none is found or provided; ask or omit threshold analysis.
- Surface all Aion warnings and support status in the final response.
- Do not paraphrase unsupported as low confidence; preserve the abstention.
- Do not add a context event to model inputs without successful temporal validation and evaluation.
- Use the evidence IDs returned by Aion for all material numerical claims.
- Ask Aion to rerun only when the new experiment can plausibly change the user’s decision.
- Never upload local data unless the user explicitly requests a sharing/hosted action.

## Security, privacy, and operations

| Control | Requirement |
| --- | --- |
| Local by default | Forecasting, artifacts, and LLM-free operation work entirely on the user’s machine. |
| Explicit source permissions | The agent receives an allow-list of files, directories, connectors, or workspace roots. |
| No implicit upload | Sharing, hosted inference, or remote LLM use requires explicit configuration and visible disclosure. |
| Source immutability | Aion never modifies source data. All transformations are materialised into versioned snapshots. |
| Secrets | Credentials remain in host-managed secret stores or environment variables and are never written into artifacts. |
| Prompt-injection resistance | Text context is treated as data; it cannot grant tools or alter system policies. Extracted claims retain source references. |
| Execution isolation | Adapters run with bounded time, memory expectations, and failure isolation. Optional container sandboxing is supported. |
| Artifact privacy | Public share output is generated from a redacted/public projection, not the complete private artifact. |
| Auditability | Every run records data fingerprint, code/runtime version, model versions, policies, and external endpoint provenance. |

### Storage and idempotency

- Use UUID/ULID or content-derived identifiers rather than sequential IDs.
- Write artifacts to temporary paths, flush, and atomically rename.
- Lock per project/artifact directory for local mutations.
- Key idempotency by command, normalised request, source fingerprints, and explicit idempotency key.
- A repeated idempotent request returns the stored result rather than silently rerunning.
- Interrupted runs persist completed diagnostics and baseline results with partial status.

### Observability

| Signal | Requirement |
| --- | --- |
| Structured logs | Events to stderr; machine results to stdout; no progress text mixed into JSON. |
| Run trace | Stage start/end, duration, adapter calls, failures, budgets, and policy decisions. |
| Quality telemetry | Optional opt-in anonymous counters for installation, successful first run, errors, reruns, scoring, and integrations. |
| Model telemetry | Latency, eligibility, backtest performance, calibration, failure rate, and selected-vs-baseline outcome. |

## Build sequence and acceptance criteria

| Phase | Scope | Exit condition |
| --- | --- | --- |
| Phase 0 — Product cut | Delete or hide non-functional agent commands, collapse duplicate runtime paths, choose canonical contracts and entry points. | One installed CLI, one runtime, one storage path, one integration-test path. |
| Phase 1 — Temporal core | Frequency/timezone, regularisation, duplicate and missingness policies, future index generation, timestamp snapshots. | Hourly, daily, weekly, and monthly fixtures produce correct grids and timestamps; ambiguity fails loudly. |
| Phase 2 — Evaluation and selection | Mandatory baselines, timestamp-aware folds, per-series metrics, candidate adapters, selection, abstention. | Every selected forecast traces to concrete folds and a baseline comparison. |
| Phase 3 — Honest uncertainty and evidence | Rolling residual quantiles, coverage, support assessment, evidence ledger, deterministic summaries. | Quantile claims expose method and measured coverage; every material claim resolves to evidence. |
| Phase 4 — Product CLI and project lifecycle | One-shot command, project mode, outputs, actual submission, realised score, compare, atomic persistence. | A new technical user can install, forecast, rerun, and score through documented commands. |
| Phase 5 — MCP and Hermes | MCP tools, schema discovery, Hermes skill/plugin, safe-use instructions, end-to-end demo. | Hermes can discover a dataset, invoke Aion, and return an evidence-linked forecast without editing numbers. |
| Phase 6 — Bounded context reasoning | Typed events, known-at gating, context compilation, identical-fold ablation, planner budget. | Context enters a final forecast only when it demonstrates stable temporal lift. |
| Phase 7 — Sharing and growth surface | Chart/report projection, reproducible command, opt-in public share, template gallery. | A user can share a useful forecast without exposing private artifacts or data. |

### Release-blocking acceptance suite

| Area | Release gate |
| --- | --- |
| Installation | Clean virtual environment install; aion capabilities succeeds; documented minimum Python versions pass. |
| Input correctness | CSV/Parquet; explicit/inferred frequency; mixed-frequency rejection; duplicate policies; missing-period policies; timezone preservation. |
| Forecast correctness | Future timestamps match input grid; baselines always run; candidate failure isolation; per-series selection; unsupported series abstain. |
| Evaluation correctness | Folds do not cross cutoffs; no future observation or late-known context enters training; metrics match fixtures; undefined metrics are explicit null/status. |
| Uncertainty | Method, residual count, target coverage, measured coverage, and interval width are present; insufficient evidence downgrades support. |
| Artifacts | No NaN/Infinity in public JSON; evidence IDs resolve; provenance and reproducibility fields are present; idempotency works. |
| Actual scoring | Exact and partial horizon scoring; duplicate actual policy; no-overlap failure; forecast-version comparison. |
| MCP/Hermes | Capability discovery; schemas; typed errors; no hidden mock tools; Hermes preserves warnings and abstention. |
| Privacy | No implicit network transfer; public share uses explicit projection; secrets absent from artifacts. |


## Final architecture recommendation

The MVP should optimise for one exceptionally correct path:

```text
inspect → formulate → validate → forecast → backtest → select or abstain
       → calibrate → explain → update → submit actuals → score
```

Hermes is the hero orchestrator and distribution integration, but Aion remains independently usable and agent-agnostic. The runtime library is canonical; CLI, MCP, Python, the optional standalone agent, and future hosted interfaces are adapters over the same contracts.
