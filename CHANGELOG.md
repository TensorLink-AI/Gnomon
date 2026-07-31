# Changelog

## Unreleased

### Enrichment adjudication

- Context events and covariates can now be supplied to the same run.
  `COMBINED_ENRICHMENT_UNSUPPORTED` is retired — a pure relaxation, since
  the combination previously could not run at all.
- New `adjudicate_enrichments` stage. After the existing ablations, every
  candidate — history-only, plus context, plus covariates, plus both — is
  scored on **identical** folds: the same rolling origins, the same
  observation prefix at each origin, and enrichment inputs gated by
  `known_at` at that origin's cutoff.
- Selection is a complexity ladder, not an argmin: each rung of added
  complexity must beat the standing champion by
  `minimum_baseline_improvement`. Enrichments that only work together are
  compared against the base and can be admitted; a second enrichment that
  re-encodes the first is compared against the winning single and cannot
  buy admission with a rounding-error gain. Ties keep the simpler
  candidate, and equal scores break by a fixed order, so runs are
  reproducible.
- The joint candidate is the covariate model's forecast carrying the part
  of the event effect that model has *not* already explained — measured on
  its in-sample residuals rather than on raw history.
- A winner that would publish intervals below the 70% coverage floor the
  history-only forecast met is rejected; the base forecast is reinstated.
- Each adjudicated series records one `enrichment_adjudication` evidence
  record: the shared folds, every candidate's per-fold scores, each rung's
  comparison with the score it needed, and the winner. The artifact proves
  the choice instead of asserting it.

## 0.3.0 — the temporal execution harness (2026-07-31)

Aion grows from a forecasting engine into a temporal execution harness:
an agent supplies an objective; Aion compiles it into validated,
snapshot-bound execution and returns typed, evidence-linked conclusions —
or a structured abstention. **Every v0.2 tool, CLI command, and artifact
schema keeps working unchanged** (see `COMPATIBILITY.md` for the frozen
set and each amendment).

### New verbs

- `aion investigate` / `aion_investigate_change` — what changed?
  Changepoints, regime shift vs transient, anomalies, and ranked
  *associational* explanations (concurrent events, cross-series
  precedence) with residual uncertainty. Never returns a cause.
- `aion decide` / `aion_decide` — what should we do? Exceedance scenarios
  from an evaluated forecast, feasibility and constraint checks, expected
  utility — or, without utilities, the feasible-action comparison as
  `conditionally_supported: missing utility inputs`.
- `aion monitor` / `aion_monitor` — when should we intervene? Sequential
  exceedance risk per step and an alert-cost-aware rule (cost-optimal with
  alert/miss costs).

### Bitemporal core

- Every observation carries `valid_time` and `known_time`; all execution
  reads through a `Snapshot` that structurally cannot serve rows published
  after its `as_of`, and logs every read.
- `aion ingest` appends revisions (corrected files become new vintage
  rows); `aion store list` inspects datasets; `store:<dataset>` inputs.
- `aion forecast --as-of <instant>` replays any historical moment; the
  artifact's `snapshot_access` evidence proves the maximum `known_time`
  touched. Backtest folds train on the series *as known at* each fold
  cutoff.

### Contracts and verification

- Five-state `SupportAssessment` (typed reasons, assumptions, sensitivity,
  recovery actions) on every result, alongside the frozen v0.2 enum.
- Typed lineage (`lineage.json`) — artifacts, evidence, claims with claim
  classes, actions, outcomes — and a deterministic claim verifier run on
  every response: causal-from-associational, uncalibrated probabilities,
  unevaluated decision constraints, and post-`as_of` citations are
  mechanical rejections.
- Structured errors now carry machine-readable `repair_options`.
- Content-addressed artifact IDs and idempotent writes; injectable clock;
  golden-artifact tests pin byte-exact output.

### Decisions and evaluation

- `DecisionArtifact` with enumerated options, constraints, and
  declared-or-absent utilities; realised-outcome scoring computes regret
  vs the best feasible action in hindsight and ex-ante optimality — bare
  `correct` is retired. v0.2 `DecisionRecord`s load as degraded artifacts.
- `aion status` — pollable open forecasts, due horizons, unresolved
  decisions, realised performance (descriptive, never causal).
- `aion eval episodes` — trap-family episode suite (temporal leakage,
  invented numbers, abstention traps, regime breaks) with mechanical
  graders and pass^k, feeding `aion eval compare`.

### Experimental (gated behind `AION_EXPERIMENTAL_PLANNER=1`)

- `TemporalPlan` DAG IR, deterministic validator, executor with step
  checkpointing / content-addressed caching / deterministic replay, and a
  bounded two-round repair loop. `aion plan compile|validate|execute` and
  the `aion_compile_task` / `aion_validate_plan` / `aion_execute_plan` /
  `aion_get_run` tools. Macros remain the default path.

### Integrations

- New tools: `aion_get_artifact`, `aion_explain_run`, `aion_status`,
  `aion_resolve_outcome`; Hermes plugin exposes the three new macros.
- Quickstart: `docs/quickstart-mcp.md`; bundled messy example datasets.

## 0.2.0

Forecasting engine: evaluated forecasts with abstention, covariate and
context-event ablation, tracking store, TSFM sandboxes, MCP server,
Hermes plugin, `aion eval compare`.
