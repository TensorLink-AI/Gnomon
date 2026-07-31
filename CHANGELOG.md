# Changelog

## Unreleased

### Enrichment adjudication

- Context events and covariates can now be supplied in the same forecast
  run. Each enrichment still passes its own independent, leakage-safe
  ablation gate; a new adjudication stage then runs a championship ladder —
  the base model against every admitted challenger (base + context,
  base + covariates, base + both) on identical selection folds — and picks
  the winner deterministically (best mean fold score, ties to fewest
  enrichments, then fixed candidate order).
- The combined challenger composes the two admitted mechanisms: the
  covariate linear forecast plus the additive event effect, fitted per fold
  under that fold's cutoff. Its winner reports
  `selected_model: "combined_enrichment"`.
- The full comparison — candidates, per-fold scores, winner, and why — is
  recorded as an `enrichment_adjudication` evidence record in the artifact
  and its typed lineage, so the artifact proves the model choice.
- The `COMBINED_ENRICHMENT_UNSUPPORTED` error is retired (a pure
  relaxation; see `COMPATIBILITY.md`). Single-enrichment runs are
  numerically unchanged.

### Messy-data repair (disclosed, capped, deterministic)

Real-world CSVs now work on first contact. `aion forecast --repair
{off,safe,aggressive}` (default `safe`):

- `safe` normalises cell text only — mixed date formats (slash dates with
  provable day/month order, month names, epoch stamps), currency symbols,
  thousands/decimal separators, percent signs, accounting negatives,
  sentinel missing values (`N/A`, `null`, …), fully blank rows, and
  byte-identical duplicate rows. It never invents a value, moves a
  timestamp, or drops a data point.
- `aggressive` opts into structural fixes: interior gaps linearly
  interpolated, jittered timestamps snapped to the inferred grid,
  conflicting duplicates resolved (last row wins), unparseable rows
  dropped, naive timestamps in mixed-timezone files assumed UTC — all
  capped (`EXCESSIVE_REPAIR` past ~30% of a series) and disclosed.
- Every fix lands in a `data_repair` evidence record; assumptive fixes
  become series warnings, so support downgrades honestly.
- Repairs fire only where strict parsing would fail: clean files remain
  byte-identical with unchanged artifact IDs.
- `aion inspect` now diagnoses instead of rejecting: `data_quality`
  reports what the file needs (`clean` / `repaired_safe` /
  `repaired_aggressive`), lists the repairs, and prints the exact
  follow-up command.
- New bundled example: `examples/filthy_requests.csv`.

### Input formats

- New always-on formats: `.tsv`, `.json` (array of objects),
  `.jsonl`/`.ndjson`, and gzip-compressed text inputs (`.csv.gz`, …).
- `.xlsx` behind a new `excel` extra (`pip install 'aion-forecast[excel]'`).
- Semicolon/tab/pipe-delimited "CSV" detected under repair when the header
  provably names the mapped columns (disclosed as `delimiter_detected`);
  non-UTF-8 files fall back to Windows-1252 under repair (disclosed as an
  `encoding_assumed` assumption; strict mode raises `INVALID_ENCODING`).
- `aion capabilities` reports the full input matrix.

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
