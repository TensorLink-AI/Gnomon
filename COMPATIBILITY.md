# v0.2 compatibility set

This file freezes the public surface of Aion v0.2. Every subsequent phase of
the temporal-harness migration must state what it **preserves**, **deprecates**,
or **breaks** against this set. `aion_forecast` keeps working unchanged
throughout the migration.

## MCP / agent tools (from `toolspec.TOOLS`)

`aion_capabilities`, `aion_inspect`, `aion_forecast`, `aion_covariate_guide`,
`aion_validate_covariates`, `aion_propose_covariates`, `aion_submit_actuals`,
`aion_list_open_forecasts`, `aion_model_performance`, `aion_record_decision`,
`aion_resolve_decision`.

Frozen per tool: name, required input properties, and the response envelope
(`schema_version`, `status`, and for forecasts `forecast_id`, `artifact_path`,
`results[]` with `series/support/selected_model/interval_coverage/warnings/`
`forecast_preview/forecast_rows/threshold/context/covariates`).

## CLI commands

`aion capabilities` · `aion inspect` · `aion forecast` · `aion covariates
guide|validate` · `aion context prompt|validate` · `aion mcp serve` ·
`aion tsfm list|install|remove|install-all` · `aion track
list|actuals|score|compare|performance|leaderboard|due|decision|export|relocate`
· `aion eval compare`. Flags listed in `cli.build_parser()` are part of the set.

## Artifact layout (`schema_version: "0.1"`)

Each forecast artifact is an immutable directory `<output>/<forecast_id>/`:

- `artifact.json` — `schema_version`, `forecast_id`, `created_at`, `status`,
  `task` (input_path, schema, horizon, quantiles, minimum_baseline_improvement),
  `source_fingerprint`, `results[]`, `evidence[]`.
- `forecast.csv` — columns `series,timestamp,point,q10,q50,q90`.
- `evidence.jsonl` — one record per line: `evidence_id`, `kind`, `series`, `payload`.
- `summary.md` — human-readable, not machine-parsed; content may evolve freely.

Support values: `supported`, `weakly_supported`, `degraded`,
`supported_ensemble`, `unsupported`.

Error envelope: `{"schema_version", "status": "error", "error": {"code",
"message", "retryable", "details"}}`.

## Amendments already adopted (Phase 0)

- `forecast_id` is now **content-addressed** (`forecast_<hex16>` derived from
  inputs + parameters + runtime version) instead of random. The format
  `forecast_<hex>` is preserved; uniqueness-per-invocation is not — identical
  tasks intentionally share an ID, and artifact writes are idempotent
  (first write wins).
- Readers of serialized artifacts accept `schema_version` N and N−1
  (`aion.versioning`).
- Phase 1 (bitemporal core), additive: `task.as_of` field in artifact.json
  (null unless a historical replay was requested); one `snapshot_access`
  evidence record per run reporting the snapshot `as_of`, whether
  `known_time` was assumed, and per-series access counts with the maximum
  `known_time` touched. `aion forecast --as-of <instant>` replays a run
  against only data known at that instant; `input` additionally accepts
  `store:<dataset>` for datasets ingested into the persistent bitemporal
  store (`aion ingest`). Numerical results for single-vintage data are
  unchanged.

- Phase 2 (contracts), additive: every result carries `support_assessment`
  (five-state status + typed reasons/assumptions/sensitivity/recovery
  alongside the frozen enum); artifact directories gain `lineage.json`
  (typed artifacts/evidence/claims); every response passes the deterministic
  claim verifier before leaving the process.
- Phase 3–4 (macros/surface), additive: new tools `aion_investigate_change`,
  `aion_decide`, `aion_monitor`, `aion_get_artifact`, `aion_explain_run`,
  `aion_status`, `aion_resolve_outcome`; new CLI verbs `investigate`,
  `decide`, `monitor`, `status`, `store list`, `track outcome`; error
  envelopes gain `error.repair_options` (machine-readable next actions).
- Phase 5 (planner), gated: `aion_compile_task` / `aion_validate_plan` /
  `aion_execute_plan` / `aion_get_run` and `aion plan …` exist only behind
  `AION_EXPERIMENTAL_PLANNER=1`; macros remain the default path.
- Phase 6 (decisions), additive: `DecisionArtifact` model with regret
  scoring; v0.2 `DecisionRecord` rows and their tools keep working, and
  load as degraded artifacts (nothing invented). Bare `correct` is retired
  in the new model only.
- Phase 7 (episodes), internal: `aion eval episodes` runs the built-in
  trap-family suite and feeds `aion eval compare` unchanged.

## Enforcement

`tests/test_golden_artifacts.py` pins byte-exact `artifact.json` output for
the example datasets under a fixed clock. A failing golden means this set is
affected; refresh only with `pytest --update-goldens` and record the change
here.
