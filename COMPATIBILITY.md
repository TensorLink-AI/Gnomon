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
- `forecast.csv` — leading columns `series,timestamp,point,q10,q50,q90`,
  then the additional quantile levels and `point_bias_correction`. `point`
  is the model's raw output; `q50` is recentred on the median backtest
  residual, and `point_bias_correction` is exactly the difference.
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
- Enrichment adjudication (championship ladder), additive: one run may now
  supply context events **and** covariates together — previously rejected
  with `COMBINED_ENRICHMENT_UNSUPPORTED`, an error code now retired. This is
  a pure relaxation: every previously-valid request behaves identically, and
  single-enrichment runs are numerically unchanged (goldens unaffected). In a
  combined run each enrichment still faces its own independent ablation gate;
  the base model and every admitted challenger (base + context,
  base + covariates, base + both) are then scored on identical selection
  folds, and the winner is chosen deterministically — best mean fold score,
  ties broken by fewest enrichments, then fixed candidate order. The full
  comparison (candidates, per-fold scores, winner, reason) is recorded as an
  `enrichment_adjudication` evidence record; the combined winner reports
  `selected_model: "combined_enrichment"`. `capabilities().features` gains
  `enrichment_adjudication: true`.

- Messy-data repair, relaxation + additive: `forecast` (Python/CLI/tools)
  accepts `repair` ∈ `off | safe | aggressive`, default `safe`. Repairs
  fire **only where the strict path would fail**, so any file that parsed
  before parses byte-identically and keeps its content-addressed ID (the
  default level is absent from the ID payload). Files that previously
  raised `INVALID_TIMESTAMP` / `INVALID_TARGET` / `DUPLICATE_TIMESTAMPS` /
  `IRREGULAR_TIME_GRID` / `MIXED_TIMEZONES` may now succeed: `safe`
  normalises cell text (date formats, currency/thousands, sentinels, blank
  rows, identical duplicates) and `aggressive` opts into structural fixes
  (gap interpolation, timestamp snapping, conflict resolution, capped row
  drops, UTC coercion). Every repair is disclosed in a new `data_repair`
  evidence record; assumptive repairs additionally appear as
  `repaired_data: …` warnings and therefore downgrade support. New error
  codes: `AMBIGUOUS_DATE_ORDER`, `EXCESSIVE_REPAIR`,
  `INVALID_REPAIR_LEVEL`. `aion inspect` now diagnoses instead of
  rejecting: it reports `data_quality` (status `clean` /
  `repaired_safe` / `repaired_aggressive`, the repair list, and the exact
  flag to pass) and raises only when no level reads the file.
- Input formats, additive: `.tsv`, `.json`, `.jsonl`/`.ndjson`, gzipped
  text inputs, and `.xlsx` (new `excel` extra) join CSV and Parquet;
  alternative CSV delimiters and a Windows-1252 fallback are detected
  under repair with disclosure. New error codes `INVALID_ENCODING` (was an
  unhandled crash) and repair options on `UNSUPPORTED_INPUT` /
  `MISSING_OPTIONAL_DEPENDENCY`. `capabilities().inputs` grew keys; the
  existing `csv`/`parquet` keys are unchanged.

- Anomaly detection + routing (TSFM infrastructure tracking), additive:
  new macro `detect_anomalies` (`aion detect` / `aion_detect_anomalies`),
  new operator `detect_anomalies` (seeded, deterministic — injection
  placement hashes the series content), and new tool/CLI `aion_route` /
  `aion route`. `aion track leaderboard` gains an optional `--task`
  filter; the tracking store migrates in place to schema v3, adding
  `task` (legacy rows read as `forecast`) and `fingerprint` columns and
  a `routing_decisions` table — every v0.2 tracking command behaves
  identically. `TSFMCapabilities` gains a `tasks` tuple (default
  `("forecast",)`, so existing selection is unchanged) and the sandbox
  worker JSON protocol gains an optional `mode` field defaulting to
  `predict` (old requests behave identically; stale worker scripts are
  refreshed on next use). `capabilities().features` gains
  `anomaly_detection`, `graded_detector_selection`, `series_fingerprints`,
  `task_conditioned_leaderboard`, `task_routing`. Forecast artifacts and
  goldens are unaffected.

- Result notes (TSFM availability disclosure), additive: every forecast
  `SeriesResult` gains `notes` (list of strings, default empty) — purely
  informational disclosures that, unlike `warnings`, never downgrade
  support. The first note: when TSFM candidates are capability-eligible
  for a series but no sandbox (or in-process adapter) is installed, the
  result names the eligible adapters and the `aion tsfm install` command
  that would add them to the same folds. `summary.md` renders notes as
  `- Note:` lines after warnings. Warnings, support semantics, forecast
  values, and artifact IDs are unchanged; goldens were refreshed for the
  new field only.

- Abstention recovery (`reduce_horizon`), additive: when a
  data-insufficiency abstention occurs and a shorter horizon is
  supportable with the observations already supplied, the support
  assessment's `recovery_actions` gain a `reduce_horizon` entry naming
  that horizon, and the abstention warning text appends the same retry
  hint. `provide_more_history` remains; existing codes, statuses, and
  supported-path outputs are unchanged (goldens unaffected — the bundled
  abstention fixtures have no supportable shorter horizon, so their
  messages are identical).

- Per-lead-time conformal intervals, **behaviour change** (Phase 1 of
  `docs/integration-plan-review-2026-08.md`): interval bounds were the
  pooled residual quantiles widened by `sqrt(step)`. The pooled
  residuals span every lead time of a horizon-h fold, so they already
  contained the growth, and scaling them again double-counted it —
  measured across 300 synthetic series, coverage was 0.96 against a
  nominal 0.80, with intervals ~2.5x too wide. Bounds are now split-
  conformal per lead time: residuals indexed by lead, the finite-sample
  `ceil((n+1)p)` order statistic instead of an interpolated quantile,
  the pooled spread borrowed where a lead has fewer than
  `MIN_RESIDUALS_PER_LEAD` residuals, and half-widths fitted monotone in
  h. Measured coverage moves to 0.88-0.91 — still conservative, as split
  conformal guarantees, but no longer by 3x. `q10`/`q50`/`q90` keep
  their names, meaning, and position; point forecasts, model selection,
  and artifact IDs are unchanged. Threshold-crossing probabilities
  follow the same per-lead scaling (their `basis` string says so).
  `Evaluation` gains `residuals_by_lead`; `interval_bounds` is retained
  for callers holding one pooled quantile set. Goldens refreshed:
  quantile columns only.

- Gate instrumentation + interval-aware coverage veto, additive: a new
  `context_gate` evidence record per series with context events, and
  `context.gate_checks` in the public context dict (each entry: `code`,
  `passed`, and where applicable `measured`, `threshold`, `detail`).
  Existing `reasons` strings are unchanged and still populated. The
  coverage condition now triggers on the Wilson upper bound of measured
  coverage rather than the point estimate, so it fires only when the
  degradation exceeds the measurement's own uncertainty — strictly fewer
  spurious rejections; no previously-rejected-for-cause run is admitted
  on any other condition.

- Conditional forecasts, additive: a result may carry
  `conditional_forecasts`, a list of answers conditioned on events the
  admission gate cannot admit (no verifiable source, so not backtestable).
  Each entry has its own `support: "conditional_on_event"`, `assumptions`,
  `forecast` rows, `measured_effect`, `effect_standard_error`, and
  `occurrences_in_history`; runs that produce one also emit a
  `conditional_forecasts` evidence record naming every declined event and
  why. The key is **omitted entirely** when empty, so a run that produces no
  conditional forecast serialises byte-for-byte as it did before the feature
  existed — the goldens are unchanged and verify this. Every existing field
  keeps its unconditional value: a v0.2 reader that ignores the key sees what
  it has always seen.

- Multivariate gate, behaviour change: `--multivariate` no longer overrides
  the forecast for every aligned series. VAR is a candidate in the selection
  folds, admitted per series under the same margin as every other model, and
  each such run emits a `multivariate_gate` evidence record. A caller that
  passed `--multivariate` and relied on `selected_model == "var"` for all
  series will now see it only where VAR won. `aion.multivariate.forecast_var`
  is removed (`VarFrame` replaces it); no MCP tool name or signature changed.

- Ensemble intervals, behaviour change: calibrated on the selection and
  calibration folds rather than a trailing window overlapping the test fold.
  Ensemble `q10`/`q90` values change (they were ~3x too narrow on a
  representative series); `point` values are unchanged. No golden covers the
  ensemble path.

- Tracking schema, additive: `forecasts.wape` and `model_performance.wape`
  columns, added by the existing in-place migration; `ScoreResult.wape`,
  `ForecastRecord.wape`, `ModelPerformance.avg_wape`; a WAPE column in
  `aion track leaderboard` and `avg_wape` in `track performance --json`.
  Existing columns and MASE ordering are unchanged.

- Quantile levels, additive (goldens refreshed, additively): forecast rows
  and `forecast.csv` gain `q05`, `q20`, `q30`, `q70`, `q80`, `q95`.
  `q10`/`q50`/`q90` are unchanged in meaning **and in value** — identical
  order statistics of identical residuals under an identical fit. The
  `forecast.csv` header retains `series,timestamp,point,q10,q50,q90` as its
  first six columns in that order, so positional readers are unaffected, and
  appends the new levels after them. The golden refresh adds keys and one
  note; no pre-existing number changed, which the refresh diff shows.

- Distributional selection loss, additive and opt-in: `evaluate()` gains
  `selection_loss` (`"wape"` default, `"pinball"`) and `Evaluation` gains
  `pinball_scores`, populated only when pinball is requested. The default
  path is byte-identical; nothing selects differently unless asked.

- **`point` is not the median, and now says so** (additive; goldens
  refreshed additively). `point` is the selected model's *raw* output.
  Every `q*` level is recentred on the median backtest residual, so when a
  model carries systematic bias `point` is not the middle of its own
  interval — in the vintage example it sits near the 30th percentile.
  This has always been the behaviour and is unchanged. What is new:

  - each forecast row and `forecast.csv` gain a `point_bias_correction`
    column, exactly `q50 - point`, appended after the quantile columns so
    positional readers are unaffected;
  - a `point_is_not_the_median` disclosure appears on any result where the
    correction is non-zero, naming the size and its share of the interval.

  **No `point`, `q10`, `q50`, or `q90` value changed.** Callers treating
  `point` as the distribution's centre were always reading it wrong; they
  should read `q50`, and `point_bias_correction` is how to detect the gap.

- Support disclosures, additive: `SupportAssessment` gains `disclosures`,
  a list of typed `SupportReason`s for correct-but-surprising facts about
  how a result was produced — `point_is_not_the_median`,
  `constant_interval_width`, `conformal_residuals_pooled_across_selection`,
  `coverage_sample_size`, `quantile_levels_collapsed`. A disclosure never
  changes `status`; that is what separates it from a `reason`. The
  free-text quantile-collapse entry moves out of `notes` into
  `quantile_levels_collapsed`, so `notes` is empty where it used to carry
  that one string. `summary.md` renders each disclosure on its own line.

- Interval provenance, behavioural fix: points and intervals now always
  come from the same model. Previously an admitted covariate model, an
  adjudication winner, and a failed-TSFM fallback each published their
  point path beside residuals belonging to a different model. Affected
  runs — those using `--covariates`, enrichment adjudication, or a TSFM
  that fails at final prediction — will report **different interval
  widths**; the covariate path was measured at ~5.8x too wide. Point
  forecasts are unchanged in every case. Runs on the default path (no
  covariates, no context, no TSFM) are unaffected.

- Conformal calibration scope, opt-in: `evaluation.pool_residuals` in
  `aion.yaml` is now read. It defaults to `true`, which is the existing
  behaviour — residuals pooled across the selection folds and the
  calibration fold. Setting it `false` calibrates on the held-out
  calibration fold alone: genuine split conformal, noisier, wider. Either
  way the choice is disclosed on the result.

- `snapshot_access` evidence, additive: gains `known_time_provenance`
  (`recorded` | `assumed` | `partially_assumed`), read from ingest
  provenance rather than inferred from the data. `known_time_assumed` keeps
  its name, type, and meaning.

- Ingest reports, additive: `IngestReport` gains `reverts_recorded`.
  `duplicates_skipped` narrows to exact repeats of an existing vintage
  (same `valid_time`, same `known_time`, same value); a restatement at a
  new `known_time` is now recorded as the vintage it is rather than being
  skipped. Stores built before this fix are missing any reverted vintages
  they were offered; re-ingesting the source files recovers them.

- Tracking schema 4, migrating: `forecasts` is keyed on
  `(forecast_id, project, series)` and `model_performance` on
  `(project, model, forecast_id, series)`. Existing databases migrate in
  place on first open, preserving every row and score. `get_forecast`,
  `score_forecast`, and `compare` gain optional `project`/`series`
  arguments; called without them they resolve to the most recent
  registration, which matches the old single-row behaviour. The unenforced
  `decisions.forecast_id` foreign key is dropped, because SQLite requires a
  uniquely-indexed parent key.

## Enforcement

`tests/test_golden_artifacts.py` pins byte-exact `artifact.json` output for
the example datasets under a fixed clock. A failing golden means this set is
affected; refresh only with `pytest --update-goldens` and record the change
here.

Golden refreshes on record: 0.4.0 (version-salt bump only — every
`forecast_id` changed, no numeric or structural output changed); result
notes (each result gained a `notes` key — no numeric output or ID
changed).

Scope of the byte guarantee: deterministic replay yields identical bytes
**per interpreter**. CPython 3.12 changed builtin `sum()` to Neumaier
compensated summation (gh-100425), shifting float results by an ulp
relative to 3.11 — so goldens are captured and byte-checked on 3.12+,
and value-checked to 1e-9 relative tolerance on 3.11. Artifact IDs hash
inputs and parameters, never outputs, so they are identical across
interpreter versions.

- Trend-shift anomaly coverage + graded-scope disclosure, additive:
  `aion.anomaly` gains a fourth candidate detector, `local_slope`
  (deviation of the windowed median first difference from the series'
  typical slope), and the injection grader gains a fourth family,
  `trend_shift` (a ramped slope change of `TREND_SCALE` robust-scale
  units, `TREND_TRIALS` per run). The three existing detectors and
  three existing families are unchanged. Every anomaly result's support
  now carries `sensitivity.graded_families` and, for synthetic-injection
  selection, an assumption naming those families — the grade vouches for
  the kinds the grader planted and says so, instead of implying
  coverage of kinds nobody tested. Detector selection is unchanged in
  mechanism (best F1, ties to the earlier candidate), but a series whose
  anomalies are slope changes can now select a detector able to find
  them, so previously-empty detections may become non-empty. Forecast
  artifacts, forecast goldens, and artifact IDs are unaffected.
