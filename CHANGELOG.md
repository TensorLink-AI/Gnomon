# Changelog

## Unreleased

- Conditional forecasts. An event without a verifiable source cannot be
  admitted to the forecast — its `known_at` cannot be shown not to leak — so
  "what if we run the promotion in March?" used to be an abstention. Such
  events now produce a clearly separated answer in a new
  `conditional_forecasts` list, each with its own `conditional_on_event`
  support and stated assumptions. The unconditional forecast is the base and
  is unchanged, so the difference between the two is the event and nothing
  else. The effect size is measured from periods in the observed history when
  an event of the same type was active — never read off the event
  description — and an event with no precedent is declined with that reason
  rather than given an invented magnitude. Intervals widen at event-active
  steps by the standard error of the measured effect, so an effect estimated
  from three occurrences is visibly less certain than one from thirty. The
  key is omitted when empty, so existing artifacts are byte-identical.
- **Behaviour change.** `--multivariate` no longer overrides the forecast.
  The VAR(1) candidate is now entered in the selection folds like every other
  model: scored on the same rolling origins, against the same baselines,
  under the same improvement margin, with its own fold-separated calibration
  residuals. Previously it overwrote the univariate point forecast outright
  for every aligned series — no fold comparison against the models it
  displaced, no evidence record, and its internal check validated on a
  trailing window that overlaps the report-only test fold. Consequences:
  the VAR is now admitted **per series** rather than imposed on all of them,
  an admitted VAR carries intervals derived from its own residuals rather
  than a different model's, and every run with `--multivariate` emits a
  `multivariate_gate` evidence record with the conditions and the one that
  decided the outcome. `aion.multivariate.forecast_var` is removed;
  `VarFrame` replaces it.
- **Behaviour change.** Ensemble prediction intervals are calibrated on the
  selection and calibration folds instead of a trailing window of the series.
  The old window overlapped both the calibration fold and the report-only
  test fold, and pooled across models at a single origin — so it measured
  model disagreement rather than error by lead time. On a representative
  series it produced intervals 3.3x too narrow. `--selection-strategy
  ensemble` now also enters the ensemble in the evaluation rather than only
  swapping the final forecast; where the ensemble still has no
  fold-separated residuals, it is declined in favour of the calibrated
  selected model rather than published with someone else's interval.
- Foundation-model capability exclusions are notes, not warnings, so they no
  longer downgrade support to `weakly_supported`. Every adapter has a
  `min_context_length` of 1, so the live trigger was frequency: `flowstate`
  supports `min`..`MS`, which downgraded every quarterly and annual series
  regardless of the evidence behind its forecast.
- `evaluate()` accepts `selection_stride`: selection origins sampled more
  finely than the horizon, so overlapping selection folds cut comparison
  variance while calibration residuals stay on the non-overlapping skeleton
  a conformal quantile needs. The default is unchanged (one origin per
  horizon) — `docs/fold-stride-measurement-2026-08.md` records why: across
  140 series the denser stride's choice beat the default's on the held-out
  test fold 19 times out of 31 changed selections (p = 0.28), for roughly 4x
  the selection compute. It also does **not** relax the four-fold cliff,
  which comes from needing calibration and test windows and is
  stride-independent.
- `evaluate()` accepts `extra_candidates`: named predictors that need more
  than the series' own history (`predictor(origin, horizon)`), scored on the
  same folds under the same margin. This is how the VAR enters the ladder.

- The context admission gate now reports itself. Every run with context
  events emits a `context_gate` evidence record: how many events were
  supplied, how many survived eligibility, each condition the gate
  evaluated with the number it was decided on, and which condition
  decided a rejection. Admission rate and rejection causes are countable
  across a corpus instead of being parsed out of prose reasons.
- The gate's coverage veto compares a Wilson upper bound rather than a
  point estimate. Coverage is measured on one test fold of `horizon`
  points, where a drop well inside sampling noise could previously veto
  context that degraded nothing.

- Forecast intervals are now split-conformal per lead time. The previous
  bounds took residuals pooled across a whole horizon and widened them by
  `sqrt(step)` — but pooled residuals already contain lead-time growth, so
  the widening double-counted it. Measured over 300 synthetic series,
  coverage was 0.96 against a nominal 0.80 and intervals were ~2.5x wider
  than the data supports. Residuals are now indexed by lead time, tails use
  the finite-sample conformal order statistic rather than an interpolated
  quantile (with a handful of residuals there is no honest 90th percentile),
  sparse leads borrow the pooled spread, and half-widths are fitted monotone
  in the horizon. Coverage measures 0.88-0.91 — conservative by design, not
  by accident. Intervals still widen with the horizon, now because the
  residuals at longer leads are wider rather than because a formula says so.

- Anomaly detection covers trend anomalies, and says what its grade
  covers. A fourth detector (`local_slope`) scores how fast the series
  moves rather than where it sits, and the grader plants a fourth family
  (`trend_shift`) so that detector must earn selection like any other.
  Found by running AnomLLM's `trend` dataset: Aion flagged nothing on
  397 of 400 series while reporting `supported`, because its grader only
  ever planted spikes, level shifts and dropouts — the detectors were
  built to treat drift as *not* an anomaly. Against that dataset's
  labels the new detector scores F1 0.755 where the best existing one
  scored 0.096. Every anomaly result now also discloses
  `graded_families` and carries an assumption naming them: a grade
  earned on planted spikes vouches for spikes, not for kinds nobody
  tested.

- Data-insufficiency abstentions now name the way out: the refusal computes
  the largest horizon the supplied observations can support and, when one
  exists, adds a `reduce_horizon` recovery action ("retry with
  `--horizon N`") to the support assessment and the warning text — in both
  the default degraded path and `--strict-abstention` mode. When no shorter
  horizon would succeed, no retry is suggested.

- Forecast results gain an informational `notes` channel (additive; never
  downgrades support, unlike warnings). When TSFM candidates are eligible
  for a series but none is installed, the result now says so and names the
  `aion tsfm install` command — a fresh install no longer silently hides
  the foundation-model tier. Notes render in `summary.md` as `- Note:`
  lines.
- README: the sandboxed TSFM tier (Chronos-Bolt, Toto, Moment, Moirai) is
  documented as a first-class capability — same folds, same mandatory
  baselines — instead of a parenthetical, and the quickstart shows the
  optional install command.

## 0.4.0 — first-contact release (2026-08-01)

The beta-readiness release: real-world files work on first contact, a
fifth verb (`aion detect`) lands with graded anomaly detectors, joint
enrichments are adjudicated honestly, tracked evidence becomes
task-conditioned with an advisory router, and the README/docs describe
the system as it is. Content-addressed IDs are salted with the runtime
version, so all artifact IDs change with this release (inputs and
parameters hash identically otherwise); golden artifacts were refreshed
accordingly.

### Evaluated anomaly detection (`aion detect` / `aion_detect_anomalies`)

- New fifth canonical macro: candidate detectors — robust z-score,
  rolling-median residual, and forecast-interval exceedance — compete on a
  deterministic synthetic anomaly-injection grader (spikes, level shifts,
  dropouts at noise-scaled magnitudes, placement seeded from the series
  content) before any of them labels the real series. Supplying labelled
  anomaly timestamps switches selection to label F1.
- Every candidate's precision/recall/F1 ships in the artifact alongside
  the winner; abstention below 16 observations is `inconclusive`, and a
  best grader F1 under 0.5 downgrades the run to
  `conditionally_supported` — if no detector can recover planted
  anomalies in this series' noise, real detections inherit that doubt.
- Registered as the `detect_anomalies` operator; surfaced through the
  CLI, agent tools, and MCP from the registry as usual.

### Series fingerprints, task-conditioned tracking, and the thin router

- Every tracked run now records a deterministic, unit-free series
  fingerprint (trend, noise ratio, intermittency, direction-change rate,
  season) and a `task` dimension. Existing stores migrate in place
  (schema v3); legacy rows read as `forecast`.
- `aion track leaderboard --task ...` and
  `TrackingStore.leaderboard(project, task=...)` condition realised
  performance on the task, so accumulated evidence transfers by data
  shape instead of restarting cold per project.
- `aion route` / `aion_route`: a disclosed, advisory routing decision —
  verified capability filter, then a fingerprint-weighted realised-MASE
  prior claimed only once ≥10 scored records exist for the task. Every
  exclusion reason and the decision itself are recorded to the store for
  replay. Evaluated runs still backtest every candidate; an explicit
  model choice always wins.

### Multi-task adapter seams

- `TSFMCapabilities.tasks` declares the tasks an adapter has verifiably
  implemented (default: forecasting only); `eligible_tsfms(task=...)`
  filters on it. MOMENT declares `forecast`, `detect_anomalies`,
  `impute`, and `embed`.
- The adapter protocol gains two optional verbs: `reconstruct(history,
  mask)` (masked reconstruction — anomaly signal and imputation) and
  `embed(history)`. Both are implemented for MOMENT in-process and in
  the sandbox worker, whose JSON protocol now carries a `mode` field;
  stale sandbox worker scripts refresh automatically.
- Installed multi-task sandboxes join the anomaly-detection candidate
  pool as reconstruction-error detectors and must win the same grader as
  the statistical detectors; a detector that cannot run scores zero with
  its error disclosed instead of failing the run.

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
