# Aion codebase review — UX and temporal correctness

**Date:** 2026-08-02 · **Reviewed at:** branch `clean/harness-integrity` (PR #31)
· **Runtime version:** 0.4.0

---

## 0. What was actually done

Findings below are marked **[run]** where a reproduction was executed and
**[read]** where the finding comes from the code alone.

**Executed:** the README 60-second quickstart; the `docs/getting-started.md`
and `docs/quickstart-mcp.md` command sequences including the `uvx --from .`
install path; all five verbs (`forecast`, `investigate`, `detect`, `decide`,
`monitor`) plus `inspect`, `route`, `capabilities`, `ingest`, `store`,
`status`, `track list|actuals|performance|compare`, and `plan
compile|execute`; a live MCP stdio session driven as a JSON-RPC client
(initialize / tools/list / tools/call, with and without
`AION_EXPERIMENTAL_PLANNER=1`); the documented Python API snippets verbatim;
two persona loops (SRE: forecast → alert policy → submit actuals → realised
performance; analyst: investigate → forecast → decide); the naive no-flags
path on every command; all six documented input formats plus Parquet; nine
adversarial inputs (future-dated rows, DST-spanning hourly aware/naive, DST
daily, unsorted, duplicate timestamps, reverted vintages, leap day,
month-end, mixed-frequency panel); three covariate vintage regimes (honest,
same-day, published-after-cutoff); eight abstention-pressure rephrasings;
a plan replayed at two `as_of` instants against a shared step cache; the
same content-addressed forecast registered in two tracking projects;
byte-identity of replayed artifacts under a pinned clock; the test suite
(443 passed, 1 skipped) and the benchmark suite (85 passed).

**Read in full:** `temporal_store.py`, `data.py`, `temporal.py`,
`verifier.py`, `lineage.py`, `support.py`, `constraints.py`,
`decision_model.py`, `fingerprint.py`, `versioning.py`, `ids.py`,
`artifacts.py`, `llm.py`, `mcp_server.py`, `covariates.py`, the selection
and calibration core of `evaluation.py`, the stage graph of `pipeline.py`,
the macro bodies of `macros.py`, and the config/ensemble/meta-model path.

**Read partially:** `tracking.py` (scoring, adaptation, actuals, registry
schema), `tsfm.py` / `tsfm_sandbox.py` (pinning and dispatch),
`execution.py` (step caching), `operators.py`, `anomaly.py`, `toolspec.py`.

**Not covered:** the `episodes.py` world-simulator internals beyond its
tests, `context_eval.py`'s gate arithmetic, `multivariate.py`,
`registry.py`'s operator table beyond its entries, and any TSFM adapter
executed for real (no sandbox is installed in this environment, so TSFM
findings are **[read]**). Of the docs, `covariates.md`, `installation.md`,
`containers.md`, `ci-cd.md`, `llm-integrations.md`, `agent-evaluation.md`,
and `persistent-tracking-design.md` were not diffed against behaviour.

---

## 1. Verdict

**The contract is upheld in its architecture and broken in its plumbing.**

The design is the real thing, and parts of it are better than the field.
`Snapshot.__init__` filters on `known_time <= as_of` before any caller holds
the object, so leakage control is a property of the type rather than of
review discipline — and `tests/test_leakage_lint.py` makes a direct file read
inside `pipeline.py`, `adjudication.py`, `operators.py`, or `macros.py` a CI
failure. The leakage-trap benchmark is not a claim but a measurement: 40/40
Aion runs clean against 13/35 leaking control runs, exact McNemar
p = 0.00024, with an `oracle-leak` arm proving the trap detects what it
says. Randomness is content-seeded and the seed is written into evidence
[run]. Replay is byte-identical under a pinned clock — `diff -r` of two runs
is empty [run]. Abstention thresholds do not bend: with 12 observations and
horizon 14, eight different rephrasings (`strict_abstention`,
`seasonal_period=1`, a negative improvement threshold, relabelled frequency)
all still abstain, and the refusal names the horizon that would work [run].
The covariate vintage gate works end to end: an honest covariate is
admitted, and both a same-day-published and an after-the-cutoff covariate
are refused with the correct diagnosis [run]. `aion_decide`'s
expected-utility arithmetic is correct, and `aion_monitor`'s alert rule is
the right decision-theoretic threshold `c_alert / (c_alert + c_miss)`,
derived from the forecast distribution [run]. None of this is decoration.

But four of the headline claims do not survive contact with the code.

**"Aion owns every number" fails wherever two models meet.** Five paths
publish a point path and an interval that came from different models. The
sharpest is measurable: with a covariate admitted, the published model is
`covariate_linear` whose own calibration residuals have sd 6.5, while every
lead time builds its interval from the *base* model's residuals at sd 37.8 —
5.8× too wide, and the covariate model's real residuals are never consulted
because each lead already has enough of the wrong ones [run]. Alongside
that: a TSFM that fails at final prediction falls back to the baseline while
keeping the TSFM's intervals; a caller constraint clamps four of ten quantile
columns and leaves `q80 > q90`; and the published `point` is uncorrected
while every quantile is bias-corrected, so in the README's own example the
"forecast" sits at roughly the 30th percentile of Aion's own distribution.

**"Model choice is decided by backtest against mandated baselines, not by
argument" is false as written.** `minimum_baseline_improvement` accepts
negative values from both the CLI and MCP. At `-5.0`, Aion selects
`window_average` (WAPE 0.0752) over the strongest baseline `seasonal_naive`
(WAPE 0.0743) — a model that *lost* — reports `baseline_improvement:
-0.0124`, and still returns `support: "supported"` with an empty `warnings`
list and no support reason [run]. One parameter turns the mandated-baseline
gate off silently.

**"A deterministic verifier … before any response leaves the process" checks
structure, never truth.** Force the ensemble with `selection_strategy:
ensemble` on a series where a baseline won the backtest, and the support
assessment states *"An ensemble of eligible models beat the strongest
baseline"* while `baseline_improvement` is `0.0` [run]. The verifier passes
it because the claim's references resolve and its class matches its evidence
kind. It has no notion of whether the sentence is true of the evidence it
cites. The same gap lets a run with 57.1% measured coverage at nominal 80%
emit `predictive` claims: `UNCALIBRATED_PROBABILITY` is satisfied by the
*existence* of a `rolling_evaluation` record, never by its quality.

**And identity is not identity.** Content-addressed IDs are the backbone of
replay and of the first-write-wins artifact store — but three of them omit
inputs that change the answer. Two decisions that select *different actions*
share a `decision_id`, and the persisted artifact records the first run's
action and feasibility while the response reports the second's [run]. A
labelled and an unlabelled `aion detect` share an `anomaly_id` despite
choosing different detectors on different bases; the artifact on disk is the
unlabelled run [run]. And for TSFM selections the id covers the model
*name* while five of six adapters call `from_pretrained` with no revision
and every sandbox pip spec is unpinned — two of them `git+https://…` with no
ref — so the same id can denote different weights and different numbers on
different days [read].

**And there is a cache that outlives the as-of boundary.** The planner's
per-step cache key is `content_id("step", {"operator", "inputs"})` — it does
not include `as_of`, though `_execute_step` receives `as_of` and the answer
depends on it, and the cache directory is shared by every run writing to the
same output path. Executing a plan against latest data and then replaying the
same plan at `--as-of 2026-05-01` returns the *latest-data forecast verbatim*
with `"cached": true`; the replay never touches a snapshot, and its artifact
records `as_of: 2026-05-01` over numbers computed from data published after
it [run]. The `run_id` correctly includes `as_of`, which is what makes the
two runs look distinct while sharing one poisoned cache entry.

The bitemporal store itself can return a wrong historical value: ingestion
deduplicates by *value*, ignoring `known_time`, so a series revised
100 → 150 → back to 100 loses its third vintage and replay after the revert
returns 150 [run]. This reproduces through the repo's own example files —
ingesting `messy_requests.csv` (which already holds final values) and then
`messy_requests_revisions.csv`, the natural reading of
`docs/quickstart-mcp.md` §4, builds a store whose vintage history is
*inverted*, permanently serving the preliminary 318.7 where the truth known
at that instant was 328.6 [run].

On the UX axis the largest finding is scope. **The bitemporal store is
unreachable from MCP** — `aion_forecast` exposes neither `as_of` nor
`store:<dataset>`, and there is no ingest or store tool among the 20 — and
**the Python API exposes one of the five verbs**: `investigate`, `detect`,
`decide`, and `monitor` are absent from `aion.__all__` and from
`docs/python-api.md`, reachable only by importing `aion.macros` [run]. Below
that sit two raw Python tracebacks on documented CLI paths, an MCP surface
that converts the same failures into JSON-RPC `-32603` rather than a
repairable tool result [run], and roughly thirty documented `aion.yaml`
options that are parsed and never read [run].

**Verdict:** the temporal reasoning is better than most of what it competes
with, and the honest-abstention behaviour is genuinely differentiating. It
is not yet trustworthy enough to call v0.2 final, because every failure above
takes the same shape — a wrong number, reported confidently, with nothing in
the output to warn the reader.

---

## 2. Findings

### Critical — a number can be wrong or leaked without the user knowing

| # | Axis | Location | Finding | Fix |
|---|---|---|---|---|
| C1 | Temporal | `temporal_store.py:341-344` | **[run]** `ingest_rows` skips a row as a duplicate when *any* existing revision shares its value, ignoring `known_time`. A revert (100 → 150 → 100) drops the third vintage: `as_of 2026-01-11` returns `150.0` where the truth was `100.0`. Reproduces through the shipped example files, inverting their vintage history. `test_ingest_appends_revisions` encodes the buggy skip as intended behaviour, so CI cannot see it. | Dedup on `(valid_time, known_time, value)`. A vintage at a new `known_time` is always a new row even when the value repeats. Add a revert test and split `duplicates_skipped` from `vintages_skipped`. |
| C2 | Temporal | `pipeline.py:499-504` + `600-607` | **[run]** The adjudication winner overwrites `points`/`residuals` but not `residuals_by_lead`. Measured on an admitted covariate run: published model `covariate_linear`, its own residuals n=7 sd=6.50; `residuals_by_lead` carries the base model's 98 residuals at sd=37.78, and every lead has 14 ≥ `MIN_RESIDUALS_PER_LEAD`, so the pooled (correct) set is never consulted. Published intervals are ~5.8× too wide and belong to a model that was not selected. The `measured_test_coverage` reported for the covariate arm describes neither. | Carry `residuals_by_lead` through `AdjudicationResult` and set it with the rest; assert before `interval_stage` that the residual set's provenance equals `selected_model`. |
| C3 | Temporal | `pipeline.py:280-287` | **[read]** When the selected TSFM fails at final prediction, `selected_model`/`points` fall back to the strongest baseline while `residuals`/`residuals_by_lead` stay the TSFM's. Only a `logger.warning` — nothing reaches `state.warnings` or the artifact. Identical in kind to the ensemble failure that `pipeline.py:228-238` was already fixed to prevent. | Mirror the ensemble fix: recalibrate from the fallback model's own fold residuals or decline and abstain, and append a user-visible warning either way. |
| C4 | Temporal / UX | `constraints.py:170` | **[run]** `apply_claims` clamps only `("q10","point","q50","q90")`; the six later-added levels are never projected. A `max: 370` bound yields `q90=370.0` beside `q80=385.3` and `q95=397.8` — a crossed distribution *and* four published numbers the caller declared infeasible. The clamped rows then feed `threshold_analysis_stage`, so the crossing propagates into `monitor` and `decide`. | Clamp every `q*` column plus `point`; re-assert monotonicity after projection; test over the full `QUANTILE_LEVELS` set, not the three legacy columns. |
| C5 | Temporal / UX | `macros.py:365-372`, `macros.py:724-729`; `artifacts.py:105-106` | **[run]** Content-address payloads omit inputs that change the answer, and `write_json_artifact` is first-write-wins. `decision_id` covers only action *names*, so `[{scale_up},{do_nothing}]` and `[{scale_up, feasible:false},{do_nothing}]` — which select **different actions** — share an id; the response says `do_nothing` while the persisted artifact says `scale_up, feasible: true`. `anomaly_id` omits `labels`, so a labelled run (detector `forecast_interval`, basis `label_f1`) is served the unlabelled run's artifact (`rolling_median_residual`, `synthetic_injection_macro_f1`). Agents are told to quote the artifact. | Put every answer-changing input into the id payload (the `forecast` payload at `runtime.py:306-327` is the correct model — copy it). Add a test asserting that changing any documented parameter changes the id. |
| C6 | Temporal | `evaluation.py:839`; `cli.py`/`toolspec.py` (`minimum_baseline_improvement`) | **[run]** The parameter is unbounded below. At `-5.0` the gate becomes `candidate <= baseline * 6`, and Aion selects `window_average` (0.0752) over `seasonal_naive` (0.0743), reports `baseline_improvement: -0.0124`, and still returns `support: "supported"` with `warnings: []` and no support reason. A caller-supplied parameter silently disables the mandated-baseline rule the README leads with. | Reject `< 0` with a structured error (or clamp to 0), and force `support` to at most `conditionally_supported` with a typed reason whenever `improvement < 0`. |
| C7 | Temporal | `evaluation.py:692-743` (meta-model); `evaluation.py:655-690` (ensemble) | **[read]** The meta-model is fit on all selection folds by `train_meta_model(mm_fold_forecasts, mm_fold_actuals)` and then scored on **those same folds**, so its selection score is in-sample and competes against its members' honest out-of-sample scores — it wins by construction. `meta_model.py:12-14` claims "trained on the calibration fold and evaluated on the test fold — the same protocol as individual models"; the code does neither. The ensemble has the milder version: each fold's inverse-error weights come from `all_valid_scores`, aggregates computed *across all selection folds including that one*. Both are `enabled: false` by default, which is the only thing holding severity here. | Score the meta-model with leave-one-fold-out refits, or reserve folds it never trains on. Weight the ensemble on fold *i* using scores from folds `< i` only. Correct the docstring either way. |
| C8 | Temporal | `tsfm.py:351,456,674,746,876`; `tsfm_sandbox.py:67-96` | **[read]** Five of six adapters call `from_pretrained(model_id)` with no `revision` (only FlowState pins `r1.1`), and `TSFM_PIP_SPECS` pins no package version at all — two entries are `git+https://github.com/ibm-granite/granite-tsfm.git` with no ref. The artifact records the model *name*, so a TSFM-selected `forecast_id` does not cover the weights or the library. Two runs with the same id can produce different numbers, and first-write-wins then discards the second. The module docstring cites numpy version conflicts, so the sensitivity is known. | Pin `revision=` on every adapter and `==` every pip spec (or a git ref); resolve the commit hash at load and put it in the id payload and in evidence. |
| C9 | Temporal | `execution.py:151-155` | **[run]** The planner's step cache key omits `as_of`: `content_id("step", {"operator": step.operator, "inputs": resolved})`. `_execute_step` takes `as_of` and the result depends on it, and `.step-cache/` is shared across every run writing to the same output directory. Verified: the same plan run at latest data and then at `--as-of 2026-05-01` produced identical cache keys, and the replay returned the latest-data forecast with `"cached": true` — a snapshot was never consulted. `run_id` *does* include `as_of`, so the two runs look distinct while sharing one poisoned entry. This is the one place where a documented `--as-of` replay demonstrably reads post-cutoff data. | Put `as_of` (and `store_path`) into the step cache key. Better: key the cache on the resolved snapshot fingerprint, so a cache hit is provably a hit on the same visible data. Add a test that replays one plan at two instants and asserts the results differ. |
| C10 | Temporal / UX | `tracking.py` (`forecasts` table: `PRIMARY KEY (forecast_id)`) | **[run]** `forecast_id` is content-addressed, so the same forecast registered in a second project **overwrites** the first project's row — project, artifact path, and every realised score move. Verified: a forecast registered and scored under `sre-demo` (MASE 0.893) was reassigned to `win-demo` by a later registration; `aion track list --project sre-demo` then returns `[]` and `aion status` reports empty `model_performance`, while the MASE earned from `sre-demo`'s actuals is now attributed to `win-demo`, whose own actuals scored nothing. This is the evidence base `aion_model_performance` tells agents to read and the router's realised-performance prior consumes. | Make the key `(forecast_id, project, series)` across `forecasts`, `model_performance`, and the adaptation log; migrate existing rows. |

### High — the agent or user is stuck or misled

| # | Axis | Location | Finding | Fix |
|---|---|---|---|---|
| H1 | Temporal | `verifier.py:37-98`; `support.py:38-39` | **[run]** The verifier checks reference integrity and claim-class/evidence-kind compatibility, never whether a statement is true of its evidence. Forcing `selection_strategy: ensemble` on a series a baseline won emits the verified claim *"An ensemble of eligible models beat the strongest baseline"* with `baseline_improvement: 0.0`. `support.py` produces that sentence unconditionally for `supported_ensemble`, which `pipeline.py:641` sets whenever `selected_model == "ensemble"`, override included. | Derive the sentence from the measurement (`improvement > 0`), and add a distinct `ensemble_forced` reason for the override path. More broadly: any claim asserting a comparison must cite the comparison's evidence record and be checked against it. |
| H2 | Temporal / UX | `pipeline.py:610-615`; `evaluation.py:277` | **[run]** `point` is the raw model output; every `q*` is `point + median_residual + offset`. In the README example `point=332.74`, `q50=366.57`, `q30=332.43` — the headline number sits at ≈ the 30th percentile of its own distribution. Nothing in the artifact, `summary.md`, `COMPATIBILITY.md`, or the docs says `point ≠ median`. `tracking.py:580-587` then scores realised MASE/MAPE/bias against `points`, so the performance history that feeds the router prior is built on the off-centre number. | Publish the bias-corrected centre as `point` (keeping the raw output as `point_raw`), or emit `point_bias_correction` and state the relationship in `summary.md` and `COMPATIBILITY.md`. |
| H3 | Temporal | `evaluation.py:181-186`, `246-249` (`MIN_RESIDUALS_PER_LEAD = 8`) | **[run]** Below 8 residuals at a lead, that lead borrows the pooled spread. With the typical 3–4 folds *every* lead borrows, so the 14-step interval is exactly as wide as the 1-step: the README example emits byte-identical `q05…q95` for all 14 rows, and `probability_above` is a constant 0.5714 across the horizon. Uncertainty visibly does not grow with distance and nothing says so. | When every lead falls back to pooled, emit a first-class support reason: "interval width is constant across the horizon; the fold count cannot resolve per-lead growth" — or apply an explicit, disclosed growth model. |
| H4 | Temporal | `evaluation.py:947` (`_pool_residuals` over `residual_origins`) | **[read]** Conformal residuals pool the selection folds *plus* the calibration fold, but the selected model was chosen to minimise error on exactly those selection folds. That is not split-conformal; the winner's residuals there are optimistically small. The docstring justifies pooling by sample size without naming the selection bias. | Calibrate on held-out origins selection never sees. If pooling stays for sample size, measure and document the bias — `docs/fold-stride-measurement-2026-08.md` is the right template. |
| H5 | Temporal | `verifier.py:65-76`; `evaluation.py:984` | **[run]** `UNCALIBRATED_PROBABILITY` is satisfied by the existence of a `rolling_evaluation` record, never by its quality. A run with measured test coverage of 57.1% at nominal 80% still emits `predictive` claims that verify; low coverage is only a warning, and `pipeline.py:641` lets `degraded` mask that warning in the support status entirely. | Gate on the measured value: refuse `predictive` claims (or force `inconclusive`) when calibration evidence reports coverage outside a stated band, and stop letting `degraded` swallow a coverage warning. |
| H6 | Temporal | `verifier.py:87` | **[read]** The leakage check is a *string* comparison: `artifact.max_known_time > as_of`. Mixed representations (`Z` vs `+00:00`, date vs datetime, differing offsets) compare lexicographically. `2026-06-03T23:00:00+00:00` against an `as_of` of `2026-06-04T00:00:00+02:00` is a real leak that compares as safe. | Parse both to `datetime` and compare instants; raise `SNAPSHOT_TIMEZONE_MISMATCH` when one is naive and the other aware. |
| H7 | Temporal | `covariates.py:57`, `144`; `lineage.py:102`; `tests/test_leakage_lint.py:19` | **[run]** + **[read]** The behaviour is correct — a post-cutoff covariate is refused with `MISSING_HISTORICAL_VINTAGES` and the forecast falls back to `last_value` [run]. The *guarantee* is not structural. The covariate snapshot is built with `as_of=None`; the run's `--as-of` never reaches it and cutoffs are enforced by convention at each call site. Covariate reads never enter `snapshot_access`, so the `max_known_time` feeding H6's check comes from the target series alone. And `GUARDED_MODULES` in the leakage lint omits `covariates.py`, the one module that reads a file outside the load stage. | Build the covariate snapshot at the run's `as_of`; merge its `access_summary()` into `snapshot_access`; bound the final cutoff by `min(timestamps[-1], as_of)`; add `covariates.py` to `GUARDED_MODULES` with an explicit allow-list entry. |
| H8 | Temporal / UX | `temporal.py:108-114`, `135-139` | **[run]** `next_timestamp` adds a fixed `timedelta`, so timezone-aware **daily** data in any DST zone fails `IRREGULAR_TIME_GRID` at the transition (verified, Europe/London). Both offered repairs are wrong: "fill the missing period" (nothing is missing; the day is 23 hours) and `repair=aggressive`, which reports `snapped: 31` — it would shift every post-transition midnight by an hour — then aborts with `EXCESSIVE_REPAIR`. There is no working path, and DST is not mentioned anywhere in `docs/data-format.md`. | Advance calendar-aware for `D`/`W`/`MS` on aware timestamps (localise → add → re-normalise); recognise a DST step as regular rather than offering to invent a row; document the behaviour. |
| H9 | UX | `toolspec.py` (`aion_forecast`, `aion_inspect`); no ingest tool | **[run]** `aion_forecast` exposes neither `as_of` nor `store:<dataset>`, unlike `aion_investigate_change`, `aion_detect_anomalies`, `aion_decide`, and `aion_monitor`, which expose both. There is no MCP tool for `aion ingest` or `aion store list`. The bitemporal store, `--as-of` replay on forecasts, and the vintage workflow `docs/quickstart-mcp.md` §4 teaches are CLI-only and invisible to agents — while MCP is the front door the README leads with. | Add `as_of` and the `store:` input hint to `aion_forecast` and `aion_inspect`; add `aion_ingest` and `aion_list_datasets`. Record as additive in `COMPATIBILITY.md`. |
| H10 | UX | `aion/__init__.py:3-9`; `docs/python-api.md` | **[run]** The Python API — one of the three declared front doors — exports `forecast`, `inspect_dataset`, and three covariate helpers. Four of the five headline verbs (`investigate_change`, `detect_anomalies`, `decide`, `monitor`) are absent from `__all__` and from the docs; they exist only via `aion.macros`. A developer using the Python API can forecast and nothing else. | Re-export the four macros from `aion/__init__.py` and document them in `docs/python-api.md` beside the CLI and MCP equivalents. |
| H11 | UX | `mcp_server.py:60-72` | **[run]** `tools/call` catches `AionError`, `KeyError`, `ValueError`, `FileNotFoundError`. Anything else escapes to the outer handler and becomes JSON-RPC `error: -32603 "internal error: string indices must be integers, not 'str'"` — not an `isError` tool result. Verified with `aion_decide` and a list-of-strings `actions`. The agent gets a protocol failure with no payload, no `repair_options`, and no way to self-correct. | Catch `Exception` in `tools/call` and return `_tool_result(AionError("INTERNAL_ERROR", …).to_dict(), True)`. A tool bug must reach the model as a repairable result, never as a transport error. |
| H12 | UX | `cli.py:163`; `operators.py:532` | **[run]** `aion decide --actions '["scale_up","do_nothing"]'` — the shape the CLI help describes ("JSON list of actions") — crashes with an unhandled `TypeError`. The MCP schema is correct (`[{name, feasible?, residual_risk?}]`); the CLI help contradicts it. Malformed JSON yields `INVALID_JSON_ARGUMENT` with empty `repair_options` and no indication of which argument failed. | Validate `actions` against the MCP schema in the CLI; raise a structured `INVALID_ACTIONS` carrying an example; name the offending argument in `INVALID_JSON_ARGUMENT`; fix the help string. |
| H13 | UX | `constraints.py:62`; `context.py:70` | **[run]** Context events *must* carry an explicit timezone offset; the shipped example data is timezone-naive; `Claim.binds` then compares aware to naive and raises a raw `TypeError` out of `aion forecast --context`. The entire context-event/constraint feature is unreachable for the repo's own examples, and there is no worked context-events file anywhere in `docs/` or `README.md`. | Normalise event timestamps to the dataset's zone at validation, or reject the mismatch as a structured error naming both sides. Ship `examples/context_events.json` and document the flag. |
| H14 | UX | `contracts.py:209` | **[run]** README: "machine-readable repair options on every error." Actual: 24 of 55 raised error codes appear in `REPAIR_OPTIONS`. The whole covariate family (11 codes) has none, as do `AMBIGUOUS_FREQUENCY`, `UNSUPPORTED_FREQUENCY`, `INPUT_NOT_FOUND`, `EMPTY_DATASET`, and `INVALID_JSON_ARGUMENT` — the codes a first run hits first. `docs/troubleshooting.md` covers several of them in prose, so the human is served where the agent is not. | Add options for every raised code and enforce completeness with a test that walks the `AionError` call sites (~15 lines). Until then, soften the README claim. |
| H15 | UX | `config.py` (whole surface) | **[run]** Roughly thirty documented `aion.yaml` options are parsed and never read anywhere outside `config.py`: `models.baselines.enabled`, `models.statistical.enabled/candidates`, `models.tsfm.overrides` (so the example's per-model `device`, `timeout`, and FlowState `scale_factors` are inert), `backends.sandbox.*`, `ensemble.max_weight_ratio/fallback/eligible/quantile_strategy`, `meta_model.ridge_alpha/lasso_alpha/min_folds/fallback` (while `_solve_nnls` hardcodes `alpha = 1e-6`), every `evaluation.*` key **including `target_coverage`, `min_observations`, and `selection`**, all six `output.*` switches, and `llm.mode/adapter/max_tokens/temperature`. A user who disables statistical models still gets all five. | Either honour them or delete them from `aion.yaml.example`, and add a test that every documented key is read. The `evaluation.*` and `models.*` groups matter most: they promise control over the abstention and selection thresholds. |
| H16 | UX | `tracking.py:739-756`, `submit_actuals` | **[run]** `submit_actuals_csv` guesses columns positionally (`ts_col = cols[0]`, `val_col = cols[1]`) and `aion track actuals` has no `--time`/`--target` flags, unlike every other input path. An operator CSV of `requests,timestamp,host` returns `{"scored": 0}` — no error, no warning, no diagnosis. Actuals for the wrong window (2027 values against a 2026 forecast) return the same `{"scored": 0}`. `scored: 0` is indistinguishable from "nothing was due", and it is the failure mode of the exact follow-up loop the product promises. | Add explicit `--time`/`--target`; on zero matches return a structured result naming the forecast's horizon window, the file's window, and the overlap count, with `repair_options`. |
| H17 | UX | `toolspec.py:585`; `router.py:154` | **[run]** `aion_route` answers "which method for this task?" with `recommendation: null` / `basis: "backtest_required"` on any cold project, and `aion_forecast` has no model parameter at all — so even a non-null recommendation cannot be acted on. The tool's output has no consumer. | Give `aion_forecast` a `candidates`/`model` parameter the router's output feeds, or re-scope `aion_route` to what it does (capability filtering and disclosure) and say so in its description. |
| H18 | UX | `toolspec.py:308-320`, `567` | **[run]** Two decision lifecycles sit side by side with no deprecation marker: `aion_record_decision` + `aion_resolve_decision` ("whether a decision was correct") and `aion_resolve_outcome` ("bare 'correct' is retired"). The descriptions contradict each other and an agent reading them cold cannot choose. | Mark the v0.2 pair deprecated in its description text, name the replacement, and say which macro produces records for which resolver. |
| H19 | UX / Temporal | `temporal.py:120-133` | **[run]** An explicit `--frequency` cannot rescue a series whose steps are not in `FREQUENCIES`: `validate_and_group` calls `infer_frequency` per series for its consistency check, which raises `AMBIGUOUS_FREQUENCY` before the requested frequency is ever applied. Month-end data (Jan 31, Feb 28, Mar 31 …) fails even with `--frequency MS`. `docs/data-format.md:63` documents the month-start requirement, but the error names neither the observed step nor the fix, and carries no `repair_options`. | Skip the inference cross-check when a frequency is given explicitly (or validate against the requested one); name the observed modal step in the error and add a `restamp_to_month_start` repair option. |
| H21 | UX | `cli.py` (`build_parser`) | **[run]** Every command requires `--time` and `--target`; `forecast` also requires `--horizon`. A two-column CSV with one obvious timestamp column and one obvious numeric column still demands both flags. Worse, the failure is a bare argparse error — `rc=2`, plain text on stderr, no JSON envelope, no `code`, no `repair_options` — so the single most common first-run failure is the one error in the product that bypasses the structured error contract entirely. Any agent wrapping the CLI must special-case it. | Infer `--time`/`--target` when the schema is unambiguous and disclose the inference as an assumption; default `--horizon` to one season. Route argparse failures through `AionError("INVALID_ARGUMENTS", …)` so every failure on every surface has the same shape. |
| H22 | UX | `docs/getting-started.md:13-17`; `install.sh:4-5` | **[run]** Getting-started says `cd /root/Aion && bash install.sh`, but `install.sh` defaults to `VERSION=main` from `TensorLink-AI/Aion` on GitHub — it installs the remote default branch, not the checkout the reader just `cd`'d into. A developer following the doc to test local changes silently runs `main`. (The `uvx --from .` path in `quickstart-mcp.md` *does* build the local checkout and works — verified.) | Add a `--local` / `--from .` mode, and make getting-started use it; or point getting-started at the `uvx --from .` path that already behaves correctly. |
| H23 | UX | `macros.py` (`investigate_change`); `artifacts.py:96-119` | **[run]** `investigate` answers "what changed?" in model-internal terms: changepoints are reported as `index: 21` with `before_mean`, `after_mean`, `shift`, `relative_gain`, and the sensitivity block carries `standardised_shift` / `standardised_tail_shift`. On the shipped example it returns `explanations: []` with `residual_uncertainty: "No candidate explanation was found; the change stands unexplained."` — honest, but the user asked what changed and gets three changepoints, one of which (`relative_gain: 0.016`) is noise beside the real one (`0.779`). And unlike `forecast`, the artifact is JSON only: no `summary.md`, so there is no human-readable rendering anywhere. | Rank or filter changepoints by `relative_gain` and mark the dominant one; drop the raw `index` in favour of the timestamp; write a `summary.md` for every macro, not just forecasts. |
| H20 | UX | `docs/quickstart-mcp.md` §4 | **[run]** The documented vintage command (`aion forecast store:requests --horizon 7 --as-of 2026-06-03`) returns `unsupported`/`inconclusive`: `messy_requests_revisions.csv` holds only 10 valid dates. The step that would make it work — ingesting `messy_requests.csv` first — is absent from the docs, and performing it triggers C1 and inverts the vintage history. A new user's first vintage run fails either way. | Ship a revisions fixture with enough history for the documented horizon and make the base ingest an explicit step — after C1 is fixed. |

### Medium — friction, inconsistency, or a misleading presentation

| # | Axis | Location | Finding | Fix |
|---|---|---|---|---|
| M1 | Temporal | `macros.py:344` | **[run]** `decide`'s `scenario_probabilities.exceed` is `max(probability_above)` — the worst single step — labelled plainly as `"exceed"` with no definition. For a horizon-level decision the relevant event is usually "at least one exceedance", which is ≥ the per-step max, so the risk feeding expected utility is understated. `summary.md:79` calls the same quantity "Peak probability above"; `decide` does not. | Name it `peak_step_exceedance` (or compute and offer the any-step probability), and state the event definition in the payload. |
| M2 | Temporal / UX | `evaluation.py:961-985` | **[run]** Interval coverage is measured on one test fold of `horizon` points and headlined as "Final-test 80% interval coverage: 100.0%". With 14 points that carries almost no information, and the `< 0.7` warning threshold is noise-dominated. | Report the count alongside the rate, or a confidence interval; suppress the headline below a minimum sample. |
| M3 | UX | `tracking.py:523-554` | **[run]** `adapted_alpha` — adaptive conformal replay over the coverage-outcome log, carefully written and covered by 12 test assertions including replay determinism — has **zero call sites** outside its tests. `record_coverage_outcome` writes to the log on every scoring pass and nothing ever reads the adapted level. A complete, tested capability is unreachable from every surface. | Wire it into `interval_stage` behind a project scope, or expose it as `aion track coverage` / an MCP field. Until then, say in the docs that it is not yet applied. |
| M4 | UX | `mcp_server.py:24-28` | **[read]** Tool results carry JSON inside a `text` content block with no `structuredContent` and no `outputSchema`, though the server advertises protocol `2025-06-18`, which supports both. Every agent must `JSON.parse` a string and has no schema to validate against. | Emit `structuredContent` alongside the text block and publish `outputSchema` per tool. |
| M5 | Temporal | `temporal_store.py:456` | **[read]** `assumed_known_time` is inferred as `all(valid_time == known_time)` over the data rather than recorded from ingest provenance. A dataset with genuine same-day publication reports as assumed; a dataset mixing an assumed ingest with a real one reports *not* assumed, hiding the assumption. Observed live: a store built from a real `--known-at` column reported `known_time_assumed: true`. | Persist `assumed_known_time` per ingest and aggregate; surface "partially assumed" instead of collapsing to a boolean. |
| M6 | Temporal / UX | `pipeline.py:117-127`; `runtime.py` (`inspect_dataset`) | **[run]** A file whose observations are entirely in the future (2027 data on 2026-08-02) forecasts happily, and `aion inspect` reports `status: valid` with no remark. For a harness built on knowledge time, "every observation postdates now" deserves a note. | Compare `max(valid_time)` to the clock in `inspect` and in the support assessment; emit an assumption, not an error. |
| M7 | UX | `temporal.py:120-122` | **[run]** For a panel file containing two genuinely different frequencies, `validate_and_group` infers one frequency from all observations pooled and then reports `IRREGULAR_TIME_GRID` against the minority series — blaming that series for being irregular rather than saying the file mixes frequencies. | Infer per series and raise a dedicated `MIXED_SERIES_FREQUENCIES` naming each series and its inferred step. |
| M8 | UX | `toolspec.py` (`aion_monitor`) | **[run]** `monitor` does not monitor — it defines an alert policy once against a single forecast. An SRE reading the name expects a running watch. Related: `decide` returns an expected-utility comparison, not an executed decision. | Rename to `alert_policy` (aliasing the old name) or lead the description with "Defines, does not run: …". |
| M9 | UX | `tracking.py:90-91, 106-107` | **[run]** The same object reports `mape: 4.279` (percent) and `avg_wape: 0.0424` (fraction), neither labelled. | Emit both as fractions, or suffix the field names (`mape_pct`). |
| M10 | UX | `toolspec.py` (default `output_dir`) | **[run]** Artifacts default to `./aion-output` in the process CWD, so the quickstart's first commands write into the user's clone. The temporal store already does the right thing with `~/.local/share/aion`. | Default to a user data directory with `--output` to override. |
| M11 | UX | `docs/python-api.md:~78` | **[run]** "Calling `forecast` also persists the four standard artifact files." There are five — `lineage.json` was added in this migration and `docs/results-and-artifacts.md` documents all five. | Update the count, or say "the standard artifact files" and link the canonical list. |
| M12 | UX | `aion.yaml.example:22-27` | **[run]** The example config lists `ets` and `arima` under "(planned)" while `ets` is implemented, is in `MODELS`, and won selection in several runs during this review. | Move `ets` out of the planned list. |
| M13 | Temporal | `temporal_store.py:86-87` | **[read]** `_comparable` checks only `observations[0]`. A dataset whose aware/naive mix starts after row 0 raises a bare `TypeError` from the comprehension instead of `SNAPSHOT_TIMEZONE_MISMATCH`. | Check the whole set, or normalise at ingest. |
| M14 | UX | `docs/` | **[run]** `--context` / `context_events_file` has a design module behind it and zero worked examples in the docs. Same for `--selection-strategy ensemble` and the `aion.yaml` surface generally. | Add a context-events example to `docs/concepts.md` and a fixture under `examples/`. |
| M15 | UX | `tracking.py` (`track actuals` response) | **[run]** The actuals response omits `schema_version` and any `support_assessment`, unlike every other envelope on every surface. | Wrap it in the standard envelope. |
| M16 | UX | `temporal.py:128` | **[read]** Out-of-order rows are silently sorted. Correct, but it is an undisclosed mutation, and the sort is why a genuinely unsorted file never raises a data-quality signal. | Record a `timestamps_reordered` note in the repair log when input order differs from sorted order. |
| M17 | UX | `covariates.py:319-324` | **[run]** `MISSING_HISTORICAL_VINTAGES` says "91 fold-time values were unavailable" without saying *why* (published too late) or what to change (`known_at` must precede each fold cutoff). The gate is correct; the message does not teach the fix. | Name the cause and the remedy, and attach `repair_options` (this is one of the 11 covariate codes from H14). |

### Low — polish

| # | Axis | Location | Finding | Fix |
|---|---|---|---|---|
| M18 | UX | `metric_scaling_cache/` | **[run]** Nine binary `.h5` files are tracked in git, while the sibling `inference_cache/` is correctly `.gitignore`d. This is the same class of artefact the two most recent commits on this branch removed ("Untrack the accidentally committed benchmark corpora and result cache"). | `git rm --cached metric_scaling_cache/` and add it to `.gitignore` alongside `inference_cache/`. |
| M19 | UX | `cli.py` (`track compare`) | **[run]** `aion track compare --a <id> --b <id>` needs two forecast ids and there is no discovery path when `track list` is empty (see C10) — and no `--project` filter on `compare` itself. The "rerun and compare" half of the follow-up loop is reachable only if you already recorded the ids by hand. | Accept `--project` plus `--latest N`, or a `--series` selector, so compare can be reached from where the user actually is. |
| L1 | UX | `docs/getting-started.md:14,32`; `docs/development.md:19,26` | **[run]** `cd /root/Aion` — a reviewer's local path shipped in the published docs. | Replace with `cd Aion`. |
| L2 | UX | `cli.py` (parser) | **[run]** Most CLI flags carry no `help=` (`--time`, `--target`, `--horizon`, `--frequency`, `--series`), while the MCP schema documents all of them well. | Reuse the MCP descriptions as `help=` strings. |
| L3 | UX | `contracts.py:209` | **[run]** `TEMPORAL_LEAKAGE` has repair options but is never raised as an `AionError` code — it appears only as a violation inside `CLAIM_VERIFICATION_FAILED`, which has none. | Attach the options to `CLAIM_VERIFICATION_FAILED`, or surface per-violation options. |
| L4 | UX | `artifacts.py:90-92` | **[read]** A no-op `except Exception: raise` leaves the `.<id>.tmp` directory behind on failure. The comment says this is deliberate for diagnosis, but nothing ever cleans it and the next run silently `rmtree`s it. | Either drop the handler or log the retained path. |
| L5 | UX | `fingerprint.py:47-50`; `router.py` output | **[run]** `season_period` is in periods, not days: weekly data reports `season_period: 7`, meaning seven weeks. Easy to misread beside `frequency: W`. | Add `season_period_unit` or render as `7 × W`. |
| L6 | UX | `pipeline.py:626-637` | **[read]** The quantile-collapse note fires on the common small-sample path and reads as a defect disclosure. Right instinct; it belongs in the typed support assessment rather than free-text `notes`. | Promote to a typed note code. |

---

## 2.5 Surprises audit

Every place Aion's actual behaviour departs from what a user would assume
without reading anything. Several are *correct* behaviour — correct-but-
surprising still needs loud, in-band communication at the moment it happens,
which is what the "Told?" column tracks.

| # | What a user assumes | What actually happens | Told? | Ref |
|---|---|---|---|---|
| S1 | `point` is the forecast — the middle of the interval | `point` is the raw model output; `q50` is bias-corrected. In the README example `point` sits at ≈ the 30th percentile of its own distribution | **No** | H2 |
| S2 | A 14-day-ahead interval is wider than a 1-day-ahead one | With the usual 3–4 folds every lead borrows the pooled spread; all 14 rows are byte-identical | **No** | H3 |
| S3 | "80% interval coverage: 100%" means the intervals are well calibrated | It is 14 points from one fold; the number is nearly information-free | **No** | M2 |
| S4 | `monitor` watches the series | It defines an alert policy once, against one forecast, and exits | **No** | M8 |
| S5 | `decide` decides | It returns an expected-utility comparison; nothing is actuated | Partly — `decision_rule` is stated | M8 |
| S6 | `P(exceed)` is the chance of breaching the threshold during the horizon | It is the maximum single-step probability, which is lower | **No** — labelled plainly `exceed` | M1 |
| S7 | Artifacts land somewhere managed | They land in `./aion-output` in the process CWD, inside the user's clone | Path is returned, but the default is never explained | M10 |
| S8 | Re-running with different parameters produces a different artifact | If the changed parameter is outside the id payload, first-write-wins silently returns the earlier run's artifact | **No** | C5 |
| S9 | Registering a forecast in a second project adds a row | It *moves* the row, taking the first project's realised scores with it | **No** | C10 |
| S10 | `--as-of` replays what was knowable then | True for `forecast` — but a planner step cache hit ignores `as_of` entirely and returns latest-data numbers | **No** — reported only as `"cached": true` | C9 |
| S11 | A revised value revised back to its original is recorded | The reverted vintage is dropped as a "duplicate" and replay returns the superseded value | Counted as `duplicates_skipped`, which reads like a no-op | C1 |
| S12 | Daily data from a European or US timezone works | Any DST transition fails the grid check; the suggested repair would shift 31 timestamps | Loudly — but the diagnosis and the offered repair are both wrong | H8 |
| S13 | Month-end series (Jan 31, Feb 28…) are monthly data | `AMBIGUOUS_FREQUENCY`, and `--frequency MS` does not rescue it | Documented in `data-format.md`; the error itself says neither | H19 |
| S14 | Rows out of order is a data-quality problem | Silently sorted | **No** | M16 |
| S15 | Data entirely in the future is suspicious | Forecast and `inspect` both proceed; `inspect` reports `status: valid` | **No** | M6 |
| S16 | A constraint I declared is enforced on the output | Four of ten quantile columns are clamped; `q95` can exceed a declared `max`, and `q80 > q90` | **No** | C4 |
| S17 | Setting `minimum_baseline_improvement` low means "be less strict" | Negative values invert the rule; a model that lost the backtest is selected and still reported `supported` | **No** | C6 |
| S18 | Options in `aion.yaml` change behaviour | ~30 documented keys are parsed and never read, including `evaluation.target_coverage` and `models.statistical.candidates` | **No** | H15 |
| S19 | Submitting actuals scores my forecast | Columns are guessed positionally and a mismatch returns `{"scored": 0}` — indistinguishable from "nothing due" | **No** | H16 |
| S20 | `aion forecast data.csv` does something | `rc=2` and a bare argparse error; three flags are mandatory and the error has no JSON envelope | Loudly, but outside the error contract | H21 |
| S21 | Following getting-started installs what I checked out | `install.sh` installs `main` from GitHub | **No** | H22 |
| S22 | The Python API is the CLI as a library | It exposes `forecast` and `inspect_dataset`; the other four verbs are undocumented `aion.macros` imports | **No** | H10 |

The pattern: sixteen of twenty-two are communicated nowhere. Aion's
disclosure instinct is strong where it has been *deliberately* applied
(`known_time_assumed`, `capability_notes`, the quantile-collapse note) and
absent everywhere it has not. The cheapest structural fix is to treat the
support assessment as the single channel for all of it — every surprise above
is a candidate `SupportReason` with a typed code.

---

## 3. Top 5 fixes before release

1. **Make identity mean identity (C5, C8, C9, C10).** One root cause, four
   symptoms: things are keyed by less than determines them. Put `as_of` in
   the planner's step-cache key — it is the only demonstrated `--as-of` leak
   in the product and it is three lines. Key the tracking registry on
   `(forecast_id, project, series)` so a second project cannot steal the
   first's realised scores. Put every answer-changing input into the
   `decision`/`anomaly` id payloads, and pin TSFM revisions and pip specs so
   an id covers the weights. Add one test per family that changing a
   documented input changes the key. Until this lands, first-write-wins is
   not idempotence, it is silent substitution.
2. **Fix the store's dedup rule and re-ship the vintage example (C1, H20).**
   Key on `(valid_time, known_time, value)`, add a revert test, regenerate
   the revisions fixture with enough history for the documented horizon, and
   make the base-file ingest an explicit documented step. This is the single
   fix that decides whether `--as-of` means anything.
3. **One invariant: points and intervals come from the same model (C2, C3,
   C4).** Assert once, before `interval_stage`, that the residual set's
   provenance equals `selected_model` and that every emitted `q*` column went
   through the same projection. Three separate bugs collapse into one
   assertion — and C2 is currently a measured 5.8× error in published
   interval width.
4. **Close the two contract holes (C6, H1).** Reject a negative
   `minimum_baseline_improvement`, and force support below `supported`
   whenever `improvement < 0`. Derive the ensemble's support sentence from
   the measurement instead of asserting it. Both are cases where the harness
   itself states something the evidence contradicts, which is precisely what
   the verifier exists to prevent.
5. **Stop crashing, and make the error promise true (H11, H12, H13, H14,
   H16).** No `aion` invocation should print a Python traceback and no MCP
   tool call should return `-32603`; catch `Exception` at the tool boundary.
   Fix `--actions` and the naive/aware context comparison, add
   `repair_options` to the remaining 31 codes with a completeness test, and
   make `scored: 0` explain itself.

Three more that are cheap relative to their value: **say that `point` is not
the median (H2)** — one line of documentation or one line of code, and it
removes a systematic misreading of every forecast Aion produces; **give
agents the store and developers the verbs (H9, H10)** — four MCP parameters,
two new tools, and four re-exports, after which all three front doors reach
the same product; and **work the surprises audit (§2.5) as a backlog** —
sixteen of twenty-two surprises are communicated nowhere, and nearly all of
them are one typed `SupportReason` each.

---

## 4. What is genuinely good

Preserve these through any refactor.

- **The snapshot as a data model, not a convention.** `Snapshot` filters at
  construction, logs every read, and returns an `access_summary()` that lands
  in the artifact. Enforcing it at the type level rather than in review
  discipline is the right call and it is correctly executed for the target
  series.
- **The leakage lint.** `tests/test_leakage_lint.py` walks the AST of the
  guarded modules and makes a direct file read a CI failure with a small,
  explicit allow-list. This is the correct way to keep a structural guarantee
  structural. (Extend `GUARDED_MODULES` to `covariates.py` — see H7.)
- **The leakage-trap benchmark.** 40/40 clean for Aion against 13/35 leaking
  for the control, exact McNemar p = 0.00024, with an `oracle-leak` arm
  proving the trap detects real leakage and four control runs caught
  transcribing post-cutoff values verbatim. A measured guarantee, not a
  claimed one — and the write-up is honest about the ceiling being optimistic
  by construction.
- **The covariate vintage gate, verified.** An honest covariate is admitted
  on measured fold lift; a same-day-published one and one published after
  the cutoff are both refused with the correct diagnosis and the forecast
  falls back cleanly. The behaviour is right even where the guarantee is not
  yet structural.
- **Abstention that names its own cure, and does not bend.** `support.py:64-74`
  computes `max_supportable_horizon` and puts "retry with horizon 2 or less"
  in the recovery actions. Eight escalating rephrasings against a 12-point
  series all still abstained. This is the best UX in the codebase and the
  clearest expression of the product's thesis.
- **The determinism seam.** Wall-clock enters only through a `Clock`;
  randomness is seeded from a content hash and the seed is written into
  evidence; `content_id` covers the runtime version. Two runs under a pinned
  clock are byte-identical by `diff -r`, and the only difference under the
  system clock is `created_at`.
- **The conformal machinery.** `conformal_quantile` taking the
  `ceil((n+1)p)` order statistic instead of interpolating; `_isotonic`
  enforcing non-decreasing width across leads; the running maximum enforcing
  ordering across levels; the explicit refusal to widen pooled residuals by
  `sqrt(step)` with the reason written down. This reasoning is better than
  most published forecasting code.
- **`dense_selection_origins`.** The distinction that overlapping folds are
  legitimate for *comparison* and illegitimate for *calibration* is subtle,
  correct, and rarely made.
- **The decision model, and `decide`'s arithmetic.** Retiring bare `correct`
  for realised utility, regret against the best *feasible* action in
  hindsight, and separate ex-ante optimality is the right shape — "a costly
  precaution can be rational when the adverse event never occurs" is exactly
  what most tracking systems get wrong. The expected-utility computation and
  the `c_alert/(c_alert + c_miss)` alert threshold both check out under
  hand-calculation.
- **The constraint admission rule.** Bounds are admissible, pinned values are
  not, and a bound the training window already breaches is rejected with the
  violating timestamps named. The right seam for caller knowledge — C4 is a
  bug in the projection, not in the rule.
- **Input handling breadth.** CSV, TSV, JSON, JSONL, gzip, and Parquet all
  work, including a European export with semicolon delimiters and
  decimal commas, detected and disclosed rather than guessed. Leap days and
  arbitrary week-start conventions are handled correctly — a weekly series
  starting on a Wednesday forecasts Wednesdays.
- **`uvx --from . aion` works exactly as `docs/quickstart-mcp.md` claims** —
  the package builds from a clean checkout and runs in one command. The
  quickstart's install step is the one first-run instruction in the docs
  that needs no correction.
- **The verifier fails closed.** `verify_or_raise` raises
  `CLAIM_VERIFICATION_FAILED` with the full violation list; there is no
  warn-and-continue path, and `runtime.py:344` applies it to Aion's own
  forecast lineage, not only to callers'. The gaps in H1 and H5 are about
  *what* it checks, never about whether it can be bypassed.
- **Error quality where it exists.** `MISSING_COLUMNS` returns the available
  columns, the missing ones, and a repair option that names the exact next
  tool to call. That is the standard the other 31 codes should meet.
- **Disclosure as a habit.** The `known_time_assumed` assumption on every
  plain-CSV run; `capability_notes` explaining that a TSFM tier was eligible
  but absent, deliberately routed away from `warnings` so it cannot downgrade
  support; the quantile-collapse note; `docs/concepts.md`'s "Current
  methodological limits". The instinct to explain rather than hide is
  consistent throughout, and `docs/data-format.md` even documents the
  month-start restriction that H19 turns into a bad error message.
- **Experimental surfaces are gated.** The four planner tools appear only
  behind `AION_EXPERIMENTAL_PLANNER=1`, so the `aion_route` /
  `aion_compile_task` confusion the review looked for does not exist by
  default.
- **443 tests in 19 seconds plus 85 offline benchmark tests**, with real
  temporal tests — `test_snapshot_structurally_excludes_future_known_rows`,
  `test_forecast_as_of_replay_uses_only_prior_data`,
  `test_verifier_rejects_post_as_of_artifact`,
  `test_join_as_of_returns_vintages_not_final_values` — rather than claims.
  The gaps identified above are gaps in coverage, not an absence of rigour.
