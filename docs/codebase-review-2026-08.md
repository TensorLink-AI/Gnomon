# Aion codebase review — UX and temporal correctness

**Date:** 2026-08-02 · **Reviewed at:** branch `clean/harness-integrity` (PR #31)
· **Runtime version:** 0.4.0 · **Method:** code read plus execution — the
README quickstart, two persona workflows, four adversarial inputs, and the
full test suite (443 passed, 1 skipped).

---

## 1. Verdict

**The contract is upheld in its architecture and broken in its plumbing.**

The design is the real thing. The snapshot is a genuine data-model
guarantee: `Snapshot.__init__` filters on `known_time <= as_of` before any
caller holds the object, so for the target series a backtest structurally
cannot read the future, and the access log proves what it touched. Model
selection is a real backtest against mandated baselines with an improvement
margin. The abstention messages name the horizon that would succeed. The
monitor's alert rule is the correct decision-theoretic threshold,
`c_alert / (c_alert + c_miss)`, derived from the forecast distribution
rather than a heuristic. Replay is deterministic — the same `--as-of` run
twice produced the same content-addressed `forecast_id`. None of this is
decoration.

But three of the headline claims do not survive contact with the code.

**"Aion owns every number" fails in four places** where the published point
path and the published interval come from *different models*. A TSFM that
wins selection and then fails at final prediction falls back to the
strongest baseline while keeping the TSFM's residuals
(`pipeline.py:280-287`); an admitted covariate model replaces `points` and
`residuals` but not `residuals_by_lead` (`pipeline.py:499-504`); a caller
constraint clamps `q10/point/q50/q90` and leaves `q05/q20/q30/q70/q80/q95`
untouched, producing a crossed distribution where `q80 > q90`
(`constraints.py:170`); and the published `point` is the raw model output
while every quantile is bias-corrected, so in the README's own example the
"forecast" of 332.7 sits at roughly the 30th percentile of Aion's own
predictive distribution. Each of these is a number the user reads as
authoritative that no single model actually produced.

**"Leakage is structural, not behavioural" is true for the target series and
not true for covariates.** `CovariateDataset._snapshot()` builds its view
with `as_of=None` (`covariates.py:57`) and relies on every call site to pass
a correct cutoff; covariate reads are never written into the
`snapshot_access` evidence, so `lineage.py:102` computes `max_known_time`
from the target series alone and the verifier's `TEMPORAL_LEAKAGE` check
cannot see a covariate published after the cutoff. The per-fold cutoffs that
*are* passed are correct — this is a missing guarantee, not an active leak
in the current call graph — but "structural" is the wrong word for it today.

**And the bitemporal store can return a wrong historical value.** Ingestion
deduplicates by *value*, ignoring `known_time` (`temporal_store.py:341-344`),
so a series revised 100 → 150 → back to 100 silently loses its third
vintage; replaying after the revert returns 150. This is reproducible with
the repo's own example files: ingesting `messy_requests.csv` (which already
contains final values) and then `messy_requests_revisions.csv` — the natural
reading of `docs/quickstart-mcp.md` §4 — builds a store whose vintage history
is *inverted*, permanently serving the preliminary 318.7 where the truth
known at that instant was 328.6. A harness whose entire premise is
"replay any past moment as it was honestly knowable" cannot ship this.

On the UX axis the strongest finding is scope, not polish: **the bitemporal
store — the differentiator — is unreachable from the MCP surface.**
`aion_forecast` exposes neither `as_of` nor `store:<dataset>`, and there is
no ingest or store tool among the 20. The vintage workflow the quickstart
teaches agents to use does not exist for agents. Below that sit two raw
Python tracebacks on documented CLI paths (`aion decide --actions`, `aion
forecast --context`), and the "machine-readable repair options on every
error" promise, which holds for 24 of 55 error codes.

Verdict: the temporal reasoning is better than most of what it competes
with, and the honest-abstention behaviour is genuinely differentiating. It
is not yet trustworthy enough to call v0.2 final, because the failures above
are all of the form the product exists to prevent — a wrong number, reported
confidently, with no signal to the reader.

---

## 2. Findings

### Critical — a number can be wrong or leaked without the user knowing

| # | Axis | Location | Finding | Fix |
|---|---|---|---|---|
| C1 | Temporal | `src/aion/temporal_store.py:341-344` | `ingest_rows` skips a row as a duplicate when *any* existing revision shares its value, ignoring `known_time`. A revert (100 → 150 → 100) drops the third vintage; replay after the revert returns 150. Verified: `as_of 2026-01-11` returns `150.0` where the truth was `100.0`. | Dedup on `(valid_time, known_time, value)`, not `value`. A vintage at a new `known_time` is always a new row even when the value repeats. Add a revert test. |
| C2 | Temporal / UX | `src/aion/constraints.py:170` | `apply_claims` clamps only `("q10","point","q50","q90")`. The six later-added levels are never projected, so a `max: 370` bound yields `q90=370.0` alongside `q80=385.3` and `q95=397.8` — a crossed distribution *and* a published number the caller declared infeasible. Verified by direct call. The clamped rows then feed `threshold_analysis_stage`, so the crossing propagates into monitor/decide. | Clamp every `q*` column plus `point`; re-assert monotonicity after projection; assert it in a test over the full `QUANTILE_LEVELS` set. |
| C3 | Temporal | `src/aion/pipeline.py:280-287` | When the selected TSFM fails at final prediction, `state.selected_model`/`state.points` fall back to the strongest baseline but `state.residuals`/`residuals_by_lead` remain the TSFM's. The baseline's point path is published with the TSFM's intervals. Only a `logger.warning` — nothing reaches `state.warnings` or the artifact. This is the exact failure the ensemble path at `pipeline.py:228-238` was fixed to avoid. | Mirror the ensemble fix: on fallback, recalibrate from the fallback model's own fold residuals, or decline and abstain. Append a user-visible warning either way. |
| C4 | Temporal | `src/aion/pipeline.py:499-504` | The adjudication winner overwrites `selected_model`, `points`, `residuals`, `coverage`, `warnings` — but not `residuals_by_lead`. `interval_stage` (`pipeline.py:600-607`) then calls `conformal_spreads(residuals_by_lead=<base model>, pooled=<covariate model>)`. Per-lead spreads and pooled spreads describe different forecasts. | Carry `residuals_by_lead` through `AdjudicationResult` and set it with the rest; fail loudly if `residuals` is replaced without it. |

### High — the agent or user is stuck or misled

| # | Axis | Location | Finding | Fix |
|---|---|---|---|---|
| H1 | Temporal / UX | `src/aion/pipeline.py:610-615`, `evaluation.py:277` | `point` is the raw model output; every `q*` is `point + median_residual + offset`. In the README example `point=332.74` while `q50=366.57` and `q30=332.43` — the headline forecast sits at ≈ the 30th percentile of its own distribution. Nothing in the artifact, `summary.md`, or docs says `point ≠ median`. | Either publish the bias-corrected centre as `point` (and keep the raw output as `point_raw`), or emit `point_bias_correction` and state the relationship in `summary.md` and `COMPATIBILITY.md`. |
| H2 | Temporal | `src/aion/evaluation.py:181-186`, `246-249` (`MIN_RESIDUALS_PER_LEAD = 8`) | With fewer than 8 residuals at a lead, that lead borrows the pooled spread. With the typical 3–4 folds *every* lead borrows, so the 14-step interval is exactly as wide as the 1-step: the README example emits byte-identical `q05…q95` for all 14 rows. Uncertainty visibly does not grow with horizon, and nothing says so. | When every lead falls back to pooled, either apply an explicit, disclosed growth model or emit a first-class note/`support` reason: "interval width is constant across the horizon; the fold count cannot resolve per-lead growth." |
| H3 | Temporal | `src/aion/evaluation.py:947` (`_pool_residuals` over `residual_origins`) | Conformal residuals are pooled from the selection folds *plus* the calibration fold — but the selected model was chosen to minimise error on exactly those selection folds. That is not split-conformal; the winner's residuals there are optimistically small. The docstring justifies the pooling by sample size without naming the selection bias. | Calibrate on the calibration fold only, or add held-out calibration origins that selection never sees. If pooling is kept for sample size, document the bias and measure it — the fold-stride note (`docs/fold-stride-measurement-2026-08.md`) is the right template. |
| H4 | Temporal | `src/aion/verifier.py:87` | The leakage check is a *string* comparison: `artifact.max_known_time > as_of`. Mixed representations (`Z` vs `+00:00`, date vs datetime, differing offsets) compare lexicographically. `2026-06-03T23:00:00+00:00` vs an `as_of` of `2026-06-04T00:00:00+02:00` is a real leak that compares as safe. | Parse both to `datetime` and compare instants; raise `SNAPSHOT_TIMEZONE_MISMATCH` when one is naive and the other aware. |
| H5 | Temporal | `src/aion/verifier.py:65-76`; `evaluation.py:984` | `UNCALIBRATED_PROBABILITY` is satisfied by the *existence* of a `rolling_evaluation` evidence record, never by its quality. A run with measured test coverage of 57.1% at a nominal 80% still emits `predictive` claims that pass verification; low coverage is only a warning, and when the run is also `degraded` that warning is masked in the support status entirely. | Gate on the measured value: refuse `predictive` claims (or force `inconclusive`) when calibration evidence reports coverage outside a stated band, and stop letting `degraded` swallow a coverage warning. |
| H6 | Temporal | `src/aion/covariates.py:57`; `lineage.py:102` | The covariate snapshot is built with `as_of=None`; the run's `--as-of` never reaches it, and cutoffs are enforced only by convention at each call site. Covariate reads are absent from `snapshot_access`, so `max_known_time` — the input to the verifier's leakage check — is computed from the target series alone. A covariate published after `as_of` is invisible to every guarantee the README advertises. | Build the covariate snapshot at the run's `as_of`; merge its `access_summary()` into the `snapshot_access` evidence; bound the final-forecast cutoff by `min(timestamps[-1], as_of)`. Add an end-to-end test with a post-cutoff covariate. |
| H7 | Temporal / UX | `src/aion/temporal.py:108-114`, `135-139` | `next_timestamp` adds a fixed `timedelta`, so timezone-aware **daily** data in any DST-observing zone fails `IRREGULAR_TIME_GRID` at the transition — verified with Europe/London. The suggested repairs are both wrong: "fill the missing period" (nothing is missing; the day is 23 hours) and `repair=aggressive`, which reports `snapped: 31` — it would shift every post-transition midnight by an hour — and then aborts with `EXCESSIVE_REPAIR`. There is no working path for daily local-time data in a DST zone. | Advance calendar-aware for `D`/`W`/`MS` on aware timestamps (localise → add → re-normalise), and recognise a DST step as regular rather than offering to invent a row. |
| H8 | UX | `src/aion/cli.py:163`; `operators.py:532` | `aion decide --actions '["scale_up","do_nothing"]'` — the shape the CLI help describes ("JSON list of actions") — crashes with an unhandled `TypeError: string indices must be integers`. The MCP schema is correct (`[{name, feasible?, residual_risk?}]`); the CLI help contradicts it. Passing malformed JSON yields `INVALID_JSON_ARGUMENT` with an empty `repair_options` and no indication of which argument failed. | Validate `actions` against the MCP schema in the CLI, raise a structured `INVALID_ACTIONS` with an example, name the offending argument in `INVALID_JSON_ARGUMENT`, and fix the help string. |
| H9 | UX | `src/aion/constraints.py:62`; `context.py:70` | Context events *must* carry an explicit timezone offset; the example datasets are timezone-naive; `Claim.binds` then compares aware to naive and raises a raw `TypeError` traceback out of `aion forecast --context`. The whole context-event/constraint feature is unreachable for the repo's own example data — and there is no worked example of a context-events file anywhere in `docs/` or `README.md`. | Normalise event timestamps to the dataset's zone at validation time (or reject the mismatch as a structured `MIXED_TIMEZONES` error naming both sides). Ship a `examples/context_events.json` and document the flag. |
| H10 | UX | `src/aion/contracts.py:209` | README: "machine-readable repair options on every error." Actual: 24 of 55 raised error codes appear in `REPAIR_OPTIONS`. The entire covariate family (11 codes) has none, as do `AMBIGUOUS_FREQUENCY`, `UNSUPPORTED_FREQUENCY`, `INPUT_NOT_FOUND`, `EMPTY_DATASET`, `INVALID_JSON_ARGUMENT` — the codes a first run hits first. | Add options for every raised code and enforce it with a test that walks the `AionError` call sites (the audit is ~15 lines). Until then, soften the README claim. |
| H11 | UX | `src/aion/toolspec.py` | `aion_forecast` — the flagship tool — exposes neither `as_of` nor `store:<dataset>`, unlike `aion_investigate_change`/`aion_detect_anomalies`/`aion_decide`/`aion_monitor`, which expose both. There is no MCP tool for `aion ingest` or `aion store list`. The bitemporal store, `--as-of` replay on forecasts, and the vintage workflow taught in `docs/quickstart-mcp.md` §4 are all CLI-only and invisible to agents. | Add `as_of` and the `store:` input hint to `aion_forecast` and `aion_inspect`; add `aion_ingest` and `aion_list_datasets`. Record the additions in `COMPATIBILITY.md` as additive. |
| H12 | UX | `src/aion/toolspec.py:585`; `router.py` | `aion_route` answers "which method for this task?" with `recommendation: null` / `basis: "backtest_required"` on any cold project — and `aion_forecast` has no model parameter at all, so even a non-null recommendation cannot be acted upon. The tool's output has no consumer. | Either give `aion_forecast` a `candidates`/`model` parameter that the router's output feeds, or re-scope `aion_route` to what it actually does (capability filtering and disclosure) and say so in its description. |
| H13 | UX | `src/aion/toolspec.py:308-320`, `567` | Two decision lifecycles sit side by side with no deprecation marker: `aion_record_decision` + `aion_resolve_decision` (v0.2, "whether a decision was correct") and `aion_resolve_outcome` ("bare 'correct' is retired"). An agent reading the surface cold cannot tell which to use, and the two descriptions contradict each other. | Mark the v0.2 pair deprecated in its description text, state the replacement, and say which macro produces records for which resolver. |
| H14 | UX | `docs/quickstart-mcp.md` §4 | The documented vintage command (`aion forecast store:requests --horizon 7 --as-of 2026-06-03`) returns `unsupported` / `inconclusive`: `messy_requests_revisions.csv` carries only 10 valid dates. The step that would make it work — ingesting `messy_requests.csv` first — is not in the docs, and doing it triggers C1 and inverts the vintage history. A new user's first vintage run fails either way. | Ship a revisions fixture with enough history for the documented horizon, and make the base-file ingest an explicit step — after C1 is fixed. |

### Medium — friction, inconsistency, or a misleading presentation

| # | Axis | Location | Finding | Fix |
|---|---|---|---|---|
| M1 | Temporal / UX | `src/aion/evaluation.py:961-985` | Interval coverage is measured on a single test fold of `horizon` points and reported as a headline: "Final-test 80% interval coverage: 100.0%". With 14 points that number carries almost no information, and the `< 0.7` warning threshold is noise-dominated. | Report the count alongside the rate, or a confidence interval; suppress the headline below a minimum sample. |
| M2 | Temporal | `src/aion/temporal_store.py:456` | `assumed_known_time` is inferred as `all(valid_time == known_time)` over the data rather than recorded from ingest provenance. A dataset with real publication times that happen to be same-day is reported as assumed; a dataset mixing an assumed ingest with a real one reports *not* assumed, hiding the assumption. | Persist `assumed_known_time` per ingest and aggregate it; surface "partially assumed" rather than collapsing to a boolean. |
| M3 | Temporal / UX | `src/aion/pipeline.py:117-127`; `cli.py` (inspect) | A file whose observations are entirely in the future (2027 data reviewed on 2026-08-02) forecasts happily and `aion inspect` reports `status: valid` with no remark. For a harness built on knowledge time, "every observation postdates now" deserves a note. | Compare `max(valid_time)` to the clock in `inspect` and in the support assessment; emit an assumption, not an error. |
| M4 | UX | `src/aion/toolspec.py` (`aion_monitor`) | `monitor` does not monitor — it defines an alert policy once against a single forecast. An SRE reading the name expects a running watch. Same class of gap: `decide` returns an expected-utility comparison, not an executed decision. | Rename to `alert_policy` (aliasing the old name) or lead the description with "Defines, does not run: …". |
| M5 | UX | `src/aion/tracking.py:90-91, 106-107` | The same object reports `mape: 4.279` (percent) and `avg_wape: 0.0424` (fraction) with no units on either. | Emit both as fractions, or suffix the field names (`mape_pct`). |
| M6 | UX | `src/aion/toolspec.py` (default `output_dir`) | Artifacts default to `./aion-output` in the process CWD. The quickstart's first commands write into the user's clone. | Default to a user data directory (as the temporal store already does with `~/.local/share/aion`), with `--output` to override. |
| M7 | Temporal | `src/aion/temporal_store.py:86-87` | `_comparable` checks only `observations[0]`. A dataset whose aware/naive mix starts after row 0 raises a bare `TypeError` from the comprehension instead of `SNAPSHOT_TIMEZONE_MISMATCH`. | Check the whole set, or normalise at ingest. |
| M8 | UX | `docs/` | `--context` / `context_events_file` is a documented flag with a whole design module behind it and zero worked examples in the docs. Same for `--selection-strategy ensemble` and the `aion.yaml` config surface. | Add a context-events example to `docs/concepts.md` and an `examples/` fixture. |
| M9 | UX | `src/aion/tracking.py` (`track actuals` response) | The actuals response omits `schema_version` and any `support_assessment`, unlike every other envelope. | Wrap it in the standard envelope. |
| M10 | UX | `src/aion/temporal.py:128` | Out-of-order rows are silently sorted. Correct, but it is a mutation the caller never learns about — and the sort is the reason a genuinely unsorted file never triggers a data-quality signal. | Record a `timestamps_reordered` note in the repair log when the input order differs from sorted order. |

### Low — polish

| # | Axis | Location | Finding | Fix |
|---|---|---|---|---|
| L1 | UX | `docs/getting-started.md:14,32`; `docs/development.md:19,26` | `cd /root/Aion` — a reviewer's local path shipped in the published docs. | Replace with `cd Aion`. |
| L2 | UX | `src/aion/cli.py` (parser) | Most CLI flags carry no `help=` (`--time`, `--target`, `--horizon`, `--frequency`, `--series`), while the MCP schema documents all of them well. | Reuse the MCP descriptions as `help=` strings. |
| L3 | UX | `src/aion/contracts.py:209` | `TEMPORAL_LEAKAGE` has repair options but is never raised as an `AionError` code — it only appears as a violation inside `CLAIM_VERIFICATION_FAILED`, which has none. | Attach the options to `CLAIM_VERIFICATION_FAILED`, or surface per-violation options. |
| L4 | UX | `src/aion/pipeline.py:626-637` | The quantile-collapse note fires on the common small-sample path and reads as a defect disclosure. Good instinct; it just needs to be a typed `support_assessment` field rather than free text in `notes`. | Promote to a typed note code. |

---

## 3. Top 5 fixes before release

1. **Fix the store's dedup rule (C1) and re-ship the vintage example (H14).**
   Key on `(valid_time, known_time, value)`, add a revert test, regenerate
   `examples/messy_requests_revisions.csv` with enough history for the
   documented horizon, and make the base-file ingest an explicit documented
   step. This is the single fix that decides whether `--as-of` means
   anything.
2. **Make points and intervals come from the same model, always (C2, C3, C4).**
   One invariant, asserted in one place before `interval_stage`: the residuals
   used to build the published quantiles were produced by the model named in
   `selected_model`, and every emitted `q*` column went through the same
   projection. Three separate bugs collapse into one assertion.
3. **Close the covariate leakage gap (H6).** Build the covariate snapshot at
   the run's `as_of`, merge its access log into `snapshot_access`, and add a
   test that a covariate published after the cutoff is refused. Until this
   lands, "leakage is structural" should read "leakage is structural for the
   target series."
4. **Give agents the store (H11).** `as_of` and `store:` on `aion_forecast`
   and `aion_inspect`, plus `aion_ingest` and `aion_list_datasets`. Today the
   MCP surface cannot reach the feature the README leads with — and MCP is
   the primary front door.
5. **Stop crashing on documented paths, and make the error promise true
   (H8, H9, H10).** No `aion` invocation should ever print a Python
   traceback. Fix `--actions`, fix the naive/aware context-event comparison,
   and add `repair_options` to the remaining 31 codes with a test that keeps
   the set complete.

Honourable mention, cheap and high-value: **say that `point` is not the
median (H1)**, or make it the median. One line of documentation or one line
of code, and it removes a systematic misreading of every forecast Aion
produces.

---

## 4. What is genuinely good

Preserve these through any refactor.

- **The snapshot as a data model, not a convention.** `Snapshot` filters at
  construction, logs every read, and hands back an `access_summary()` that
  goes into the artifact. Getting this right at the type level rather than
  in review discipline is the correct call and it is correctly executed for
  the target series.
- **Abstention that names its own cure.** `support.py:64-74` computes
  `max_supportable_horizon` and puts "retry with horizon 2 or less" in the
  recovery actions. Verified end to end: an under-supported run returns a
  refusal an agent can act on without guessing. This is the best UX in the
  codebase.
- **The conformal machinery.** `conformal_quantile` taking the
  `ceil((n+1)p)` order statistic instead of interpolating; `_isotonic`
  enforcing non-decreasing width across leads; the running maximum enforcing
  ordering across levels; the explicit refusal to widen pooled residuals by
  `sqrt(step)` and the comment explaining why. The reasoning in these
  docstrings is better than the reasoning in most published forecasting code.
- **`dense_selection_origins` and the selection/calibration separation.** The
  distinction that overlapping folds are legitimate for *comparison* and
  illegitimate for *calibration* is subtle, correct, and rarely made.
- **The decision model.** Retiring bare `correct` for realised utility,
  regret against the best *feasible* action in hindsight, and separate
  ex-ante optimality is the right decision-theoretic shape — "a costly
  precaution can be rational when the adverse event never occurs" is
  precisely the thing most tracking systems get wrong.
- **The constraint admission rule.** Bounds are admissible, pinned values are
  not, and a bound the training window already breaches is rejected with the
  violating timestamps named (`constraints.py:97-116`). This is exactly the
  right seam for caller knowledge — the bug in C2 is in the projection, not
  in the rule.
- **The monitor's alert rule.** `alert_cost / (alert_cost + miss_cost)`,
  derived from the forecast distribution, with `alert_rule_basis` stated in
  the output and the uninformative 0.5 default explicitly flagged when costs
  are missing.
- **Disclosure as a habit.** The `known_time_assumed` assumption on every
  plain-CSV run; `capability_notes` explaining that a TSFM tier was eligible
  but absent, deliberately routed away from `warnings` so it cannot downgrade
  support; the quantile-collapse note. The instinct to explain rather than
  hide is consistent throughout.
- **443 tests passing in 19 seconds**, with real leakage tests
  (`test_forecast_as_of_replay_uses_only_prior_data`,
  `test_verifier_rejects_post_as_of_artifact`, the leaktrap benchmark) rather
  than a claim. The gaps identified above are gaps in coverage, not an
  absence of rigour.
