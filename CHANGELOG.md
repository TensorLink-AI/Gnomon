# Changelog

## Unreleased

- **Artifact ids cover the runtime version: a different build is a
  different answerer.** Every artifact id payload (forecast, and the
  investigation/decision/monitor/anomaly macros) now includes
  `versioning.RUNTIME_VERSION`, and the planner's step cache keys it too.
  Content ids previously named only the question — same file, same task,
  same id — so after a code fix the stale artifact kept answering under
  the fixed build's identity: observed live when `decide` served the
  pre-fix threshold artifact (`P(above 340) = 0.61` on disk, 0.0714 from
  every fresh run) because first-write-wins kept the old file. Artifacts
  stay immutable — a new build writes a new id rather than overwriting —
  and `artifact.json` carries the stamp (`runtime_version`, additive).
  `gnomon_get_artifact` adds a `runtime_note` when serving an artifact
  produced by a different or pre-stamp build, since agents are told to
  quote artifacts verbatim. **Ids change once per release** (this is the
  point); tracking rows keep their stored ids and are unaffected. Goldens
  are updated by textual patch — id and new field only — so their
  3.12-generated float bytes stay byte-exact on CI; `__version__` and
  `gnomon capabilities` now read the same constant.

- **Every caller-settable parameter is classified, and moving an evidence
  rule now always leaves a trace.** `contracts.PARAMETER_AUTHORITY` tags
  each front-door parameter as *intent* (what the caller wants — free),
  *data* (what the caller's data is — validated and disclosed), or
  *epistemic* (what counts as evidence — priced), and
  `tests/test_parameter_authority.py` walks the CLI parser and the MCP
  schemas so a new parameter cannot reach a front door unclassified.
  The one unpriced epistemic knob is now priced: a
  `minimum_baseline_improvement` below the default 0.02 weakens the
  mandated-baseline gate, so the run carries a typed
  `nonstandard_evaluation` reason and its support is capped at
  `conditionally_supported`; above the default it is disclosed without a
  cap, and negative values were already refused
  (`INVALID_MINIMUM_IMPROVEMENT`). `EPISTEMIC_TRACES` records, per
  epistemic parameter, exactly where the artifact discloses the deviation
  — repair, forced ensembles, best-effort, and the context lanes were
  already priced; the table makes the promise auditable.

- **Artifacts record the channel the data arrived through.** The task
  block now carries `provenance: "inline"` when the observations were
  typed by the caller through the MCP `observations` channel (and
  `"store"` for bitemporal-store reads); file runs serialise
  byte-identically, with no key. Inline rows were already validated,
  fingerprinted, and repaired exactly like a file — but a file at least
  existed outside the conversation, and a reader weighing the numbers is
  owed the difference: an agent cannot invent a forecast, and until now
  nothing recorded that it may have invented the history. Every tool
  response built on inline rows (observations, covariates, actuals, or
  context events) also carries the fact in
  `support_assessment.assumptions`. Support is *not* downgraded — the
  caller's file was always the caller's word too; the fix is that the
  channel is visible, not that one channel is pretended trustworthy.
  (Also fixed here: the schema-inference branch of the MCP wrapper was
  overwriting assumptions accumulated before it, so channel notes and
  inference notes now compose.)

- **A restricted candidate pool is disclosed and cannot share an id with
  an open contest.** `--candidates` runs now carry a
  `candidate_pool_restricted` support reason naming the pool (support is
  not capped — the mandatory baselines compete regardless), and
  `statistical_candidates` joins the config fingerprint, so a restricted
  run and an open run over the same file no longer collide on one
  `forecast_id` under first-write-wins. A pool restricted via
  `gnomon.yaml` is disclosed identically.

- **Structural event failures are named, not blamed on the source.**
  `events_excluded` used to report every structurally invalid event as
  "no verifiable source" — an event rejected for a bad `expected_shape`
  was told to fix its source. Exclusion reasons now quote the actual
  contract violations. Relatedly, `parse_mapping` accepts the covariate
  mapping as a list of entries alongside the documented comma-joined
  string: an agent handing a JSON array is guessing the unambiguous
  thing, and the previous behaviour was a raw `AttributeError` with no
  repair path.

- **Context-event proposers may nominate an effect shape.** The event
  attribute `expected_shape` (`level` | `decay` | `ramp`) narrows the
  ablation's shape contest to the nominated shape — one comparison
  instead of three, strictly *less* room to look good by luck — and the
  nominated shape must still beat the history-only baseline on identical
  folds or the event is excluded outright, never silently switched to a
  shape that fits. Conflicting nominations across events cancel and the
  full contest runs. The artifact records `nominated_shape` beside
  `effect_shape` and `shape_scores` (absent when nothing was nominated,
  so existing artifacts are unchanged); an unknown shape is a structural
  contract violation named by `validate_context_event`. The proposer
  ledger can attribute shape-nomination skill later without a schema
  change, since the nomination rides in the admission evidence.

- **Threshold-crossing probabilities describe the published quantiles
  again.** `threshold_analysis_stage` recentres the backtest residual
  cloud by its own median before comparing it to the threshold, pinning
  the cloud's location to `point + centre_shift` — the published q50 —
  under either recentring policy. Previously the cloud was shifted by
  `centre_shift` alone, which was correct only while recentring was on;
  once fold-starved runs began suppressing recentring (`centre_shift` = 0)
  the probabilities kept the model's median backtest error while the
  intervals discarded it, and the README's own monitor scenario published
  `P(above 340) = 0.61` in the same artifact as `q80 = point + 6.1`
  (`first_timestamp_point_above: null` beside a per-step exceedance
  probability of 0.61). On that scenario the probability is now 0.0714,
  coherent with the quantiles beside it; the 20× miss-cost alert still
  fires, now for the right reason (the cost ratio sets the alert bar at
  0.048). Fold-rich recentred runs are unchanged up to the difference
  between the per-lead and pooled medians. Every `monitor` and `decide`
  payload on short-history series inherits the fix. Found by an agent
  dogfooding the README (`docs/agent-dogfood-review-2026-08.md`, F1).

- **`evaluate_threshold_risk` runs.** The plan operator passed
  probability-keyed residual quantiles where `threshold_analysis_stage`
  expects step-keyed spreads, so every invocation raised `KeyError: 1` —
  surfaced through plan execution as an unrepairable
  `OPERATOR_ERROR: "1"`. It had no test and can never have completed. The
  spreads are now read off the forecast rows themselves
  (`q50 − q10`, `q50 − point`, `q90 − q50` per row), so its probabilities
  agree with the quantiles printed beside them under whatever recentring
  policy produced the rows (F2 in the same review, now tested).

- **`--cost-ratio` is understood as a guess for the cost pair.** The
  README frames monitor costs as a ratio ("costs us 20x a false alarm"),
  so `--cost-ratio` is the natural first flag to try; lexical distance
  cannot map it to `--alert-cost`/`--miss-cost`. The flag-synonym table
  now supports one-to-many replacements: the structured error suggests
  both flags and the `rename_flag` repair carries them as arguments
  (a ratio of R is `--alert-cost 1 --miss-cost R`). `--cost`, `--costs`,
  `--false-alarm-cost`, and `--penalty` are mapped alongside.

- **The tool surface infers the schema the way the CLI does, additive.**
  `gnomon forecast data.csv` has worked without flags since v0.4 — exactly
  one column parses as timestamps, exactly one other as numbers, every
  inference disclosed — but the MCP tools, the surface the README calls
  preferred for agents, still hard-required `time_column`, `target_column`,
  and (for forecasts) `horizon`. Now the same eleven data-reading tools the
  CLI infers for (`gnomon_inspect`, `gnomon_forecast`, the four verb
  macros, `gnomon_route`, `gnomon_preflight_context`, and the three
  covariate tools) accept a bare `input` (or inline `observations`):
  missing columns are filled only when the file leaves no choice, the
  inference rides in `support_assessment.assumptions` (or a top-level
  `assumptions` for payloads without results), and `gnomon_forecast`
  additionally defaults `horizon` to one seasonal period of the inferred
  grid, disclosed the same way. Ambiguity still refuses — now with
  `AMBIGUOUS_SCHEMA` errors that speak tool-parameter names
  (`target_column`, candidates listed, `target_column: "auto"` offered for
  forecasts) instead of CLI flags an MCP client cannot pass.
  `store:<dataset>` inputs still require explicit columns (no header to
  infer from), and `gnomon_ingest` never infers: a write to the store
  under guessed columns would persist the guess. Every previously valid
  call is untouched — this only widens what is accepted — and explicit
  arguments disclose nothing. The Hermes plugin follows suit: its schema
  copies drop the same `required` entries and its handlers forward
  `--time`/`--target`/`--horizon` only when supplied, letting the CLI it
  wraps do the inferring and disclosing.

- **General frequencies: any whole-second sub-daily step.** The named grid
  is now a set of defaults rather than the boundary. Inference accepts any
  strictly regular series — one unique spacing — at a whole-second step
  shorter than one day, canonicalised as `<N>s`/`<N>min`/`<N>h` (`60s` is
  `min`, `120min` is `2h`), and the same codes are accepted explicitly via
  `--frequency` and the tool schemas (enum → pattern). Default seasonal
  periods derive from the one rule the curated table already encoded — the
  daily cycle when it fits in 288 lags, else hourly, else a minute cycle —
  which reproduces every hand-picked value exactly. `AMBIGUOUS_FREQUENCY`
  now means what it says: its message and details distinguish "spacing
  varies" from "regular but unrepresentable step" (sub-second, or a day or
  more without being exactly `D`/`W`/`MS` — a fixed 48-hour duration and
  "every second calendar day" diverge at the first DST transition, so
  Gnomon refuses to guess). An unusual step with gaps also stays a refusal:
  it is indistinguishable from a heavier grid with jitter.

- **`--best-effort`: an abstention with numbers attached (default off).**
  When the evaluation abstains for lack of history — the CiK bucket where
  six monthly points meet a seven-step horizon — the run can now publish a
  clearly labelled naive fallback instead of empty results, for callers
  that must have numbers: the last observed value carried flat, with
  random-walk intervals scaled from the history's dispersion. Nothing about
  it claims accuracy, three ways: support `best_effort` (additive enum
  value), a verbatim `NO RELIABLE FORECAST` warning beside the abstention's
  original reasons (the supportable horizon included), and a descriptive —
  never predictive — lineage claim, so the verifier's calibration gate is
  unreachable from a fallback. The support assessment stays `inconclusive`
  with the same recovery actions as the plain abstention. Flag-off runs are
  byte-identical, IDs included (the payload carries the flag only when on).

- **Frequency grid widened: `s` (1 second) and `10min` (10 minutes).** The
  CiK abstention analysis found 45 task-seeds refused with
  `AMBIGUOUS_FREQUENCY` on data with exactly one unique spacing across the
  whole series — solar irradiance at a 10-minute step, sensor channels at a
  1-second step. Those refusals came from a missing grid entry, not from any
  property of the data; inference on a perfectly regular series should never
  be called ambiguous. Both codes flow through inference, validation, repair
  snapping, season defaults (`s`: 60, a minute cycle; `10min`: 144, a daily
  cycle), the CLI, and the MCP/registry/Hermes tool schemas. Aliases: `S`,
  `1s`, `sec`, `second`; `10T`, `10m`. Regular data at a spacing the grid
  still does not carry (e.g. 10 seconds) keeps failing loudly.

- **Per-proposer calibration ledger** (tracking schema **5**; additive
  tables, existing stores migrate by creation). Context-event proposals
  now leave a scoreable record: `register_artifact` accepts the run's
  `context_events` and writes `event_proposals` / `event_admissions`
  rows joined to the gate verdicts the artifact already recorded (CLI
  `--project` runs and the MCP forecast tool pass them automatically);
  the pipeline persists the history-only point path as
  `enrichment_counterfactual` evidence when an enrichment is admitted
  (previously computed during adjudication and discarded); and
  `submit_actuals` resolves `event_outcomes` with the realised lift —
  error(counterfactual) − error(published), in WAPE, attributed
  set-level because the gate admits event sets. `gnomon track
  proposers` and the `gnomon_proposer_skill` MCP tool report skill
  shrunk toward no-skill priors (k = 10 pseudo-observations; hit rates
  toward 0.5), so small-n cells cannot outrank measured ones. Proposal
  identity is content-addressed from the *claim* (type, window, scope,
  source) and version-independent, so ledgers survive upgrades;
  `parse_context_response` accepts a caller-supplied `proposer`
  identity and discards any model-written one (the impersonation
  channel), recorded under `attributes.proposer` only when given.
  **No proposal earns forecast influence from these numbers** — the
  ledger is the measurement substrate the news-regime design
  (docs/design/news-regime.md, mechanisms c–d) requires before any
  influence lane may be built; enrichment-free artifacts are
  byte-identical.

- **TSFM tier truthfulness fixes** (no artifact changes): the sandbox
  root no longer resolves to the current working directory
  (`Path("")` is truthy), so sandboxes land in the documented
  `~/.cache/gnomon-tsfm-venvs` and `gnomon capabilities` stops
  answering per-cwd; sandbox workers now load Hub weights at exactly
  the pinned revisions the forecast id records (`resolved_weights`
  travels in the worker request; a missing pin is a refusal, and
  FlowState's movable `r1.1` branch pin is replaced by its commit);
  `capabilities()["models"]["tsfm"]` includes ready sandboxes instead
  of reporting `[]` after a successful `gnomon tsfm install`; and
  `models.tsfm.candidates` actually restricts the competing pool
  (it was parsed, documented, and never passed to `evaluate`).

- **Short-history guardrail** (always on, degraded runs only; no flag —
  the change is that under-powered evidence is no longer acted on as if
  it ranked anything). Below 2 disjoint selection folds the selection
  margin rises to a measured 75% single-fold improvement bar: zero of 50
  near-martingale 30-point benchmark series produced a spurious win that
  large, while a plain linear trend clears it easily — so noise stops
  winning and deterministic structure still can. Candidates that miss
  the bar stay in the artifact as evidence with a
  `selection_underpowered` typed reason and `selection_fold_count` in
  sensitivity; a candidate that clears it is disclosed as a single-fold
  selection. Degraded runs also centre quantiles on the point path
  (`point_bias_correction` = 0, disclosed as
  `point_recentring_suppressed`): the median-residual recentring at ≤ 2
  folds measured as a ~1σ shift in a coin-flip direction. On the
  pre-registered 50-series benchmark this took gnomon-pure from
  MSE 7.42 / MAPE 2.37% to 2.995 / 1.71% (`last_value` floor:
  2.56 / 1.61%) and pooled q10–q90 coverage from 63.7% to 79.1% against
  the 80% nominal (`results/short-history-guardrail/`, three registered
  iterations with falsifications reported). Runs with ≥ 4 rolling
  origins are byte-identical, IDs included; the three degraded goldens
  are deliberately refreshed.

- **Verifiable future-context events** (`gnomon.future_context`), behind
  `context.future_events` (default **off**). Two typed event classes whose
  admission is by textual verifiability instead of fold ablation, for
  events that are structurally untestable on history (future-dated, no
  overlap with the observed window): `constraint:*` bounds parsed
  deterministically from a quoted source span and projected onto the
  emitted quantiles, and `override:*` stated states (value from the text,
  never estimated) that set their window's steps with boundary-widened
  intervals. The proposer never supplies a number that is applied; a
  claimed bound must re-parse from the span; a bound the recent history
  already violates is rejected as suspect. Fold-testable events still go
  through the ablation gate, and a fold-tested failure stays rejected.
  Disclosure: influenced forecasts report the new (flag-gated, additive)
  support value `context_trusted`, record the history-only counterfactual
  in `future_context_applied` evidence, and carry the admitted events in
  the artifact-ID payload — absent when the flag is off, so every existing
  ID and golden artifact is byte-identical. Motivated by the CiK result
  where the gate rejected 100% of proposed events (see
  `results/future-context-ab/HYPOTHESIS.md`, pre-registered).

- Two flag-independent consistency fixes in the caller-claims lane,
  surfaced by the same review that hardened the new lane (artifact IDs
  are unchanged by both; no golden moves):
  - Threshold-crossing probabilities now centre on the published point
    path read back from the rows. Previously they centred on the
    pre-clamp points, so a capacity cap at the threshold still reported
    near-certain exceedance of a value the published rows never exceed —
    the constraint stage was explicitly ordered before the threshold
    stage "so the threshold analysis sees the same numbers the caller
    will", and the probabilities were the one output that didn't.
  - A clamped row restates its own `point_bias_correction`. The row
    promises `pbc == q50 - point`; a clamp that moved one of them
    without restating the gap shipped a row that lied about itself.

- `gnomon context validate` discards a model-supplied `claim` attribute,
  exactly as it discards a model-supplied `source_span`. The
  caller-claims channel applies its number with no span parsing at all;
  that authority is the caller's, never the model's.

- **Multi-target batching, additive.** One invocation forecasts several
  columns of a wide file: `gnomon forecast data.csv --target hr,spo2,resp`
  (comma list) or `--target auto` (every numeric non-time column); the MCP
  `gnomon_forecast` tool accepts the same specs in `target_column`. One
  shared load pass, per-target evaluation on a thread pool (worker count
  adapts to whether the interpreter can actually parallelise the work —
  under a GIL the statistical path runs one worker, because measured
  contention made more workers slower, not faster; sandboxed-TSFM and
  free-threaded runs use `min(channels, cpus)`), one combined artifact
  reusing the `results[]` shape with one entry per target column. Each
  channel's numbers are identical to a single-target run (pinned by parity
  tests), the artifact is identical at any worker count (pinned), and a
  channel that fails to load or abstains is disclosed verbatim in its own
  result — code, message, repair options — without blocking the others.
  Single-target invocations are byte-identical to before (goldens
  unchanged); multi-target IDs hash the ordered target list, which no
  single-target payload can collide with. The TemporalBench adapter now
  batches all of a row's channels into one run and the TimeSage toolbox
  accepts a comma list of columns — same benchmark numbers, fewer
  invocations (measured 4.0x wall-clock on a 240-row 6-channel file where
  invocation overhead dominates; parity within noise when backtests
  dominate).
- **Brief output mode, additive.** `gnomon forecast --brief` (CLI) and
  `format: "brief"` (MCP) shrink the response to the q50 path with one
  q10–q90 interval per step, the selected model, and — verbatim, never
  summarised — the support state, every warning, abstention reason,
  recovery action, and disclosure; an abstention serialises the same
  structured support assessment as full mode, and tests pin that nothing
  epistemic can be dropped. The full artifact is written to disk
  unchanged, and the default stdout format is untouched. Brief stdout is
  compact JSON (no pretty-printing); a 6-channel horizon-24 response
  shrank from ~46 KB to ~28 KB, a single-channel horizon-7 one from
  ~5.7 KB to ~3.1 KB.
- **Discoverability, additive.** `AMBIGUOUS_SCHEMA` errors now include a
  minimal working invocation built from the actual file (only the refused
  flag needs a value; everything else is inferred) plus the candidate
  columns, and offer `--target auto` when the ambiguity is the target.
  Unrecognized flags get a nearest-valid suggestion (`--column` → "did
  you mean --target?"), via a synonym table plus a typo fallback, in the
  message, `details.flag_suggestions`, and `repair_options`. README and
  `docs/cli-reference.md` lead the forecast examples with the minimal
  invocation and state which flags are inferred and when inference
  refuses; `gnomon capabilities` reports `multi_target_batching` and
  `brief_output` under `features` plus a machine-readable
  `forecast_surface` block.

- **Renamed from Aion to Gnomon, and bumped to 0.5.0.** The old name
  collided with AION (Zhan et al., arXiv:2605.25045) and could not be
  claimed on PyPI. Every public identifier moved: distribution
  `aion-forecast` → `gnomon-forecast`, import package `aion` → `gnomon`,
  console script `aion` → `gnomon`, MCP tools `aion_*` → `gnomon_*`,
  environment `AION_*` → `GNOMON_*`, default output `aion-output/` →
  `gnomon-output/`, config `aion.yaml` → `gnomon.yaml`, image
  `ghcr.io/tensorlink-ai/aion` → `ghcr.io/tensorlink-ai/gnomon`, and the
  repository itself. **This breaks the frozen v0.2 tool set**, deliberately
  and without aliases — serving `aion_*` would have kept the name the
  rename existed to remove. Porting a v0.4.0 client is a prefix
  substitution: no tool input, response envelope, artifact layout, support
  value, or error code changed. Golden artifact IDs move because they are
  salted with the runtime version, not the project name, and the version
  bumped. See [the rename record](docs/rename-impact-inventory.md).

- Temporal-leakage trap family (`benchmarks/leaktrap/`), with results. On 40
  generated trap tasks where reading past the cutoff is worth ~78% of the
  honest ceiling, a GLM-5.2 control told plainly that "a value is only
  knowable on or after its publication date" leaked on **13 of the 35 tasks
  it answered** and reproduced post-cutoff values verbatim on **4**. Gnomon
  through the snapshot path: **0 of 40**, with the no-read-past-cutoff claim
  proven **40/40** from each run's own `snapshot_access` evidence rather than
  asserted. Exact McNemar p = 0.00024. See
  `docs/leakage-trap-results-2026-08.md`, including why the accuracy columns
  do *not* carry the finding.
- Adaptive-conformal state, additive and bitemporal. The tracking store gains
  a `conformal_adaptation` table and `record_coverage_outcome` /
  `coverage_outcomes` / `adapted_alpha`. It is an append-only log rather than
  a mutable current level, and each row carries the `known_time` of the
  outcome that caused it, so `adapted_alpha(..., as_of=T)` is a fold over the
  rows known by T: adding an outcome tomorrow cannot change what a replay of
  yesterday reports, and insertion order is irrelevant. Scoring feeds the log
  automatically; nothing reads it into a published interval yet.
- `benchmarks/run_all.py` gains `--output-root`, so one config can be run in
  two environments (with and without TSFM sandboxes) without editing it.
- **Additive, with a golden refresh.** Forecast rows and `forecast.csv` now
  carry nine quantile levels (q05, q10, q20, q30, q50, q70, q80, q90, q95).
  `q10`/`q50`/`q90` keep their exact meaning *and their exact values* — they
  are the same order statistics of the same residuals, fitted the same way,
  verified across randomised cases and by the goldens, whose only
  non-additive change is a new note. The `forecast.csv` header keeps its
  first six columns in order and appends the rest. Where the residual sample
  cannot resolve adjacent levels they report the same number; that is
  disclosed as a note rather than left to look like a defect.
- `evaluate()` accepts `selection_loss`: `"wape"` (default, unchanged) or
  `"pinball"`, the proper scoring rule for a quantile, scored on fold *i*
  using only residuals from folds before it and reusing the fold forecasts so
  it costs no extra fits. The default is unchanged because the measurement
  does not support changing it —
  `docs/selection-loss-measurement-2026-08.md` records that across 50 real
  series the pinball-selected arm scored *worse* held-out pinball (0.891 vs
  0.871) than the WAPE-selected arm, losing on its own metric.
- Context events may carry typed numeric claims. A `min` or `max` bound in
  `ContextEvent.attributes["claim"]` (reserved `constraint:` event_type
  namespace) is projected onto the emitted quantiles — monotone, so it
  cannot reorder them, and idempotent. A claim that supplies a *value*
  rather than a bound is refused, and the refusal names the admissible route
  (a covariate, where Gnomon estimates the coefficient on identical folds).
  Bounds the training window already breaches are rejected with the
  violating timestamps rather than enforced. Each run with claims emits a
  `constraint_applied` evidence record. No new dataclass; runs without
  claims are byte-identical.
- `assess_context()` accepts `shrink`: continuous admission via an
  empirical-Bayes factor λ = max(0, 1 − (standard error / mean)²) applied to
  the measured effect, pinned to zero below 0.1. λ is **always computed and
  disclosed** as `context.shrinkage`; applying it is off by default, and
  `context.admitted` stays exactly `λ > 0` in that mode so the frozen boolean
  keeps its meaning. λ governs strength only — eligibility, fold
  availability, candidate fit, and coverage stay hard vetoes, because no
  measured improvement makes a leaking event admissible.
  `docs/shrinkage-admission-measurement-2026-08.md` records why the default
  is off, and here the evidence is significant *against* rather than merely
  absent: across 120 planted-event series, shrinkage produced the better
  held-out forecast on 5 of the 28 series where the arms differed
  (p = 0.0009). The gate's three strength conditions have already removed the
  candidates whose gains are noise, so shrinking on top discards a median 36%
  of effects independently established as real.
- Context effect shapes. The intervention model grows from a single level
  shift to `level`, `decay` (geometric, for a pull-forward) and `ramp`
  (linear build). The shape is chosen by the *same* identical-fold ablation
  that decides admission, never by the caller, and both the winner
  (`context.effect_shape`) and every shape's fold score
  (`context.shape_scores`) are disclosed. Shapes are normalised to deliver
  the same measured total, so they differ in timing rather than magnitude
  and `decay` cannot win by simply being smaller. On planted data the
  ablation recovers the true shape 3 times out of 3.
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
  decided the outcome. `gnomon.multivariate.forecast_var` is removed;
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
  Found by running AnomLLM's `trend` dataset: Gnomon flagged nothing on
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
  `gnomon tsfm install` command — a fresh install no longer silently hides
  the foundation-model tier. Notes render in `summary.md` as `- Note:`
  lines.
- README: the sandboxed TSFM tier (Chronos-Bolt, Toto, Moment, Moirai) is
  documented as a first-class capability — same folds, same mandatory
  baselines — instead of a parenthetical, and the quickstart shows the
  optional install command.

## 0.4.0 — first-contact release (2026-08-01)

The beta-readiness release: real-world files work on first contact, a
fifth verb (`gnomon detect`) lands with graded anomaly detectors, joint
enrichments are adjudicated honestly, tracked evidence becomes
task-conditioned with an advisory router, and the README/docs describe
the system as it is. Content-addressed IDs are salted with the runtime
version, so all artifact IDs change with this release (inputs and
parameters hash identically otherwise); golden artifacts were refreshed
accordingly.

### Evaluated anomaly detection (`gnomon detect` / `gnomon_detect_anomalies`)

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
- `gnomon track leaderboard --task ...` and
  `TrackingStore.leaderboard(project, task=...)` condition realised
  performance on the task, so accumulated evidence transfers by data
  shape instead of restarting cold per project.
- `gnomon route` / `gnomon_route`: a disclosed, advisory routing decision —
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

Real-world CSVs now work on first contact. `gnomon forecast --repair
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
- `gnomon inspect` now diagnoses instead of rejecting: `data_quality`
  reports what the file needs (`clean` / `repaired_safe` /
  `repaired_aggressive`), lists the repairs, and prints the exact
  follow-up command.
- New bundled example: `examples/filthy_requests.csv`.

### Input formats

- New always-on formats: `.tsv`, `.json` (array of objects),
  `.jsonl`/`.ndjson`, and gzip-compressed text inputs (`.csv.gz`, …).
- `.xlsx` behind a new `excel` extra (`pip install 'gnomon-forecast[excel]'`).
- Semicolon/tab/pipe-delimited "CSV" detected under repair when the header
  provably names the mapped columns (disclosed as `delimiter_detected`);
  non-UTF-8 files fall back to Windows-1252 under repair (disclosed as an
  `encoding_assumed` assumption; strict mode raises `INVALID_ENCODING`).
- `gnomon capabilities` reports the full input matrix.

## 0.3.0 — the temporal execution harness (2026-07-31)

Gnomon grows from a forecasting engine into a temporal execution harness:
an agent supplies an objective; Gnomon compiles it into validated,
snapshot-bound execution and returns typed, evidence-linked conclusions —
or a structured abstention. **Every v0.2 tool, CLI command, and artifact
schema keeps working unchanged** (see `COMPATIBILITY.md` for the frozen
set and each amendment).

### New verbs

- `gnomon investigate` / `gnomon_investigate_change` — what changed?
  Changepoints, regime shift vs transient, anomalies, and ranked
  *associational* explanations (concurrent events, cross-series
  precedence) with residual uncertainty. Never returns a cause.
- `gnomon decide` / `gnomon_decide` — what should we do? Exceedance scenarios
  from an evaluated forecast, feasibility and constraint checks, expected
  utility — or, without utilities, the feasible-action comparison as
  `conditionally_supported: missing utility inputs`.
- `gnomon monitor` / `gnomon_monitor` — when should we intervene? Sequential
  exceedance risk per step and an alert-cost-aware rule (cost-optimal with
  alert/miss costs).

### Bitemporal core

- Every observation carries `valid_time` and `known_time`; all execution
  reads through a `Snapshot` that structurally cannot serve rows published
  after its `as_of`, and logs every read.
- `gnomon ingest` appends revisions (corrected files become new vintage
  rows); `gnomon store list` inspects datasets; `store:<dataset>` inputs.
- `gnomon forecast --as-of <instant>` replays any historical moment; the
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
- `gnomon status` — pollable open forecasts, due horizons, unresolved
  decisions, realised performance (descriptive, never causal).
- `gnomon eval episodes` — trap-family episode suite (temporal leakage,
  invented numbers, abstention traps, regime breaks) with mechanical
  graders and pass^k, feeding `gnomon eval compare`.

### Experimental (gated behind `GNOMON_EXPERIMENTAL_PLANNER=1`)

- `TemporalPlan` DAG IR, deterministic validator, executor with step
  checkpointing / content-addressed caching / deterministic replay, and a
  bounded two-round repair loop. `gnomon plan compile|validate|execute` and
  the `gnomon_compile_task` / `gnomon_validate_plan` / `gnomon_execute_plan` /
  `gnomon_get_run` tools. Macros remain the default path.

### Integrations

- New tools: `gnomon_get_artifact`, `gnomon_explain_run`, `gnomon_status`,
  `gnomon_resolve_outcome`; Hermes plugin exposes the three new macros.
- Quickstart: `docs/quickstart-mcp.md`; bundled messy example datasets.

## 0.2.0

Forecasting engine: evaluated forecasts with abstention, covariate and
context-event ablation, tracking store, TSFM sandboxes, MCP server,
Hermes plugin, `gnomon eval compare`.
