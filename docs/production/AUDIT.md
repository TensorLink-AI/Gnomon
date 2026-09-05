# Initial implementation audit and cull decisions

## Latest: iteration 20, scoped forecast comparison evidence

Added the 11-task retrospective cohort and explicit complete-horizon forecast
loss grading. MAE/RMSE/nonseasonal MASE and completion coverage accompany all-task
success/cost; missing forecasts never become zero-error successes. Existing corpus
hashes/scoring remain compatible when no forecast contract is declared.

Rejected draft Statsmodels-derived sources because their full datasets were bundled
into the agent image. Final tracked BreachBench source CSVs were checked against
pinned Prophet examples; three matched directly and temperature matched after 12
nonfinite exclusions. All selected windows have checked cadence. Source hashes,
cutoffs, transformations and limitations are in matched-retrospective.manifest.json.
Historical LLM-training contamination remains possible; no decontamination or SOTA
claim is made. Known source CSVs/Prophet are absent from the ordinary image, and
actual Naive/AutoETS work on all four public windows without Gnomon adapters.

Final suite: 3,541 passed, 13 skipped, 355.49s. Focused 169 checks passed on each
Python 3.11 and 3.13 with real containers; 25 cohort tests and 119 documentation/
boundary/progress checks passed. Ruff, compilation and whitespace checks passed.

Scoped Paracast source search also found a candidate deployment URL. Two
unauthenticated health/models requests returned 401; this does not verify serving
models. The reference warns of billable authenticated cold starts, so confirmation,
token environment-variable name and spending approval were requested. No credential
read, authenticated call, forecast POST, warmup or deployment occurred. Score stays
97/100 pending actual agent comparison and authenticated live service verification.

## Latest: iteration 19, reported-cost controls and experiment configuration

The existing matched runner now supports an optional per-arm reported-service-cost
stop. It uses the existing attempt journal, including failed, resumed and unfinished
work; unknown charges stop further dispatch. The shared driver receives the
remaining allowance privately and checks it between model/tool/phase operations.
Serial execution, zero retries and a persistent journal are required. Stopped tasks
remain explicit zero-work failures in the complete cohort. One operation can
overshoot before reporting: this is not a provider-enforced hard dollar ceiling.

Operator-filled templates bind the actual ordinary/lean/full backends, common
prompt/model controls and equal per-arm allocations. They do not enable paid
requests unchanged. No new production dependency or storage service was added.
All 22 existing context cases were checked: their oracles encode legacy
context_behavior contracts, so they cannot simply be relabelled as tool-neutral
agent comparison cases. Historical regressions are preserved.

Verification: 33 new spending tests pass; 144 focused checks pass each on Python
3.11 and 3.13 with actual containers. Full production/historical suite: 3,516 passed,
13 skipped, 360.63s. Documentation/boundary/progress checks: 119 passed. Ruff,
compilation and whitespace checks pass; no test containers remain. Score stays
97/100: actual agent comparison and deployed Paracast verification are unearned.
No paid calls, remote writes, commits, deployment or material deletion occurred.

## Latest: iteration 18, committed agent episodes

The matched path now supports durable, unchanged agent commitments before later
data is revealed. It preserves one transcript, backend and cumulative budget;
crash recovery retains partial usage and never replays an episode after a prior
start. The append-only attempt journal upgrades to schema 2 without rewriting
existing attempts. This is evaluation infrastructure, not a second production
forecast ledger. The production package and its dependencies are unchanged.

Actual container tests cover saved Python forecasts and lean-ledger predictions
followed by operator-supplied actuals and agent-requested evaluation. Three smoke
cases cover revisions, permission and bitemporal replay. All phases are graded;
later answers cannot erase earlier errors. These scripted cases establish protocol
behavior, not LLM improvement. The implementation and limits are documented in
benchmarks/workflow/MATCHED.md and the current HANDOFF.md.

Final production/historical suite: 3,483 passed, 13 skipped, 340.81s. Focused
111 checks passed on each Python 3.11 and 3.13 with real containers; 119
documentation/boundary/progress checks passed. Ruff, compilation and whitespace
checks passed; no test containers remain. Score stays 97/100, with actual agent
comparison and deployed Paracast verification unearned. No paid calls, remote
writes, commits, deployments or material deletions occurred this iteration.

Baseline: `0d3d72e`, inspected 2026-09-05. Evidence comes from executed code,
schemas, import/call tracing and tests, not the roadmap's claims.

## Baseline verification

- `.venv/bin/python -m pytest -q`: 2,991 passed, 13 skipped, 329.61 s.
- `.venv/bin/ruff check src/gnomon`: passed.
- 64,261 lines of Python source; 22 registered MCP tools, 10 default.
- Traced `mcp_server._handle -> toolspec.runner_for -> _run_forecast ->
  runtime.forecast -> pipeline.evaluate_stage -> evaluation.evaluate ->
  pipeline.predict_stage -> interval_stage -> artifacts`, and the optional
  publication, temporal-question, tracking and decision branches.

## Keep, consolidate, archive

| Area | Decision | Code-based reason |
|---|---|---|
| Snapshot/valid/known-time handling | Keep and extend | Real fold-time isolation; add local recording-time replay. |
| Adapter output conformance and baseline evaluation | Keep | Prevent invalid shapes/nonfinite predictions and unsupported numerical claims. |
| Forecast tracking/artifacts | Consolidate into revision-preserving execution ledger | Scores update/replace rows; registration rereads source data instead of frozen inputs. |
| APIAdapter and ForecastRequest/Result | Extend existing boundary | HTTP path exists but remote discovery is coupled to local names; data capabilities need end-to-end propagation. |
| Tool registration and response projections | Consolidate | 4,926-line toolspec performs materialization, caching, execution and several output transforms. |
| General run/track experimental tool profile | Candidate for removal during surface work | Broad union schemas hide operation-specific requirements; not in default profile. |
| Context dossier/transformation/publication ladders | Isolate and reduce after dependency coverage | Ordinary inference does not require these; retain regressions while separating optional execution. |
| ReasoningBench | Archived, active route removed | Catalogue already marks it retired for answer exposure, but run_all still registered it. |
| Independent numeric, leakage and adapter tests | Keep | Removing these would discard coverage for the retained production behavior. |
| Other small synthetic suites | Consolidate release selection, retain relevant cases | Many are regression instruments, not independent evidence of agent uplift. Further culls need caller mapping. |
| Curated historical releases | Preserve | Referenced by product claims and integrity tests, including negative findings. |
| Legacy design docs and version-loop protocols | Historical; consolidate current documentation later | Old direction statements conflict with the newly authorized product, and tests pin some prose. |
| User scratch/untracked files | Preserve | Pre-existing user state, not benchmark source or task-owned output. |

## Reproduced correctness findings

1. `temporal_question._PROPERTY_ALIASES` mapped average/mean to level without
   retaining a measure; `answer_descriptive_question` answered latest. Public
   describe returned 100 for 27 ones and 100 (mean 4.535714), marked supported
   and automation-eligible. Exact-measure regressions now cover this path.
2. `evaluation.evaluate` defaults to pooling selection residuals into interval
   calibration. Keep disclosed tradeoffs separate from claims of independent
   calibration; evaluate statistical performance rather than relying on seals.
3. `tracking.score_forecast` updates forecasts and replaces model_performance;
   `submit_actuals` skips scored forecasts. Revised actuals need immutable
   evaluation records, not destructive rescoring.
4. `tracking.register_artifact` reads the original input again, bypassing the
   completed run's frozen snapshot. Metadata can diverge from the prediction.
5. Router leaderboard reads lack a uniform historical cutoff. Source
   availability and system-recorded history need distinct replay semantics.
6. `agent_eval.compare_runs` excludes budget-exhausted pairs. Preserve a
   conditional diagnostic if needed, but add overall task completion metrics.

## Paracast integration source

The connected GitHub account can read `TensorLink-AI/paracast`. Read revision
`f5d3f53b17e56d8c7ed0cbae61e043b8667f236a`, specifically
`paracast_common/src/paracast_common/schema.py`, `router/app.py`, and
`router/service.py` (service dispatch still being traced).

The plain transport exposes POST /forecast, GET /models, GET /health, and
POST /feedback. Forecast requests use mode (explicit/route/ensemble), series
arrays, freq, horizon, quantiles and optional named covariates. Responses use
per-series quantile-keyed arrays and meta.models_used/request_id/notes.
This differs from Gnomon's generic point/history API adapter. No sample paths
or immutable weight revision is promised by this inspected response schema.
The deployed base URL is intentionally configurable and remains unverified.

## Iteration 2: concrete changes and limits

- Added the public callable/provider inference boundary over the existing
  ForecastRequest/Result types. Fresh fitting factories, alignment validation,
  optional batch execution and explicit-version/deterministic cache reuse are
  tested. Inference is separate from measured accuracy and action authority.
- Implemented Paracast's actual wire mapping after tracing router/service.py
  and chute.py. Discovery is deployment-owned. Median-as-point is disclosed;
  weights revisions remain unknown. Unsupported multivariate/sample/fixed-season
  requests are refused. No live deployment was contacted.
- Replaced duplicated legacy urllib/retry logic with the same bounded transport
  used by Paracast. Forecast POSTs no longer retry implicitly. Existing generic
  wire shape remains compatible; configuration URLs may not contain credentials.
- Added optional SQLite execution/outcome ledger, not a general event framework.
  Small immutable content-addressed payloads and metadata are committed together.
  Legacy mutable tracking is retained pending explicit migration/integration.
- Added system-recording-time replay to TemporalStore. Historical timestamps
  cannot be safely reconstructed from replaceable old ingest records; migration
  leaves them unknown and strict recorded replay excludes/discloses them.
- Found and fixed another provenance bug: dataset_fingerprint hashed only
  counts/latest time/row ID, so different values could share a fingerprint. It
  now hashes observation content, and snapshots have visible-vintage identities.
- Added lazy legacy Python exports so direct inference does not import the
  optional evaluation/publication/context stack. This is not yet common CLI/MCP
  isolation; those integration checks remain open.
- Golden artifact changes are limited to recorded_as_of, replay_mode,
  unknown_recorded_times and snapshot_id. Forecast numeric values are unchanged.

Iteration 2 verification: fresh full pytest passed 3,057 tests with 13 skips in
331.62 seconds. A 133-test focused run also includes the seven added progress
framework checks. Ruff and diff whitespace checks pass. Isolated sdist/wheel
build and fresh no-index/no-deps installed CLI/MCP/callable/ledger smoke pass.
This does not verify other supported Python/OS platforms or the live TSFM service.

## Iteration 3: shared execution and historical migration

- GnomonSession owns explicit provider configuration and optional ledger; Python,
  CLI infer/ledger, and MCP execution-profile calls use its same typed dispatch.
  The local HTTP integration test launches real CLI/MCP subprocesses. Agent tool
  arguments cannot load imports, resolve credential variables or replace URLs.
- Outcome writes and imports require an operator startup setting. Session import
  and direct inference remain free of optional context/evaluation stack imports.
- Historical artifact/tracking import is read-only at the source and idempotent.
  Import provenance gaps remain explicit. Model weights identities are preserved
  only when the source artifact recorded them. Missing histories are not fabricated;
  mutable old score summaries are not recast as revision-preserving evaluations.
- A sealed history.json now survives source changes/store URI inputs and drives
  legacy registration. Canonical JSON predictions remain scoreable without CSV.
- Intermediate full-suite failures caught undocumented commands/error repairs,
  missing parameter classifications and the changed artifact file set; these were
  fixed in the implementation contracts/docs rather than dropping coverage.
- Final full suite: 3,086 passed, 13 skipped, 300.30 s. Source lint and diff checks
  pass. Fresh installed wheel smoke covers old and new CLI/MCP/provider/ledger
  journeys. No live TSFM inference or supported-platform claim is implied.

## Iteration 4: shared frozen inputs and scope integrity

- Extracted dataset loading from pipeline.py into datasets.py without copying
  the evaluation/context stack. Legacy pipeline exports refer to the same loader.
  Extended the leakage lint to cover the shared loader and wide-file reads.
- Session-local DataReferences freeze file/store vintages under bounded retention;
  source edits and later database revisions cannot rewrite an existing reference.
  Recorded-time replay is exposed for store inputs and explicitly refused for
  files that lack recorded-time history. Panel selection is exact, not implicit.
- Execution-profile inspect/describe and CLI infer --input use the shared loader.
  Summaries compute exact scalar statistics over inclusive timestamp windows and
  report the selected count, unit label and window. Units are not guessed or
  converted. Forecasts cannot override frozen history and reject stale-history
  grids that would call a pre-as_of timestamp a future prediction.
- Traced ignored semantics in the legacy question compiler: unknown fields,
  custom comparison/validation dictionaries, unsupported options and duplicate
  aggregate members could survive compilation without corresponding execution.
  These now fail explicitly. Registered statistical method selection still
  reaches the capability planner, which refuses unsupported methods unchanged.
- Fixed default-horizon inheritance and explicit-future fallback parsing. An
  observed sub-question cannot become predictive just because its parent forecast
  has a horizon. Generic supported envelopes no longer claim calibrated automatic
  use; numeric policy qualification remains separate from action permission.
- Expanded parameter-authority tests through session schemas, including nested
  request fields and discriminated ledger/forecast unions. This found parameters
  absent from the old top-level-only audit and prompted explicit classifications.
- The first broad run found two expected integration misses (classifications and
  the expanded answer envelope); both were corrected with retained coverage.
  Fresh distribution smoke now exercises file-based inference and frozen summaries
  outside the checkout as well as the existing provider/ledger journeys.

Final iteration 4 verification: 3,122 passed, 13 skipped in 301.55 seconds;
131 focused checks pass. Ruff, whitespace checks and fresh installed-wheel smoke
pass. Scope/authority checks earn six points, bringing verified progress to 72/100.

Default legacy core still exposes its older ten tools; the five-tool execution
profile (with a ledger) is not the completed surface cull. Next: budgeted evaluation
from frozen vintages, cutoff-bound routing, default/experimental profile consolidation,
general temporal primitives and completion-aware matched measurements. In particular,
fold evaluation must not treat a file repaired using later rows as a historically
available prefix, nor use the final recorded-time boundary for every historical fold.

## Iteration 5: budgeted matched evaluation

- Added optional provider-neutral rolling-origin evaluation, not hidden work in
  inference. Baselines, failures and partial dispatch consume explicit budgets.
  Operator session ceilings cannot be raised by tool arguments. Cancellation/time
  limits are honestly dispatch-bound; internal remote fan-out remains unknown.
- Historical folds use frozen source vintages and independently narrowed recording
  boundaries. Missing history is unscored, not replaced with the final prefix.
  Data-dependent repairs are refused until their historical availability can be
  reconstructed. Per-fold factories remain isolated; inference cache is bypassed.
- Point metrics compare the same all-provider matched fold intersection, preserving
  requested/planned/failed/attempted counts. Declared provider versions and returned
  metadata are retained; unknown training/weights identities are never attested.
- Ledger schema 3 persists immutable full studies and execution references without
  inventing online actual revisions. Compact Python/CLI/MCP projections support
  full retrieval. v2 migration preserves existing executions and append-only guards.
- Full pytest: 3,150 passed/13 skipped in 288.84s; 183 focused checks pass. Real local
  HTTP/MCP/CLI journeys match Python cohorts. Ruff, diff checks and installed wheel
  smoke pass, including evaluation and full study retrieval. Verified score 77/100.

The legacy router is still unsafe as a historical prior: _tracking_prior reads a
mutable leaderboard and per-model summaries with no source or recorded cutoff.
Its model recommendation needs replacement, not relabelling as ledger evidence.

## Iteration 6: cutoff-bound routing and recoverable prior cull

- Replaced the mutable TrackingStore performance prior with an explicit ledger
  study route. The legacy positional route retains structural suggestions only;
  archived pre-cull implementation is `archive/legacy/router_pre_ledger.py`.
- Traced a second unsafe path: shadow outcomes use INSERT OR REPLACE and have no
  local recording-time vintage. Shadow routing now retains the explicit champion,
  with statistical diagnostics but no historical promotion authority. Its former
  implementation is preserved in `archive/legacy/adapter_promotion_pre_ledger.py`.
  No tracking databases or historical outcomes were deleted.
- New routing checks both query clocks, immutable study/execution recording times,
  exact task/provider identities and reconstructed historical inputs. Revised
  store outcomes produce a new immutable rescore; original forecasts/studies remain
  unchanged. Unknown versions, unresolved timezones, unavailable historical inputs
  and unattested pretrained training cutoffs fall back or refuse explicitly.
- Recommendations require at least three matched folds, an explicit baseline and
  an improvement threshold. They are advisory, not calibrated action permission.
  Rescoring makes zero provider calls and reports requested/effective cutoffs.
- Python, typed CLI and ledger-configured MCP share this implementation. The new
  execution profile has seven tools with a ledger and five without. Switching the
  ordinary default from the old ten-tool surface remains a separate pending gate.
- Corrected adapter conformance: stochastic outputs are not invalid merely because
  they differ across calls, request IDs are excluded from numeric repeatability,
  and input mutation is checked on the actual requests. Determinism can be required
  explicitly; AdapterBench's deterministic adversarial gate now does so.
- Focused routing/conformance/benchmark checks and the fresh installed-wheel
  evaluation/routing/rescore-retrieval journey pass. The first full run found the
  stale deterministic benchmark assumption (3,165 passed, 13 skipped, one failure);
  that caller was corrected. Final full pytest:3,166 passed/13 skipped in 288.13s.
  Ruff, diff checks and seven progress-checker tests pass. The routing acceptance
  gate earns three points; verified progress is 80/100, not production completion.

## Iteration 7: actual default switch and isolated compatibility checks

- Default CLI/MCP startup now uses one owned GnomonSession. Five default tools
  support frozen inspection, exact statistics, explicit provider forecasting and
  optional evaluation; a configured ledger adds routing and ledger operations.
  Startup/CLI capabilities do not import the legacy formatter or advanced runtime.
- Retired duplicate describe and experimental mega profiles. Removed311 lines from
  the active registry, preserving its exact former source under archive/legacy.
  Unregistered diagnostic helpers are isolated in legacy_experiments.py; independent
  numerical and provenance regression coverage remains. Explicit legacy profiles
  retain advanced evaluated/context workflows, with migration instructions.
- Current README, quickstart, compatibility policy, product claims and packaged
  skill teach the actual provider/session contract. Skill-creator guidance keeps
  substantial legacy instructions in a conditional reference; the old quickstart
  remains archived. The skill validator and installed reference-file checks pass.
- Historical instruments explicitly bind retained legacy profiles. Their scores
  are not default-session evidence. Original broad tests exposed13 stale selectors
  and obsolete registry/skill assumptions; every identified cause was corrected,
  not skipped. Helpers restore serial legacy environment selection; subprocess
  adapters explicitly select their profile instead of inheriting a product default.
- CI separates production regressions across declared Python versions from
  historical adapter compatibility on3.12. Four shipped batch configs pass dry-run
  validation, with no paid inference or external benchmark downloads. Local isolated
  runs pass: production2302/7 skips/78.06s; historical868/6 skips/221.46s. Aggregate
  current regression coverage:3170 passed/13 skipped. No supported-platform matrix
  result is inferred from local3.12 success or an edited CI file.
- Focused223 surface/session/docs checks,165 final docs/workflow checks and119 final
  documentation/progress/CI checks pass. Source lint and whitespace checks pass.
  Fresh no-index/no-deps installed-wheel smoke verifies both ordinary defaults and
  explicit advanced/provider/ledger journeys, including packaged skill references.

Thin default boundaries and suite separation earn five points; final fresh build
and installed smoke passed after all shipped edits. Verified progress is85/100;
bounded result projections, general temporal operations, unbiased matched agent
measurements and remaining release checks stay unearned.

## Iteration 8: bounded results, exact retrieval and resilient framing

- One default projection leaves small answers unchanged and retains exact canonical
  JSON for large answers. Its partial receipt never presents shortened forecast
  arrays as complete. One gnomon_read tool supports JSON pointers and Unicode-aware
  text pages with a root integrity digest; retrieval makes no provider calls.
- Private temporary session storage has per-result, aggregate-byte and count limits,
  LRU expiry and cleanup. It is not the durable ledger. Oversized retention errors
  disclose completed work/available durable IDs; expired/corrupt receipts have
  specific recovery guidance, not an instruction to blindly repeat inference.
- Short-lived CLI commands and explicit full Python calls return unabridged results.
  Initial CLI evaluation now emits the full study rather than a dead session ID.
  No full-result resource is claimed when the configured retention limit rejects it.
- Stdio bounds/drains physical frames and recovers from malformed/non-object JSON,
  nonfinite numbers, invalid UTF-8 and invalid IDs/parameters. Supported-version
  negotiation was checked against the official
  [MCP lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle)
  and [stdio transport](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
  specifications. It no longer echoes arbitrary client protocol versions as supported.
- Default count is6 tools/8 with a ledger. README/quickstart/CLI/config/compatibility
  documentation and packaged skill reflect actual behavior. Skill-creator guidance
  keeps targeted retrieval and duplicate-work warnings concise. Byte limits cover
  compact UTF-8 payloads; repeated MCP forms add wire overhead. They do not constrain
  arbitrary provider memory, parser peak allocation, full output or legacy projections.
- Initial production run exposed one missing error-recovery mapping gate; corrected
  with meaningful guidance and tests. Final production2316 passed/7 skips/67.00s;
  historical868 passed/6 skips/228.69s. Final56 focused result/session/evaluation
  and26 error/reliability/recovery checks pass. Combined3184 passed/13 skipped.
  Final fresh sdist/wheel and offline installed smoke passed real stdio large-result
  reads, full CLI studies and retained default/legacy/provider/ledger journeys.
  Source lint, whitespace checks and packaged-skill validation pass.

Result projection earns3 points, packaging is reverified, and verified progress
is88/100. Temporal primitives, unbiased agent measurements and remaining release
gates are still required; no production-complete or agent-uplift claim is made.

## Iteration9: optional explicit temporal calculations

- Added one independent standard-library module, temporal_ops.py, with closed
  normalize/duration/shift/interval/order_events operations. Existing forecast-grid
  helpers were not reused as general calendar semantics. No provider or ledger calls.
- UTC comparisons and elapsed arithmetic avoid wall-clock DST subtraction traps.
  Local named-zone times require disambiguation; round-trip validation rejects gaps,
  including half-hour transitions and skipped dates. Month-end clamping and target
  folds require explicit choices. Dates do not become implicit midnight instants.
- Durations use exact decimal strings; intervals are nonempty and half-open with
  Allen13 relations. Events are bounded, uniquely identified, grouped by instant
  and stably ordered without causal/availability-time claims. Zone rules are honestly
  host-supplied and unversioned; missing rules fail rather than silently substituting.
- The lazy Python export and CLI command share the operation implementation.
  Operator-only enable_temporal adds one tool, leaving the ordinary6/ledger8 count
  and ordinary import graph unchanged. Large event outputs reuse exact result reads.
- Skill-creator guidance kept packaged agent instructions conditional and concise.
  Current docs distinguish supplied facts from verified facts and executable temporal
  assistance from measured LLM-reasoning improvement. Neither accuracy uplift nor
  independently attested timezone rules is claimed.
-83 independent temporal checks and114 final focused boundary checks pass. Full
  production2399 passed/7 skips/83.23s; historical868 passed/6 skips/234.81s.
  Combined3267 passed/13 skipped. Final clean sdist/wheel plus installed Python/CLI/
  real MCP temporal/default/provider/ledger/result/legacy journeys pass at
  /tmp/gnomon-temporal-build-VvV6qJ/. Source lint, whitespace and skill validation pass.

Temporal gate earns3 points; packaging was reopened and reverified. Score91/100.
Remaining measurement5 and release4 points stay unearned. Read-only tracing for
the next iteration confirms survivor-biased denominators in agent_eval.py and
benchmarks/report.py; current historical tests preserve those semantics and must be
revised with explicit all-task versus conditional-quality distinctions.

## Iteration10: completion-aware measurement, producer accounting still pending

- Replaced the biased agent comparison with schema0.2 all-task delivered success,
  exact matched binary tests, explicit completion uncertainty, failure/budget/
  abstention measurement coverage and exact paired safety cohorts. Unknown grades
  no longer become clean safety results merely because another arm measured a field.
- Resource reporting retains supplied unsuccessful attempts, distinguishes partial
  observed totals from complete totals and rejects invalid values. Conditional
  accuracy and probability/Brier/bin diagnostics remain separate from delivery.
  Probabilities are supplied facts, not attested pre-outcome measurements.
- Historical report import now refuses malformed/duplicate normalized records;
  it shares the corrected comparator instead of maintaining a second denominator
  rule. Conditional score means and hypothetical baseline imputations are labelled.
  Legacy workflow promotion refuses incomplete or mismatched case sets.
- Added strict bounded JSON/type/nesting checks, independent failure/cost/coverage/
  calibration/binomial tests and installed CLI comparison smoke. Full production
 2437 passed/7 skips/83.92s; historical871 passed/6 skips/243.98s. Combined3308
  passed/13 skipped. Fresh distribution at /tmp/gnomon-metrics-build-Zkk1KS/ passes
  offline installed journeys. Source lint and whitespace checks pass.
- Further producer tracing found that Workflow Bench retries and checkpoint resume
  retain only the final observation's resources. Executable mock diagnostic:
  failed3 calls/100 tokens then successful1 call/10 tokens reports1/10, not4/110,
  in both paths. Timeout/subprocess observations also default unobserved spend to0.
  The new report cannot recover lost producer records. Documented this limitation
  explicitly and left the full measurement gate pending rather than crediting a
  narrower passing report-only implementation. Next iteration must preserve attempt
  accounting through stages/resume, distinguish unknown totals and test consumers.

Packaging reverified; score remains91/100. This is implementation progress without
new gate credit. Matched ordinary/lean/full measurement and remaining release gates
remain required; production-complete and agent-accuracy/cost-uplift claims remain unproven.

## Iteration 11: attempt accounting and executed release checks

- Workflow Bench now commits an attempt start before each external invocation and
  appends its completion in a separate benchmark-only SQLite journal. Retry, stage
  and resume receipts preserve original usage, errors and leakage. Interrupted
  starts survive checkpoint loss; historical missing measurements stay unknown.
  Scalar compatibility fields are explicitly lower bounds, not complete totals.
- Shared LLM usage accounting distinguishes absent/invalid usage and cached history
  from new-process spend. Unknown costs remain null; small costs are not rounded to
  free. Accounting and safety gates require complete measurement, not merely an
  absence of observed failures. These are producer receipts, not billing attestation
  or a sandbox for arbitrary agent code. One orchestrator owns an output directory.
- 122 focused checks pass, including 21 independent accounting checks with real
  subprocess timeouts, parallel workers, resume, crash-before-checkpoint and CLI
  export. Historical compatibility: 897 passed, 6 skipped; four runnable YAML batch
  configurations pass dry-run validation without datasets or paid calls.
- Added three ledger release checks beyond existing thread/replay/migration tests:
  separate spawned processes initializing and appending deduplicated revisions;
  failed study writes rolling back payload and metadata; fault-injected migration
  rollback preserving schema version and records, followed by a successful retry.
- Final full production suite: 2,440 passed, 7 skipped on each of Python 3.11.14,
  3.12 and 3.13.11 (83.28s, 82.17s and 79.82s respectively). Combined unique
  production and historical coverage: 3,337 passed, 13 skipped. Source compilation
  passes on all three. These are Linux checks, not untested OS or optional-model
  certifications. Source Ruff, installer ShellCheck 0.11.0 and whitespace gates pass.
- Fresh isolated sdist/wheel build at `/tmp/gnomon-release-checks-ublF4A/` passes
  distribution size and packaged-skill checks. The wheel's complete installed
  CLI/MCP smoke passes in a local Docker container with networking disabled in
  6.89s; no checkout imports, package-index access or model service is needed.
  Unchanged shipped source also passed installed journeys on all three Python
  versions. CI job equivalents were executed locally, not remotely published.

Measurement metrics earn 3 points and release checks earn 2: score **96/100**.
Matched ordinary/lean/full evidence (2), executable onboarding/operations examples
(1), and the actual deployment contract check (1) remain unearned. A real Paracast
base URL and token environment-variable name have not been supplied. No agent
superiority, real weights inference, production completion or paid calls are claimed.

## Iteration 12: matched experiment controls, surface experiment still pending

- Traced the historical adapter: it forces a preferred legacy tool, compiles
  arguments and recovers engine-authored answers. Historical outcome stages also
  compile bookkeeping in the host. Neither path is evidence of autonomous use of
  the current lean tools or agent-owned forecast retention/scoring.
- Added an optional matched contract to the existing process runner rather than
  a second scorer/model client. One exact driver command, declared local files,
  shared model/generation/prompt/provider settings and budgets govern three explicit
  arm descriptions. Concrete source/benchmark/corpus/interpreter content and
  dependency versions distinguish different worktrees, not just Git dirty markers.
- Resume and post-run checks refuse changed identities. Original public cases omit
  oracles and gain experiment settings without preferred-tool/argument compilation.
  Historical staged cases are refused until genuine driver state/reveal handling
  exists. Driver compliance, remote weights, environment values and malicious edits
  are not independently attested; tool/token/round limits still need the real driver.
- A dedicated comparator verifies scored observation/journal bytes, exact task rows
  and shared controls before all-task paired comparisons. Errors and unknown costs
  remain visible. Historical report/promotion consumers refuse matched runs instead
  of silently bypassing the stricter comparison path. This is not statistical proof
  that tasks are independent, execution order randomized, or Gnomon improves an LLM.
- 26 independent control tests pass on Python 3.11/3.12/3.13 with real scripted
  subprocesses. They cover file edits, hidden overrides, changed resume inputs,
  mid-run changes, missing tasks, stale artifacts and reporting bypasses. The test
  driver performs arithmetic only and explicitly does not claim to exercise Gnomon
  profiles. Final focused workflow/accounting/report coverage: 121 passed.
- Full production: 2,440 passed, 7 skipped in 77.34s. Final historical/benchmark
  suite: 923 passed, 6 skipped in 255.14s. Combined: 3,363 passed, 13 skipped.
  Source and changed-benchmark Ruff, compilation and whitespace pass; 118 final
  documentation/progress tests pass. Shipped package source is unchanged.
- Corrected the installation guide's checkout instructions to use `--local` after
  tracing the script's GitHub-download branch. This does not by itself satisfy the
  pending executable onboarding/operating-examples release gate.

Score stays **96/100**. Actual ordinary/lean/full driver inventories, enforced shared
limits and the matched forecast/decision/temporal task experiment remain required.
The user was asked asynchronously for a live agent/model and total spending limit;
offline implementation can continue. The separate Paracast deployment URL/token
environment-variable name is still missing. No paid calls, remote writes or new
production-complete/agent-uplift claims were made.

## Iteration 13: executable provider and ledger operating instructions

- Added a separately installable, dependency-free example provider package with a
  stateless callable and fresh mutable fitting object. The same operator TOML works
  through installed Python, CLI and MCP; no per-library Gnomon adapter was added.
  Its elementary reference calculations are explicitly not real TSFM/library lift.
- A runnable synthetic walkthrough preserves original forecasts while appending
  pending/partial/complete/revised and historical-vintage scores. It records a
  proposed decision without action authority, then checks backup and restore.
  Agent outcome writes remain disabled; operator Python writes do not grant agents
  that permission. Existing directories and backup destinations are not overwritten.
- The backup utility uses a read-only SQLite source and a private, exclusively
  created destination with integrity/foreign-key checks. Tests prove committed WAL
  inclusion and preupgrade source preservation followed by candidate-only migration.
  Failure/partial-backup, distributed-storage and administrator-tampering limits
  remain explicit. The documented legacy-import block executes without modifying
  its source registry or inventing missing historical input/score information.
- Updated operations/onboarding/offline guides and canonical links against these
  executed paths. Verified isolated local pip, Bash `--local`, uv and actual
  Dockerfile installations. Docker's build context now admits only package inputs,
  not unrelated checkout/credential/data directories; runtime inference also passes
  with no network and a read-only container root.
- The offline-wheel smoke now verifies the separately built example wheel and
  removes ambient checkout/profile overrides. CI requires the installed Python/CLI/
  MCP and ledger/backup journey in a network-disabled container. No example module
  or dependency is included in the core wheel. No remote CI/publishing was performed.
- 131 focused checks pass; 11 independent onboarding tests execute the examples and
  failure boundaries. Final production: 2,451 passed, 7 skipped on each supported
  Python version (3.11:84.28s, 3.12:83.41s, 3.13:81.60s). Historical: 923 passed,
  6 skipped in 231.06s. Combined: 3,374 passed, 13 skipped. Compilation on all three,
  source/example/script lint, installer ShellCheck and whitespace checks pass.
- Fresh core distributions at `/tmp/gnomon-operator-build-41nNwV/` pass size/skill
  boundaries; installed core+example journeys pass on all three Python versions
  and with Docker networking disabled. Concrete artifact hashes and local install
  directories are recorded in HANDOFF.md. Shipped runtime Python is unchanged.

Executable operating documentation earns 1 point: score **97/100**. The actual
ordinary/lean/full agent/task experiment and live Paracast deployment verification
remain required. No paid model calls, real-model accuracy claims or production
completion were inferred from synthetic examples.

## Iteration 14: shared unsteered agent loop, not a completed ablation

- Traced a hidden-budget problem in the shared LLM client: one chat call could
  retry truncated responses with increasing output allowances and top up absent
  choices. Added opt-in single-attempt behavior without changing historical
  defaults. Tests prove one transport request despite retry configuration,
  truncation or missing choices, bypassed caches, bounded response reads and
  strict refusal of duplicate/nonfinite/nonobject response JSON.
- Added one backend-neutral loop that discovers actual tools and leaves choice,
  arguments and final numbers to the model. Scripted tests include real current
  Python inference and execution/full MCP discovery/calls. Original incorrect
  submitted numbers stay incorrect; there is no host answer recovery.
- Shared per-attempt limits count malformed/unknown calls and prevent excess
  batched dispatch. Failed calls retain unknown service charges; incomplete token
  usage stops additional work. Tokens/time are post-work observations, not a
  hard billing quota or guaranteed remote cancellation. Operator callbacks are
  trusted, not a sandbox, and no ordinary software backend has yet been bundled.
- Followed budget flags through the actual downstream receipt, checkpoint,
  scoring and normalized-row path. Prior violations survive resume; older
  receipts and unmeasured attempts do not become clean measurements. Actual host
  subprocess timeouts are positive findings; retry totals still lack a global
  spending cap. The JSON receipt extension reads old journals without migration.
- Documented concrete factory interfaces and limits in MATCHED.md. Unsupported
  generation controls are rejected; the eventual driver must still bind the
  declared model/config/arm to those interfaces. This module does not itself
  constitute the ordinary/lean/full experiment or genuine staged agent execution.
- Focused verification:161 passed on Python3.12;110 each on3.11 and3.13. Production
  2451 passed/7 skipped on3.12;118 docs/progress checks pass. Source/changed-file
  Ruff, three-version benchmark compilation and whitespace checks pass. Final
  historical959 passed/6 skipped in251.82s; combined3410 passed/13 skipped. All
  iteration14 process handles are terminal. No shipped
  runtime code or package inputs changed, and no new build/paid run is claimed.

Score remains **97/100**: ablation2 and deployed Paracast1 are still pending. This
iteration is concrete progress toward the existing experiment requirement, not
an additional score item or evidence that an LLM became more accurate.

## Iteration 15: executable binding and bounded driver processes

- Added a real shared driver script that validates pinned prompt/provider bytes,
  uses the declared model/generation/budget, imports only the selected operator
  backend factory and adds only that arm's declared guidance. Public case data is
  separated from experiment/configuration paths before model/backend submission.
  The current checkout, not an ambient installed Gnomon version, is used by the
  absolute script. Backend source must be declared in driver_files; undeclared
  dependencies and remote weight changes are not magically attested.
- Credentials must come from the named environment variable. No key, URL or .env
  fallback; no ambient proxy or redirect forwarding. Local scripted HTTP is
  possible without opting into a remote origin. Remote calls require both HTTPS
  and --allow-model-requests, which is not a global dollar quota. Existing shared
  OpenRouter callers retain their default transport unless an opener is supplied.
- Traced the previous subprocess.run boundary: timeout killed only the immediate
  driver and capture_output buffered unbounded data. Replaced it with bounded
  POSIX pipe reads and owned-process-group cleanup on timeout, output limit and
  normal completion. Input/output/error ceilings are enforced during I/O. Tests
  launch real descendant processes, including an exited parent, and exercise both
  output floods and closed pipes. Escaped sessions, Docker workers and remote
  model work require separate lifecycle/isolation controls; this is not a sandbox.
- Real shared-command/local-HTTP tests call current Gnomon inference and export
  results through actual attempt journaling and scoring. All three binding markers
  are checked, but deliberately use the same actual backend: they are not a
  surface ablation or a claimed strong ordinary-software baseline. Missing tokens,
  mutated pinned files and redirects are independently rejected.
- Actual local dependency metadata has none of numpy/pandas/scipy/statsforecast.
  No broad library baseline was inferred from that environment. No new image,
  container, deployment, paid request, Git commit or push was performed.
- Focused182 passed on3.12;106 each on3.11 and3.13. Production2451 passed/7 skipped
  in87.45s;118 docs/progress checks pass. Source/changed-file Ruff, benchmark
  compilation on all three versions and whitespace checks pass. Historical980
  passed/6 skipped in245.40s; combined3431 passed/13 skipped. All handles terminal.
  Shipped runtime/package inputs are unchanged.

Score remains **97/100**. Next is actual ordinary/lean/full backend provisioning,
appropriate tasks and genuine staged agent-owned forecast/ledger execution, then
the controlled experiment. Deployed Paracast verification remains required too.

## Iteration16: usable ordinary software, isolated from the checkout

- Added the real driver-compatible ordinary Python factory, with no Gnomon
  adapter or agent-generated host execution. It requires an explicit local Docker
  socket and installed immutable image ID. Public case data travels over stdin;
  there are no host mounts, credentials or future-stage inputs. Python files can
  persist within the task, and errors return stderr/status for agent repair.
- Built a separate Linuxx86-64/Python3.12 image from a remotely verified immutable
  Python base, with26 hash-locked wheel dependencies and pip check. Actual direct
  StatsForecast SeasonalNaive/AutoARIMA/AutoETS forecasts execute. NumPy/pandas/
  SciPy and the standard library are available, without a Gnomon installation.
  Other architectures are not inferred from this build. No core dependency changed.
- Tested non-root agent execution, denied external network/root writes/timer
  signals and absence of host secrets/checkouts. Container bounds memory, CPUs,
  processes, tmpfs and captured output. A protected UID0 PID1 timer terminates
  independently of the host; exact-owner cleanup removes only created containers.
  Missing/ambiguous daemon state is not declared successful cleanup. These controls
  are not VM isolation or certification against kernel/container exploits.
- The actual shared loop uses the container in a scripted test and records its
  image/software/input provenance. Scripted responses are not LLM-quality evidence.
  Local service charges0 explicitly exclude infrastructure, not free-compute claims.
- Separate CI software/isolation job added.51 focused checks pass on each supported
  host Python3.11/3.12/3.13, using the same final Python3.12 container. Final production
  2451 passed/7 skipped; historical992 passed/6 skipped with real containers enabled;
  combined3443 passed/13 skipped.119 boundary/docs/progress checks pass. Ruff,
  three-version benchmark compilation and whitespace pass. All handles terminal.
- Final imageID sha256:fa55aa0eb37dd59d1ee258ab37d3acab35507d4f28192f3c159f813c17dd65b8
  is retained locally, along with the initial build and private build/cache folders;
  exact artifact paths/hashes are in HANDOFF.md. Final listing found no leftover
  owned test containers. No user containers/images were pruned or overwritten,
  and no paid model calls, pushes or deployments were made. Remote CI was not run.

Score remains **97/100**. Remaining experiment work is fair ordinary/lean/full
combinations with isolated Gnomon services, appropriate tasks and genuine staged
agent-owned forecast/ledger journeys, plus total spending controls and the actual
model experiment. Live Paracast deployment verification remains required.

## Iteration17: actual isolated lean/full combinations

- Built and exercised current lean and retained full MCP services beside identical
  ordinary software, each in an owned network-disabled container. Actual original
  schemas/results are preserved. Full's legacy evaluated forecast is explicitly
  distinct from lean direct inference; added compute/profile functionality is not
  presented as a pure tool-count or equal-compute intervention.
- Added package-wide installed-content verification (including non-Python data),
  dependency-version parity checks, persistent framed MCP state and bounded IO/ID
  validation. No arbitrary host Gnomon filesystem, environment or Docker socket
  reaches agent code. Cleanup attempts every component and preserves uncertainty.
- Public files are an explicit bounded text mapping copied identically to isolated
  /tmp/data directories. Path traversal is rejected before daemon access. There
  is no inferred dataset conversion or implicit sharing of newly generated files.
- Optional lean ledger/temporal startup switches are typed, pinned and rejected
  for legacy full. Root-owned readonly configuration does not enable agent outcome
  writes. Real MCP save/retrieve/operator-actual/score checks preserve prediction[7]
  while recording MAE1. This is not a staged agent experiment or host-filled answer.
- Real current and legacy forecasts, persistent data references, error repair,
  original discovery contracts, package mismatch and lifecycle failures are tested.
 70 focused checks pass on each3.11/3.12/3.13 host with actual containers. Final
  production2451 passed/7 skipped; historical1011 passed/6 skipped with both image
  opt-ins; combined3462 passed/13 skipped.119 boundary/docs/progress checks, source/
  changed-file Ruff, three-version benchmark compilation and whitespace pass.
- Local service image7ffb55e2aa02c29d374cec15b96f1cb4a275a5a6d3ffaa806138b6f8d70a40c8
  retains the verified iteration13 core wheel with the same locked ordinary
  software. pip check passes; exact package/build/image hashes are in HANDOFF.md.
  The dedicated CI job now builds/checks both images; remote CI was not run. All
  handles terminal and no test containers remained; local images were retained.

Score stays **97/100**. Next is an appropriate task corpus and genuine staged
agent save/reveal/score protocol, total spending controls and actual matched model
runs. No paid model run, deployment change or live Paracast verification is claimed.
