# Changelog

## 1.1.9 — 2026-09-10

- Record concise execution-bound decision summaries with assumptions, invalidation
  conditions, verified ledger references and explicit context availability.
- Compare complete matched production forecasts under exact context filters,
  with exclusions, provenance, sample counts and descriptive error ranges.
- Generate read-only outcome review packets and save immutable versioned lessons;
  preserve original decisions, old actual evidence and exact latest retries.
- Export compact hypotheses with verification calls and provide a dependency-free
  adapter for caller-owned LangGraph-compatible stores. No background agents,
  external memory services or causal/business-value claims are implied.
- Expose the workflow through Python, CLI and configured MCP ledger tools, with
  bounded schemas, write opt-in, measured diagnostics and an installed example.

## 1.1.8 — 2026-09-09

- Expose canonical forecast completions bound to the full request and a public
  host-side final-selection resolver. Preserve a single matching execution
  after prose finals, require explicit selection for ambiguity, and report
  strict final conformance separately from recovered completion.

- Keep assumed source availability independent of known recording timestamps
  for CSV-store evaluation, routing and immutable rescoring. Report missing
  origin observations separately from insufficient history, with exact cutoff
  and endpoint visibility diagnostics.
- Return task-preserving frozen-snapshot retries and temporal correction
  templates, with authoritative recovery causes and explicit semantic choices.
- Default capabilities to shared schemas and errors to a canonical error
  object; retain `--expanded`, Python `brief=False`, and expanded-error options.
- Report all independently checkable study mismatches and invalid configuration
  field paths. Correlate provider failures without exposing exception secrets.
- Measure provider dispatches, forecast requests and committed ledger row
  writes at session boundaries; budget MCP read pages including these fields.
- Install a complete controlled-clock `compare_history` example.

## 1.1.7 — 2026-09-09

- Explain compare-history exclusion causes and historical replay visibility per
  fold; add evaluation preflight and explicit CLI replay selection.
- Keep capability discovery from opening ledgers, validate providers before
  ledger creation, and support configured built-ins with a caller-owned ledger.
- Bind CSV timezone declarations transactionally to dataset identity. Disclose
  completed backtest executions as scored in their saved study.
- Recommend admissible safe jitter alignment, gate impossible drop advice and
  expose combined fill/conflict costs alongside independent repair scopes.
- Save rejected CLI results, warn on default routing fallback, identify mismatch
  fields, and preserve task parameters in column/provider argument corrections.
- Add cache counters, brief discovery projections, raw provider examples,
  auditable self-check family contracts and detailed case evidence.
- Clarify completion, returned evidence, result shapes and byte-budget scopes;
  preserve compatibility for strict snapshots, compact-call defaults and legacy
  errors. The external Arena adapter/protocol is not changed by this patch.

## 1.1.6 — 2026-09-09

- Canonicalize validated cache identities across numeric/container/default and
  timestamp representations without collapsing large integers or calendar DST
  grids. Add cache diagnostics and engine execution cache/provenance metadata.
- Add immutable revision-aware `evaluate --rescore` and `evaluate --compare`
  operations through CLI, Python and MCP, preserving original executions/origins,
  refreshing rankings, ties and optional arithmetic derivations with zero calls.
- Accept timezone declaration at CSV store ingestion; disclose both visibility
  cutoffs in empty snapshots and preserve explicit versus defaulted error inputs.
- Add resolved-configuration inspection without provider imports or secrets,
  installed provider/MCP examples, effective-season disclosure, centralized exit
  and cutoff semantics, and optional compact error envelopes.
- Add bounded all-mode repair diagnosis, separate bad-cell and affected-row
  counts, shared recovery/budget metadata, optional arithmetic verification and
  selectable finite self-check families for revisions, recording, folds,
  covariates, series, DST, repairs and cache equivalence.

- Label early gap-run rejection counts as lower bounds with incomplete-scan
  metadata; retain the bounded scan and existing repair limits.
- Add optional strict routing (`--require-evidence` / `require_evidence=true`),
  explicit evidence/fallback status and task-preserving follow-up actions.
- Link compact evaluations directly to full saved fold evidence; distinguish
  full response retrieval from full study retrieval. Retained receipts include
  character counts, hashes and pagination guidance.
- Mark temporal examples mechanically runnable or unresolved, list preserved
  fields, retain numbers in proposed integer type corrections, and restrict
  month-end policy guidance to relevant failures.

## 1.1.5 — 2026-09-09

- Preserve request identity in installed custom-provider examples and rejection
  guidance; explain empty session construction and add a unit-bearing cache example.
- Add `forecast` as a CLI alias for `infer` and a tested local evidence workflow.
  Route by study ID loads omitted task parameters at the requested recording cutoff.
- Disclose routing fallback, selection and ties, recompute rescore tie metadata,
  distinguish snapshot cutoffs from ledger evidence cutoffs, and make insufficient
  fold guidance reflect whether any folds were excluded.
- Keep units and temporal context in input errors, describe both MCP forecast
  forms, and report batch actual units per item rather than as a unitless batch.
- Preserve temporal facts through explicit fold, target-fold and month-end policy
  examples; retain clock-gap requests as templates needing source correction.
- Scan early parsing failures for complete unparseable-drop costs on inputs up
  to 100,000 rows, without changing data or promising grid/conflict admissibility.
- Self-check response schema 0.2 replaces `structural_claim_proven` with
  `checks_passed`, varies synthetic temporal cases, and explicitly limits its claim.

## 1.1.4 — 2026-09-09

- Preserve supplied temporal parameters in recovery examples, identify changed
  fields and missing semantic choices, and label generic fallback illustrations.
- Return current ledger coverage alongside immutable score evidence, including
  newly visible wrong-unit actuals on reused scores; distinguish supplied query
  cutoffs and units from operation defaults.
- Include actual matched-fold counts and persistence in evaluation routing
  readiness, with the default three-fold requirement visible in CLI help.
- Add `gnomon schemas`, offline cache-enablement examples, series selector help,
  and typed custom-provider return guidance.
- Preserve zero/fractional actuals, batch shape and forecast history in recovery
  templates; enforce 2–100 execution IDs in comparison schema/runtime and label
  generic CLI illustrations. Label reconstructed legacy coverage separately
  from saved evidence and retain scoring/coverage summaries in large responses.

## 1.1.3 — 2026-09-09

- Preserve bounded, decision-relevant summaries when a large structured result
  moves behind a session result reference. Historical comparisons now keep their
  matched/observed counts, aggregate model rankings, duplicate/provider-call
  diagnostics, and origin/exclusion counts in the first agent-visible response;
  detailed per-origin evidence remains available through `gnomon_read`.

## 1.1.2 — 2026-09-08

- Expose ledger scoring completion, coverage, missing timestamps and visible
  other-unit actuals. Explain the partial-scoring default and retained CLI exit
  semantics; preserve exact retries and existing unit/cutoff selection rules.
- Preserve ledger operations and task parameters in recovery guidance across
  CLI and MCP. Explain stored forecast timezone failures and distinguish missing
  studies from studies unavailable at a recording cutoff.
- Publish repair budgets and denominators in help/schema and predict whether
  gap interpolation fits both fraction and consecutive-gap limits. Give useful
  duplicate, timestamp, target and seasonal-history recovery, and avoid suggesting
  a business calendar when the input contains observed weekends.
- List available providers, series and configuration keys on relevant errors;
  add `capabilities --config-schema`, explain CLI/MCP forecast argument mapping,
  group temporal shift field errors, and clarify Python forecast call/type errors.

See [scoring and recovery contracts](docs/scoring-and-recovery.md) for examples
and the operation defaults matrix. These changes address verified first-use
interface friction; they do not establish forecasting superiority.

## 1.1.1 — 2026-09-08

- Accept request dictionaries in Python session/engine forecasts and batches;
  explain invalid types, missing fields and how to initialize built-in providers.
- Publish per-provider request schemas in capabilities, including season defaults
  and declared feature/history/horizon/frequency limits. Show nested command
  choices in top-level help and explain ledger write authorization and query filters.

- Expose executed request cutoffs and frozen snapshot provenance directly on
  forecast responses across CLI, Python and MCP, independently of ledger recording.
- Match temporal error examples to the attempted operation and include runnable
  examples in each operation's schema.
- Disclose evaluation ranking and tie policy, with tied providers and guidance
  that does not imply equivalent predictions or future performance.

## 1.1.0 — 2026-09-08

- Complete infer/ledger schema discovery and provider-name recovery. Require an
  existing ledger for CLI reads/routing so path typos cannot create empty evidence.
- Preserve dependency versions during managed updates and skip unchanged builds.
  Protect live Python/MCP environments during pruning using process inspection
  and interpreter startup leases, preserving environments with unknown usage.

- Bump to 1.1.0 and embed commit/source fingerprints in distributions;
  expose qualified build IDs in CLI/MCP and use them in provider/content identity.
- Add Python environment discovery and interpreter passthrough, plus managed
  install listing, updates, rollback and previewed pruning. Standalone installs
  retain source receipts and clean up failed environments.
- Add temporal JSON Schema discovery and actionable field errors. Explain repair
  modes in help and gap errors, including why safe repair does not interpolate.

- Add direct evaluate/route flags, discoverable JSON schemas and examples,
  `inspect`/`describe --input` aliases, and concise CLI usage errors.
- Expose season and quantile options for file forecasts. Disclose session-only
  cache behavior in capabilities and forecast responses.
- Report inspection readiness, support explicit IANA timezones, and provide
  an observed contiguous-window recovery for gapped data. Independent formatting
  repairs can be evaluated; future-dependent preparation remains blocked.
- Add portable frozen `.gnomon` snapshots, direct ledger paths and saved result
  files for evaluation-to-routing across processes without copying study IDs.
- Return exit 2 for unscored CLI evaluations and 3 for partial evaluations, with
  actionable history and budget diagnostics in the result. MCP marks unscored
  evaluations as failed tool calls while retaining their diagnostic reports.

## 1.0.1 — 2026-09-07

CLI onboarding and diagnostics patch.

- Keep the CLI's single structured success or error response on stdout and use
  exit status for failure, so callers can parse one stable channel.
- Report safe, actionable provider entrypoint import failures without exposing
  arbitrary provider exceptions, credentials or private endpoints.
- Put local-provider environment isolation and the exact ledger scoring identity,
  timestamp and timezone requirements in the first-run documentation.

## 1.0.0 — 2026-09-07

First stable provider-neutral execution API.

- Keep one Python/CLI/MCP execution contract with optional evaluation, temporal
  data and durable ledger capabilities.
- Remove pre-1.0 artifact and TrackingStore imports, schema-upgrade paths and
  migration documentation. Fresh 1.0 ledger and temporal-store files have explicit
  identities; incompatible databases fail without mutation.
- Preserve the dependency-free forecast path and explicit evidence limits.

## 0.9.0rc1 — 2026-09-06

Breaking release candidate: use the provider-neutral execution interface and
review its documented limits. Live-service verification and real-agent
performance evidence remain pending; this is not a stable-production claim.

- Repair the publishing workflow to verify current CLI commands and the separately
  installed provider example. Publish matching wheel/sdist and prerelease metadata.

- Validate production comparison grids and lead times; keep fallback sample counts
  consistent with per-origin evidence. Preserve full missing-step lists in
  `pending()` and allow unknown/null units in the ledger tool schema. Reject
  oversized cursors and unrepresentable actuals with structured argument errors;
  avoid intermediate overflow when averaging extreme finite per-origin losses.

- Add bounded ledger discovery with derived feedback status, atomic actual/score
  batches and retry-safe scoring. Compare matched production forecasts across an
  explicit origin window without model calls or automatic routing. Preserve provider
  lifecycle/capabilities in new executions; return actionable evidence diagnostics.
  Refresh the agent skill for cross-session discovery and outcome feedback.

- Reject invalid repair/regrid policies before input reads and nonboolean
  partial-scoring flags before ledger writes. Use shared overflow-resistant point
  losses for backtests and ledger scores. Remove unused snapshot accessors and
  obsolete MCP output fields; correct repair defaults and encoding guidance.

- Remove the stale 0.7 dependency lock and unused repair-log helpers. Consolidate
  benchmark token/cost counters and answer grading, replacing duplicate yield
  fields with `answered_rate`. Correct stale descriptions of removed workflows.

- Remove obsolete benchmark promotion/audit/generation runners, smoke cases,
  publication grading, host-generated follow-up answers, response caches and
  hidden model retries/sample fan-out. Keep one matched agent workflow and
  independent accuracy, failure, cost and cutoff regressions. Case schema v2
  requires fresh runs; old benchmark code is recoverable at `1642cb2`.

- 0.9 development removes the old evaluated runtime, legacy profiles, context/
  publication/effect-learning stack, model catalogues/installers and per-library
  adapters. One provider-neutral session remains. This is a breaking change;
  Historical records remained importable read-only in this candidate.
- Reduce the CLI to session operations. Move agent-comparison metrics into the
  benchmark harness; full now means the same session with optional ledger/time
  tools, not the retired runtime. Old-arm evidence is not current-product evidence.

## 0.8.0rc3 — 2026-09-06

- Shorten the README around a plain-English introduction, runnable quick start
  and optional integrations.
- Refresh the packaged agent skill, its UI metadata and legacy guidance against
  the current session. Add an executable offline skill-example regression.
- Retain the first candidates' validation limits; live-service verification and
  the real-agent comparison remain pending.

## 0.8.0rc2 — 2026-09-06

- Lead with Gnomon's provider-neutral toolkit; present Ephemeris as one optional
  connector and remove implementation-specific naming throughout the active tree.
- Generate container tags for PEP 440 prereleases without promoting them to latest.
- Supersede the already-published first candidate; the live evidence gates below
  remain pending.

## 0.8.0rc1 — 2026-09-06

### Changed

- Make provider-neutral execution the default Python/CLI/MCP session: six tools,
  eight with a ledger. Temporal calculations remain optional.
- Add typed callable/factory providers, bounded data/result references and
  explicit budgeted backtesting. User-selected libraries need no Gnomon adapter.
- Name the inference integration Ephemeris (`EphemerisProvider`, provider kind
  `ephemeris`). The deployment URL is configurable.
- Preserve forecast executions, actual revisions, evaluations and cutoff-bound
  routing evidence in the optional append-only SQLite ledger.
- Retain advanced evaluated workflows as explicit legacy profiles; remove
  duplicate describe and experimental mega profiles.
- Refresh documentation and remove superseded designs, old benchmark families
  and duplicate archive source. Git checkpoint `2cba20e` preserves their history.
- Retain one matched ordinary/lean/full evaluation harness with failure-aware
  accounting, bounded execution and committed multi-phase episodes.

### Migration and limitations

This is a release candidate, not a fully verified production declaration.
Read [compatibility](COMPATIBILITY.md) before upgrading an existing MCP client.
Authenticated live Ephemeris verification and an actual matched agent comparison
remain pending. Tests and publication do not establish forecast superiority,
calibrated provider uncertainty or improved LLM reasoning. Older benchmark results
are not labelled as evidence for the current default surface.

## 0.7.0 — 2026-08-30

Forecast-quality and agent-boundary hardening: temporal answers, context recall,
anomaly attribution, decision integrity and crash-safe evaluation. Advanced
evaluated workflows preserve the immutable primary and human-review boundary.

## 0.5.0 — 2026-08-30

Governed context intelligence: strict/best-effort/scenario publication modes,
source-grounded context, prospective outcome learning and compact MCP profiles.

## 0.4.0 — 2026-08-01

First-contact improvements: messy-data repair, evaluated anomaly detection,
enrichment adjudication, tracking and wider input-format support.

## Earlier versions

0.3 introduced temporal execution, point-in-time snapshots and typed evidence.
0.2 introduced evaluated forecasts, abstention, enrichment and tracking.

The detailed historical development log, previous design decisions and benchmark
reports are recoverable at Git commit `2cba20e`. These older records describe their
own versions, not claims about this release candidate.
