# Changelog

## Unreleased

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
