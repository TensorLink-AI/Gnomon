# Changelog

## 0.8.0rc1 — 2026-09-06

### Changed

- Make provider-neutral execution the default Python/CLI/MCP session: six tools,
  eight with a ledger. Temporal calculations remain optional.
- Add typed callable/factory providers, bounded data/result references and
  explicit budgeted backtesting. User-selected libraries need no Gnomon adapter.
- Name the inference integration Ephemeris (`EphemerisProvider`, provider kind
  `ephemeris`). Paracast is its backend; the deployment URL is configurable.
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
