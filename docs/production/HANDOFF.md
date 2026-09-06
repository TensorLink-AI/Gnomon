# Delivery checkpoint — iteration 28

Updated 2026-09-06. The user requested fixing the final review findings and checking
for bugs. Completed local implementation and verification; PR CI is next.

## Current scope

One provider-neutral Python/CLI/MCP execution session, explicit evaluation,
optional observation vintages/forecast ledger and temporal arithmetic.
Ephemeris is one connector; users own model libraries and callable/factory providers.
Runtime:26 modules/5,778 lines (original118 modules/67,013 lines).
One matched ordinary/lean/full agent workflow:3,309 implementation lines
(`workflow/*.py` plus `common/*.py`; before benchmark cull5,289).

Iteration28 rejects invalid repair/regrid values before reading files or opening
a store, and nonboolean partial-scoring flags before appending evaluations.
Ledger evaluation/comparison, backtesting and study routing share finite point
losses that avoid intermediate overflow. Unrepresentable errors fail before writes.
Removed unused repair constants, Snapshot.variables/access_log and obsolete MCP
envelope fields. Corrected repair defaults and encoding guidance. Actual snapshot
access summaries, saved forecasts, migration support and explicit repairs remain.

## Verification

- Full suite:819 passed,29 opt-in skips in30.42s;70 new cases, no tests removed.
- Actual ordinary/service container suite:77 passed in34.74s.
- New regressions cover invalid policy types/values through Python and real MCP,
  unchanged inputs/no database creation, strict partial flags, large finite losses
  and refusal without appending when errors are unrepresentable.
- Ruff, compilation and whitespace pass.
- Rebuilt wheel/sdist; clean installed CLI/MCP/provider/ledger/example-plugin
  journeys pass. Artifacts: `/tmp/gnomon-fix28-final-dist.vVrj9k`.
- Current rebuilt service image:
  `sha256:6beb711ee4bdeb80d20ba3b2d15750ddaa8655bee2c3550c53164d5d322a009d`.
- Initial full-suite/package checks caught a missed study-routing import after
  metric consolidation; fixed it and rebuilt/reran successfully, not waived.
- Previous baseline0826117 passed all seven PR checks. Current fixes still need
  push and fresh CI; do not reuse baseline CI as evidence for new source.
  Do not edit source or commit during identity-pinned harness tests.

## Remaining external evidence

[progress.json](progress.json) stays97/100. Fewer lines earn no extra points.

- Actual matched agent comparison needs model/endpoint selection, credential
  environment-variable name and spending approval.
- Live Ephemeris needs deployment URL, credential environment-variable name
  and budget for potentially billable wake-up/inference.

Neither paid experiment nor authenticated inference has been performed. Scripted
agents/local HTTP fixtures do not satisfy these gates. Do not extract credentials
or repeat unauthenticated probes to bypass the authorization boundary.

## Distribution and recovery

Source0.9.0.dev0 is breaking and unreleased. Published0.8.0rc3 remains immutable;
no PyPI upload. No user database, saved forecast or user input was deleted.
Final-fix recovery0826117; residue-pass recovery53482fc; benchmark-cull recovery1642cb2; runtime recovery333ed2c;
earlier benchmark/design archives2cba20e.

Branch `codex/gnomon-ephemeris-ledger`, [PR99](https://github.com/TensorLink-AI/Gnomon/pull/99).
It is now ready for review rather than draft; do not change that status. Base remains
`claude/enterprisebench-multi-domain-n4xyjf`; do not silently retarget or merge.
Unrelated root scratch files are untouched and must not be staged:
`-`, `Continue`, `Current`, `Immediate`, `Use`, `accelerate`, `actual`,
`cases.`, `optimizing`, `that`.

Continue from this checkpoint after automatic compaction, not from scratch.
