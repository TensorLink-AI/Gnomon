# Delivery checkpoint — iteration 27

Updated 2026-09-06. The user requested clearing the remaining identified residue.
Completed the focused follow-up; do not restore retired features for their tests.

## Current scope

One provider-neutral Python/CLI/MCP execution session, explicit evaluation,
optional observation vintages/forecast ledger and temporal arithmetic.
Ephemeris is one connector; users own model libraries and callable/factory providers.
Runtime:26 modules/5,774 lines (original118 modules/67,013 lines).
One matched ordinary/lean/full agent workflow:3,309 implementation lines
(`workflow/*.py` plus `common/*.py`; before benchmark cull5,289).

Iteration27 removes the obsolete0.7 root dependency lock and uncalled RepairLog
clone/has_actions/warnings_for helpers. Dependencies remain in pyproject.toml;
the current ordinary-software benchmark's hash-locked requirements are unchanged.
Model transport now owns one set of token/cost counters. Answer and typed-fact
checks are computed once; forecast metrics are shared by the report. One
`answered_rate` replaces duplicate initial/final yield fields. Correctness,
missing answers, episode completion and resource completeness remain separate.
Fixed stale documentation advertising the removed forecast workflow/adapter bridge.

## Verification

- Full suite:749 passed,29 opt-in skips in30.23s; no tests removed this iteration.
- Actual ordinary/service container suite:77 passed in34.63s.
- 1,200 deterministic before/after scorecard comparisons: all retained fields
  identical, excluding explicitly removed duplicate fields.
- Ruff, compilation and whitespace pass.
- Rebuilt wheel/sdist; clean installed CLI/MCP/provider/ledger/example-plugin
  journeys pass. Artifacts: `/tmp/gnomon-cull27-dist.FFGwuz`.
- Current rebuilt service image:
  `sha256:d201f169172bb64aa75daeb3ebb95a082522f89daadb63d450065c715c5a8619`.
- Cleanup `e056823` is pushed to PR99; all seven checks passed:
  [CI34007584412](https://github.com/TensorLink-AI/Gnomon/actions/runs/34007584412)
  (Python3.11/3.12/3.13, harness, real containers and installed package/plugin)
  and [Container34007584434](https://github.com/TensorLink-AI/Gnomon/actions/runs/34007584434).
  This follow-up records results only. Do not edit source or commit during
  identity-pinned harness tests.

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
Residue-pass recovery53482fc; benchmark-cull recovery1642cb2; runtime recovery333ed2c;
earlier benchmark/design archives2cba20e.

Branch `codex/gnomon-ephemeris-ledger`, [PR99](https://github.com/TensorLink-AI/Gnomon/pull/99).
It is now ready for review rather than draft; do not change that status. Base remains
`claude/enterprisebench-multi-domain-n4xyjf`; do not silently retarget or merge.
Unrelated root scratch files are untouched and must not be staged:
`-`, `Continue`, `Current`, `Immediate`, `Use`, `accelerate`, `actual`,
`cases.`, `optimizing`, `that`.

Continue from this checkpoint after automatic compaction, not from scratch.
