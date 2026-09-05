# Current delivery checkpoint

Updated 2026-09-06. Gnomon is the provider-neutral toolkit; Ephemeris is one optional
inference connector. The user explicitly requested PR creation, PyPI
publication and removal of obsolete documentation/benchmarks.

## Implemented

Typed provider-neutral inference, callables/fresh factories, Ephemeris HTTP mapping,
shared Python/CLI/MCP session, bounded data/result references, append-only ledger,
budgeted backtests, cutoff-bound routing, optional temporal calculations and a
matched ordinary/lean/full evaluation harness.

The cleaned release candidate is reverified at 97/100. Publication does not earn
the missing live-evidence points.
See [PLAN.md](PLAN.md) and [progress.json](progress.json).

## Pending external evidence

1. Actual agent comparison: model, endpoint, credential environment-variable name
   and spending approval.
2. Live Ephemeris: confirmed base URL, credential environment-variable name and
   permission/budget for billable wake-up and inference.

No authenticated service inference, paid agent comparison or automatic action has
been performed. Do not extract keys from repository instructions or treat HTTP401,
scripted agents or local fixtures as completion of these gates.

## Cleanup and recovery

Commit `2cba20e` preserves the pre-cull implementation, obsolete docs, old
benchmarks and detailed audit/handoff logs. Unrelated root scratch files are
neither removed nor included in the PR.

First-candidate local verification: 2666 passed, 7 skipped in128.19s with actual isolated
software/service tests.224 focused tests pass on each Python3.11/3.13;54 final
documentation/progress/discoverability checks pass. Final wheel/sdist metadata,
clean installed CLI/MCP/provider/ledger and separately rebuilt plugin smoke pass.
An initial stale-image mismatch and the plugin's old version range were corrected
and reverified, not waived. Numeric goldens changed only version-derived identities.

The active docs now contain24 files and the benchmark tree39 Python files.
Tracked historical result files were removed; untracked user files are untouched.
The CI retains production, current harness and isolated-container gates.

Draft [PR #99](https://github.com/TensorLink-AI/Gnomon/pull/99) is open against the
original checkout branch. Its first-candidate CI passed across Python 3.11–3.13.
Version 0.8.0rc1 published successfully through trusted publishing before the latest
naming correction reached the workflow; cancellation found it already completed.
Its immutable tag and artifacts are not overwritten.

Iteration 22 prepares 0.8.0rc2: remove implementation-specific naming from active
source and docs, introduce Gnomon independently of connectors, and fix container
tag generation for PEP 440 prereleases without promoting them to latest.
Corrected-candidate checks: 74 focused tests pass on each Python 3.11, 3.12 and
3.13. Wheel/sdist metadata and naming scans pass, as does the clean installed
CLI/MCP/provider/ledger/plugin walkthrough. Ruff, compilation and whitespace checks
pass. The full suite with the rebuilt isolated service image passed: 2667 tests,
7 skipped in 134.34s. All PR CI jobs passed on commit `6ddfa8f`.
The corrected local wheel SHA256 is
32fd32a797a0bac4a9197fd901fd3d879a0abf4b9297f211299ba730a4119cd7.
Version 0.8.0rc2 published successfully: release run 33993727253 and container run
33993727297 both passed. The CI wheel and source archive match the local builds.
Publication still does not complete either live-evidence gate.

Iteration 23 incorporates the user's final plain-English README and agent-skill
request in 0.8.0rc3. The skill's tool-call example executes against the real offline
session; guidance distinguishes local models from optional connectors, retrieval
from repeated inference, and recorded decisions from action permission. UI metadata
no longer overclaims trusted answers. Skill validation and 50 targeted checks pass
before the version bump. Rebuild and verify the final packaged skill before release.
