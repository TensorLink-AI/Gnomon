# Current delivery checkpoint

Updated 2026-09-06. Gnomon is the toolkit; Ephemeris is the public inference
integration, backed by Paracast. The user explicitly requested PR creation, PyPI
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

Final local verification: 2666 passed, 7 skipped in128.19s with actual isolated
software/service tests.224 focused tests pass on each Python3.11/3.13;54 final
documentation/progress/discoverability checks pass. Final wheel/sdist metadata,
clean installed CLI/MCP/provider/ledger and separately rebuilt plugin smoke pass.
An initial stale-image mismatch and the plugin's old version range were corrected
and reverified, not waived. Numeric goldens changed only version-derived identities.

The active docs now contain24 files and the benchmark tree39 Python files.
Tracked historical result files were removed; untracked user files are untouched.
The CI retains production, current harness and isolated-container gates.

The requested release is0.8.0rc1, explicitly a prerelease while live evidence is
pending. The final local wheel SHA256 is
191fcd80c1076f005db1b19039930a3b6d433a08b052ace87d272fb014280bae;
sdist036cbfa1a73d5d42e2c527e0b8c04f516940e214aeae0da4611786119a25bd14.
Build artifacts live under /tmp/gnomon-080rc1-final-dist locally, not in Git.

Next: push the review branch, open the PR, publish the explicitly authorized
prerelease through trusted publishing, and confirm actual PyPI files. No completed
PR/publication is claimed in this pre-push checkpoint.
