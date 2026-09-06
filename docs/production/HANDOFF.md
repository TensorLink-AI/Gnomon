# Delivery checkpoint — iteration 26

Updated 2026-09-06. The user requested removing obsolete benchmark machinery,
following the breaking runtime cull. Do not restore retired features for their tests.

## Current scope

One provider-neutral execution session with explicit input/time semantics, budgeted
evaluation, optional temporal ledger and optional time arithmetic. Ephemeris is one
connector; local models and user libraries use operator-owned callables/factories.
Runtime remains 26 modules/5,801 lines, down from 118 modules/67,013 lines.

One matched ordinary/lean/full agent evaluation remains. Benchmark implementation
(`workflow/*.py` plus `common/*.py`) is 3,351 lines, down from 5,289 (about37%).
Removed promotion/audit/generation runners, old smoke cases, publication grading,
host-generated repair/outcome answers, response caches, sample fan-out, hidden
transport retries/token escalation and old journal migration. Model transport is
188 lines instead of1,020. Dead report fields no longer label tool calls redundant.

Keep independent numerical, cutoff, failure, unknown-cost, isolation and committed
episode regressions. Case and score schemas are now v2; current attempt receipts
are required for resume. Old formats fail explicitly. Use fresh run directories.
The retained cohort's inputs, oracle answers and reveals were checked unchanged
against1642cb2; only retired fields and version differ. Manifest hashes regenerated.

## Verification

- Full local suite:742 passed,29 opt-in skips in31.11s.
- Real software/service containers:76 passed in33.12s, no paid model calls.
- Ruff, compilation and whitespace pass.
- Current service image rebuilt from the prior final0.9 wheel:
  `sha256:1f67cc447f1aa37ac7e2ed64755346381c9a6842b80f2a8f2b43a39564b3c509`.
- Baseline1642cb2 had all7checks green: CI34004767226 and Container34004767228.
  The benchmark-cull update must pass its own PR checks; do not substitute baseline CI.
- No runtime, provider plugin or packaged skill behavior changed this iteration.

Do not commit or edit source while matched-harness tests pin their identity.
[progress.json](progress.json) remains97/100; fewer lines earn no extra points.

## Remaining external evidence

- Real matched agent comparison: confirmed model/endpoint, credential environment
  variable and spending approval.
- Live Ephemeris: confirmed deployment URL, credential environment variable and
  budget for potentially billable wake-up/inference.

No paid comparison or authenticated inference has been performed. Local HTTP
fixtures and scripted agents do not satisfy these gates. Do not extract credentials
or repeat unauthenticated probes to bypass this authorization boundary.

## Distribution and recovery

Source is0.9.0.dev0, breaking and unreleased. Published0.8.0rc3 and its tag remain
immutable; this cleanup does not upload to PyPI. No user database, saved forecast
or user input data was deleted. Removed benchmark source/tests/fixtures are
recoverable at1642cb2; runtime cull recovery333ed2c; earlier archives2cba20e.

Branch: `codex/gnomon-ephemeris-ledger`; draft
[PR99](https://github.com/TensorLink-AI/Gnomon/pull/99).
Base remains `claude/enterprisebench-multi-domain-n4xyjf`; do not change it silently.
Unrelated root scratch files remain untouched and must not be staged:
`-`, `Continue`, `Current`, `Immediate`, `Use`, `accelerate`, `actual`,
`cases.`, `optimizing`, `that`.

Continue from this checkpoint after automatic compaction; do not restart the cull.
