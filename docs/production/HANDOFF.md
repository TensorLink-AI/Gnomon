# Delivery checkpoint — iteration 25

Updated 2026-09-06. The user explicitly requested removing features outside the new
goal. This supersedes the earlier legacy-compatibility constraint. Recovery:
`333ed2c`. Do not restore retired code merely to satisfy its old tests.

## Current scope and implementation

One provider-neutral execution session with exact input/time semantics, explicit
budgeted evaluation, optional durable evidence and optional temporal arithmetic.
Local models are operator-owned callables/factories; Ephemeris is one connector.

Runtime: 118 modules/67,013 lines before this cull → 26 modules/5,801 lines now
(about 91% fewer lines). Removed the old runtime/registry, context/scenario/publication/
effect stack, model catalogues/installers/adapters, monitoring and feedback.
No second legacy package or compatibility dispatch remains.

Python/CLI/MCP, README, concise current docs and packaged agent skill now match the
retained contract. Legacy commands/profiles fail explicitly. Repair, frequency,
migration and nonfinite-input tests now exercise retained code directly. Historical
imports use frozen fixtures, not the old writer; manifest escape/tamper is refused.

Context-engine benchmark cases/graders are gone. Agent metrics live only in the
benchmark harness. Full now means the same execution contract with ledger/time
tools enabled. Old full-arm results are not current evidence. Removing the empty
context oracle field changes the retrospective corpus hash, not its data or answers.

## Verification and next step

Current complete local suite: 760 passed, 29 opt-in skips in 29.96s. Separate pre-stdin-follow-up
software/service container suite: 77 passed in 29.53s.
Ruff, compilation, skill validation and whitespace checks pass.
Wheel/sdist and example plugin build; metadata and clean installed Python/CLI/MCP/
provider/ledger/plugin journeys pass against the final reformatted package.

Cull commit `aceeaf1` is pushed to draft
[PR99](https://github.com/TensorLink-AI/Gnomon/pull/99). CI34004475000 passed the
Python3.11/3.12/3.13, harness and real-container jobs; Container34004475029 passed.
The package job exposed a lost piped-CSV path in the reduced CLI. Piping serves
the new goal: restored it with an8 MiB bound and four current CLI regressions.
All760 local tests pass after the fix; rebuilt package/install checks pass.
Push the stdin follow-up and verify all current CI checks before crediting the gate.
Do not commit or change source while matched-harness tests pin their identity.
Supported-Python CI must pass before recrediting the release-check gate; current
score is 95/100, with that gate and the two external-evidence gates still unearned.
[progress.json](progress.json) tracks exactly100 possible points, not a guarantee
beyond the acceptance contract in [PLAN.md](PLAN.md).

## Remaining external evidence

- Real matched agent comparison: confirmed model/endpoint, credential environment
  variable and spending approval.
- Live Ephemeris: confirmed base URL, credential environment variable and budget
  for potentially billable wake-up/inference.

No paid agent experiment or authenticated service inference has been performed.
Local HTTP fixtures, scripted agents and previous unauthenticated401 responses do
not satisfy these gates. Do not extract credentials or repeat probes to avoid this
authorization boundary.

## Distribution and recovery

Source is `0.9.0.dev0`: breaking, unreleased, not automatically uploaded to PyPI.
Published `0.8.0rc3` and its tag remain immutable. No input data, user database or
saved forecast was deleted. Old implementation/tests/docs remain in Git at333ed2c;
earlier design/benchmark archives remain at2cba20e.

Branch: `codex/gnomon-ephemeris-ledger`; PR base remains
`claude/enterprisebench-multi-domain-n4xyjf`. Do not silently change the base.
Unrelated untracked root scratch files remain untouched and must not be staged:
`-`, `Continue`, `Current`, `Immediate`, `Use`, `accelerate`, `actual`,
`cases.`, `optimizing`, `that`.

No manual compaction operation is available. Continue from this checkpoint and the
progress file after automatic compaction; do not restart completed work.
