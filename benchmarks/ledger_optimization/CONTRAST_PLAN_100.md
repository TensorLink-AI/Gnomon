# Candidate 100 prospective development comparison

Frozen before any candidate-100 paid dispatch. This is a development experiment,
not permission to open the untouched final set or a claim of 20% improvement.

The primary comparison is fresh Hermes + Gnomon + ledger versus fresh Hermes +
Gnomon without ledger. Hermes alone is secondary. All arms receive the same
current-CV table and calculated ranks. Only the ledger arm receives the contrast
with historical evidence it already requested. No additional provider calls,
history queries, data, numerical budgets or model choices are introduced. The
agent still selects the forecast. Existing native-memory availability remains
matched and memories are isolated by arm/series.

The trial uses the same development cohort and schedule as frozen 097: four
series, 26 sequential origins, three arms, seed 7. All 312 sessions are planned;
36 are the first-three-origin pilot and 276 are continuation. The model remains
Engy `deepseek-v4.1-flash`, temperature 0.2, 3,072 output tokens, with published
Gnomon 1.2.0 and the pinned build. Budgets remain 16 agent requests, 60 numerical
attempts, 480 agent seconds and the existing selection reserve. A fresh collection
costs three CV fits plus one unselected production forecast. No earlier run's
executions, predictions, selected configurations or memory are reused.

Continuation uses only the existing completion gate: all 12 cases valid per arm,
at least 11 full workflows per arm, and no audit failures. It does not use RMSLE.
The pilot is retained once in the final development report. Failures and unknown
API usage remain in evidence; accuracy results are never a reason to selectively
restart, stop early, or omit a session. Uncertainty is exploratory because this
cohort has informed development. Prior 097 scores are a descriptive reference,
not matched controls for estimating candidate 100's improvement.

## Admission and validation

`contrast_plan_100.py` verifies the exact parent plan, candidate capsule, task
source and synthetic worker proof. It checks the retained runtime/report hashes,
six complete synthetic sessions and 36 independently audited annotations before
writing a new plan using exclusive creation. Code, task, proof or runtime changes
require a new frozen plan. It never accesses credentials or launches a process.

`contrast_readiness_100.py` checks the predecessor on the worker host. It checks
both process identities, the complete 312-session terminal record, archive and
inventory hashes, every inventoried file, source/task identity, exact cohort and
cost coverage. A live PID/start-tick/boot match overrides claimed completion.
This is a read-only prerequisite check, not a substitute for the final independent
numerical audit or candidate runtime verification. A launcher still must enforce
those checks before dispatch.

Ten tests passed: unchanged comparison/gate preservation, no prior-state reuse,
changed parent/tasks/modules/proof/runtime rejection, symlink rejection, complete
synthetic predecessor admission, live-process rejection, altered retained evidence,
wrong cohort, wrong source and missing cost/completion records. Synthetic terminal
receipts used in tests are fixtures, not evaluation outcomes.

## Evidence and remaining work

- Plan: `results/contrast-100-prospective-plan-002/plan.json`.
- Tests: the same directory's `tests.stdout` and `tests.stderr`.
- Committed receipt: `evidence/contrast-100-prospective-plan-002.json`.
- Plan 001 is retained and explicitly superseded before dispatch. Plan 002 adds
  the original objective's prior-ledger comparison requirement; neither has paid
  results. This is not an outcome-driven amendment.

Wait for 097's terminal archive and independent audit, then finish and test the
exact launch/continuation integration. Current synthetic integration demonstrates
correct summaries and transport, not a model's behavioral or accuracy response.

The untouched final evaluation still requires at least 20% lower mean per-case
RMSLE than matched no-ledger, a paired 95% interval excluding zero improvement,
and improvement over a frozen pre-optimization ledger baseline. The goal's old
1.1.9 baseline wording and the user's newer 1.2.0 runtime instruction must both
be accounted for by freezing baseline behavior and its integration separately.
Historical scores alone cannot satisfy that comparison. Final/protected data
remain closed; main and PyPI remain unchanged.
