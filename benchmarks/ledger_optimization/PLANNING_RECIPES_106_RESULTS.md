# Planning-time recipes 106 — supported history, blocked immediate action

Do not deploy this version. It retrieves repeatable historical recipes at the
planning point, but its immediate backtest call is blocked by a missing initial
checkpoint. A dependency-aware action sequence is needed before an agent test.

The fixed replay covers all 186 audited sessions (62 ledger sessions) in the
prospectively specified candidate-100 prefix. All 63 actual ledger reviews are
included; no current backtest, checkpoint, budget or additional page was invented.

| Structural observation | Count |
|---|---:|
| Actual review calls | 63 |
| Reviews before a current backtest | 62 |
| Reviews without an initial checkpoint | 62 |
| Cold reviews with no historical catalog | 4 |
| Reviews with eligible historical recipes | 46 |
| Eligible recipe entries | 137 |
| Displayed entries under the three-recipe limit | 133 |
| Reviews with an immediately admissible backtest | 0 |
| Partial source pages | 1 |

All 133 displayed backtest calls correctly say `checkpoint_required`. The one
review with a checkpoint (item 1047756/store 23, origin index 4) had no eligible
untried recipe. Repeated reviews, configurations and origins are not independent
samples. Support is a union of distinct origins within each recipe, not a shared
cohort that licenses a global model ranking.

The renderer orders by support count and configuration identity, not outcome
magnitude. It retains losses/ties through the original evidence references and
never selects a forecast. The 59 rendered payloads were 1,712–3,937 compact JSON
bytes, median 3,881. This is potential additional context cost, not a measured
token saving. The four cold responses are counted separately, not dropped.

## Verification and retention

Protocol freeze: `123e8ca8`; implementation before real replay: `a3c95991`.
Ten unit tests passed. The real replay passed 1,089 checks over source hashes,
actual call order, task/configuration identity, independent support unions,
recipe order, shared budget capacity, evidence pointers and unchanged inputs.
It used the exact frozen common configuration validator extracted from its
AST, without importing or running any numerical prediction function.

No model fitting, Engy requests, extra ledger queries, accuracy measurement or
live-source changes occurred. The replay does not establish that an agent would
use these recipes, save calls, or forecast better. The 20% target remains unmet;
held-out numerical data stayed closed.

Raw results and per-review outputs are under
`results/planning-recipes-106-offline-001/`; argv, complete stdout/stderr and exit
status are under `results/planning-recipes-106-launch-001/`. Test output is under
`results/planning-recipes-106-tests-001/`. The compact authenticated receipt is
[evidence/planning-recipes-106-offline-001.json](evidence/planning-recipes-106-offline-001.json).

## Next mechanism to test

Preserve these negative usability results. A separate prospective amendment can
offer the common required `start` action, followed by a task-bound conditional
backtest after successful checkpoint creation. Charge both batches and retain
the final-fit reserve; recheck time, phase and budget between calls. Exclude the
baseline recipe that `start` itself executes. Never mark a future conditional
step as executable now, substitute a checkpoint that does not exist, or count
this mechanical fix as evidence of lower forecast error. Neither this result
nor that proposed amendment changes the currently running experiments.
