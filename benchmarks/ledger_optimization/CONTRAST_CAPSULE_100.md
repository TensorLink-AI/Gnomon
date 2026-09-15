# Candidate 100 worker integration design

This is a separate capsule built from the exact frozen 097 worker. No live
worker modification or paid dispatch is authorized by building it.

All arms receive a current-CV table after successful `start`, `backtest`,
`review`, `status` and `commit`. The table includes every complete current
configuration (maximum imposed by the unchanged numerical budget), its canonical
settings, mean RMSLE, rank and three fold scores. This also exposes the baseline
scores that `start` previously computed without returning. Scores are recomputed
from the actual log prefix and the three visible input folds. Incomplete groups
are labeled; no forecasts, outcomes, additional fits or model choices are added.

A bounded pair view compares the newly backtested configuration with the lowest
current-CV alternative. Other operations show the two lowest current-CV means.
Ties use configuration ID only to choose presentation order; ranks remain tied.
An explicit existing review pair is honored if both configurations have current
complete CV. The focus rule and pair IDs are returned; it selects a comparison,
never a forecast. All other current configurations remain visible in the table.

The ledger arm may add historical comparisons only from the latest review that
the agent already requested and received for this exact task. No hidden query,
page fetch, new numerical observation or retrieval budget is introduced. An
absent pair/page is unknown. A prior-task cache is never reused as current.
The common CV view is the same function in all arms; raw facts and native-memory
availability remain matched under the unchanged 097 rules.

The host stores full content-addressed artifacts before returning their
references. Artifact hashes and an annotation audit log bind each response to
the exact existing execution-log prefix and input-file hashes. Agent tools may
read but cannot overwrite artifacts. No new general-purpose write tool is added.
All added source modules are included in the host's protected-source manifest.

Required before any prospective paid test: builder source-identity checks,
synthetic real-model lab tests across all arms and two origins, real Hermes
transport/retrieval tests, independent reconstruction of annotations, unchanged
fit and call budgets, clear response-size accounting, and a fresh frozen plan.
Wait for the existing 097 development run to complete and pass its full audit.
No accuracy-dependent continuation or final-set access is permitted by this
prototype. The 20% final target remains unestablished.
