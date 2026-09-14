# Development091: valid historical comparisons restored

The fix excludes each execution recorded at or after its first target before
fingerprint ambiguity and duplicate checks. A retrospective backtest can no
longer invalidate an earlier prospective forecast just because its history
differs. The exclusion identifies execution, provider, recording time and first
target. Conflicts between eligible prospective forecasts still reject the origin.
Identical eligible retries still count once. No caller retrospective label is
trusted as the eligibility rule.

The product change is committed only on dev/ledger-optimization. The original
published installation was not modified. For the original-data replay, a narrow
port of its history comparison runs as separately hashed development code over
the pinned1.2.0storage and engine. This is not an unmodified published comparison
or a released1.2.0fix. Main/PyPI remain unchanged.

## Evidence coverage on the frozen sixteen queries

All16queries from090passed the new091audit. Original final ledgers and logs
were copied and checked against their original030hash inventory. The same
series, origins, candidates, pages, cutoffs and actuals apply to both versions.

| Series | Original saved review entries | Corrected entries |
|---|---:|---:|
| item_1047756_store_23 |4|52|
| item_1304243_store_32 |4|52|
| item_1372862_store_12 |4|47|
| item_1457251_store_14 |6|53|
| Total |18|204|

An entry is a matched pair/origin in a queried lifetime window. These counts
repeat historical origins across query dates and may repeat an origin across
provider pairs; they are not204independent observations. The original saved
reviews retain18entries. The unmodified comparison applied to copied final
ledgers returned zero at these same16queries because the additional retrospective
executions made every eligible pair ambiguous. The corrected comparison restored
204, including all18original entries without changing their execution IDs,
actual IDs, predictions, counts or metrics. It adds186over the originally shown
evidence. For the first series at round25, lifetime support changes from1to25.

The original review happened before current-origin backtests. In the synthetic
replay all operations within that origin share a recording timestamp, so the
final database includes those later-in-sequence backtests at the same cutoff.
The new eligibility rule resolves the contamination using actual recording and
target times, without inventing timestamps or editing history.

## Verification

Source frozen6629297before corpus replay. Product ledger regression69tests passed,
including differing/identical late retries, exact first-target boundary, genuine
prospective ambiguity, changed eligible provider identity, late identity changes,
duplicates and existing cutoff/identity checks. Context, decision-memory and
evidence-summary regressions added28passes,97total. No installed package edits.

Independent replay audit passed19,856checks. It verifies the exact old scored
entries survive, all new scored executions precede their first targets and are
recording-visible, histories/cutoffs are ex-ante, actuals meet both availability
cutoffs, provider/task/request identities match, and RMSLErecomputed directly
from saved predictions and actuals matches each returned score. It does not use
the proposed eligibility helper to decide those validation conditions. Forecast
dispatch was replaced with a rejecting guard during replay; no forecasts ran.
Original source hashes and each initialized working ledger remained unchanged.

The16paired queries took69.178seconds. Reads across filtering plus both query
versions:1,224execution reads,284history comparisons,16actuals reads. Independent
verification added396execution reads and16actuals reads. SQLite internal reads
are not counted as public calls. No new forecast fits or API calls were made by
the corpus replay; regression-test fixture forecasts are test setup, not new
experimental predictions. Raw outputs, corrected/baseline full queries, copied
ledgers, checks and source hashes are archived with a separate receipt.

The pinned runtime build is1.2.0+ga38cd0cad353.s9723394ccb6d. The development
history module hash is82fbac7134d17e3c4dfb1d091f10f59a7b5cb73555f435c7007fcc97d8ea3f42.
It uses pinned private storage APIs for this prototype and must not be described
as a public standalone extension contract. The narrow product fix and prototype
diff are both retained for review.

## Implication for the goal

This identifies a concrete reason accumulated ledger evidence was underused in
the sampled original workflow. It does not establish how often the issue affected
every original query or how an agent would choose models with the restored
history. The original030forecasts and comparative scores remain unchanged.
No new agent accuracy, no20%improvement and no held-out95%claim are established.

090's original exact-replay gate remains a preserved failure. This is a new
correctness experiment, not a reclassification of that failure. A prospective
agent workflow should now use the corrected comparison with explicit development
code provenance and measure whether the additional valid history changes decisions
usefully.089's token result applies to the old saved reports; richer restored
windows can change payload sizes. Final/protected data stay closed and the full
forecasting objective remains unmet.
