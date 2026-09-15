# Observed search ceiling: selection alone cannot reach 20% in this subset

The calculation frozen in commit `52c27353` completed on the pilot and audit
batches 001–004: 186 authenticated sessions, including 61 complete three-arm
cases. The selected-score means exactly reproduce the independent audit.
Every matched case remains included; three unmatched sessions stay pending.

| Selection | Mean per-case RMSLE | Reduction versus actual no-ledger |
|---|---:|---:|
| Actual no-ledger agent | 0.472588 | Reference |
| Actual ledger agent | 0.482512 | −2.10% |
| Hindsight best of ledger's executed configurations plus fallback | 0.447582 | 5.29% |
| Hindsight best of no-ledger's executed configurations plus fallback | 0.444575 | 5.93% |
| Hindsight best of all three arms' executed configurations plus fallback | 0.435842 | 7.78% |

Even the combined hindsight set cannot reach the 20% target on these cases.
This limits selection-only changes within the observed searches. It does not
bound untried configurations, more effective search or another dataset, and
does not show that historical evidence could identify the hindsight winners.
The combined set merges different agent search trajectories; it is not a
budget-matched fourth arm or a deployable ledger policy.

For the two series with all 26 origins completed, the combined hindsight
reductions are 7.87% and 5.27%. The other series have only five and four matched
origins, with reductions 16.01% and 8.24%. These subsets were not selected for
their accuracy and do not define a new development or final population.

## Consequence

Do not launch another paid presentation-only variant on the claim that choosing
better among these same recorded forecasts can deliver 20%. Keep candidate 100
running unchanged and retain its full result. Further development must examine
search opportunity and whether the already frozen M5 task population offers
useful historical decision evidence; it must not silently weaken the control,
change budgets or select a final cohort after seeing its outcomes. The final
gate remains closed and the objective is unmet.

## Verification and retained failures

The loader authenticated 516 successful production forecasts across all 186
sessions, their exact origin-bounded requests, provider revisions, result
correspondence, host actuals and the common seasonal fallback. It passed 14,175
authentication, identity and selected-score checks. Source hashes remained
unchanged; no negative forecast values required clipping. Six pure-function
tests cover independent metric arithmetic, differing per-case/arm hindsight
winners, arithmetic rather than pooled means, failed workflows, pending arms,
identity conflicts, duplicate executions, invalid vectors, clipping and zero
denominators. The original five-test run is retained too.

Two initial loader attempts failed before producing a report. The first assumed
plain Hermes included Gnomon's status/result metadata; its frozen execution
object contains only execution ID, provider, revision and point. The second
mistakenly equated model configuration season with request metadata season.
The corrected loader checks the plain representation explicitly and checks
every request field against its recorded host request. Model configuration and
provider revision are verified separately. These were diagnostic-driver errors;
no original forecast, score or execution was changed or rerun.

Raw scripts, all attempts, input hashes and complete per-case results are in
`results/observed-search-ceiling-104-offline-001/`. Compact receipt:
`evidence/observed-search-ceiling-104-offline-001.json`. No provider/API calls,
ledger mutations or reserved-data access occurred. Main/PyPI are unchanged.
