# Candidate 100: ongoing later-origin development comparison

The continuation retains the complete 36-session pilot once and adds 276 later
sessions under the same three-arm Gnomon 1.2.0/DeepSeek v4.1 Flash protocol. The
pilot's negative result remains in the denominator. This is reused development
data, not the reserved final evaluation. The 20% target remains unestablished.

## First continuation audit

The first five completed continuation sessions passed 5,094 independent checks,
including current-CV comparisons, recording-visible historical evidence and
execution accounting. All 579 files and the 4,181,543-byte snapshot archive
verified. All five forecasts and workflows are valid. The snapshot includes
every session then finished with grade and memory receipts, without filtering
on success or accuracy; the running worker was not modified or restarted.

Each session restored three prior outcomes. The audit checked seven to ten
matured production executions per session, including four to seven unselected
executions. Evidence carry-forward started no additional numerical calls. These
checks establish correct historical inputs and presentation on the audited
subset, not that the agent benefits from them.

The completed pilot and this disjoint batch cover 41 sessions and 38,008 checks.
Only the 13 cases completed by all three arms enter the combined comparison:

| Arm | Matched mean RMSLE | Matched reported tokens | API requests | Fits |
|---|---:|---:|---:|---:|
| Hermes | 0.525462 | 1,591,617 | 113 | 152 |
| Hermes + Gnomon | 0.530707 | 1,524,159 | 110 | 136 |
| Hermes + Gnomon + ledger | 0.533071 | 1,546,090 | 111 | 148 |

There is no ledger accuracy advantage on this small matched subset. Incomplete
arm groups are retained and await their counterparts; they are not selectively
added to the comparison. The live continuation is not complete.

The five-session batch contains 42 agent requests/responses and 628,627 reported
tokens, plus five readiness requests and 70 tokens. No API errors, missing usage
or orphan responses occurred in this batch. Dollar costs remain unknown. These
requests are part of the continuation, not additional audit executions. The
collector, independent analyzer and summary each passed on their first attempt.

Receipt: `evidence/contrast-100-development-audit-001.json`; raw retained inputs,
source/hash receipts and executable verification are in
`results/contrast-100-development-audit-001/`. Full pilot results are in
`CURRENT_HISTORY_CONTRAST_100_PILOT_RESULTS.md`. Main/PyPI and reserved data are
unchanged.
