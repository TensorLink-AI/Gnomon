# Candidate 100: ongoing later-origin development comparison

The continuation retains the complete 36-session pilot once and adds 276 later
sessions under the same three-arm Gnomon 1.2.0/DeepSeek v4.1 Flash protocol. The
pilot's negative result remains in the denominator. This is reused development
data, not the reserved final evaluation. The 20% target remains unestablished.

## Fourth continuation audit: 186 verified sessions

The next disjoint batch contains 93 valid full workflows and passes 128,144
independent checks. Combined coverage is 186 sessions and 227,677 checks. The
61 complete three-arm cases have mean per-case RMSLE 0.484505 (Hermes), 0.472588
(Gnomon), and 0.482512 (ledger): ledger is 2.10% worse than no-ledger. All 183
matched sessions completed without fallback; three other sessions remain
pending. These are audited snapshot counts, not the latest live progress.

The new batch preserves 784 agent requests/responses, 14,144,325 reported
tokens and one recovered API 502 with unknown usage. Its 93 readiness requests
report 1,302 tokens. The failed request and its unknown cost remain included;
the session completed successfully without a forecast-run restart. Dollar
costs remain unknown. Collection and audit added no provider or Engy calls.

The initial SSH observation timed out while preparing the evidence archive.
The existing remote archive subsequently verified in full against its inventory
and was transferred in 202 independently hashed chunks, all without chunk
failures. All 16,678 files and the 211,366,945-byte archive verified. The original
timeout, archive, scripts and corrected transfer are retained; no snapshot or
forecast was regenerated. The same documented offline cache-order audit
correction applies. Receipt: `evidence/contrast-100-development-audit-004.json`.

## Third continuation audit: 93 verified sessions

The next disjoint batch contains 33 completed, valid workflows and passes
39,597 independent checks using the separately documented verifier correction.
All 4,749 files and the 45,849,125-byte archive verified. Across the pilot and
three batches, 93 sessions and 99,533 checks are now independently verified.
This is an audit snapshot, not the latest live completion count.

Thirty cases have all three arms. Their arithmetic mean per-case RMSLE is
0.498401 (Hermes), 0.480151 (Gnomon), and 0.493400 (ledger): ledger is 2.76%
worse than Gnomon without ledger. All 90 matched sessions completed the full
workflow with no fallback. The three unmatched sessions remain pending; no
success or accuracy filtering is used. The new batch's 269 agent calls report
4,552,701 tokens, with no API errors or missing agent usage. Its 33 readiness
calls report 462 tokens. Dollar costs are unknown. These costs belong to the
existing experiment; collecting and analyzing it added no paid agent calls.

The per-origin analysis and search diagnostics are documented in
[CURRENT_HISTORY_CONTRAST_100_HISTORY_DIAGNOSTICS.md](CURRENT_HISTORY_CONTRAST_100_HISTORY_DIAGNOSTICS.md).
They provide no clear accumulating accuracy advantage yet. They also show why
three illustrative losses must not be generalized into less exploration by the
ledger arm: both Gnomon arms average 2.87 complete CV configurations per task.
Receipt: `evidence/contrast-100-development-audit-003.json`.

## Second continuation audit: 60 verified sessions

An additional disjoint batch of 19 completed sessions passed 21,928 independent
checks after correcting an offline verifier defect described below. Together
with the pilot and first batch, this covers 60 valid, complete workflows and
59,936 checks. All 20 matched cases enter the following comparison:

| Arm | Matched mean RMSLE | Matched reported tokens | API requests | Fits |
|---|---:|---:|---:|---:|
| Hermes | 0.512245 | 2,517,023 | 171 | 224 |
| Hermes + Gnomon | 0.498703 | 2,589,785 | 175 | 228 |
| Hermes + Gnomon + ledger | 0.516325 | 2,573,279 | 171 | 232 |

Ledger is 3.53% worse than no-ledger on this developing matched subset. No
accuracy target or final gate has passed. The run continues unchanged.

The new batch contains 165 agent requests/responses and 2,731,063 reported
tokens, with no agent API errors or missing agent usage. Its 20 readiness
requests report 266 tokens; one has unknown usage. Audits make no Engy or
provider calls. These are original-run costs, not additional audit costs.

### Offline audit correction, with original failure retained

The original audit exited 1 on a presentation comparison. A per-session
diagnostic isolated four later-origin ledger sessions. Their recent and longer
history windows first had different matched cohorts. The annotator uses an
in-memory review during `review`, then reloads its sorted-key JSON cache on
later operations. This changes the order of distinct labelled history groups.
The auditor incorrectly reused the original dictionary order. Scores, labels,
counts and the content-addressed evidence bytes matched; array order did not.

`contrast_audit_100.review_at_operation` now reproduces that serialization
boundary. Exact view equality and artifact-byte equality remain required;
output arrays are not arbitrarily sorted or compared as sets. Twenty-five
synthetic regression tests pass. On separate copies of an affected real
session, reordered groups, fabricated scores and fabricated sample counts
still reject after changing both response copies. Original evidence hashes
remain unchanged. The corrected full analyzer then passed all 19 sessions.

The frozen worker and its bundled original auditor were not edited. Its eventual
controller audit may therefore report incomplete because of this known verifier
defect; preserve that terminal record and reconcile with the separately hashed
offline correction. Do not restart sessions or describe the original audit as
passing. The raw failure, diagnosis, corrected audit and tests are retained in
`results/contrast-100-development-audit-002/` and
`results/contrast-audit-cache-replay-100-001/`. Receipt:
`evidence/contrast-100-development-audit-002.json`.

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
