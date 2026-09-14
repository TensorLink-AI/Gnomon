# Development090: response compatibility fixed; original replay gate fails

The actual v4 Hermes core logs GnomonSession.forecast responses. The088filter
was tested with complete engine execution envelopes and incorrectly required
an embedded request in both forms. A frozen public1.2.0probe confirmed two
representation failures: the session response omits request; live engine
execution dictionaries can contain tuples while the saved ledger returns JSON
lists. Both valid recorded executions were rejected.

Correction33349cf makes the embedded request optional for session responses
while checking it completely when supplied. Provider, revision, fingerprint
and forecast result remain mandatory and must match ledger.execution. Arrays
and equivalent integer/float values compare canonically; booleans do not equal
numbers. Every supplied event-request field is checked, including season,
covariates and all cutoffs. Required series/unit/history/origin/target fields
remain mandatory. Stored result metadata.config is also authoritative when
present. Missing unverifiable references still fail closed.

The corrected synthetic probe accepted both response forms with one public
execution read each, no provider calls during filtering, and no query mutation.
Twenty-three unit tests passed, including six new representation/identity
checks. The088recording-visibility regression passed all38public-runtime checks.
Each of the original and corrected090probes made2synthetic forecasts,4total;
regression/replay made zero. No paid calls were used for090.

## Actual-ledger replay stopped on the first mismatch

Frozen320acb9then attempted the sixteen already-selected089queries against
copies of original030final ledgers and original event logs. It stopped on the
first query rather than relaxing the comparison: item_1047756_store_23, round1,
origin2016-08-30. Candidate identity and pagination matched, but previously valid
cards became insufficient_evidence with ambiguous_inputs_at_origin. The failure,
full response, copied ledger and source hashes are retained. Only the first
series's final ledger/log and first query were accessed; the remaining15query
checks were not completed and are not passed.

Read-only inspection of that copied database identifies the conflict:

| Executions for the same providers and August16origin | Recorded at | First target |
|---|---|---|
| Original prospective seasonal and Ridge forecasts | August16 | August17 |
| Later retrospective seasonal and Ridge backtests | August30 | August17 |

Public1.2.0compare_history applies the requested August30recording cutoff, so
both sets are visible to that query. It groups by provider/origin and tests all
fingerprints for ambiguity before checking ex-ante eligibility of only the
first selected execution. The retrospective runs therefore make the valid
prospective runs appear ambiguous. They must not supply scored production
evidence, but their presence should not invalidate unrelated earlier forecasts
solely through a different retrospective input fingerprint.

This is distinct from088's future-recording catalogue issue: here the later
backtests are already recording-visible. In the synthetic replay clock, every
operation within an origin has the same timestamp. The original saved review
was produced before those current-origin backtests; replay from the final
database includes them. The observed mismatch does not establish that original
scores or agent results were numerically wrong, and none were changed. It does
establish that the complete integration gate remains unsatisfied.

The next experiment must isolate ex-ante eligibility before ambiguity detection,
retaining genuine conflicts between two eligible prospective executions. Preserve
the original090failure; do not change its exact-equality gate after seeing it.
Any corrected query that recovers previously excluded older origins is new
development evidence, not a rewrite of the saved original studies.

Evidence roots: results/session-review-090-initial, corrected,
visibility-regression and replay, with logs under results/session-review-090-logs.
Runtime was published1.2.0build1.2.0+ga38cd0cad353.s9723394ccb6d. Package inspection
was used for development diagnosis after reproducing the failure; this was not
a first-time black-box acceptance test. No main/PyPI release or protected data
access. The formatting089gate is independent and does not override this failure
or the unchanged20%forecasting objective.
