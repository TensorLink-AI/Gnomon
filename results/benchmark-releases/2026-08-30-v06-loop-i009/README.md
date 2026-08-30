# v0.6 loop I009: executable value-of-information recovery

Decision: **complete / promoted.** Gnomon now projects existing recovery prose
into a ranked `recovery_plan`. Exact forecast patches are emitted only when the
current response determines them; missing paths, frequency choices, malformed
schemas, and other external choices remain explicit user-input actions.

The frozen six-case RecoveryBench completed serially. All three deterministic
forecast repairs succeeded through the real public runner in one call and
published only the tier each rerun earned. All three external-choice failures
withheld argument patches. Manual inspection also exposed and fixed two general
boundary contradictions: malformed decision actions now return
`INVALID_ACTIONS`, and sufficiency/resolution now agree for refused and usable
forecasts.

Canonical engine fields were unchanged except for those two documented typed
contract corrections and the additive plan/reference. The final full local
suite passed 2,574 tests with 11 skips; the focused suite passed 82 tests. Raw
resumable evidence remains under `results/v06-p9-i009-recovery-*`.
