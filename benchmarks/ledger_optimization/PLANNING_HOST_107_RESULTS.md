# Candidate 107: complete host preflight and paid pilot admission

The full synthetic host preflight passed before paid admission. It retained the
72-session pilot exactly once and executed 240 continuation sessions, covering
four series, 26 sequential origins and three arms. All 312 workflows completed.
There were 2,496 numerical attempts, 1,248 scripted responses and zero Engy calls.

The original workflow audit passed 231,836 checks with no failures or shutdown
gaps. The separate recipe audit passed 40,492 checks over 416 published
annotations, including 176 actionable next calls. These audit counts overlap in
their underlying evidence and should not be added as independent observations.
Actual recipe uptake is separately measured in [PLANNING_UPTAKE_107.md](PLANNING_UPTAKE_107.md).
All uptake in this preflight was scripted, not a measured agent response.

The pilot archive contains 7,578 files and has SHA-256
`35939a355f82a5d84c6c3edfe1683318686c31c7fd96a22f97acb4e8dc46a4c7`.
The complete-stage archive contains 45,439 files and has SHA-256
`9409e42b139ac175efb7511c28192756a6ca1bdd08e262113b92a638fb509fe3`.
The bundle builder reverified both archives against their unchanged saved
evidence before copying the exact source, proof and frozen plan. The committed
receipt is `evidence/planning-host-107-complete-001.json`; complete raw evidence
is under `results/planning-host-107-synthetic-001`.

The isolated 76-file dispatch bundle passed copied-input checks and archive
verification. Its SHA-256 is
`5639c48768c7872018e8225f6bf9ed4edbc0d34eab9f8fd0cab75dcb5b8327c5`.
Pod-side extraction verified every member. Read-only admission then checked the
actual pinned runtime, original-host terminal predecessor, preserved failure
and corrected report, source identities and complete preflight. It returned
checks_passed with no credentials read, reservation or provider/API calls.

After that check was inspected, the paid pilot controller started at
2026-09-15 15:54:28 UTC. Its one-shot reservation was consumed and the original
host subsequently confirmed both controller and worker live, with matching
PID/start-time/boot identities. At the first status check, two agent requests
were recorded and no case had finished. A later 16:03 UTC checkpoint records
10/72 sessions, all valid and workflow-complete, with zero fallback/API errors
in those completed sessions. The first two cases matched across all three arms
have identical mean RMSLE 0.411990. They are cold-start cases and establish no
ledger advantage; remaining-session costs and terminal audits are pending. See
`evidence/planning-107-pilot-progress-001.json` for that immutable live snapshot.
The launch is preserved in `evidence/planning-107-pilot-launch-001.json` and
`results/planning-107-pilot-dispatch-001`.

The pilot follows the [frozen protocol](PLANNING_SEQUENCE_107_DEVELOPMENT.md):
72 fresh sessions, three arms, published Gnomon 1.2.0, DeepSeek v4.1 Flash,
requested seed 7, identical numerical tools/current information/native-memory
access and budgets. Only the ledger arm receives optional plans based on history
it requested. Completion and integrity, not accuracy, govern continuation; no
pilot cases may be selectively rerun or removed. A later accepted continuation
adds 240 sessions while retaining the pilot and its memories once.

The previous [candidate-100 negative result](CONTRAST_100_NUMERICAL_RESULTS.md)
remains preserved. Passing this infrastructure preflight does not establish
ledger value, predictive superiority or the 20% objective. Main, PyPI and the
reserved final data remain unchanged.
