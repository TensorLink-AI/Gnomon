# Candidate-107 paid admission

`launch_planning_107.py` admits only the prospectively frozen development plan
`99683f980b5a4b31ff39886d385e57c87b1f0d0b5e714c638323638974c4cb48`.
It has no final-data operation and cannot change the three arms or cohort.

Before any reservation or credential access it verifies worker/protocol proof
hashes, the complete 312-session synthetic host proof, both synthetic terminal
archive identities, the synthetic plan's matching worker/runtime/stage settings,
and the actual local runtime/build. An admission manifest additionally binds
the launcher source, host proof, synthetic plan and predecessor reconciliation.

The narrow candidate-100 predecessor path preserves the original audit failure.
It requires the original host boot, dead original controller/worker identities,
the complete original archive, the separately reproduced known audit failure,
the corrected 312-session report, unchanged costs and recording of the original
nonzero exit. It never deletes INCOMPLETE or pretends the original audit passed.
This is an explicit admission amendment supported only by that exact evidence;
other failures remain rejected.

Fixed remote destinations are under `/root/gnomon-ledger-ml-v3/code/results`:
`planning-107-pilot-001`, `planning-107-pilot-launch-001`,
`planning-107-development-001`, and `planning-107-continuation-launch-001`.
`planning-107-dispatch-registry-001` holds atomic one-shot stage reservations.
Each worker consumes its reservation once while its admitting controller is
still live. A failed or partially launched stage cannot be silently restarted.

Continuation additionally checks the terminal 72-session pilot before reserve,
then checks copied state and re-audits it before reading credentials. Each stage
is supervised and archived separately. Pilot quality, rather than accuracy,
controls continuation. There is no automatic continuation to paid or final work.

Eight admission tests pass, including pilot-only proof rejection, source drift,
missing terminal evidence, count-type errors, duplicate reservations/workers,
changed bindings and cross-stage reservation misuse. Raw attempts are retained
under `results/planning-dispatch-107-tests-001` and `-002`; the latter also treats
Python warnings as errors. These unit tests do not establish the full host
preflight, which remains independently running.

No paid candidate-107 reservation has been made at this commit. The complete
host preflight and candidate-100 terminal reconciliation must exist before a
source-bound admission manifest and dispatch bundle can be finalized.
