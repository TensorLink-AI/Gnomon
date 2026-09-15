# Candidate-107 paid admission

`launch_planning_107.py` admits only the prospectively frozen development plan
`99683f980b5a4b31ff39886d385e57c87b1f0d0b5e714c638323638974c4cb48`.
It has no final-data operation and cannot change the three arms or cohort.

Before any reservation or credential access it verifies worker/protocol proof
hashes, the complete 312-session synthetic host proof, both synthetic terminal
archive identities, the synthetic plan's matching worker/runtime/stage settings,
and the actual local runtime/build. An admission manifest additionally binds
the launcher source, its indirect host imports, host proof, synthetic plan and
predecessor reconciliation.

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

Nine admission tests pass, including pilot-only proof rejection, source drift,
missing terminal evidence, count-type errors, duplicate reservations/workers,
changed bindings, indirect-import source binding and cross-stage reservation
misuse. Raw attempts are retained under `results/planning-dispatch-107-tests-001`
through `-003`; the latter two treat Python warnings as errors. These unit tests
did not establish the full host preflight; it subsequently passed separately
as recorded in [PLANNING_HOST_107_RESULTS.md](PLANNING_HOST_107_RESULTS.md).

At the initial admission implementation checkpoint, no paid candidate-107
reservation had been made. The completed host preflight and candidate-100
terminal reconciliation subsequently allowed the source-bound admission
manifest and dispatch bundle to be finalized.

`build_planning_bundle_107.py` prepares that bundle only after the complete host
proof and predecessor reconciliation exist. It verifies both original synthetic
archives against their current evidence, copies the exact frozen plan, worker,
proofs and transitive host imports, and checks the copied inputs in an isolated
Python process. The archive includes a SHA-256 inventory; every member is read
back and checked before a build receipt is written. No credentials, runtime,
final dataset or stage reservation are bundled. Original-host process, archive
and runtime checks remain required at dispatch.

Six archive integrity tests reject changed bytes, duplicate/omitted/extra files
and symlink substitutions. Together with the nine admission tests they pass
with warnings treated as errors (`results/planning-bundle-107-tests-001`).
These tests do not stand in for full host proof or authorize a paid launch.
The full build, copied pod checks and inspected one-shot pilot admission have
since completed; [the result](PLANNING_HOST_107_RESULTS.md) preserves their
identities and the paid controller's start. This does not authorize automatic
continuation, selective retries or access to final data.
