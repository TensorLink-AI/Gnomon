# Terminal-gated dispatch for candidate 100

The next candidate is already frozen as plan 003, capsule 003 and staged bundle
002. `wait_contrast_100.py` can wait for the original 097 controller and worker,
then invoke that bundle's existing candidate controller exactly once. It does
not change the trial, candidate, runtime, tasks, evidence, or API budgets.

The waiter checks boot ID and process start ticks as well as PID. Missing,
reused and zombie processes are terminal identities; missing completion evidence
still prevents dispatch. It hashes the original process receipts and stops if
they change. Polling creates timestamped liveness records every 30 seconds.
It reads no credential file and strips inherited API keys and PYTHONPATH from
the child environment.

At startup and after waiting, the waiter checks the exact bundle archive hash,
binds the payload inventory to that archive, checks all 79 files, rejects extra
files and symlinks, and checks the corrected plan hash. A retired bundle stops
dispatch. It requires separate fresh state, pilot and controller paths outside
old evidence, runtime and source directories. A global exclusive reservation in
the bundle is written before spawning. Failure never triggers another attempt.

The waiter requires the old controller's successful terminal receipt before
starting the new controller. That controller's existing launcher then performs
the full original-run inventory, cohort, cost and independent numerical re-audit,
checks the candidate's source and tested runtime, and only then reaches its
credential callback. The waiter does not substitute its lighter terminal check
for those gates. Rejected admission is retained without a provider call.

This authorizes only the frozen 36-session development pilot after the 312-session
097 predecessor has completed. The candidate controller archives success or
failure and the waiter records its exit. Neither starts continuation, another
candidate, an extra arm, or a final evaluation. A failed pilot completion gate
remains an observed result; it does not authorize a retry.

Tests exercise live waiting, missing/incomplete terminal evidence, changed
process receipts, bundle revalidation, reservation before failed spawn, rejection
without retry, protected paths, PID reuse/zombies/boot changes, the actual staged
bundle, simultaneous payload/inventory tampering and retirement. Synthetic tests
mock the child controller; the paid launcher itself retains its separate tested
gates. Starting a waiting process is not evidence that a pilot has launched.
