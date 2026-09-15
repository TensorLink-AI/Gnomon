# Candidate-107 host execution preflight

The prospective protocol is `PLANNING_SEQUENCE_107_DEVELOPMENT.md` (commit
8c8d223c). The new host primitives use the unchanged tested worker capsule and
common chain/copy functions, with an explicit host-only six-origin prefix.
They do not invoke the old worker main's three-origin pilot path.

`planning_host_107.py` runs a 72-session pilot or a 240-new-session continuation.
It checks exact job identities, source/runtime/build hashes, terminal pilot
processes on their original boot, archived bytes, copied-prefix audits, each
retained grade and score, full recipe annotations and cost coverage. It never
drops failed sessions or gates on accuracy. Pilot quality requires 24 valid
forecasts and at least 22 complete workflows per arm. A quality failure is
retained as completed evidence with a failed promotion gate.

`probe_planning_host_107.py` runs both full stages through actual subprocess
controllers, Hermes tools, numerical backtests, memory, archives and audits.
Only synthetic jobs, their authentication hash and upstream API responses are
substituted. Four scripted model replies and eight numerical attempts per new
session imply 1,248 scripted responses and 2,496 numerical attempts across 312
unique workflows. Copied pilot sessions are not executed again. The test checks
native-memory continuation/isolation, two parallel series, and mature recipe
handoffs. No genuine credentials or Engy requests are needed.

The eight unit checks passed in `results/planning-host-107-unit-001`. They are
not a substitute for the full host preflight, whose result is initially pending.
Once that preflight starts, its source-identity files must stay unchanged until
it finishes. Failures and their costs must remain; no automatic retry exists.

These are internal execution primitives, not a public paid dispatcher. Paid
admission still requires an exact machine-readable plan, full host proof,
terminal candidate-100 predecessor verification/reconciliation, a one-shot
exclusive reservation, and credential access only after those checks pass.
No held-out data, main branch or PyPI changes are authorized by this preflight.
