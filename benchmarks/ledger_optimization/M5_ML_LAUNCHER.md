# M5 pilot and continuation launcher

`python -m benchmarks.ledger_optimization.launch_m5_ml --help` exposes the
fixed-seed pilot and continuation entry points. No final stage is supported.
The launcher verifies the exact frozen seed plan, development bytes, worker
preflight, host integration proof, stopped candidate-100 predecessor and actual
host runtime/build before reserving a launch. Output/controller roots cannot
nest in source, runtime or retained evidence directories.

Reservation precedes the controller subprocess. The child rechecks admission
and consumes that reservation exclusively, so manually invoking it again cannot
reuse the reservation. The pilot stamps the requested seed and host source
identity into its generated manifest. It records a failed quality gate without
pretending the archived evidence is missing.

Continuation verifies the complete terminal pilot and archive, copies its state,
and independently audits that copy against the fixed 72-session prefix before
reading credentials. It then runs 552 new sessions through two series chains,
preserving the original pilot, isolated arm memory and original outcomes. Its
full-stage audit and count checks must pass before a successful exit is saved.
Exceptions set the stop flag and retain incomplete receipts; no retry occurs.

## Current verification and dispatch status

Thirty-seven tests passed across launcher, inputs/reservations, controller,
terminal/archive and stage checks. Five new focused launcher tests cover
incomplete host proof, nested outputs, credential access after copied audit,
seed stamping/quality-gate distinction and live-predecessor rejection. These
tests use controlled worker substitutes where stated; they do not constitute
the full integration below.

The real CLI was also deliberately given the earlier two-series seed-19 proof.
It authenticated the real fixed plan, then rejected that insufficient host
proof with the expected cause. No output/controller directories, credentials,
worker execution, forecasts or API calls resulted. Exact commands and errors
are retained in `results/m5-ml-launcher-001`.

**Paid dispatch remains closed.** The required integrated host proof must bind
the current launcher/helper hashes and both frozen plan hashes, exercise eight
synthetic series per seed with two concurrent series chains, retain 72 pilot
plus 24 resumed full workflows per seed, prove unchanged/copied prefix state,
seed isolation and reservation re-entry rejection. The existing separate
two-series and controller probes cannot substitute for that test. A further
source-bound bundle/remote deployment is needed before any paid dispatch.

Candidate 100 must also finish first. Its original controller contains the
known cache-order audit issue; if it records INCOMPLETE, this launcher refuses
to override it. Retain and explicitly reconcile that original failure using
the corrected independent auditor before implementing any corresponding
predecessor-admission amendment. Do not rerun forecasts to repair an audit.

Source and evidence receipt: `evidence/m5-ml-launcher-001.json`. Main/PyPI,
current paid sources and final targets remain unchanged. No new accuracy result
or claim of meeting the 20% objective follows from launcher implementation.
