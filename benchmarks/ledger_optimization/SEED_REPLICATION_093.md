# Prospective seed replication for the guarded ML runner

The live guarded093 development experiment deliberately uses requested seed 7.
Both its worker constructor and its metering proxy hard-code that value. Changing
only a manifest label or worker argument would not produce a seed-19 replication:
the proxy would still forward seed 7. The live source and experiment are unchanged.

`seed_capsule_093` prepares separate source copies for requested seeds 7 and 19.
It requires the exact audited 24-file guarded093 source inventory, whose canonical
SHA-256 is `4ead8b05a2b15b2e9cb2a03c33fd7555dc04d1b985d23db2b5993cba1b61b121`.
It copies only those sources, with no datasets, credentials, runtime environment,
prior executions or agent memory. It refuses an existing output path, altered
source, symlinks and invalid seed settings before preparing a capsule.

For seed 7, all 24 source files are byte-identical. For seed 19, exactly one
declared setting in each of `worker.py` and `transport.py` changes. All model
families, tool implementations, budget limits, correction policy and prompts
remain identical. `capsule.json` records the base and derived hashes and requested
parameters. The copied historical `PROTOCOL.md` describes the seed-7 base; the
capsule manifest explicitly identifies the replication seed.

Use a fresh process rooted in each capsule so the original canonical package
path resolves to that copy. Never import both copies into a process with a cached
`benchmarks.hermes_ml_checkpoint_v6` module. Each seed also needs separate run,
project and Hermes-home directories; memory persists within an arm/series/seed
chain, and must not cross seeds or arms. All three arms in a matched replication
use the same requested seed.

The pair 7/19 is available for the prospective final design and must be included
explicitly in its final freeze. A requested seed is not proof that Engy honors
it or that repeated outputs are deterministic. Seed repetitions do not create
independent stores in the final uncertainty calculation.

## Local verification, no Engy calls

The test builds both capsules and launches a fresh probe process for each.
`probe_seed_capsule_093` verifies source hashes and import locations, constructs
the actual worker's agent factory with a mocked agent, and checks the requested
seed on three constructions representing successive remaining budgets. It then
exercises the actual local HTTP proxy with scripted upstream responses. All 16
forwarded requests carry the declared seed and unchanged model/temperature/token
limit, even if the incoming payload supplies a different seed. The last four
requests retain selection-phase guidance, and request 17 is rejected before any
upstream call. Original and forwarded payloads remain available in the probe
artifact, with no real credential or model response.

Tests also prove seed-7 byte equivalence, the two exact seed-19 substitutions,
original source preservation and rejection of modified or invalid inputs.

## Remaining gate

This is source preparation and a scripted settings/budget probe, not an actual
Hermes/Engy replication or final-run launcher. Full worker integration, state
isolation, source/data gating, runtime parity and complete-grid accounting still
need validation before final dispatch. In particular, the copied host's original
task-source hash guard and exact-source preflight remain in place; this helper
does not bypass them or authorize M5 target access.

Do not launch a second paid development run from this artifact while guarded093
is running. Do not open reserved data until the development and final-freeze
gates pass. Main, PyPI and the live pod's files remain unchanged.

## Full integration queued after development

A separate synthetic integration controller is queued on the pinned pod runtime.
It waits for development controller PID 3981368 (boot/start identity retained)
to exit cleanly with its complete audit before doing any numerical work. Queued
controller PID 4004888 was verified alive and waiting. It will run six synthetic
sessions for each requested seed, covering all three arms over two origins, with
real model fits, ledger maturation, native-memory restoration and cross-arm/seed
isolation checks. Upstream replies and service probes are scripted: zero Engy
requests. It preserves each subprocess, source hashes, transcripts, independent
audit and any failure. Pending integration is not passing integration evidence.

Receipt: `evidence/seed-integration-093-queued-001.json`; pod directory:
`/root/gnomon-ledger-ml-v3/code/results/seed-integration-093-queued-001`.
Before starting another paid run or changing the pinned runtime, inspect this
controller and wait for its terminal outcome so the tests cannot contend with
timed agent sessions. It never launches a paid or final evaluation.
