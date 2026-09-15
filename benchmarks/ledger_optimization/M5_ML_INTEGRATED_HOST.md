# Full synthetic M5 host integration

Freeze the harness before running it. Both requested seeds use eight synthetic
series, the actual pinned 1.2.0 runtimes and frozen seed-specific workers. Each
seed executes a 72-session pilot and the complete 552-session continuation:
1,248 unique agent workflows total, 9,984 expected local numerical fits and
4,992 scripted model replies. No Engy requests or real final targets are used.

The earlier integration admission design proposed only one resumed origin per
series. That would not exercise the production continuation's exact 552-session
completion and archival path. The gate now requires the full continuation, with
624 full workflows per seed. This strengthens local verification; paid task,
model, budget and final-accuracy rules do not change.

`probe_m5_integrated_host` runs each controller and stage worker as separate real
subprocesses. The actual launcher `execute_stage`, prefix copy/audit, stage score
checks, controller, cost collector and archive verifier are exercised. Series
chains run two at once. Upstream callbacks use a separate request counter per
session and are keyed by synthetic transport credentials, so concurrent replies
cannot share the earlier sequential probe's global counter.

Test substitutions are explicit: the data source and its hash authenticator
accept only generated synthetic fixture bytes; the capsule's in-memory source
hash matches those bytes. Real worker source files remain unchanged. Upstream
model replies and readiness replies are scripted locally; any unexpected URL
rejects. The production CLI's frozen-real-plan and stopped-predecessor admission
are separately tested; the harness does not claim to execute a paid predecessor
or authenticate synthetic bytes as the real M5 data.

The harness verifies requested seed, four replies/eight fits per session, full
workflow completion, two-way concurrency, exact original/copy preservation,
memory restoration and isolation, exclusive reservation consumption, independent
score audits and complete controller archives. It retains all failed attempts
and stops without automatically retrying a failed stage or starting the next
seed. Success alone emits the integrated preflight receipt. Its source hashes
also bind the harness and synthetic fixture generator.

The synthetic reservation registry is under the test output. Production launch
reservations are untouched. The source files must not be edited during a live
integration test. No passed receipt may be fabricated from the smaller previous
probes, unit tests, process launch alone or partial progress.

This is operational reliability evidence, not an accuracy experiment. It cannot
establish ledger value or open the held-out final partition. The live paid
candidate-100 run, main and PyPI remain unchanged.
