# Persistence audit and initial deployment envelope

Audited base: remote `main` commit `82d9766a621cb3a568ea0c86b3890d5d6c76c74c`.
This is the clean `ditto` worktree, not the earlier development checkout. No production
storage behaviour is changed by milestone A.

## What exists

| Area | Evidence in source | Consequence for hosting |
| --- | --- | --- |
| Schema | `src/gnomon/ledger.py`, `TemporalLedger`, schema 4 | SQLite tables, JSON expressions, foreign keys and immutable UPDATE/DELETE triggers; not a backend-neutral repository. |
| Opening | `TemporalLedger.__init__`, `_connect` | Local path; `create=False` checks identity/version and opens read/write. It is not read-only authorization. Connections use a 30-second lock timeout. |
| Actual revisions | `append_actual`, `_append_actual` | `BEGIN IMMEDIATE`; content identity deduplicates exact actual retries; revision allocation is serialized. No submitting principal is persisted by core. |
| Execution writes | `inference.py`, `_finish`; ledger `_insert_execution` | Provider completes before ledger transaction; payload and execution commit together. Lost provider/commit acknowledgement needs a service receipt. |
| Decision review | `decision_memory.py`, `review_decision` | Explicit `BEGIN` read snapshot. Packet is computed, not persisted as a standalone review. |
| Lessons | `record_lesson`, `export_lesson` | Immutable outcome records embed exact review; predecessor validation and identical-latest-retry reuse. |
| Model comparison | `ledger_history.py`, `compare_history` | Explicit read transaction with direct SQLite/JSON queries. Replacing the ledger connection alone is not a Postgres port. |
| Memory bridge | `memory_bridge.py`, `EvidenceMemory` | Neutral exports exist; implementation also uses private ledger reads. Lesson enumeration and individual current reviews use separate connections; the entire recall is not one snapshot. Logical `ledger_ref` is not an authenticated remote resolver. |
| Studies | `session.py` `_studies`; `ledger.study` | Session retains a small cache; ledger-recorded studies can be reopened independently. |
| Data handles | `data_refs.py`, `DataReferences` | Session-owned preparation state; local file paths accepted. Requires remote-safe upload/references or an initial inline-only subset. |
| Result handles | `result_refs.py`, `ResultReferences`; `session.close` | Session-owned retained values are cleared. Not a cross-agent reference. |
| Provider registry | `session.py`, `_configure_provider` | Can import local Python entrypoints and use environment credentials. Configuration must remain operator-owned remotely. |
| MCP | `mcp_server.py`, `serve` | Stdio server only, no hosted authentication or tenant isolation. |
| Migration | `TemporalLedger.__init__` | Unsupported schema versions are rejected; no general upgrade/rollback machinery. |

Gnomon also records some evaluations durably, but a decision review is a different
operation. Do not present every computed review as already having a persisted review ID.

## Candidate envelope

The first implementation target is one Python server instance on one machine, a
server-owned database per project on persistent local disk, and several authenticated
clients. There is no active-active replica, shared network filesystem or R2-mounted
database. Providers run outside write transactions; only the service writes databases.

Initial functional probe: four writer processes, 25 unique observations per process,
all revising the same series/time/unit, each with one identical retry. Separate spawned
processes forecast, append outcomes, review, save a lesson and verify after restore.
A WAL-only barrier test forces an actual revision to commit during a review. Concurrency
is measured separately under DELETE (current default) and WAL journal modes.

These are reproducible probe parameters, not limits promised to users. A 30-second
SQLite busy timeout is not an HTTP timeout or an acceptable latency target. Do not
switch core journal mode globally based on this microbenchmark.

Proposed beta qualification workload, subject to explicit measurement in B/D:

- Four active clients in one project; two projects exercised for isolation.
- 10,000 executions/project using representative bounded histories and quantiles.
- At least 1,000 mixed reads/writes, with a disclosed read/write ratio and payload sizes.
- Report median/p95/p99 latency, lock waits, failures, bytes/execution and backup size.
- Suggested initial target: no lost committed writes, no mixed-snapshot reviews, and
  p95 under one second for bounded metadata operations, excluding provider inference.
- Proposed recovery target: a clean restore within five minutes for that fixture.
  Backup cadence and acceptable recovery-point loss must be selected by the operator;
  milestone A does not establish a production RPO or power-loss guarantee.

If the workload fails, first diagnose query plans, transaction duration and storage.
Choose a different backend only against a demonstrated requirement. New backends must
preserve reference identity, cutoffs, immutable records and revision semantics.

## Decisions deferred to implementation

1. Persist review IDs/snapshots through public core interfaces versus service-owned
   immutable packets; either must use core calculations and retain exact actual IDs.
2. Transactional coupling between operation receipts and ledger mutations. Tests must
   cover a crash between separate stores; a duplicate check before execution is inadequate.
3. Server-owned artifact versions and migration manifests. Inline forecast history
   already survives in an execution, but general uploaded datasets need a durable store.
4. Credential issuance/revocation and project mapping. Current core has no ACL layer.
5. Remote API surface. Local path and entrypoint features cannot be exposed unchanged.
6. Provider and export reconciliation for uncertain outcomes. No cross-service
   exactly-once guarantee is assumed.

## Backup method

The probe uses SQLite's online backup API, then checks integrity and foreign keys and
opens the restored database in a new process. It retains old and revised actuals plus
the original lesson. It does not copy a live `.db` file and ignore its WAL.

The probe manifest is explicitly a synthetic validation artifact, not the production
project-export format. It contains no external artifacts or credentials. A full service
backup needs coordinated manifests and artifact retention, encryption/access policy,
identity remapping and recovery tests; those remain milestone D work.
