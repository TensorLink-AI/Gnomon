# Milestone A validation evidence

Result: the existing SQLite ledger passes the bounded offline feasibility probe. This
supports proceeding to a single-server demonstrator. It does not qualify a hosted beta.

Base code: `82d9766a621cb3a568ea0c86b3890d5d6c76c74c` (remote main at branch creation).
The exact probe SHA-256 and environment are included in the retained JSON reports.
Recorded 2026-09-30. No accounts, external services or real user data were accessed.

## Reproduce

From the repository root, using Python with Gnomon and the development test dependencies:

```bash
PYTHONPATH=src python -m benchmarks.hosted_ledger.validate /tmp/new-ledger-delete --workers 4 --writes 100 --journal-mode DELETE
PYTHONPATH=src python -m benchmarks.hosted_ledger.validate /tmp/new-ledger-wal --workers 4 --writes 100 --journal-mode WAL
python -m pytest -q benchmarks/tests/test_hosted_contract_fixtures.py benchmarks/tests/test_hosted_ledger_validation.py tests/test_ledger.py tests/test_decision_memory.py tests/test_memory_bridge.py
```

Each output directory must be new. It contains a synthetic ledger, an online backup,
identity/checksum manifest and JSON report. The committed reports contain no database
contents or credentials. The schema-fixture test requires `jsonschema`, already part
of the project's development extra.

## Observed results

Environment: Python 3.13.11, SQLite 3.50.4, Linux/WSL2. Runs used the same host alongside
other validation work; they are correctness probes, not a controlled performance
comparison. Timings include filesystem and process scheduling effects.

| Measurement | DELETE | WAL |
| --- | ---: | ---: |
| Writer processes | 4 | 4 |
| Distinct contended revisions | 400 | 400 |
| Exact retries | 400 | 400 |
| Missing/duplicate revision numbers | 0 | 0 |
| Elapsed write phase, including process startup | 2.507 s | 1.916 s |
| Median write plus exact retry | 5.61 ms | 3.96 ms |
| p95 write plus exact retry | 8.56 ms | 12.01 ms |
| Maximum write plus exact retry | 1,337.59 ms | 968.69 ms |
| Backup plus fresh-process verification | 0.128 s | 0.138 s |
| Restored database size | 249,856 bytes | 249,856 bytes |

Raw reports: [DELETE](evidence/delete.json), [WAL](evidence/wal.json).

The maximum latency demonstrates contention even in this tiny workload. These values
must not be converted into a user-count estimate or an SLA. The current core default
journal mode has not changed.

## Correctness established by the probe

- A forecast and decision survive process exit; another process supplies outcomes.
- Before actuals, the review is pending and a premature lesson is rejected.
- The baseline prediction `[12, 12]` scores MAE 1.5 against `[13, 14]`.
- Concurrent writers revising one series/time/unit receive a complete unique revision
  sequence. Identical actual retries reuse IDs rather than adding revisions.
- Existing append-only triggers reject UPDATE and DELETE of actuals.
- A separate WAL barrier test pauses a review, commits `[13, 20]` through another
  connection, then repeats the actual query inside the open review transaction. Both
  reads see `[13, 14]`; the packet does not mix revisions. This test uses WAL regardless
  of the earlier write-throughput mode, as labelled in each report.
- Online SQLite backup passes integrity and foreign-key checks. A new process opens
  the restored file: old cutoffs still score 1.5, newer cutoffs score 4.5, and the entire
  original exported lesson remains byte-for-byte equivalent as a decoded JSON object.

Targeted regression result: **47 passed** (new probe/schema tests plus ledger,
decision-memory and memory-bridge tests). Ruff and whitespace checks also pass.

## Not established

- No hosted server restart, real MCP connection or authorization boundary has been
  tested. Process reopening tests the core persistence component only.
- Unauthorized-access cases are design fixtures, not passing security tests. The
  schema test rejects invalid identity examples; it is not an access-control test.
- There are no saved standalone hosted reviews, persistent uploaded dataset references,
  request receipts, durable jobs or production export queue yet.
- No process was killed during a commit or external operation. Provider timeout and
  response-loss reconciliation remain milestone B/C requirements.
- Backup tests cover this synthetic database only, not external artifacts, credentials,
  power loss, full disks or an upgrade across schema versions.
- No real Ditto save/search/fetch occurred, and no forecasting benefit is claimed.

## Next gate

Implement milestone B's principal/project boundary, transactional operation markers
and durable reference resolution. Exercise real client/server restarts and failure
injection, then qualify a representative workload before making deployment guarantees.
