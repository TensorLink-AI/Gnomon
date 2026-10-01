# Hosted MCP tool examples

Use `gnomon_hosted` with `{"action":"info"}` for service/project/ledger IDs and
server time. All values below are examples; use actual timestamps and IDs.

## Forecast, record, score, share

Call `gnomon_forecast`:

```json
{
  "provider": "last_value",
  "idempotency_key": "sales-2026-10-01-forecast",
  "request": {
    "series_id": "sales", "unit": "widgets", "history": [10, 11, 12],
    "timestamps": ["2026-09-28T00:00:00Z", "2026-09-29T00:00:00Z", "2026-09-30T00:00:00Z"],
    "horizon": 1, "future_timestamps": ["2026-10-02T00:00:00Z"]
  }
}
```

Save `result.execution_id` and `result.reference`. A forecast response has
`request_id`, `state` and `result`. Only `state=completed` indicates completion.
Same principal/key/input returns the prior receipt; different input conflicts.
A retry can return running or outcome_unknown rather than repeat provider work.

Call `gnomon_ledger`:

```json
{
  "operation": "record_decision_summary", "execution_id": "EXECUTION_ID",
  "rationale": "Use the baseline as the current planning estimate.",
  "assumptions": ["No known capacity change"],
  "invalidation_conditions": ["A capacity change is reported"], "context": [],
  "idempotency_key": "sales-2026-10-01-decision"
}
```

Core results are under `result.result` inside mutation receipts. Keep its
`decision_id`. An independently authorized outcomes process later calls:

```json
{
  "operation": "append_actual", "series_id": "sales", "unit": "widgets",
  "valid_time": "2026-10-02T00:00:00Z", "value": 13,
  "source_available_at": "2026-10-03T00:00:00Z", "source_ref": "daily-sales-feed",
  "idempotency_key": "sales-2026-10-02-actual-revision-0"
}
```

Run this only after those times. Server recording time is authoritative;
source availability is a caller provenance claim. New observations at the same
valid time append revisions and preserve the earlier record.

Review with `gnomon_hosted`:

```json
{
  "action": "review.save", "decision_id": "DECISION_ID",
  "source_as_of": "2026-10-04T00:00:00Z", "recorded_as_of": "2026-10-04T00:00:00Z",
  "idempotency_key": "sales-review-1"
}
```

This saves a core-generated review with a durable reference. A later review can
use later cutoffs without overwriting it. Call `gnomon_ledger`:

```json
{
  "operation": "record_lesson", "decision_id": "DECISION_ID",
  "lesson": "Forecast 12 versus actual 13; MAE 1 widget. Cause is unverified.",
  "source_as_of": "2026-10-04T00:00:00Z", "recorded_as_of": "2026-10-04T00:00:00Z",
  "idempotency_key": "sales-lesson-1"
}
```

Export using successive `gnomon_hosted` calls:

```json
{"action":"export.enqueue","lesson_id":"LESSON_ID","recorded_as_of":"2026-10-04T00:00:00Z","idempotency_key":"sales-export-1"}
```

```json
{"action":"export.deliver","export_id":"EXPORT_ID"}
```

A fresh agent, with an evidence.read credential for the project, calls:

```json
{"action":"memory.recall","query":"sales planning baseline","source_as_of":"2026-10-04T00:00:00Z","recorded_as_of":"2026-10-04T00:00:00Z"}
```

Use cutoffs after the actual lesson export for present-time recall. Historical
cutoffs are for evidence eligible at that time. Returned `current_review`,
`evidence_changed` and `narrative_verified=false` distinguish checked evidence
from interpretation. Unknown or modified remote contents are excluded.

## Other hosted operations

- `dataset.put`: supply `request` and `idempotency_key`; returns a durable
  dataset snapshot/version ID. Pass it as `dataset_id` instead of inline `request`
  to `gnomon_forecast`. The snapshot includes the target time grid.
- `resolve`: supply the complete returned `reference`; returns `result`. Supports
  dataset_version, execution, decision, actual, review, lesson and export_receipt.
  Actual IDs name individual immutable revisions. Service identity is preserved
  on restore; the client chooses the new server address independently.
- `request.get`: supply `request_id`; only the originating principal can read
  its operation receipt, including after credential rotation.
- `export.get`: supply `export_id` to inspect state without a remote call.
- `export.reconcile`: supply `export_id` and `memory_id` for an uncertain save.
  It fetches and verifies the memory before marking delivery acknowledged. A
  missing or mismatched memory leaves the export uncertain.

All lookups recheck project authorization. Possessing a reference is not access.
Tool schemas expose parameter names portably; the runtime validates the exact
operation-specific contract. Unsupported local tools and filesystem paths are
rejected. A 409/conflict is not a success response.
