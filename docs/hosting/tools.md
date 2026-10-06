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

This saves a core-generated review with a durable reference. Check `review_ready`
and `scoring_status`: a successful save may contain a pending review if the cutoff
precedes an observation’s recording time. A later review uses later cutoffs and a
new idempotency key, without overwriting the pending review. This matters after
late ingestion or a backwards wall-clock step; never backdate observations to
force them into an earlier review. Call `gnomon_ledger`:

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

## Local computation

See [local computation](local-computation.md) for `forecast.submit`, `snapshot.save`,
`analysis.submit`, the `gnomon-shared` CLI, and exporting a client analysis using
`export.enqueue` with `analysis_id`. Client analyses remain numerically unverified,
including on Ditto recall; original snapshots and revised evidence are separate.

## Discover exact contracts and deployment capabilities

Start with `gnomon_hosted`:

```json
{"action":"info"}
```

The response identifies hosted/core versions, build identity, project permissions,
available actions, built-in providers, explicitly configured provider names and
Ditto configuration/credential presence. Configuration discovery performs no
inference, imports no provider plugins and makes no external calls. Configured
providers and Ditto are **not health-checked** by this operation. Dynamic remote
catalog aliases and saved onboarding providers are not enumerated.

The flat MCP discovery schema is an index for portable harness support. Before
using an action, retrieve its exact structural contract:

```json
{"action":"schema.get","target_action":"snapshot.save"}
```

Or set `target_action` to `analysis.submit`, `forecast.submit`, or any name from
`info.actions`. The returned schema and runtime structural validation come from
the same definition, including required fields, allowed fields and types.
Authorization, evidence cutoffs, forecast alignment and numerical checks still
apply. `export.enqueue` requires exactly one of `lesson_id` or `analysis_id`.

## Trace a decision through export

```json
{"action":"decision.status","decision_id":"YOUR_DECISION_ID"}
```

This read-only, project-scoped call reports current actual coverage, snapshot IDs,
saved reviews/evaluations, client analyses, structured lessons and associated
Ditto export receipts/memory IDs. It also returns permission-aware next steps.
It needs `evidence.read` and makes no Ditto calls or ledger writes.

- `pending` actuals means no matching observations; `partial` means some arrived.
- Saved analyses remain explicitly unverified client arithmetic.
- `evidence_changed` flags snapshots/analyses/reviews/lessons whose actual IDs
  differ from current eligible evidence, including revised observations.
- An acknowledged export is a stored delivery receipt, not a fresh remote fetch;
  use `memory.recall` to verify the memory still exists and matches its evidence.
- An uncertain delivery calls for reconciliation, not an automatic retry.
- The view reports current state with server-selected evidence cutoffs. It does
  not reconstruct historical export transitions. Large histories fail explicitly
  at the documented per-category limits; resolve individual references instead.

Refreshing MCP discovery after an upgrade exposes these descriptions to Hermes
and other clients. The calls also work via `gnomon-shared call`.
