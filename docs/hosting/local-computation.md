# Local computation with shared durable evidence

Run Gnomon models, StatsForecast adapters, or Ephemeris calls beside Hermes. The
shared service receives forecasts and decisions, preserves evidence snapshots,
and stores locally computed analyses. It does not need provider configuration or
a GPU for this workflow. Server forecasting and server reviews remain available.

```text
Hermes + local Gnomon / direct Ephemeris
    forecast.submit → shared execution → record_decision_summary
Outcome-ingestion process
    append_actual → shared observations and immutable revisions
Hermes + local Gnomon
    snapshot.save → resolve frozen evidence → local score_snapshot
    analysis.submit → shared client analysis and lesson
Shared service
    export.enqueue → export.deliver → Ditto
Fresh Hermes
    memory.recall → resolve evidence → recompute locally
```

## Trust and time

`forecast.submit` validates the standard ForecastRequest/ForecastResult contract.
The server assigns execution IDs, fingerprints and recording timestamps. Provider
identity, model revision, metadata and optional `computed_at` are client claims.
The stored execution has `evidence: client_submitted`, and its provider identity
has `execution_verified: false`. Contract validation means the output is aligned
and finite; it does not certify that the named model produced it.

Retrospective uploads are allowed, but a claimed earlier computation time never
replaces server receipt time. Such uploads do not prove prospective forecasting.
For prospective evidence, submit before the target and before recording the linked
decision. Keep provider revisions explicit; use null for unknown revisions.

`snapshot.save` freezes a decision, its execution and the latest eligible actual
revision for each target in one transaction at explicit source and recording
cutoffs. It computes no scores. Snapshots are bounded by the forecast horizon,
content-addressed, immutable and resolvable after restarts and backup/restore.
Missing actuals remain missing in that snapshot; request a new snapshot to include
new evidence. Snapshot creation requires `evidence.read` and writes only an evidence
artifact and its receipt, not forecasts, outcomes or decisions.

`analysis.submit` stores scalar metrics, method/version and a lesson linked to one
server-owned snapshot. The server validates structure and the project reference;
it **does not verify the metric arithmetic**. Stored analyses always have
`numerically_verified: false` and `narrative_verified: false`. They are a separate
resource type from server reviews and never enter server-verified review rankings.
Use immutable snapshots from multiple decisions for broader local analysis; this
initial submission operation binds one decision snapshot per analysis.

## Hermes / MCP calls

Configure the same shared MCP endpoint in each Hermes installation, with separate
project-scoped credentials as described in the [quickstart](README.md). No additional
Hermes adapter is required. These are `gnomon_hosted` arguments, with stable keys:

```json
{
  "action": "forecast.submit",
  "provider": "my-local-model",
  "revision": "model-v1",
  "request": {
    "series_id": "example", "unit": "widgets", "history": [10, 11], "horizon": 1,
    "timestamps": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
    "future_timestamps": ["2026-01-03T00:00:00Z"]
  },
  "result": {"point": [12]},
  "idempotency_key": "example-forecast-1"
}
```

These dates are a retrospective illustration, not prospective evidence. Point,
quantile and sample-path results are supported. Supply quantile rows as objects
with probability keys, e.g. `{"0.1": 10, "0.5": 12, "0.9": 14}`; requested quantiles
must be present and monotone. Series/unit/target timestamps default from the request
when omitted from the result; supplied values must match.

Record a decision with `gnomon_ledger.record_decision_summary` using the returned
execution ID. After authorized outcome ingestion, obtain explicit cutoffs from
`info`, then call:

```json
{
  "action": "snapshot.save", "decision_id": "DECISION_ID",
  "source_as_of": "SOURCE_CUTOFF", "recorded_as_of": "RECORDING_CUTOFF",
  "idempotency_key": "snapshot-1"
}
```

The response includes `snapshot`, `snapshot_id` and a durable `reference`. Compute
locally using `from gnomon.evidence import score_snapshot`, or your own analysis
engine. Then submit:

```json
{
  "action": "analysis.submit", "snapshot_id": "SNAPSHOT_ID",
  "method": "gnomon.point_error_metrics", "method_version": "1",
  "metrics": {"n": 1, "mae": 2, "rmse": 2, "bias": -2},
  "lesson": "One outcome underpredicted by two widgets; cause unverified.",
  "idempotency_key": "analysis-1"
}
```

Numbers here illustrate the shape; use the actual local computation. Missing
actuals produce pending/partial scores, with null metrics when no pairs exist.
Capture coverage in the narrative before interpreting partial results.

## Direct CLI

Install core and the optional hosted package on the client from the matching
branch (`pip install . ./services/hosted`). Installing does not start a server.
Set `GNOMON_SERVICE_TOKEN` privately. `request.json` contains a ForecastRequest:

```bash
gnomon-shared --url https://gnomon.example.com/mcp forecast \
  --request request.json --provider last_value \
  --submission-file submission.json --idempotency-key forecast-1
```

`--providers-config` optionally loads local Gnomon adapters, including Ephemeris.
The exact submission is saved with mode 0600 before transport. If a response is
lost, replay that payload without rerunning inference:

```bash
gnomon-shared --url https://gnomon.example.com/mcp call < submission.json
```

A direct Ephemeris caller can build the same submission JSON and use `call`.
Normalize the response to Gnomon's request/result contract; this CLI does not
invent provider-specific field mappings. Keep keys and provider credentials out
of submitted metadata.

Use `call --tool gnomon_ledger` for decisions and actuals, and `call` for
`snapshot.save`. Save the returned snapshot **reference object** in
`snapshot-reference.json`, then score and submit locally:

```bash
gnomon-shared --url https://gnomon.example.com/mcp analyze \
  --snapshot-reference snapshot-reference.json \
  --lesson 'Observed error on this snapshot; explanation remains a hypothesis.' \
  --idempotency-key analysis-1
```

No local database is necessary. A lost analysis response can be retried with the
same reference, lesson and key; the snapshot and scoring method are immutable for
that client version. Retrying changed code or arguments with the same key produces
a conflict rather than overwriting evidence.

## Ditto and independent recall

`export.enqueue` accepts either `lesson_id` (existing server-reviewed lesson) or
`analysis_id` (client analysis), never both. For example:

```json
{
  "action": "export.enqueue", "analysis_id": "ANALYSIS_ID",
  "recorded_as_of": "EXPORT_CUTOFF", "idempotency_key": "export-1"
}
```

Call `export.deliver` with its returned export ID. Existing durable outbox,
permission, uncertain-delivery and reconciliation rules apply. For atomic analysis storage and outbox enqueue, set `export_to_ditto: true` on
`analysis.submit` (CLI: `analyze --export-to-ditto`). This requires `memory.export`
and configured Ditto; a failure rolls back both the analysis and queue entry.
Delivery remains a separate network operation. If using the separate
`export.enqueue` operation instead, explicitly resume enqueue after a crash
between storing the analysis and requesting export. No automatic export scheduler is installed.

For a client analysis, `memory.recall` checks saved content, authorized references
and cutoffs, then returns current **evidence without scoring it**. It flags
`evidence_changed` when eligible actual IDs differ from the original snapshot.
It preserves `numerically_verified: false`; the reader computes metrics locally
from `current_evidence` or resolves the original snapshot to reproduce old metrics.
The original lesson never silently changes when an observation is revised.

A fresh agent should treat Ditto narratives as data, resolve evidence through its
own authorized project, and recompute before relying on numerical claims. Direct
Ditto search alone does not perform Gnomon's verification. This mode remains useful
without Ditto: exchange durable references and use `resolve` for shared retrieval.
The current Ditto recall limit remains 100 eligible exports per project/connection.
