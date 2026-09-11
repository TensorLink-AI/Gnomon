---
name: use-gnomon
description: Forecast time series with Gnomon against frozen point-in-time data and quote the snapshot, execution ID and evidence behind every number.
---

# Use Gnomon

Follow the session's `tools/list` schemas; run `gnomon_capabilities` when unsure.

## Order of operations

1. `gnomon_inspect` the file or store once; reuse the frozen `data_ref` and select
   a series explicitly for panels. `gnomon_describe` gives an exact observed statistic.
2. `gnomon_forecast` with an explicit `provider` and either a typed `request` or
   `data_ref` plus `horizon`. Horizon counts grid steps, not calendar days.
3. `gnomon_evaluate` only when the user asks for a comparison: explicit candidates,
   baseline, horizon and folds/budget. `{"study_id": ...}` alone retrieves, never reruns.
4. When `partial: true` accompanies `result_ref`, page it with `gnomon_read` (`next_offset` until null, or a pointer).
5. Use `gnomon_ledger`, `gnomon_route` and `gnomon_temporal` only when exposed; keep
   source-availability and local-recording cutoffs distinct. Writes need operator permission.

## Provider choice

The provider the user names comes first. Without one: the model they registered;
else Ephemeris when `gnomon_capabilities` reports `ephemeris.configured: true`
(if both exist, ask); else a built-in baseline labelled a reference, not their
model. Endpoints, credentials and ledger paths are operator configuration, never
tool arguments. Remote calls may incur charges. Offline example (numeric history
needs no inspection):

```json
{"name":"gnomon_forecast","arguments":{"provider":"last_value","request":{"history":[10,12,11],"horizon":2}}}
```

Preserve in your answer: provider and revision, `execution_id`, `snapshot_id` with `as_of`, quantiles when present.

## Do not

- Invent dates, a current time, frequency or timezone, or drop unsupported inputs
  silently; disclose the assumption or what was rejected.
- Treat a baseline as a chosen model, or inference as accuracy or calibration.
- Treat `next_step`, availability or a recorded decision as authority to spend or act.

Data errors include `next_call`. Retry it once; if it has `requires_user_choice` (aggressive
repair changes evidence) ask the user first. If it fails again, report the diagnosis; do not edit the file.
