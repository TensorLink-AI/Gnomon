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
4. When `partial: true` accompanies `result_ref`, page the full result with
   `gnomon_read` (follow `next_offset` until null, or use a JSON pointer).
5. Use `gnomon_ledger`, `gnomon_route` and `gnomon_temporal` only when exposed; keep
   source-availability and local-recording cutoffs distinct. Writes need operator permission.

## Provider choice

Use the registered Ephemeris provider when `gnomon_capabilities` reports
`ephemeris.configured: true`; otherwise the model the user registered; only with
neither, a built-in baseline labelled as a reference, not the user's model.
Endpoints, credentials and ledger paths are operator configuration, never tool
arguments. Remote inference and evaluation may incur charges. Offline example
(supplied numeric history needs no inspection):

```json
{"name":"gnomon_forecast","arguments":{"provider":"last_value","request":{"history":[10,12,11],"horizon":2}}}
```

## What to preserve in your answer

Provider and revision, `execution_id`, `snapshot_id` with `as_of`, quantiles when present.

## Do not

- Invent dates, a current time, frequency or timezone; ask or disclose the assumption.
- Drop unsupported inputs silently; report what the provider rejected.
- Treat a baseline as a chosen model, or inference as accuracy or calibration.
- Treat `next_step`, availability or a recorded decision as authority to spend or act.
