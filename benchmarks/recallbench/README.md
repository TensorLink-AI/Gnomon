# RecallBench

Is a hosted model's forecasting edge on public series *skill*, or
*recall*? The two have opposite product implications: genuine skill
justifies a governed LLM-forecast candidate lane; memorized futures are
a lookup that will not transfer to a client's private data, and a lane
built on them ships an illusion.

Matched arms over **identical real windows** (the BreachBench corpus
plus the supported-cadence DossierBench series; yearly/quarterly series
are excluded rather than mislabelled daily):

| Arm | Sees |
| --- | --- |
| `raw` | the true recorded values — skill and recall both help |
| `anon` | the same window under a seeded positive affine transform — recall is defeated, pattern-based forecasting survives |

Scoring is **MASE** (MAE scaled by the in-sample seasonal-naive error
of the same history), chosen because it is invariant under positive
affine transforms — the transform itself cannot move the score, and the
harness verifies pre-flight that the seasonal-naive reference lands
identically in both arms. Gnomon's production forecast, the seasonal
naive, and last-value run as deterministic references per arm at zero
API cost.

Two paired verdicts:

- `memorization_delta` — model MASE anon minus raw. Near zero: the raw
  edge is transferable skill. Large and positive: it was recall.
- `skill_vs_gnomon_anonymized` — engine minus model inside the anon
  arm: the leakage-controlled version of "the model forecasts better
  than the engine", the only reading that justifies admitting LLM
  forecasts as governed candidates.

Run (~2 model calls per case; Gnomon references are local and free):

```
uv run python benchmarks/recallbench/run_recallbench.py \
  --model <model> --reasoning-effort none --cases 120 \
  --output-dir results/recallbench/<run>
```

Reasoning mode is explicit and part of resume identity. The default is
`none`, matching the low-latency agent lane; a reasoning-enabled run is a
separate treatment and cannot reuse non-reasoning rows.

Operational guarantees match the sibling harnesses: held-out futures
verified absent from every prompt (history-excised sentinel), durable
resumable rows stamped with the full dataset identity and answering
model, one failed API call fails loudly after saving completed rows,
malformed forecasts are recorded rather than crashing, and API usage
and cost land in `summary.json`.
