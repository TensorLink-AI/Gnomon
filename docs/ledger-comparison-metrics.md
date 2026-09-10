# Development: metric-aligned ledger comparisons

Available on `dev/ledger-optimization`; not a published release feature.

Use the metric relevant to the decision. A provider that minimizes MAE need
not minimize RMSLE. The ledger can calculate both comparisons from the same
complete, matched production origins without executing providers or writing
new scores.

```python
comparison = ledger.compare_history(
    series_id="item_123_store_4",
    unit="unit_sales",
    horizon=14,
    providers={"model_a": "revision-a", "model_b": "revision-b"},
    start="2026-01-01T00:00:00Z",
    end="2026-06-30T00:00:00Z",
    source_as_of="2026-07-15T00:00:00Z",
    recorded_as_of="2026-07-15T00:00:00Z",
    metric="rmsle",
    recent_origins=4,
    negative_predictions="clip_zero",
)
```

The IDs, revisions and timestamps above are a template, not an executable
example with existing evidence. For executable setup with controlled recording
times, run `python -m gnomon.examples.compare_history` in a fresh directory,
then add these optional metric arguments to its comparison call.

CLI and MCP `gnomon_ledger` accept the same fields for operations
`compare_history` and `compare_context`. Discover them with
`gnomon ledger --schema`. `compare_context` retains its additional explicit
label filters and source/recording provenance requirements.

## Metric and cohort contract

- `metric="mae"` preserves the existing default and permits negative values.
- `metric="rmsle"` computes natural-log RMSLE within each complete horizon,
  then averages those per-origin RMSLE values equally. This differs from
  calculating RMSLE over all pooled predictions.
- Negative actuals exclude the entire matched origin from an RMSLE query.
  They are never silently set to zero.
- Negative predictions exclude the entire matched origin by default. The caller
  can explicitly select `negative_predictions="clip_zero"`; per-origin model
  records disclose how many predictions were clipped. Original forecasts remain
  unchanged, and their untransformed MAE is retained.
- All providers use the same eligible origin/actual cohort. A metric-domain
  failure for one candidate excludes that origin for every candidate and appears
  in `excluded`; it must not make a candidate appear better by shrinking only its
  denominator. Compare exclusions when comparing different metric queries.
- Source and recording cutoffs select available evidence, not hindsight-best
  revisions. The original execution-identity, forecast-time, history-timestamp
  and provider-revision requirements remain in force.

## Calculated evidence cards

`evidence_summary` contains:

- `metric`, `aggregation` and `lower_is_better`.
- `lifetime`: all eligible origins within the explicit query window. This does
  **not** necessarily mean the entire lifetime of the ledger or provider.
- `recent`: the latest `recent_origins` eligible origins in that same window.
- Per-window ranks, exact-score ties, origin/point/unique-actual counts, and
  observed start/end origins. Exact ties share competition rank (1, 1, 3);
  provider input order only stabilizes display.
- Pairwise `left_minus_right` scores. A negative difference favors left.
  `left_relative_improvement_over_right` divides the improvement by right's
  score; when that score is zero, the ratio is null with
  `relative_status="undefined_zero_reference"`.
- `recent_lifetime_disagreement` and explicit `rank_changes`.

`origins` retains execution IDs, actual IDs and per-origin model scores. RMSLE
entries include a SHA-256 of the original scored prediction/actual pairs. Use
these records for independent arithmetic; the summary does not replace them.

Ranks and score differences are descriptive. They do not establish statistical
superiority, causal explanations or a mandate to select a provider. A small
recent window may disagree with the longer window through noise or a real
change; the caller still needs to assess relevance.

The development experiment and its 20% improvement target are documented in
[the protocol](../benchmarks/ledger_optimization/PLAN.md). Current results are
development evidence; no 20% or held-out superiority claim has been established.
