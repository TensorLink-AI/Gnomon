# Scoring and recovery contracts

`gnomon ledger` reports operation success separately from scoring completion.
For example, a two-step forecast with one matching actual returns exit 0 and:

```json
{
  "status": "ok",
  "scoring_status": "partial",
  "complete": false,
  "allow_partial": true,
  "result": {
    "status": "partial",
    "complete": false,
    "n": 1,
    "horizon": 2,
    "coverage": {
      "required_steps": 2,
      "matched_steps": 1,
      "fraction": 0.5,
      "missing_steps": [1],
      "unit": "widgets"
    }
  }
}
```

This is an excerpt. Coverage also includes missing timestamps and other unit
labels found at missing steps, filtered by the same source and recording cutoffs.
Steps use zero-based indexes. Units match exactly; omitted actual units are
unitless, and Gnomon never converts them. With no matching actuals the score is
`pending`, n=0, and metrics are null. With all actuals it is `complete`.
Batch evaluation returns a result array; top-level completion requires every
execution to be complete. Aggregate scoring status is partial if any points
were scored while any horizon is incomplete.

`allow_partial` defaults to true. To require complete scoring:

```sh
gnomon ledger --ledger-path outcomes.db --arguments '{"operation":"evaluate","execution_id":"EXECUTION_ID","allow_partial":false}'
```

Strict incomplete scoring rejects with exit 2 and coverage details. Its example
keeps the evaluate operation and explicitly enables partial scoring; use it only
when incomplete coverage is acceptable. Otherwise supply the missing observed
actuals, preserving their real units and availability times. Change cutoffs only
when appropriate for the analysis. An early recording cutoff cannot see a
forecast created later. A stored forecast with naive future timestamps needs a
new forecast from input with a declared source timezone; changing scoring cutoffs
cannot fix the immutable forecast.

Scoring persists evaluation evidence without `allow_outcome_writes`. That opt-in
controls actual/decision writes. Exact scoring retries reuse the evaluation ID;
changed evidence or cutoffs can produce new records. Existing stored scores are
not rewritten. Older scores read via `evaluations` may lack the new coverage
fields; evaluating them again returns coverage with the original ID when the
evidence and cutoffs match.
Coverage on a stored score describes the evidence at scoring time; exact retries
retain it. Search/pending coverage is recomputed for the current query and can
show other-unit actuals added since that score. Search bounds diagnostic lists
to 20 entries and marks truncated lists.

## Omitted query defaults

| Operation | Source cutoff | Recording cutoff | Unit selection |
|---|---|---|---|
| search | current clock | current clock | all units |
| pending | current clock | current clock | each execution's unit |
| actuals_as_of | unbounded | unbounded | unitless |
| evaluate, compare | unbounded | unbounded | each execution's unit |
| compare_history | required | required | unitless |
| study, evaluations | not applicable | unbounded | not applicable |
| decision | unbounded | unbounded | not applicable |

Source cutoffs filter `source_available_at`, not valid time. Recording cutoffs
filter local recording time. Explicit cutoffs support reproducible queries.
Ledger responses expose a `query` description; search and pending include the
effective clock values. Route continues to require both cutoffs explicitly.
Its baseline fallback distinguishes `study_not_found` from
`study_unavailable_at_recorded_cutoff` and makes no model calls.

## Repair limits

- Aggressive fills plus conflicting-row resolutions may affect at most 30% of
  original observations in each series, before deduplication or filling.
- Each consecutive gap may contain at most `max(3, observations // 10)` fills.
- Unparseable-row drops may affect at most 5% of all input rows.
- Timestamp alignment is disclosed but does not count toward the value-repair
  budget. Structural regridding has separate limits and requires a source-calendar
  declaration.

One fill for three observed rows is 1/3 and exceeds 30%, even though it would be
1/4 of the resulting grid. Initial grid errors report the missing count, budget
and eligibility. They recommend aggressive interpolation only when those grid
limits permit it; other input checks still apply. Source correction or
`--window latest_contiguous --frequency D` can provide observed history instead.
Filled values remain assumptions and cannot serve as historical outcomes.

Strict identical duplicate errors suggest safe deduplication. Conflicting rows
require source correction or an explicit admissible aggressive choice. Seasonal
history errors retain the intended season and required/observed counts rather
than suggesting an unrelated request. No repair mode creates seasonal evidence.

## Interface discovery

`gnomon capabilities --config-schema` describes operator TOML keys without
loading providers or opening a ledger. Configuration is still TOML, not JSON.

Python forecasting uses `session.forecast(provider, request)`. MCP uses
`gnomon_forecast` with `provider` and `request`, or a previously inspected
`data_ref` and a horizon. CLI `infer --input` performs that inspection step for
you. MCP errors explain the mapping and include an example. `series_id` selects
an existing series; use `series_column` to read labels from source data.

The default session's `temporal.enabled=false` describes that session's MCP tool
visibility. The standalone `gnomon temporal` command is always available.
