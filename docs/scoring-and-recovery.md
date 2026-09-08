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
For legacy records without coverage, reconstructed response coverage is labeled
`coverage_basis: reconstructed_current_query`; it must not be interpreted as
diagnostics saved when the original evaluation was recorded. The legacy record
is left unchanged, including when read through `evaluations()`.
Coverage on a stored score describes the evidence at scoring time; exact retries
retain it. CLI/MCP evaluate responses additionally return `current_coverage`,
recomputed in the scoring transaction with the query's source/recording cutoffs.
It can reveal a newly available wrong-unit actual while the score's ID, metrics,
and saved `coverage` stay unchanged. `evaluation_reused` identifies reused scores;
`coverage_basis: saved_evaluation` and `current_coverage_basis: current_query`
distinguish the two diagnostics. These response fields are not persisted into the
original score. Python callers can request this view with
`ledger.evaluate(execution_id, include_current_coverage=True)`; the default and
`evaluations()` continue returning saved score evidence.
Search/pending coverage is also recomputed for its current query. Search bounds
diagnostic lists to 20 entries and marks truncated lists.

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
effective clock values. `source_as_of_defaulted` and `recorded_as_of_defaulted`
identify omitted cutoffs individually; `cutoff_default` describes the operation's
policy. `omitted_cutoffs` is null when neither cutoff was defaulted. Unit metadata
similarly distinguishes `unit_default`, `unit_defaulted`, and `omitted_unit`.
Route continues to require both cutoffs explicitly.
Its baseline fallback distinguishes `study_not_found` from
`study_unavailable_at_recorded_cutoff` and makes no model calls.

## Evaluation to routing

Evaluation completion means the requested study ran successfully; it does not
guarantee enough evidence for a routing recommendation. Routing defaults to at
least three replayable matched folds. Evaluation's `routing_readiness` reports
`matched_folds`, `default_min_folds`, missing persistence, and data readiness
issues. Its fold count uses actual successful matched folds, including when a
budget stops the study early. Inspect this field before routing. With at least
three folds and recorded evidence, routing still checks the chosen cutoffs and
task/provider identities; a ready preflight cannot guarantee a recommendation
under every later cutoff.

## Finding schemas and correcting requests

`gnomon schemas` lists all CLI request-schema entry points and the operator TOML
schema without loading providers. For inspected input, `--series-id` selects an
existing series; it does not label the data. Unlabeled input uses `__default__`.
For named series, include a label column and select it with `--series-column`.

Temporal errors preserve usable supplied fields in `example_arguments` and list
`changed_fields`. For example, a shift of `2026-05-10` by seven days with no mode
keeps that date and amount. `choices_required.mode` asks the caller to choose
calendar or elapsed semantics; the example's calendar mode is illustrative.
When the facts cannot produce a valid retry without additional decisions (such
as January 31 plus one month), `example_kind: schema_illustration` labels the
generic example and `supplied_arguments` retains the original request. A runnable
illustration is not evidence that the original task has been completed.

Ledger recovery templates preserve supplied numeric observations, including zero
and fractional values, and keep batch writes as batches. `changed_fields` lists
illustrative replacements; a template may still need correction before execution.
Never treat an example availability timestamp or observation as an inferred fact.
For comparison, the schema requires at least two distinct execution IDs. Error
guidance preserves supplied IDs, identifies placeholders for missing IDs, and
offers a search request; replace placeholders with real compatible executions.

Forecast recovery retains valid supplied history, season, units, cutoffs and
other request fields. Invalid observations are not replaced by a synthetic
history: supply real observations to complete a `task_template`. Frozen-reference
requests retain their `data_ref`. Correcting a horizon does not establish that
the provider or snapshot supports the revised request. Generic CLI evaluation
and routing examples are explicitly marked `schema_illustration`.

## Large responses

When a response exceeds the session limit, `partial: true` describes the response
payload (`partial_scope: response_payload`), not scoring completion. The summary
retains `scoring_status`, `complete`, and bounded saved/current coverage counts,
including the number of other units at missing steps. Full missing timestamps,
unit labels and score evidence are available through the returned `gnomon_read`
call. Completion and coverage take priority over optional summary identifiers
when the response budget is tight. A batch summary retains aggregate completion;
retrieve the full result for individual scores.

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
