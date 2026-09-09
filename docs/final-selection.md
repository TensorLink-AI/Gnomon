# Preserve successful forecasts across an agent's final answer

A tool executing successfully and an agent returning valid final JSON are
different events. The host should retain successful tool evidence and resolve
the final selection against it, instead of discarding a valid forecast just
because the agent described it in prose.

Every successful forecast response includes `completion`, a canonical object:

```json
{
  "operation": "forecast",
  "operation_succeeded": true,
  "provider": "last_value",
  "execution_id": "execution UUID",
  "revision": "provider revision",
  "series_id": "item_123_store_4",
  "unit": "widgets",
  "horizon": 2,
  "future_timestamps": ["2026-01-04T00:00:00+00:00", "2026-01-05T00:00:00+00:00"],
  "point": [3.0, 3.0],
  "request_fingerprint": "sha256:<64 hexadecimal characters>"
}
```

`final_selection` supplies machine-readable preservation guidance and an exact
`{"provider": ..., "execution_id": ...}` example for that execution. The
existing response fields, including `result.point`, remain available. Python
engine callers can use `execution.completion()` or
`gnomon.forecast_completion(execution)`.

For a retained MCP response, follow `forecast_completion` to `gnomon_read` at
`/completion`, concatenating pages until `next_offset` is null. The root hash
applies to the full retained payload, not to an individual JSON-pointer slice.
Retrieve the root and verify its hash when verifying full-payload integrity.
A summary or unassembled page is not successful execution evidence.

## Host integration

```python
from gnomon import GnomonSession, resolve_final_selection

request = {
    'history': [1, 2, 3], 'horizon': 2,
    'timestamps': ['2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z', '2026-01-03T00:00:00Z'],
    'future_timestamps': ['2026-01-04T00:00:00Z', '2026-01-05T00:00:00Z'],
    'series_id': 'item_123_store_4', 'unit': 'widgets',
}
with GnomonSession.from_config() as session:
    execution = session.forecast('last_value', request)
    # Capture this from the agent host's tool transcript, not the final answer.
    executions = [execution]
    resolution = resolve_final_selection(
        final_answer='Here is the forecast in prose.',
        successful_executions=executions,
        expected_request=request,
    )
    assert resolution['engine_execution_succeeded']
    assert not resolution['final_answer_conformant']
    assert resolution['resolution_status'] == 'recovered_single_execution'
    assert resolution['end_to_end_completed']
    points_to_score = resolution['execution']['point']
    assert points_to_score == [3.0, 3.0]
```

The helper accepts host-owned `ForecastExecution` objects, their serialized
forms containing `completion`, complete forecast response payloads, or canonical
completion objects. For MCP, pass `structuredContent` after complete retrieval.
Integrity-checked ledger reads expose reconstructed completions without storing
invocation IDs in deduplicated request/result payloads. Full legacy execution
records with `request`, `result` and `result_contract_validated:true` can also be
adapted by the resolver; old compact tool outputs without an established full
request cannot be safely bound by guessing a fingerprint.
Explicit error/failed/cancelled/unscored records may be included; they are ignored
as candidates. At most 1,000 records may be passed per resolution.

Supply the full request used for dispatch. For data references, obtain it with
`session.data.request(data_ref, horizon=..., season=..., series_id=...)` using the
same forecasting arguments. This preserves snapshot identity, cutoffs and the
generated future timestamps. A hand-written subset is insufficient. Remote MCP
hosts must retain an equivalent full request or trusted execution request; a
session-local `data_ref` string is not a portable request identity.

`forecast_request_fingerprint(request)` hashes the validated canonical request:
history, horizon, season, units, series, timestamps, all cutoffs, snapshot ID,
covariates and remaining request fields/defaults. It uses the engine's numeric
and timestamp canonicalization. It excludes provider/revision identity, unlike
the existing execution/cache `fingerprint`, so different providers can be
matched to the same task.

Hashes are bindings, not signatures. Only supply execution evidence retained
by the host from successful tools or the engine. Never populate
`successful_executions` from the LLM's final JSON, and never manufacture a
request fingerprint for old evidence whose request cannot be established.
The resolver cannot authenticate forged caller-owned records.

## Resolution rules

| Input | Resolution |
|---|---|
| Canonical JSON/dict selecting a successful matching execution ID | `explicit_selection` |
| Provider identifying exactly one successful execution, matching the task | `explicit_selection` |
| Missing answer, prose or Markdown table; one matching execution | `recovered_single_execution` |
| Multiple matching executions without selection | `ambiguous_multiple_executions` |
| Same provider with multiple successful execution IDs | Require `execution_id` |
| Failed, unknown or unrelated execution selected | `conflicting_final_selection` |
| No matching execution and no explicit selection | `task_mismatch` |
| No successful execution | `no_successful_execution` |
| Contradictory identity/points or malformed successful evidence | Fail closed; no selected execution |

Repeated identical receipts with the same execution ID are deduplicated. Two
different IDs remain distinct even when their points or providers are identical.
If a provider name refers to multiple successful IDs, including another task,
the caller must select by ID. Without an explicit selector, automatic recovery
considers only executions matching the expected task.

Canonical final JSON may contain `provider`, `execution_id`, optional completion
fields and optional string `rationale`. At least one selector is needed for
strict conformance. Documented final-answer aliases `selected_provider` and
`candidate` normalize to `provider`, with a recorded normalization and
`final_answer_conformant:false`. Tool inputs still use `provider`.

Unsupported/misspelled fields, conflicting aliases, duplicate JSON keys and
malformed JSON-looking objects are not silently repaired. Any echoed completion
fields must agree with the selected trusted execution. The resolver never
scrapes provider names from prose, Markdown or code fences.

`final_answer_conformant` describes the resolver's canonical final schema,
independently of whether the referenced execution exists or matches the task.
For an existing benchmark with a different final schema, retain its original
strict-conformance metric alongside this field; do not retrospectively count
recovery as strict conformance. For an unresolved case, `execution` is null.

## Arena wiring and reporting

Arena must call the helper after the agent finishes, keep the raw final answer
and original strict-grade result, and score `resolution.execution.point` when
`resolved` is true. Log execution ID, provider, revision and request fingerprint.
Do not replace a recoverable typed execution with seasonal naive for a prose
final. Multiple ambiguous executions remain unresolved; this helper does not
choose a provider or execute a fallback.

Report engine success, strict conformance, recovery, ambiguity and end-to-end
completion separately. The code includes a synthetic 96-case accounting test
with 59 strict selections, 35 recovered singles and 2 unresolved multiple-call
cases. That is a regression fixture for the reported pattern, not a replay of
the Favorita run or evidence of changed RMSLE. The actual frozen run must be
replayed by the matching Arena harness to establish its recovery totals.

Adopt the resolver as an explicit, versioned integration policy and test that
policy in a separate run. Preserve original strict-parser scores, raw answers,
execution records and budgets. A retrospective replay is a recovery analysis,
not a replacement for the original benchmark. A bounded correction opportunity
must be reported separately from automatic single-execution recovery. Neither
mechanism validates narrative claims about stockouts, lost demand or seasonality.
