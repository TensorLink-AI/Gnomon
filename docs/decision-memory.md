# Decision summaries, context evidence and lessons

Gnomon 1.1.9 links short decision summaries and versioned lessons to recorded
forecasts. It verifies numerical evidence and preserves its history. Rationale,
context labels and lessons remain caller assertions; a low error does not prove
an explanation, a stockout, recoverable lost demand or a business saving.

Run the complete offline example in a **fresh directory**:

```bash
python -m gnomon.examples.decision_memory
```

It records two prospective built-in forecasts with a controlled clock, saves a
promotion-labelled decision, checks pending outcomes, appends synthetic actuals,
compares the matched forecasts, reviews the decision, and exports a lesson to a
local memory sink. It makes no network or model calls beyond the two built-ins.

## Operations

All five operations are Python `TemporalLedger` methods and JSON operations of
`gnomon ledger` / MCP `gnomon_ledger`. The MCP tool appears when a ledger is
configured. `gnomon ledger --schema` describes all fields and required cutoffs.
CLI/MCP summary and lesson writes require `allow_outcome_writes = true` in the
operator's TOML. Python ledger methods are explicit application-owned writes;
they do not use session permission flags. Read operations do not write scores.

| Operation | Purpose | Writes |
| --- | --- | --- |
| `record_decision_summary` | Bind a concise rationale, assumptions, invalidation conditions, context and evidence references to an execution | One decision |
| `compare_context` | Filter matched production forecasts using exact labels | None |
| `review_decision` | Compare the selected forecast with actuals at explicit cutoffs | None |
| `record_lesson` | Save a bounded hypothesis and its complete immutable review | One outcome; identical latest retry writes nothing |
| `export_lesson` | Return compact JSON and numerical verification calls | None |

These reuse ledger schema 4. Existing decisions and outcomes remain readable;
no migration or rewrite is required. The structured `kind` markers are reserved:
use the typed operations rather than inserting them into generic decision data.

## Record the decision

The execution must already exist, with a named series, history timestamps and
future timestamps. Provider, revision, series, unit, horizon and canonical full
request fingerprint are derived from it, not supplied by the agent. Supporting
`evidence_refs` use `{ "kind": "execution" or "study", "id": "..." }` and must
already exist at recording time.

```python
summary = ledger.record_decision_summary(
    execution_id=forecast["execution_id"],
    rationale="Use the latest observed level during the planned promotion.",
    assumptions=["The supplied promotion schedule applies to this series."],
    invalidation_conditions=["The promotion is cancelled."],
    context=[{
        "key": "promotion", "value": "planned",
        "valid_from": "2026-01-20T00:00:00Z",
        "valid_to": "2026-01-23T00:00:00Z",
        "source_available_at": "2026-01-19T00:00:00Z",
        "source_ref": "promotion-calendar:revision-1",
    }],
)
```

This is a template following an actual forecast, not a standalone command.
Rationale is limited to 1,000 characters, assumptions/invalidation conditions to
8 statements each, and context/evidence references to 16 items each. Store a
concise decision summary, never a private internal reasoning transcript.

Context values are exact strings. Keys are unique within a summary. Valid-time
ranges are half-open; source availability is separate. The ledger assigns local
recording times and never accepts caller-supplied recording timestamps. Future
source-availability assertions are rejected until they become available. A
future planned promotion is valid: its plan can be known before its valid time.

## Compare comparable evidence

`compare_context` accepts the existing `compare_history` arguments plus a nonempty
`context_filters` object such as `{"promotion": "planned"}`. Series, unit, horizon,
explicit provider revisions and origin range still define the comparison.

Every included origin must satisfy the existing prospective execution, matched
request, provider identity and complete actual-coverage checks. It must also have
matching context attached to at least one of those executions:

- The label applies at the forecast origin.
- It was source-available by that origin and the query's source cutoff.
- Its decision was recorded strictly before the first forecast target and no
  later than the query's recording cutoff.
- Eligible summaries agree on the requested label values; conflicting labels
  exclude the origin. Duplicate agreeing labels do not increase the sample size.

Recording before the first target is the declared production comparison rule;
it does **not** prove a label was recorded before every candidate execution.
Labels attached retrospectively after the target cannot be used to claim an
ex-ante advantage. Unknown labels are excluded, never inferred from zero sales.
Only requested labels determine the filter; this is not similarity retrieval.

Results include origin ranges, matched counts, distinct actual counts, exclusions,
context decision references and per-provider mean/min/max origin MAE. Means are
recomputed over the filtered cohort. Bounds are **descriptive ranges**, not
confidence intervals: origins can overlap and share outcomes. No statistical
superiority or causal claim is made. Comparisons are bounded to 1,000 executions
and 1,000 summaries in the requested series/origin window; exceeding a limit
rejects the query rather than returning a partial ranking.

## Review and version the lesson

Poll `review_decision` with explicit source and recording cutoffs. It returns
`scoring_status: pending/partial/complete`, coverage, exact scored pairs and
actual IDs. `review_ready: true` invites the agent to review its assumptions once
matching-unit actuals cover the full horizon. There is no background scheduler
or automatic LLM call. Applications decide when to poll and whom to notify.

A `record_lesson` call repeats the numerical review inside its write transaction,
requires complete coverage and cutoffs no later than the recording clock, and
stores the review with the hypothesis. It never accepts agent-supplied metrics.
The original decision remains unchanged. For a new version, supply the current
`previous_lesson_id`; stale references reject instead of forking the chain.
An exact retry of the latest lesson with the same predecessor and evidence reuses
its ID. A retry after another version exists must retrieve that version first.
Later actual revisions change a new review, not the old saved review or lesson.
No scoring outcome sets `business_explanation_validated` to true.

```python
cutoffs = dict(source_as_of=source_cutoff, recorded_as_of=recording_cutoff)
review = ledger.review_decision(decision_id=summary["decision_id"], **cutoffs)
if review["review_ready"]:
    saved = ledger.record_lesson(
        decision_id=summary["decision_id"],
        lesson="Underprediction occurred; whether the promotion caused it remains untested.",
        **cutoffs,
    )
    memory = ledger.export_lesson(
        lesson_id=saved["lesson_id"], recorded_as_of=ledger.clock.now().isoformat()
    )
```

## Store the compact memory elsewhere

Exports carry lesson/decision/execution IDs, provider revision, task identity,
context provenance, metrics, actual IDs, hashes and two explicit ledger calls:
`verification_call` reproduces the numerical review; `immutable_evidence_call`
retrieves the recorded decision and saved lesson evidence. Keep the ledger
reachable under an application-owned identifier alongside the export. Gnomon
does not export local database paths, authenticate remote ledgers or sign hashes.
Hashes check content consistency within trusted ledger evidence, not authorship.

```python
from gnomon import put_lesson

# store is a caller-owned LangGraph-compatible object with put(namespace, key, value).
put_lesson(store, (tenant_id, "gnomon-lessons"), memory)
```

The adapter has no optional dependency and performs exactly one explicit `put`.
The lesson ID is the key, so versions coexist and retrying a put uses the same
key. Supply a persistent store for durable storage; an in-memory store is only
for testing. Namespace/tenant access control belongs to the application. Store
errors propagate; Gnomon does not silently retry or claim an external write
succeeded. Memory text is untrusted hypothesis data, not system instructions.

LangGraph's namespaced JSON store fits this protocol; see its
[long-term memory documentation](https://docs.langchain.com/oss/python/langchain/long-term-memory).
The same JSON can be carried into other memory systems, including Letta, by
application code; no Letta-specific API adapter ships in this release.

## Evaluate the value separately

Compare no memory, ledger only, narrative memory only, and both, with the same
raw observations, temporal visibility, provider candidates and call/token budgets.
Predeclare which tools each arm receives and the lesson creation/retrieval policy.
Report warm-up and later phases separately, plus independent series/seeds and
uncertainty that respects shared origins. Preserve strict final conformance,
resolved completion, forecast loss, unsupported explanations and agent effort as
separate measures. Evaluate savings only with explicit costs and constraints.
This release supplies the mechanisms; it contains no new paid evaluation or
claim that memory improves forecasting or business decisions.
