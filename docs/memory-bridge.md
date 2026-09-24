# Evidence that agent memory can verify

`EvidenceMemory` connects existing `TemporalLedger` decision summaries, reviews
and immutable lessons to caller-owned memory systems. It has no Hermes, LangGraph
or network dependency. It makes no provider calls. `HermesMemoryAdapter` is a thin
formatter over the same contract; another integration can consume the JSON directly.

Run the complete offline example in a fresh directory:

```bash
python -m gnomon.examples.memory_bridge
```

The example records one forecast, rejects an incorrect provider claim, scores
actuals, exports a proposed Hermes update, ingests a revision, and demonstrates
that saved metrics stay unchanged while current evidence is flagged as changed.
It retains `memory-bridge.db` and makes no external memory writes.

## Bind memory to the submitted forecast

```python
from gnomon import EvidenceMemory, HermesMemoryAdapter

bridge = EvidenceMemory(ledger, ledger_ref="retail-evidence")
execution = session.forecast("last_value", request)
decision = bridge.record_decision(
    execution_id=execution["execution_id"],
    expected_request=request,
    claimed_provider="last_value",
    rationale="Only the initial baseline was submitted before the budget ended.",
    assumptions=["The latest level is informative."],
    invalidation_conditions=["The level changes."],
)
```

`ledger`, `session` and `request` are application-owned objects; the executable
example shows their construction. The session must record executions in this
ledger. `expected_request` must come from the host's current task, not from the
agent's claimed result. The full canonical request fingerprint must match.
Unknown executions and conflicting task/provider claims fail before a decision
write. The existing named-series, history and target timestamp requirements apply.

Packets separate `verified_execution` from `narrative`. Provider, revision, task
identity, season, frequency and forecast hashes come from the execution. Arbitrary
custom-model hyperparameters cannot be inferred from an opaque provider revision.
Rationale, assumptions and lesson text are retained as **unverified hypotheses**.
Gnomon does not parse prose or endorse a narrative that names a different model.

## Check outcomes and explicit claims

`bridge.decision(decision_id=..., source_as_of=..., recorded_as_of=...)` returns a
read-only review with pending/partial/complete scoring and exact matching-unit
coverage. MAE, RMSE, bias and RMSLE refer to its disclosed scored pairs. RMSLE is
unavailable for negative predictions/actuals; nothing is clipped implicitly.

`bridge.check_claims(..., claims=[{"field": "provider", "value": "last_value"}])`
uses the same explicit cutoffs. Supported identity and metric fields are checked;
contradictions retain both the supplied and observed value. Unscored metrics,
causal explanations, rankings and other unsupported claims remain `unverified`.
Numeric metric tolerance is 1e-9 relative / 1e-12 absolute; integer identity/count
fields require exact integer equality. Claims about one forecast are not model
rankings. The check is returned to the caller; it does not append a claim audit to
the ledger or rewrite the explanation.

After complete outcomes, call `bridge.record_lesson(decision_id=..., lesson=...,
source_as_of=..., recorded_as_of=...)`. This reuses the existing immutable lesson
chain. An identical latest retry reuses its ID; a changed lesson requires
`previous_lesson_id`. Original decisions and prior versions remain unchanged.

## Retrieve a small amount of relevant evidence

```python
memories = bridge.retrieve_lessons(
    series_id="sales", unit="widgets", horizon=2,
    source_as_of=source_cutoff, recorded_as_of=recording_cutoff,
    context_filters={"promotion": "planned"}, limit=3,
)
```

Use explicit source and recording cutoffs. Unit matching is exact; `None` selects
unitless data. Context filters are optional exact, caller-declared labels, not
similarity search or causal evidence. Use `compare_context` for prospective,
matched model comparisons. Retrieval chooses the newest visible version per
decision and returns at most three records by recording order, never best score.
Counts and exclusions describe this query, not statistical confidence. A bound of
1000 matching versions prevents an unbounded scan; use explicit lesson IDs if
the bound is exceeded.

Saved scoring stays under `scoring`; `current_evidence` separately reports current
metrics and `changed_since_lesson`. A changed actual revision calls for review,
not automatic rewriting of a lesson. Packets retain ledger/decision/execution IDs,
hashes, evidence cutoffs, context provenance and an exact verification call.
Hashes provide consistency within trusted evidence, not authentication.

## Adapt to a memory system

For a caller-owned namespaced JSON store:

```python
bridge.put(store, (tenant_id, "forecast-lessons"),
           lesson_id=lesson_id, recorded_as_of=recording_cutoff)
```

The store implements `put(namespace_tuple, key, value)`. Gnomon re-reads the lesson
from the ledger before writing. The key includes the ledger identifier and lesson
ID, so retries replace the same external key and new versions coexist. Namespace
access control, authorization and durable storage belong to the application.
Store exceptions propagate; no silent retries or writes occur during retrieval.

For Hermes:

```python
adapter = HermesMemoryAdapter(bridge)
proposal = adapter.lesson_update(lesson_id=lesson_id, recorded_as_of=recording_cutoff)
# Host decides whether to call proposal["tool"] with proposal["arguments"].
```

`decision_update` similarly prepares a factual handoff immediately after a
submission, before actuals mature. These methods return Hermes's `memory` batch
operation shape; they **do not call it** or claim delivery succeeded. Use an
application-owned task-scoped Hermes home for episodic lessons, not a user's
global profile. `adapter.recall(...)` accepts the same retrieval arguments and
returns task-context JSON instead, without persistent memory mutation.

To refresh the same external entry, supply its exact previous text through
`existing_entry`. A stable record marker guards replacement of another entry.
Add calls are not assumed idempotent; inspect the memory tool response and retain
delivery state in the host. New lesson versions append distinct references.
Compact narrative excerpts are explicitly truncated at 400 characters; full
assumptions and text remain in the verified ledger packet. Excerpts are untrusted
data, not system instructions.

## Keep workflow policy separate

This bridge does not change agent budgets or forecasting algorithms. An agent
runner should reserve time to submit and validate its selected execution, then
create memory from that execution. Test that policy separately under unchanged
total budgets. This implementation supplies provenance and verification; it does
not establish that memory improves forecast error or efficiency.

## Compact comparisons and arithmetic checks

`EvidenceMemory.comparison_card(execution_ids=..., source_as_of=..., recorded_as_of=..., metric="rmsle")`
returns ranks and differences only for complete matched inputs and actuals. It
selects the latest admissible origin, preserves provider revisions, and reports
insufficient evidence when no comparison exists. Recent versus lifetime evidence
uses the existing historical comparison contract; changing custom revisions are
never pooled. These are descriptive comparisons, not proof of superiority.

`HermesMemoryAdapter.recall_compact(execution_ids=..., series_id=..., unit=...,
horizon=..., source_as_of=..., recorded_as_of=...)` returns one comparison and
at most two recent exact-filter lessons. The short narrative excerpts remain
unverified hypotheses; the full records remain retrievable by ID. Context filters
can be supplied when the application has valid descriptive labels.

`EvidenceMemory.check_comparison_claim(left=[.01], right=[0], actuals=[0],
metric="rmsle", relation="lower")` returns `contradicted`. The helper checks
arithmetic on caller-supplied arrays; it does not verify their provenance, parse
prose, or establish a business explanation. Supported relations are `lower`,
`equal`, and `lower_or_equal`; metrics are `mae` and `rmsle`. A caller can expose
this helper as a tool and allow a bounded correction using its existing budget.

`HermesMemoryAdapter.recall_brief(...)` provides progressive disclosure without
workflow instructions: a small overview of matched scores, sample counts and
unverified hypotheses, plus the full comparison for applications to expose on
demand. Applications can attach exact retrieval commands and hash-checked
submitted-code references. A matching source artifact does not prove the file
was executed. The bridge does not require model reuse, a particular experiment
sequence, memory writing, or use of the optional arithmetic checker.
