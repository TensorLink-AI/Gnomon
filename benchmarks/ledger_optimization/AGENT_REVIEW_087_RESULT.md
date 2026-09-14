# Infrastructure087: smaller reviews with unchanged evidence

This is a ledger-interface improvement, not another numerical forecasting
method. The completed030 agent run already used pairwise RMSLE review cards.
The new adapter presents those same pairs, metrics, counts, windows and exact
ties with one configuration index and references to immutable full evidence.
Equal windows are shared only when their complete underlying objects match,
including actual/execution identities; equal numbers alone are insufficient.

The full replay covers100persisted reviews from100of104completed030tasks,
25per series. The four round0tasks have no persisted full review and remain
explicitly listed. No cases or grades were rescored or removed.

| Compact UTF-8 measure | Existing summary | Brief summary |
|---|---:|---:|
| Total bytes over100reviews |1,851,512|792,153|
| Median bytes per review |21,618|9,255|

Mean per-review byte reduction is56.2052%. Total original logged bytes including
the old outer task/record-count fields were1,857,495; the table compares the
reconstructed existing compact_cards payload to the brief view.1,978windows
share identical evidence with an earlier window. The full original response
remains available by file path/SHA-256 and per-card/window JSON pointers.

All33,670reference, metric, identity, count, tie, pagination and immutability
checks passed for105original source files. The adapter never creates global
ranks across different pair cohorts, selects a forecast or calls a provider.
Pagination disclosure distinguishes all available pairs from the returned page.
This measures bytes and semantic fidelity. It does not measure model tokens,
latency, improved reasoning, task completion, forecasting accuracy or ledger ROI.

## Preserved correction and runtime checks

Original helper/audit frozen atb7ca1d7; initial001replay retained. Subsequent
interface review found its next call expanded custom page sizes to12. Correction
930f87f preserves the requested page size and rejects inconsistent next offsets.
The separately frozen002replay passed33,670checks again. All100default-page
views are exactly unchanged. Eight unit tests cover ties, missing/cold-start
support, differing actual revisions, pagination, invalid identities/metrics,
future origins and immutable evidence files.

Public-runtime probes used gnomon-forecast1.2.0,
build1.2.0+ga38cd0cad353.s9723394ccb6d in the pinned isolated environment. The new
review function produced the correct three-origin synthetic comparison, saved
hash-verifiable full evidence, executed a limit1next-page call, and preserved
the earlier evidence after a physically present future-available revision.
The ledger file stayed byte-identical during each review; provider calls during
review were zero. Future revision ingestion was an explicit synthetic setup
mutation, not a query side effect.

Costs retained: old public-card regression30synthetic forecast executions,
initial adapter probe6, corrected probe9 (45total), zero API calls. Both review
corpus replays and runtime logs/reports are archived. The old030agent run's costs
and accuracy results remain unchanged; no new Engy/Hermes comparison was run.

## Development usage and remaining gate

The benchmark adapter is available on the development branch:

```python
from benchmarks.ledger_optimization.agent_review import review
from benchmarks.hermes_ml_checkpoint_v4 import dev_evidence_summary

packet = review(
    db,                   # existing published TemporalLedger
    records,              # recorded v4 execution events
    task,                 # series_id, unit, horizon, explicit origin
    dev_evidence_summary, # separately pinned development metric helpers
    "review-evidence/new-query.json",
    offset=0, limit=12,
)
```

Use a fresh evidence path; existing files are never overwritten. Follow
pagination.next_call only for its stated task origin. Resolve a window's
same_as within that card, or read the original full window via its pointer.
Missing support remains insufficient_evidence, not a zero score or a winner.
This does not replace the already completed v4 runs or change their transcripts.

Evidence bundle: results/agent-review-087-bundle, with both replay attempts,
frozen initial sources, real-runtime synthetic ledgers, logs and per-file hashes
in evidence/agent-review-087.json. Main/PyPI/protected data unchanged. This helper
can be integrated into a prospectively frozen agent workflow; it is not evidence
for relaxing the unchanged20%/95% accuracy objective or launching confirmation.

A clean replay using only the source subset copied into the bundle passed33,670
checks and reproduced all100views byte-for-byte. The initial six-forecast runtime
probe's harness was revised before a contemporaneous source hash was saved;
its outputs/ledger remain retained. The corrected nine-forecast probe's complete
harness is committed at930f87f and is the reproducible integration check.
