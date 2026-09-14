# Development088: recording visibility fixed before catalogue construction

The synthetic public1.2.0 probe confirmed a defect in our development review
adapter, not in Gnomon's score arithmetic. Two forecasts recorded on January30
with a January8 origin entered a January10 query's configuration catalogue.
Both target horizons had closed by the query date, but the executions did not
yet exist at that recording cutoff. The public comparison correctly excluded
their scores. Nevertheless, catalogue ordering put their empty pair first.

| January10 query, page size1 | Original records | With later-recorded events |
|---|---|---|
| Old adapter candidate pairs | 3 | 10 |
| Old adapter first pair | a / b | late_d / late_e |
| Old adapter status | evidence_available | insufficient_evidence |
| Corrected adapter candidate pairs | 3 | 3 |
| Corrected adapter first pair | a / b | a / b |
| Corrected adapter status | evidence_available | evidence_available |

The new `visible_agent_review.review` reads authoritative public
`ledger.execution` records before catalogue construction, checks the inclusive
recording cutoff and validates visible event envelopes against the saved
provider, revision, fingerprint, request and forecast result. A separately
supplied event request cannot change its series, unit, horizon, history or
timestamps. Each unique execution reference is read once for this filter;
those extra reads are included in query accounting. The old adapter remains
unchanged for reproduction of earlier results.

All three corrected pages preserve the original cards, matched scores, global
origin windows and pagination exactly. Future-recorded configurations are absent
from the candidate index. Full operator evidence identifies excluded references
and the recording boundary; those exclusions are not eligible forecast evidence.
At the exact recording boundary the new executions become discoverable, while
public compare_history still rejects their non-ex-ante scoring. Visibility is
not equivalent to historical forecast eligibility.

## Verification and preserved test-harness failure

The first reproduction was frozen at feaa1f5, and the corrected adapter/checker
at43b21f4. Seventeen unit tests passed, including nine new checks for recording
visibility, tampered references, equivalent timezones, inclusive cutoffs and
duplicate read accounting. The initial integration checker reached its final
source-byte invariant and failed: opening the published TemporalLedger constructor
changed the original synthetic database's file bytes. Its failed output and
source are preserved. That initial source ledger is no longer byte-identical to
the hash recorded when the reproduction finished; do not claim otherwise.

The revised checker82c1810 copies the database before opening it, records
constructor and query effects separately, and preserves all source artifacts.
A new fixture/probe reproduced the original catalogue failure. All38corrected
integration checks passed. The constructor changed file bytes but an independent
SQLite logical dump remained identical; subsequent queries changed neither the
initialized working database nor any original source artifact. Saved evidence
hashes and refusal to overwrite existing files were checked. This does not
establish that every public ledger read is free of constructor side effects.

Both probes used the pinned published distribution1.2.0,
build1.2.0+ga38cd0cad353.s9723394ccb6d. No dirty local product module was imported.
Each probe made11synthetic setup forecasts,22total; review queries made zero
provider calls. The final check counted120execution reads,8comparison reads and
7actuals reads across its tests and control comparisons. The failed check's
total read count was not retained. API calls and paid spend were zero. Raw stdout,
stderr, command statuses, both synthetic databases and full query payloads remain
in the088bundle with hash inventory. Initial source-byte mutation is an explicit
limitation; the corrected source remains intact.

## Scope and next use

Use the separate development adapter:

```python
from benchmarks.ledger_optimization.visible_agent_review import review

packet = review(db, records, task, evidence_math, "new-evidence.json", limit=12)
```

An event envelope must contain the typed execution's request, result, provider,
revision and fingerprint, plus the original production request and configuration
identity. Missing or conflicting references fail rather than generate a score.
This check does not authenticate arbitrary configuration labels independently
of the event log; existing catalogue checks still enforce stable label/revision
relationships. A production integration must retain trusted event provenance.

No claim is made that this hypothetical late-recorded event occurred in030,
that correcting it changes any previous forecast result, or that the helper
improves agent decisions. It is a necessary visibility boundary for using a
long-lived ledger safely. The latest actual-agent scores remain030; numerical
development086 remains0.249618,2.93%below its matched control. The20%/95%goal is
unmet. No API evaluation or final confirmation was launched. Main/PyPI and
protected data are unchanged.
