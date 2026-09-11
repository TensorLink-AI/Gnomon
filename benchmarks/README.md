# Agent evaluation

One workflow compares the same agent model using ordinary software, Gnomon's
lean session, and that session with optional ledger and temporal tools.

Start with [matched controls](workflow/MATCHED.md),
[operator configuration](workflow/experiment/README.md), and the
[11-task retrospective cohort](workflow/cases/MATCHED_RETROSPECTIVE.md).

For the development experiment on accumulating and querying experience, see
[the longitudinal evidence-workflow benchmark](experience_workflow/README.md).
It compares Gnomon with a persistent SQLite control, independently audits revised
evidence and forecast-input visibility, and tracks a preregistered agent-work
objective separately from forecast quality. It has not established superiority.

```bash
pytest -q benchmarks/tests
```

These tests verify the harness, not improved agent performance. Real container
checks need locally built image IDs; see [software](workflow/software/README.md)
and [service isolation](workflow/service/README.md). Model calls need explicit
configuration and spending approval. CI makes no paid model calls.

`workflow/` contains the runner, agent loop, scoring, isolation and attempt journal.
`common/` contains transport, checkpoint and provenance helpers.
Frozen forecast inputs retain their source hashes and attribution.
Production accuracy, cutoff and ledger regressions live separately in `tests/`.

The old promotion/audit/generation runners, smoke corpus, host-generated follow-up
answers and response-cache machinery are removed. Recovery: Git commit
`1642cb2` (earlier benchmark archives: `2cba20e`). Case schema v2 and current
attempt receipts are required; old results are not comparable with this workflow.
