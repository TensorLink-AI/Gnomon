# Current evaluation harness

One retained workflow compares the same agent model using ordinary software,
Gnomon's lean session and the same session with optional ledger/temporal tools.

Start with [matched controls](workflow/MATCHED.md), the
[operator configuration](workflow/experiment/README.md) and the
[retrospective cohort](workflow/cases/MATCHED_RETROSPECTIVE.md).

```bash
pytest -q benchmarks/tests
```

These are harness regressions, not evidence that an LLM improves. Real container
checks require explicit locally built software/service image IDs; see
[software isolation](workflow/software/README.md) and
[service isolation](workflow/service/README.md). Model calls require separate
configuration and spending approval. The repository has no scheduled paid run.

The active benchmark code is `workflow/`, with shared transport/provenance helpers
in `common/`. Fixtures for older summary/schema compatibility remain tests, not
alternative recommended experiments. The forecast data under `workflow/data/`
retains source hashes and provenance.

Superseded benchmark families, their runners and copied archives were removed.
They and older results are recoverable at Git commit `2cba20e`; they are not
evidence for the new default session. Production numeric/cutoff regressions remain
in `tests/`.

The 0.9 cleanup also removes context-engine cases and grading branches (recovery:
`333ed2c`). Old full-arm results describe a different runtime and cannot establish
the value of today's optional tools.
