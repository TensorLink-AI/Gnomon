# Matched agent evaluation

Compare ordinary software, the lean Gnomon session and the same session with
optional ledger/temporal tools. The agent chooses its tools, arguments and answers;
the host does not manufacture or repair answers from tool results.

Start with [matched controls](MATCHED.md) and [experiment configuration](experiment/README.md).
The [11-task cohort](cases/MATCHED_RETROSPECTIVE.md) combines retrospective forecast
windows, quantity/decision/temporal tasks and committed multi-phase episodes.

```bash
python -m benchmarks.workflow.run_workflow --help
python -m benchmarks.workflow.matched --help
```

The runner defaults to that cohort. Episodes require a matched experiment and the
shared driver. Single-phase custom cases can also be run or scored independently;
that alone does not establish a controlled comparison.

Inputs, code, requested model settings and software identities are pinned.
Each model call makes one bounded transport attempt. Failed work, missing answers
and unknown costs remain visible; current journal receipts are required for resume.

Case schema v2 has no host-generated repair/outcome stages or publication grading.
The score report uses one `answered_rate` for answered statuses; correctness and
episode completion are separate measurements.
Use fresh output directories for the new format. Old formats are rejected rather
than migrated or silently reinterpreted.

Templates cannot call a real model without configuration and spending approval.
Tool/time/token limits and reported-cost stopping are not hard billing caps.
