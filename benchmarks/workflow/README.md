# Matched agent evaluation

Use [MATCHED.md](MATCHED.md) for the controlled ordinary/lean/full comparison.
The built-in driver preserves agent-selected tools, arguments and final answers.
It does not recover answers from Gnomon artifacts or force preferred tool calls.

Configuration starts from [experiment templates](experiment/README.md).
The [11-task cohort](cases/MATCHED_RETROSPECTIVE.md) combines retrospective
forecast windows, utility/temporal tasks and committed multi-phase episodes.

Run `python -m benchmarks.workflow.run_workflow --help` for the runner and
`python -m benchmarks.workflow.matched --help` for matched comparison.
Inputs, code, models/settings and software identities are pinned; attempts and
reported costs remain recorded through failure and resume.

The unconfigured examples deliberately cannot call a real model. Tool/time/token
limits and reported-cost stopping are not a provider-enforced billing cap.
Follow the operator guide before enabling remote calls.

Old synthetic/context cases and generic summary comparison remain solely for
schema/scoring regressions. Their old forced-tool adapter has been removed;
they must not be relabelled as measured improvement of the new default.
