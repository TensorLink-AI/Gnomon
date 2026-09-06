# Validation and limitations

Gnomon establishes executable contracts and regression behavior, not forecasting
superiority, improved LLM reasoning or permission to act.

## Tested behavior

Production tests cover request/result alignment, unsupported inputs, cutoffs,
immutable ledger revisions, migrations, bounded evaluation, Python/CLI/MCP,
numerical regressions and installed-wheel operation. Local HTTP tests exercise
Ephemeris wire mapping, credentials, response validation and no-retry POSTs.
They do not test a live TSFM.

## Current agent comparison

The [matched workflow](../benchmarks/workflow/MATCHED.md) compares ordinary agent
software, the lean Gnomon session and that same session with optional ledger/time
tools enabled. It pins common
model/settings, code, corpus and software identities. Agent choices and answers
remain unmodified. Failures, missing answers and costs stay in the denominators.
Episodes commit answers before later observations arrive.

The 11-task retrospective cohort has four historical forecast windows, four
quantity/decision/temporal tasks and three episodes. This is a small comparison,
not a SOTA benchmark. Model training may have included the historical data;
transformation does not prove decontamination. Forecast reports retain
MAE/RMSE/MASE, complete-horizon coverage and baseline comparisons.

## Still unverified

- An actual matched ordinary/lean/full run with a user-selected agent model.
- Authenticated inference against the intended Ephemeris deployment and its served models.

These need endpoint configuration, credential environment-variable names and
spending approval. A reachable authentication gateway is not a live forecast check.
Reported-cost stopping can overshoot by one operation; hard billing limits are
service-operator responsibilities.

Old benchmarks and reports remain at Git checkpoint `2cba20e`. Their older
surfaces do not establish gains for the new default. Removing obsolete instruments
does not turn negative or missing evidence into success.
The [weighted delivery record](production/PLAN.md) is separate from publication.
