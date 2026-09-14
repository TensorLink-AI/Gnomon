# Development070: shared one-hour blends with unchanged evidence retrieval

069found range limitations account for only23.6%of pooled squared ledger error,
but its floor allowed hindsight per-hour weights. This experiment tests a
prospective common per-hour action without supplying future-derived projections.
Freeze protocol, numerical module, wrapper and tests before scoring source data.

The only learner change from068:24one-hour simplex weight vectors in place of
four six-hour vectors. Same seven slots (original six plus066search-selected),
mean smoothed case RMSLE, .01penalty averaged over weight vectors,045current-only
global anchor extended by0,16eligible-neighbor retrieval, masses, nominal
availability/recording clocks and temporal filters. Repeated four-block weights
must have identical objective in both formulations. Each arm uses the same
24-hour learner. No hourly action parameter is selected using production scores.

Reuse exactly066/068sources and416tasks;125completed warm-up raw forecasts,
three known unavailable warm-ups. No new raw forecast, historical model refit,
API call or protected-data access. Historical selected-role forecasts retain
actual historical configuration IDs. Ledger uses prior-backtest search plus
matured production retrieval; control neither. This remains a combined numerical
memory treatment, not an isolated causal component effect or Gnomon/Hermes run.

SLSQPftol1e-14,max500,up to32same-objective Frank-Wolfe refinements, acceptance
success=true, feasible simplex, gap<=1e-5, objective no worse than initializer.
New dimension168weights; do not relax convergence after seeing failures. Stop
and retain failure if any fit fails. Two fits/task,832new fits, no silent drop
or fallback. An observation timeout never authorizes restarting the process.

Execute an isolated copy of frozen068runner with fit/combine replaced. Retain
its full original base-report and manifest, plus a070amendment identifying both
code hashes. Keep068source unchanged. After fitting, separately join all416saved
068control/ledger scores as fixed comparators by task identity and actuals.
Keep this comparison in a separate file; no new predictions enter any history.

Gate:>=20%improvement versus matched hourly control ANDfixed strong050block-CV;
positive improvement versus061AND068ledger overall; all four comparisons
positive in both domains. Preserve early/later means and existing068control
comparison too. Source search attempts/fits, surrogate solves,045anchors and
068comparison preparation costs remain disclosed. No paid confirmation unless
the gate passes; a passed reused-development gate still would not establish
the final target, which requires1.2.0/Engydeepseek-v4.1-flash matched agents on
untouched final cases with paired95%uncertainty excluding zero. Main/PyPI unchanged.
