# Use the user's backtest engine

Accept an existing repository, package, CLI, API or simulator. Reuse its execution
model and adapt Gnomon forecast delivery to its callbacks, timestamped signal
table or strategy interface. Identify the installed version and the actual contract;
explain missing capabilities rather than silently replacing the engine.

At each decision event, expose only then-available data to the provider/policy.
Record forecast origin, input cutoff, target timestamps, policy revision and engine
order/fill IDs. Historical forecasts computed today retain today's ledger recording
time. Keep replay runs separate from prospective evidence; never backdate their
recording clock to obtain eligibility in a production history comparison.

Precomputed forecasts are suitable if each was generated from an isolated earlier
window and the engine receives it only after its simulated availability. Include
the inference delay if material. Freeze the selection/update procedure before the
evaluation window, including feature transformations, LLM adjustments and thresholds.
Preserve a trial log and treat a repeatedly inspected holdout as selection data.

Verify the engine's calendar, adjusted/raw price mapping, signal timing, order
eligibility, fill model, position accounting, costs, partial fills and overlapping
positions at the detail needed by the strategy. A close-derived signal cannot fill
at that already-passed close. With bar data, declare assumptions when stop/limit
ordering or intrabar liquidity is unknowable. Include cross-validation gaps for
overlapping targets where needed and account for dependent returns in uncertainty.

Use forecast loss/coverage to diagnose models and net returns, drawdown, exposure,
turnover and execution shortfall to diagnose policies. `review_decision` checks
point forecast error; it does not score a quantile gate's calibration, a stop-loss
rule or trading profitability. Calculate those with the retained quantiles and the
engine's outcome/fill history. For log-return forecasts, preserve return units and
return endpoints; do not append raw prices as their actuals.

## Evidence and measurement

The ledger is a maintained implementation of temporal evidence contracts on SQLite.
An SQL application can implement the same checks; storage technology is not the
differentiator. The repository's `benchmarks/hermes_ledger/RESULTS.md`
reported no performance benefit over a strong supplied SQL helper: the voluntary
pilot used no ledger calls and the explicit-ledger check exposed scope/schema
problems. These small synthetic tests did not measure trading performance.
That report is a repository artifact, not bundled with an installed skill.

A useful next evaluation measures whether agents preserve evidence cutoffs across
delayed/revised actuals, reject retrospective labels and unversioned providers,
retain exact context filters, reproduce saved reviews, and recover decision/order
identity after restart. Give comparison arms the same raw records, policies,
budgets and data visibility; include failures and count agent effort. Measure
forecast accuracy and net trading performance separately. Do not claim a ledger
advantage, profitability or reduced token cost until the measurements support it.

Recording times rely on the operator's clock and database integrity. Neither a
hash nor an application-assigned timestamp independently attests source truth or
unknown training data. Prospective recording is useful evidence under those trust
assumptions. Record model provenance and treat historical LLM decisions with
unknown training overlap as exploratory; use later, prospectively recorded outcomes
to assess the deployed policy.

Research background: [backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
and [LLM look-ahead bias](https://arxiv.org/abs/2309.17322).
