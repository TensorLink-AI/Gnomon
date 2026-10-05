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

## Check causality, pre-register, seal a holdout

- **Leakage check.** Before trusting any backtest, run
  [`scripts/leakage_check.py`](../scripts/leakage_check.py): `check_causal(decide, rows)`
  re-runs your decision function on data truncated at sampled times and fails if any
  decision changes. It catches centred windows, full-sample normalisation, fits on all
  data and forward-filled future values. Also run the policy on a zero-drift null
  (simulated or block-shuffled returns): a positive gross result there is a bug.
- **Pre-register.** Before evaluating candidates, commit the pass/fail thresholds
  (net Sharpe or utility, drawdown, cost and delay stress, regime split, parameter
  plateau) and keep an append-only trial log of every variant run. Tie each result to a
  hash of the strategy code and parameters so it cannot drift from the code that made it.
- **Seal a holdout.** Keep the final period unreadable in code until one recorded,
  one-shot unseal names the single revision allowed to run on it. Every look at a
  validation period turns it into selection data, so count those looks.
- **Twin for every added layer.** Evaluate a memory router, an overlay or a filter next to
  an identical twin without it; for routers use `memory_ablation` (route-with-gnomon).

## Live state must match the backtest

Path-dependent state (a drawdown brake's peak equity, a volatility-scaling history, an
incumbent position) must be carried live exactly as in the backtest. Recomputing on a
rolling window resets that state and changes the strategy: in one multi-asset test a
400-day window cut validation Sharpe from 0.82 to 0.58. Keep full history (or persist the
state) and verify that live decisions equal the backtest's at the same timestamps.

## Evidence and measurement

What the ledger establishes: Gnomon assigns local recording times through its
configured clock (public writes accept no caller-supplied `recorded_at`); keeps
outcome valid time, declared source availability and recording time separate;
selects evidence at explicit cutoffs, so revised actuals change later reviews while
saved lesson reviews keep their basis; and enforces matched requests, timing
eligibility and exact context labels instead of widening cohorts. This holds only
under a trusted clock and an intact, operator-controlled database; it does not prove
source correctness or model training provenance.

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
