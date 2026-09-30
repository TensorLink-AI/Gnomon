---
name: trade-with-gnomon
description: Make and review forecast-led trade decisions with Gnomon—forecast, record the decision in the ledger, then score it against realised outcomes. Use when researching, backtesting, or operating a trading strategy with local models, StatsForecast, or optional Ephemeris on a user-selected venue.
---

# Trade with Gnomon

Use forecasts as the primary quantitative basis for trade selection and new
exposure. Apply your reasoning to hypotheses, model choice, interpretation, sizing,
timing and abstention. Choose the strategy, targets and tools that fit the user's
objective; StatsForecast and Ephemeris are independent options. Gnomon supplies
forecast and decision evidence; the user's venue or backtest engine handles trades.

## The working loop

Start from the user's research, paper, competition or live mandate. Resolve only
consequential gaps in instruments, capital/risk limits, execution authority and
operating scope. Within delegated bounds, choose and test the policy yourself.
Discover installed capabilities. Use a configured ledger by default for the loop;
summary/actual/lesson writes through CLI/MCP require `allow_outcome_writes=true`.
If recording is unavailable, continue research, disclose the missing audit trail,
and resolve setup before claiming the workflow is prospectively recorded.

1. **Forecast.** Use `session.forecast` / `gnomon_forecast` with a named series,
   exact units, history and target timestamps, and a decision-time cutoff. Keep
   `execution_id`, provider/revision and the returned uncertainty. Select relevant
   targets—prices, returns, spreads or volatility—and validate the policy using
   those outputs. Reuse saved evidence where appropriate. Unless the user names
   models, choose a target-appropriate baseline and one versioned candidate
   (StatsForecast or your own registered model). Use matching supported requests
   for `compare_history`. Built-ins such as `historical_mean` are point-only:
   a quantile-based policy needs a compatible quantile provider. A separate
   point-only comparison can assess point accuracy, but it does not validate the
   quantile gate or automatically match the quantile decision forecast. Use Ephemeris when requested or when
   its benefit justifies the latency/cost budget; direct `compare_history` cannot
   rank it until its connector reports a revision, but an operator router with
   `identity_policy = "prospective_unattested"` can, disclosed. If a router is
   configured (`ledger.routers` in capabilities), forecast through it: it serves the
   model that ledger evidence and declared costs favour and shadows the rest. Never
   silently substitute another provider for an unavailable one.
2. **Decide and record.** Translate forecasts into an explicit intent under the
   selected policy. In Python, call `record_trade_decision` from
   `scripts/trade_decisions.py`; it checks the mode, clock and live promotion, then
   calls `record_decision_summary`. Over CLI/MCP, call `record_decision_summary` on a
   server whose operator config sets `decision_context.trading_mode` (see below).
   Either way, bind `execution_id` to a concise `rationale`, `assumptions`,
   `invalidation_conditions`, context and evidence refs.
   Capture relevant context, such as a regime label, with its valid interval,
   `source_available_at` and source. Persist `decision_id` before submitting orders.
   A context label remains an assertion; define its calculation before evaluation.
3. **Execute within the mandate.** Record the structured order intent in the order
   journal, linked to `decision_id` and `execution_id`. Derive a venue-compatible
   client ID from account, decision and leg identity, and reuse it on recovery.
   Reconcile positions and open orders before changing exposure. The summary's
   invalidation conditions are evidence for the policy, not automatically running
   exit orders. Protective exits follow the risk policy even without a new forecast.
4. **Append outcomes and review.** Use `append_actual` when observed outcomes arrive,
   preserving series, target timestamp, unit and source availability. For return
   forecasts, transform observed prices into the same returns before appending.
   Call `review_decision` with explicit `source_as_of` and `recorded_as_of`; inspect
   coverage and `review_ready`. It computes forecast-error evidence, not strategy
   P&L or proof that the trade thesis was correct. Score fills/costs in the engine
   or order journal; ledger `evaluate` can additionally persist a forecast score.
5. **Compare comparable history.** Use `compare_history`, or `compare_context` for
   exact pre-recorded context labels, using the
   [ledger skill's worked calls](../use-gnomon-ledger/SKILL.md#worked-calls). These operations require 2–8 explicitly versioned providers;
   unknown revisions, including the current Ephemeris connector, cannot be made
   eligible by inventing a version. Pretrained providers also need eligible training-
   cutoff metadata. Review individual executions or use a separately
   scoped evaluation when strict history matching is unavailable.
6. **Carry forward a checked lesson.** When outcomes are complete, `record_lesson`
   saves a hypothesis together with a ledger-computed immutable review. Use
   `export_lesson` for IDs and verification calls; retrieve the evidence on reuse.
   Version later lessons using `previous_lesson_id`. Let measured results guide
   the next experiment; one trade or cohort does not establish future superiority.

Run the [worked lifecycle example](references/trade-lifecycle.md) for a log-return
forecast, quantile-based intent, decision-linked client ID, delayed/revised actuals,
context comparison and exported lesson. It is an offline API demonstration, not
an endorsed strategy. Build real workflows from `scripts/trade_decisions.py`, not
the demo harness's fixture clock or toy policy. Read only the references you need:
[forecast providers](references/forecasting.md), [backtest engines](references/backtesting.md),
and [venue/execution](references/venue-and-execution.md).

## Backtest, paper, live

Keep one operator-configured ledger per mode, such as `backtest.db`, `paper.db`
and `live.db`; never mix modes in one ledger. Both paths store the mode as a
`trading_mode` context label and refuse a ledger holding another mode:

- **Python:** `record_trade_decision` accepts only `backtest`, `paper` or `live`.
- **CLI/MCP:** the operator gives each mode its own TOML, e.g.
  `[decision_context.trading_mode]` with `value = "paper"` and a `source_ref`. The
  server adds that label to every summary and rejects a caller-supplied
  `trading_mode`. Check `gnomon_capabilities` → `ledger.decision_context` before
  recording; if it is absent, ask the operator rather than recording unlabelled
  decisions. Only the operator can start a live-configured server.

Progress through the modes in order, skipping only what the data makes impossible:

1. **Backtest** when history exists, with the policy frozen first. This is simulation
   evidence only.
2. **Paper-trade prospectively** in the paper ledger until the user's minimum is met
   (a number of decisions or days with complete outcomes). Agree that minimum and the
   promotion criteria before paper trading starts; paper is mandatory when no
   backtest was possible. Also exercise the execution failure cycles in
   [venue/execution](references/venue-and-execution.md).
3. **Promote only on user approval** of a review against those criteria: net result
   after costs, the policy's forecasts against a simple baseline, reconciliation
   health. `promotion_review` counts paper decisions with complete outcomes at
   explicit cutoffs; add the engine/journal results and give it to the user. The
   user writes the promotion record (`review`, `min_complete_decisions`, `criteria`,
   `approved_by`, `approved_at`); never set `approved_by` yourself. Live decisions
   in Python require that record: `verify_promotion` validates its fields, a
   positive integer decision minimum, and a timezone-aware approval time at or
   after the review cutoffs and no later than now. It rechecks completion counts
   and IDs at those saved cutoffs; it does not verify performance criteria or
   approver identity. Review scores, costs, fills and any later revisions separately.
   A minimum expressed in days also needs a separate duration check. Over MCP,
   `source_ref` only names the promotion record: it does not load or validate it.
   The operator/execution layer must verify approval before enabling live orders.
4. **Go live small**, with the user's size cap and a stated condition for returning
   to paper.

A live mandate from the user sets the limits; it does not skip steps 2–3 unless the
user explicitly waives them for this strategy.

Replayed forecasts are recorded when you run them, after their targets, so
`compare_history` and `compare_context` exclude them from comparisons of prospective
evidence automatically. That protection depends on a real clock: backdating recording
times through a fixture clock defeats it.

## What the ledger establishes

Gnomon assigns local recording times through its configured clock; public write
operations do not accept a caller-supplied `recorded_at`. It keeps outcome valid
time, declared source availability and local recording time separate. Reviews
select evidence at explicit cutoffs; revised actuals change later reviews while
saved lesson reviews retain their original basis. Context comparisons enforce
matched requests, timing eligibility and exact labels instead of silently widening
cohorts. These contracts are the reason to use the ledger beyond storing rows.

This holds only under a trusted clock and intact, operator-controlled database; it
does not prove source correctness or model training provenance. See
[evidence limits](references/backtesting.md#evidence-and-measurement).

## Evaluation and reasoning boundaries

Keep the LLM creative in research and disciplined about decision-time evidence.
New exposure needs relevant forecast support under the policy. Context can explain,
challenge, delay or reduce an intent; remembered market outcomes cannot supply the
signal. Record material contextual adjustments and their sources. Risk limits and
protective exits remain authoritative. Machine-validate final intents before use.

For historical performance claims, freeze the policy before evaluation or predefine
walk-forward updates using only earlier observations. Include LLM vetoes, sizing,
exits and model selection in that boundary. When an LLM or pretrained forecaster's
training overlap is unknown, label its historical contribution exploratory and
validate it prospectively. A dated prompt alone cannot remove training knowledge.
Keep a trial record; choose a new holdout after inspecting results.

Compare forecast quality and net trading utility separately, with realistic costs
and fills ([backtest engines](references/backtesting.md)). A median is not an
expected return, and interval width is not a probability of profit.

## Related skills

[Ephemeris setup](../setup-gnomon-ephemeris/SKILL.md), [Gnomon usage](../use-gnomon/SKILL.md)
and [ledger](../use-gnomon-ledger/SKILL.md) cover setup and schemas in detail.
