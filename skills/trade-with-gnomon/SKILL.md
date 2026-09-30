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
   (StatsForecast or your own registered model). Built-ins such as
   `historical_mean` are point-only: a quantile-based policy needs a quantile
   provider, and a point-only comparison does not validate a quantile gate. Use
   Ephemeris when requested or when its benefit justifies the cost; for ranking it,
   see the [ledger skill](../use-gnomon-ledger/SKILL.md#bounded-recovery). If a
   router is configured (`ledger.routers`) or you build one (see memory below),
   forecast through it and record its
   reason, `evidence_level` and (with memory) `effective_n` in the rationale; size
   down when evidence is weak. Never silently substitute an unavailable provider.
2. **Decide and record.** Turn forecasts into an explicit intent under the policy.
   In Python call `record_trade_decision` (`scripts/trade_decisions.py`), which checks
   mode, clock and live promotion; over CLI/MCP call `record_decision_summary` on a
   server whose operator config sets `decision_context.trading_mode`. Bind
   `execution_id` to a concise rationale, assumptions, invalidation conditions and
   sourced context (valid interval, `source_available_at`); a context label is an
   assertion, so define it before evaluation. Persist `decision_id` before orders.
3. **Execute within the mandate.** Journal the order intent with `decision_id` and
   `execution_id`, derive the client ID from account, decision and leg (reuse it on
   recovery), and reconcile before changing exposure. Invalidation conditions are
   evidence, not automatic exits; protective exits follow the risk policy regardless.
4. **Append outcomes and review.** Use `append_actual` when observed outcomes arrive,
   preserving series, target timestamp, unit and source availability. For return
   forecasts, transform observed prices into the same returns before appending.
   Call `review_decision` with explicit `source_as_of` and `recorded_as_of`; inspect
   coverage and `review_ready`. It computes forecast-error evidence, not strategy
   P&L or proof that the trade thesis was correct. Score fills/costs in the engine
   or order journal; ledger `evaluate` can additionally persist a forecast score.
5. **Compare comparable history.** Use `compare_history`, or `compare_context` for
   exact pre-recorded context labels, using the
   [ledger skill's worked calls](../use-gnomon-ledger/SKILL.md#worked-calls). They need
   2–8 explicitly versioned providers; never invent a version. When strict matching is
   unavailable, review individual executions or run a separately scoped evaluation.
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

## Choosing models with memory

With several related instruments and candidate models, let a Gnomon router choose per
decision instead of fixing one model: `pool` the instruments and add `memory`, which
retrieves the most similar past situations (volatility, trend, level shift) across
them and scores each model on what followed. In Python build it yourself with
`GnomonSession(..., routers={...})`; over MCP ask the operator. Replay it on your
backtest folds against the best single model first; if nothing beats a zero or naive
forecast, no router will. Put each forecast's `evidence_level`, `effective_n` and top
neighbours in the decision rationale, and size down when evidence is weak. Setup,
settings and replay: [route-with-gnomon](../route-with-gnomon/SKILL.md).

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
   after costs, forecasts against a simple baseline, reconciliation health.
   `promotion_review` counts complete paper decisions; the user writes and approves
   the promotion record, and you never set `approved_by`. What `verify_promotion`
   does and does not check: [promotion](references/promotion.md).
4. **Go live small**, with the user's size cap and a stated condition for returning
   to paper.

A live mandate from the user sets the limits; it does not skip steps 2–3 unless the
user explicitly waives them for this strategy.

Replayed forecasts are recorded when you run them, after their targets, so
`compare_history` and `compare_context` exclude them from comparisons of prospective
evidence automatically. That protection depends on a real clock: backdating recording
times through a fixture clock defeats it.

The ledger's value is its time contracts (ledger-assigned recording times, separate
valid/available/recorded times, exact cutoffs and cohorts), under a trusted clock
and database; see [evidence and measurement](references/backtesting.md#evidence-and-measurement).

## Evaluation and reasoning boundaries

Be creative in research and disciplined about decision-time evidence. New exposure
needs forecast support under the policy; context can explain, delay or reduce an
intent, but remembered market outcomes cannot supply the signal. Record material
adjustments and sources; risk limits stay authoritative; machine-validate intents.

For historical claims, freeze the policy (including LLM vetoes, sizing, exits and
model selection) before evaluation or predefine walk-forward updates. Label
contributions with unknown training overlap exploratory and validate them
prospectively; a dated prompt cannot remove training knowledge. Keep a trial record
and choose a new holdout after inspecting results.

Compare forecast quality and net trading utility separately, with realistic costs
and fills ([backtest engines](references/backtesting.md)). A median is not an
expected return, and interval width is not a probability of profit.

## If something fails

- No ledger or writes disabled: continue research read-only, disclose the missing
  audit trail, and ask the operator; never edit configuration to enable writes.
- `ledger.decision_context` absent or showing another mode: stop recording and ask
  the operator for the right ledger.
- A journal file already exists for an event: that event was decided; reconcile it
  instead of recording a new decision or order ID.
- Order or venue errors: follow [venue/execution](references/venue-and-execution.md);
  reconcile positions before any retry.

## Report

State the mode, the strategy and its revision, what was actually executed or only
recorded (with decision/execution IDs), evidence against a baseline after costs,
unresolved limits and the next experiment. Say plainly when no edge survived costs.

## Related skills

[Ephemeris setup](../setup-gnomon-ephemeris/SKILL.md), [Gnomon usage](../use-gnomon/SKILL.md)
and [ledger](../use-gnomon-ledger/SKILL.md) cover setup and schemas in detail.
