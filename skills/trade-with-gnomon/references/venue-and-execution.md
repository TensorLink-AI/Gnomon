# Venue integration and execution

Read when building a venue adapter or operating an authorized strategy. Use only
the sections relevant to the requested mode; a research task needs no deployment.
Reuse an existing verified adapter and supervisor when available.

## Venue contract

Read the venue's current official API/SDK documentation and available schemas.
Keep a small adapter and a record of the verified contract:

- Instrument identity: venue ID, symbol, asset type, base/quote and settlement
  currencies, price/quantity precision, increments, minimum size/notional, and
  contract multiplier where relevant. Identical ticker text need not identify
  the same instrument.
- Market data: timestamps, timezone, bar open/close convention, finality, calendar,
  gaps, adjustments, and when each observation became available.
- Account: balances, buying power, positions, pending orders, borrowing, margin
  and net/hedged position mode. Distinguish quantity, notional and posted margin;
  missing position fields mean unknown state, not zero exposure.
- Execution: supported order types, sides, time-in-force, extended sessions,
  reduce-only/close semantics, fees, funding, rate limits, authentication renewal,
  client IDs, cancellation, and order/fill reconciliation.

Discover support rather than translating every venue into an assumed
`buy/sell + positionSide` API. Check the relevant instrument details: equities
need trading sessions and corporate-action handling; derivatives may need expiry,
rolls, multiplier, funding, exercise or liquidation semantics; spot crypto and
perpetual contracts are different products. Add only the constraints that apply.
Unsupported instrument semantics block execution for that instrument until mapped.

Keep secrets in the host's secret facility or injected environment, not source,
logs or skill files. Use permissions sufficient for the task; trading does not
require withdrawal access. Follow the venue's documented token expiry/refresh
mechanism instead of assuming a fixed token lifetime.

## Execute and recover

For each decision event, use this sequence as a starting point and adapt its cadence
to the strategy:

1. Acquire a lock or durable lease for the account/strategy scope. Reconcile account,
   open orders and fills, including manual/external activity and uncertain prior
   submissions. Verify fresh data and the correct execution environment.
2. Derive new-exposure intent from relevant forecast evidence under the recorded
   policy, then apply permitted decision-time context, portfolio/risk limits,
   instrument constraints and venue precision rules. Record forecast IDs and any
   contextual adjustment. Protective actions may proceed under their risk policy
   without a new forecast. Recheck exposure after rounding.
3. Persist the decision and a stable local submission ID before submission; use
   the venue's client order ID facility where supported. Derive
   identity from account, strategy/version, instrument and decision event, with
   distinct IDs for separate legs; a process-local cycle counter is insufficient.
4. Submit only authorized intents through the venue adapter. Record acknowledgments
   separately from fills, and reconcile partial fills, fees, rejections, expiry and
   cancellation. A cancellation request is not a completed cancellation.
5. After a timeout, query by client/venue ID where available and reconcile before
   resubmitting. Without an ID lookup, use the venue's order/fill history and stop
   for unresolved ambiguity; do not infer rejection from a missing acknowledgment.
   A `409` is not universally success: verify the existing order matches the intent.
   If acceptance is unknown, stop conflicting submissions until resolved. Do not
   promise exactly-once execution where the venue cannot support it.
6. Persist the resulting state and structured diagnostics. On restart, reconcile
   with venue state before permitting new exposure. Recover multi-leg partial
   execution according to the strategy's explicit contingency plan.

Missing/stale state should block new exposure; handling existing positions follows
the user's defined protection policy. A kill switch must distinguish stopping new
orders, canceling outstanding orders, and liquidating positions. Do not assume
those are equivalent or that all have been authorized.

## Operate and improve

Select a container, managed job, persistent service or external driver that the
host and venue support. Verify restart behavior, clock synchronization, secret
injection, dependency availability and single ownership of execution. Do not assume
memory survives a restart, runtime package downloads work, or a particular cron
service lives/dies with a terminal. Run heartbeats only when required by the actual
contract, using its cadence and timestamp format.

For competitions, inspect registration, wallet, submission and reward rules as
venue-specific setup. Use an external driver only if permitted; do not submit an
idle container as a substitute for the strategy the rules require. Wallet creation,
funding and on-chain fees are separate operations from developing a strategy.

Test representative full cycles in a simulator/paper environment: ordinary fills,
partial fills, rejected/unknown orders, expired authentication, stale data, model
failure and restart recovery. A compile check or platform build smoke test is not
an execution test. Paper results also do not establish live fill quality. Enable
live execution only within the established mandate and validated venue contract.

Monitor liveness, data freshness, forecast/LLM failures, reconciliation drift,
exposure, turnover, costs and P&L separately. Use periodic aggregate reviews to
detect churn and silent degradation. Tune alert thresholds to the strategy; keep
notification and scheduling actions within the requested scope.

Use the configured Gnomon ledger by default for forecasts, decisions and outcome reviews; use
an order/fill journal for execution truth. Link both through decision/execution IDs.
Forecast error, realized P&L and counterfactual policy performance are distinct.
Retain model/config revisions and review changes on held-out evidence before
deploying them. Do not promote one competition's settings into universal rules.

Deliver the strategy rationale, venue mapping, runnable implementation/config,
validation evidence and operating instructions appropriate to the task. State the
current mode, actual actions taken, unresolved limits and next useful experiment.
