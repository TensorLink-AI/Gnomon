# Worked forecast-to-decision lifecycle

Run the bundled [trade_lifecycle.py](../scripts/trade_lifecycle.py) with Gnomon in
the same Python environment and a **new** output directory:

```bash
python /path/to/trade-with-gnomon/scripts/trade_lifecycle.py /tmp/trade-lifecycle-demo
```

From a source checkout without installing Gnomon:

```bash
PYTHONPATH=src python3 skills/trade-with-gnomon/scripts/trade_lifecycle.py /tmp/trade-lifecycle-demo
```

Choose another output path if it exists. The example requires no extra packages,
credentials or network. It writes a synthetic ledger, an intent journal and an
evidence packet; it submits no orders. Its controlled clock is explicitly a test
fixture, not evidence that these forecasts existed on the simulated date.

## Adapting this to a real strategy

The example has two files. Build from [trade_decisions.py](../scripts/trade_decisions.py);
treat `trade_lifecycle.py` as a demo harness, not a template.

- **Reuse** everything in `trade_decisions.py`: `record_trade_decision` (checks the
  mode, claims the event's journal, records the summary with its `trading_mode`
  label, persists a decision-linked client order ID before any submission),
  `client_order_id`, and `promotion_review`/`verify_promotion` for going live.
- **Replace** everything the demo defines itself: the synthetic prices, the
  `empirical`/`zero-return` toy providers, `choose_intent`'s
  `q10 > log1p(cost_hurdle)` policy, `assess_outcome`'s fixed-fraction cost model
  and `classify_regime`, with validated choices for the user's instruments and venue.
- **Never copy** `FixedClock` or `ledger.clock = ...`. Open the ledger with its
  default system clock. `record_trade_decision` refuses a non-system clock unless
  `require_system_clock=False`, which only the demo passes.

On real hourly returns the example policy will rarely clear its hurdle; choose a
policy for the actual forecast target and costs rather than inheriting this one.

## The decision

The script generates prices, converts them to consecutive one-hour log returns,
and forecasts one step's 0.1/0.5/0.9 quantiles using historical empirical quantiles.
It also records a zero-return baseline with degenerate quantiles on the identical
request. Both are toy providers with explicit revisions; neither claims calibrated
uncertainty or predictive skill. Swap the selected provider for StatsForecast or
Ephemeris through the [forecasting contract](forecasting.md).

The illustrative policy assumes a flat long-only account and a one-hour holding
horizon. It permits a LONG intent only when `q0.1 > log1p(cost_hurdle)`, where the
cost hurdle is the all-in gross simple return needed to cover the assumed round trip.
The logarithm puts that hurdle in the forecast's units. This is a one-step example;
summing marginal quantiles does not yield a multi-step return quantile.

The regime rule is fixed before the outcome: a positive mean over the last 24
completed hourly log returns gives `positive-trailing-24h`; otherwise the label is
`nonpositive-trailing-24h`. The script computes it from the forecast history and
records the rule version in the context source reference.

The decision summary saves the policy, signal and assumptions. Its `decision_id`
determines a stable account/leg-specific client order ID in `intent.json`. Quantity
is deliberately unset, and the journal says `submitted=false`: an authorized
executor must supply sizing, verify account state and implement the time/risk exits.
The ledger does not execute its text `invalidation_conditions`.

On recovery, reuse the persisted event → decision → client ID mapping. Calling
`record_decision_summary` again creates a new ID; it is not an idempotency primitive.
An uncertain write or order submission requires reconciliation, not a new decision
and fresh order ID. Multiple legs need distinct IDs and a persisted mapping.

## Outcomes, comparisons and lessons

Before the outcome arrives, `review_decision` is pending. When the synthetic closing
price arrives, the script converts it to the same log-return target and appends it
with its real *fixture* availability time. The review becomes complete. It compares
the two explicitly versioned providers using the decision's exact context label,
records a bounded lesson, and exports it with verification calls.

The default fixture exposes a bad call: the selected empirical model's MAE is
approximately **0.0012**, versus **0.0004** for the zero-return baseline. Stdout
prints the ledger's ranking with zero-return first. The LONG intent also misses
the cost hurdle. Under the explicit toy cost model—round-trip costs fixed at
0.001 of entry notional—the hypothetical net simple return is
`expm1(observed_log_return) - cost_hurdle`, approximately **−0.00059992 (−6 basis
points)**. This is a scenario calculation, not realised P&L: there are no fills,
and the script submits no orders. A `NO_NEW_POSITION` intent has zero policy return.

The saved lesson derives “exceeded”, “matched” or “missed” by comparing the actual
log return with `log1p(cost_hurdle)`. The ranking and hypothetical cost-adjusted
result use the original outcome cutoff. The later revised review is separate;
it does not rewrite that lesson or retroactively change the printed evaluation.

A subsequent actual revision produces a new review. Assertions check that the
earlier cutoff reproduces the earlier review, the saved lesson remains unchanged,
and the journal's client ID can be reconstructed from its persisted decision ID.
The comparison has one matched origin: that demonstrates mechanics, not a reliable
ranking. Point MAE does not evaluate interval calibration or realised trading P&L.

For production, use the system clock, real timestamped data and the existing
configured ledger. With CLI/MCP, summary/actual/lesson writes need operator-enabled
`allow_outcome_writes`; direct Python ledger calls are explicit application-owned
writes. Preserve the same units and target endpoints for revised outcomes.

## Equivalent MCP decision recording

Over MCP the mode comes from the operator's configuration, not the call. Each mode's
server uses its own TOML and ledger, for example:

```toml
schema_version = 1
ledger_path = "paper.db"
allow_outcome_writes = true

[decision_context.trading_mode]
value = "paper"
source_ref = "operator-config"
```

For live, the operator sets `value = "live"` and names the user-approved promotion
record in `source_ref`. This is provenance metadata; the MCP server does not load
or verify that record or its performance criteria. The operator/execution layer
must check approval before enabling live orders. Confirm `gnomon_capabilities` shows the expected
`ledger.decision_context` before recording.

After `gnomon_forecast`, use its returned execution ID in a `gnomon_ledger` call.
This illustrates the same summary operation; substitute the actual intent,
assumptions and invalidation conditions. Empty context is valid when no sourced
label is available; the Python example shows the full timestamped regime label.

```json
{
  "name": "gnomon_ledger",
  "arguments": {
    "operation": "record_decision_summary",
    "execution_id": "EXECUTION_ID_FROM_FORECAST",
    "rationale": "policy=q10-cost-v1; intent=LONG because the returned q0.1 exceeds the log cost hurdle.",
    "assumptions": ["Flat long-only account; one-hour forecast and cost horizon."],
    "invalidation_conditions": ["Data stale or account not flat: cancel entry intent."],
    "context": []
  }
}
```

The response's `result.decision_id` links to the order journal. Recording the
summary does not submit an order. The server must have the same configured ledger
as the forecast and `allow_outcome_writes=true`. Use the host's discovered tool
name if it prefixes `gnomon_ledger`; inspect the operation schema before calling.

Ephemeris forecasts can be recorded and reviewed; for ranking them see the
[ledger skill](../../use-gnomon-ledger/SKILL.md#bounded-recovery). Strict comparisons
also need eligible training cutoffs for pretrained providers and matching request
features: a point-only baseline and a quantile-requested forecast are not
automatically a matched cohort. Design comparable requests or disclose the limitation.
