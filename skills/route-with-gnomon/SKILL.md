---
name: route-with-gnomon
description: Choose among your forecasting models per forecast with a Gnomon router that learns from recorded outcomes, including episodic memory that retrieves similar past situations across related series; test it by replay before trusting it. Use when several models or several related series are available and the task is which model to trust when; for a single forecast use use-gnomon or forecast-with-gnomon.
---

# Route with Gnomon

A router serves one of *your* models per forecast and learns from outcomes recorded in
the ledger. With `memory` it asks "when a series looked like this before, which model
did best?", across the series in its `pool`. Measured so far: plain rolling selection
did not beat choosing one good model; memory with a pool did (Favorita retail, 96
series, 0.794 vs 0.812 relative MAE). On unpredictable targets (5-minute crypto
returns) no model beat zero, so no router can help. Replay on your data first.

## 1. Decide whether routing fits

- Worth trying: several candidate models, and several related series with the same
  unit and horizon (stores, products, instruments), or one long series whose regime
  changes.
- Not worth it: one model, one short series, or no outcomes arriving to learn from.
- Every routed forecast also runs the other models (shadows) so evidence accrues:
  cost and latency multiply by the number of models unless `shadow_every` samples.

## 2. Test by replay before deploying

Build folds with a Gnomon evaluation study per series, then replay policies with
`gnomon.adaptive_router.replay_router` on one shared clock. The runnable example is in
[replay on your data](references/replay.md). Compare the router with the **best single
model**, not just the baseline, on origins after a warm-up, and prefer the simplest
policy that wins. Replay needs many folds: raise `[evaluation_limits]` (default 8).

## 3. Configure

In Python you can build a router yourself; over MCP routers are operator
configuration, so ask the operator (they appear under `gnomon_capabilities` →
`ledger.routers`).

```python
from gnomon import GnomonSession

series = ["store-1", "store-2", "store-3"]
policy = {
    "candidates": ["last_value", "historical_mean"], "baseline": "seasonal_naive",
    "metric": "mae", "recent_origins": 20, "min_origins": 5, "min_improvement": 0.02,
    "lookback_seconds": 90 * 86400,           # must exceed the horizon; outcomes must mature
    "pool": {"series": series, "own_weight": 2.0},
    "memory": {"features": ["volatility_ratio", "trend", "level_shift", "cv"],
               "short_window": 7, "long_window": 56, "k": 32, "min_effective_n": 8, "own_weight": 2.0},
}
base = GnomonSession.from_config(ledger_path="routing.db")
session = GnomonSession(base.engine, ledger=base.ledger, routers={"router/stores": policy})
```

Forecast with `session.forecast("router/stores", request)`; requests need `series_id`,
`timestamps` and `future_timestamps`. Append actuals as they arrive (`append_actual`).

Settings that matter:
- `lookback_seconds` longer than the horizon plus enough origins to score; the
  default 7 days never yields evidence at horizons of 7 days or more.
- `short_window`/`long_window` count observations: 56 is 8 weeks of daily data but
  under 5 hours of 5-minute bars.
- `k` a small fraction of episodes in the window; `min_effective_n` guards against a
  few near-duplicate neighbours. Start with the dynamics features above; add
  `covariate_share` (with `future_covariate`) when a known-future driver matters.
- Declare costs (`costs`, `cost_weights`, `limits`) when models differ in price or
  latency. Formulas and scoring: [how memory scores](references/scoring.md).

## 4. Read and report each routed forecast

`reply["routing"]` gives `served_provider`, `reason`, `evidence_level` (`memory`,
`context` or `all`), `effective_n`, the top `memory_neighbours` and the utility
`table`. Report the served model and reason. Treat a low `effective_n`, neighbours
from unrelated series, or `insufficient_evidence` as weak evidence: say so, and in
decisions reduce size or confidence. A baseline fallback is not proof the baseline is
best. Record `routing_decision_id` with any decision that used the forecast.

## If something fails

- Always `insufficient_evidence`: lookback shorter than horizon + scoring origins,
  actuals not appended, or `shadow_every = 0`.
- Never `memory`: too few matured episodes, features null (history shorter than
  `long_window`, flat series), or memory settings changed (older episodes ignored).
- `evidence_incompatible`: request shape or a provider revision changed in the window.
- `evidence_unavailable`: read `evidence_error`; the baseline is served meanwhile.
- Unknown provider or router name: use names from capabilities; never substitute.

Report the policy (or its `router_revision`), replay results against the best single
model, what was served and why, and what is not yet established.
