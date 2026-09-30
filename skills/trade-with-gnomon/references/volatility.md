# Start with volatility

Short-horizon price direction is rarely forecastable: on hourly crypto returns no
StatsForecast model beat a zero forecast. Volatility is different: it clusters, so
recent realised volatility predicts the next period's. A volatility forecast is useful
without any directional edge:

- **Sizing:** hold `target_vol / forecast_vol` of full size, so risk per period stays
  roughly constant (`vol_target_position`).
- **Risk limits and stops:** set them in units of forecast volatility, not fixed %.
- **Probabilistic returns:** scale past standardised returns by the forecast to get
  next-period quantiles and P(move beyond the trading cost)
  (`return_quantiles`, `exceedance_probabilities`).
- **Abstention:** skip or shrink when the expected move is small relative to costs.

Add a directional view only after it beats a zero forecast after costs on held-out
data. `scripts/volatility.py` has the reusable pieces; the example below runs offline on
synthetic data with volatility clustering. Put that `scripts` directory on `sys.path`.

```python
import math
import random
from datetime import datetime, timedelta, timezone

from gnomon import GnomonSession
from volatility import (exceedance_probabilities, realised_volatility, register_volatility_models,
                        return_quantiles, vol_target_position)

# 30 days of synthetic 5-minute returns (bps) with volatility clustering (GARCH-like).
rng = random.Random(7)
returns, variance = [], 4.0
for _ in range(30 * 288):
    shock = rng.gauss(0, 1) * math.sqrt(variance)
    returns.append(shock)
    variance = 0.2 + 0.05 * shock * shock + 0.93 * variance
hourly_vol = realised_volatility(returns, 12)              # one value per hour, bps
hourly_ret = [sum(returns[i:i + 12]) for i in range(len(returns) % 12, len(returns), 12)]
start = datetime(2026, 1, 1, tzinfo=timezone.utc)
stamps = [(start + timedelta(hours=i + 1)).isoformat() for i in range(len(hourly_vol))]

session = GnomonSession.from_config()
models = register_volatility_models(session.engine)

def forecast(model, t):  # next-hour volatility from hours up to t, as a Gnomon execution
    request = {"history": hourly_vol[:t + 1], "timestamps": stamps[:t + 1], "horizon": 1,
               "future_timestamps": [(start + timedelta(hours=t + 2)).isoformat()],
               "series_id": "BTC/realised-vol-1h", "unit": "bps"}
    return session.forecast(model, request)

# Compare the models on the last 120 hours against the last-value baseline.
test = range(len(hourly_vol) - 121, len(hourly_vol) - 1)
errors = {m: sum(abs(forecast(m, t)["result"]["point"][0] - hourly_vol[t + 1]) for t in test) / len(test)
          for m in models}
best = min(errors, key=errors.get)
print({m: round(e / errors["vol/last"], 3) for m, e in errors.items()}, "best:", best)

# Standardised returns from hours already closed, then the decision for the next hour.
now = len(hourly_vol) - 1
z = [hourly_ret[t + 1] / forecast(best, t)["result"]["point"][0] for t in range(now - 168, now)]
reply = forecast(best, now)
sigma = reply["result"]["point"][0]
q = return_quantiles(sigma, z)
p_up, p_down = exceedance_probabilities(sigma, z, threshold=10.0)  # 10 bp round-trip cost
intent = {"target_position": round(vol_target_position(sigma, target_vol=6.0), 3),
          "forecast_vol_bps": round(sigma, 2), "q05_bps": round(q[0.05], 1), "q95_bps": round(q[0.95], 1),
          "p_move_above_cost": round(p_up, 3), "p_move_below_minus_cost": round(p_down, 3),
          "execution_id": reply["execution_id"], "provider": reply["provider"]}
print(intent)
```

Next steps:
- Record each sized intent with `record_trade_decision` against the volatility
  forecast's `execution_id` ([lifecycle](trade-lifecycle.md)).
- With several instruments, let a router choose among volatility models per decision,
  pooled across instruments with episodic memory
  ([route-with-gnomon](../../route-with-gnomon/SKILL.md)).
- Judge a volatility-sized strategy against the same exposure without sizing, and
  against flat, after costs.
