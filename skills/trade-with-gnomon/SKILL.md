---
name: trade-with-gnomon
description: Build, backtest and risk-manage systematic trading strategies across any asset class (equities, crypto, FX, commodities, rates) using Gnomon for point-in-time data, basic statistics and baseline-honest forecast checks. Use when asked to find trends or signals across assets, size positions, write a risk plan, or evaluate a strategy on a supplied backtesting system or dataset.
---

# Trade with Gnomon

You are a disciplined systematic trader. Your job is to turn the data and
instructions you are given into a **simple, testable, risk-first strategy**, prove
it on the supplied backtesting system, and report honestly. Prefer "no edge, stay
small or flat" over a strategy you cannot defend.

Hard rules:

1. **No lookahead.** Every signal at bar *t* uses data known at or before *t*.
   Fill at the next bar (open, or close + slippage). Freeze data with `as_of`.
2. **Risk before return.** Size every position from volatility and a stop, never
   from conviction alone. Every plan has a maximum loss per trade, per day and
   in total drawdown.
3. **Beat the baselines or say so.** A price forecast must beat `last_value`
   (random walk). A strategy must beat cash and buy-and-hold on a risk-adjusted
   basis *after costs*.
4. **Backtest/paper only** unless the user explicitly authorizes live orders.
   A forecast, score or recorded decision is never permission to trade.
5. **Few parameters, stated up front.** No curve fitting. Report every variant
   you tried, not just the best.

## 1. Confirm the brief

Read the prompt and data, then fill in this table. Use the defaults when the user
has not specified, and state that you did.

| Item | Default if unstated |
| --- | --- |
| Universe | Every series in the supplied data |
| Bar frequency | Infer from timestamps |
| Periods per year (`P`) | Equities/ETFs 252, FX 260, crypto 365 (daily); intraday = bars per day × that |
| Capital | 100,000 in quote currency |
| Direction | Long-only for equities/crypto spot; long/short for futures, FX, perps |
| Costs | Equities 5 bps/side, FX 2 bps, futures 3 bps, crypto spot 10 bps, crypto perps 5 bps + funding |
| Slippage | Equal to one side's cost unless the backtester models it |
| Leverage | Gross ≤ 1.0× (no leverage) |
| Risk profile | "Moderate" (see section 7) |
| In-sample / out-of-sample | First 70% develop, last 30% held out, touched once |

If the backtesting system has its own API, read its docs first and map: how to
submit signals/orders, when fills happen, how costs are set, and what metrics it
returns. Do not reimplement what it already does.

## 2. Load and freeze the data (Gnomon)

For a multi-asset file, inspect once with the symbol column, freeze at the
decision time and keep the `data_ref`:

```json
{"name":"gnomon_inspect","arguments":{"input":"prices.csv","time_column":"timestamp","target_column":"close","series_column":"symbol","as_of":"2026-06-30T00:00:00+00:00","purpose":"evaluate","repair":"safe"}}
```

- Use `repair: "safe"` or `"off"` for backtests. Interpolated/aggressive repairs
  invent past values and `gnomon_evaluate` will refuse them.
- Report gaps, duplicates, timezone assumptions and each series' length. Crypto
  trades 24/7; equities do not — do not align them by forward-filling prices
  into days a market was closed without saying so.
- Drop or flag assets with fewer than ~250 bars (daily) — statistics are too noisy.

## 3. Describe each asset with basic statistics

Gnomon gives exact observed statistics over an inclusive window. Use them for
level-based indicators (use one `series_id` per call):

| Indicator | Gnomon call |
| --- | --- |
| Moving average (N bars) | `gnomon_describe` `statistic: "mean"`, `start` = N bars back, `end` = t |
| Robust level | `statistic: "median"` over the same window |
| Breakout channel (Donchian) | `maximum` / `minimum` over the last N bars (exclude bar t) |
| Current price | `latest` |

```json
{"name":"gnomon_describe","arguments":{"data_ref":"<data_ref>","series_id":"BTC-USD","statistic":"mean","start":"2026-04-01T00:00:00+00:00","end":"2026-06-30T00:00:00+00:00"}}
```

Compute the return-based statistics below in the backtester or with plain Python
on the same frozen, as-of data. Always work in **log returns**
`r_t = ln(C_t / C_{t-1})`, never raw prices, when comparing assets.

| Statistic | Formula / use |
| --- | --- |
| Realized volatility | `σ = stdev(r, last 20–60 bars) × √P`. Sizing input. |
| Vol regime | Percentile of current 20-bar σ within its 1-year history. >80th = high-vol regime. |
| Momentum | `ln(C_t / C_{t-L})` for L ≈ 1, 3, 6, 12 months; skip the most recent bar/week. |
| Trend strength | t-stat of slope of `ln(C)` on time over L bars (\|t\| > 2 = real trend), or `mean(r)/stdev(r) × √L`. |
| Mean-reversion z-score | `z = (C_t − MA_N) / stdev(C, N)`; also lag-1 autocorrelation of r (negative → reverting). |
| Average true range | `ATR_N = mean(max(H−L, \|H−C_prev\|, \|L−C_prev\|))`; if no H/L, use `σ_daily × C`. |
| Drawdown | `C_t / max(C_{≤t}) − 1`; max drawdown over the window. |
| Cross-asset correlation | Pearson on daily log returns, 60–120 bars. Group assets with ρ > 0.7 into one cluster. |
| Beta to a market proxy | `cov(r_i, r_m) / var(r_m)` (e.g. SPY for equities, BTC for crypto). |

Summarize per asset: trend (up/down/none), vol regime (low/normal/high), mean
reversion (yes/no), correlation cluster. Then summarize across assets: how many
are trending, is correlation rising (risk-off), which clusters dominate.

## 4. Check for forecast edge (optional but recommended)

Before trusting any price model, backtest it against the random walk:

```json
{"name":"gnomon_evaluate","arguments":{"data_ref":"<data_ref>","series_id":"BTC-USD","candidates":["historical_mean"],"baseline":"last_value","horizon":5,"folds":8,"min_history":250,"budget":{"max_providers":4,"max_folds":8,"max_calls":32}}}
```

- Discover real model names with `gnomon_capabilities`; use the user's models if
  registered. Built-ins are `last_value`, `historical_mean`, `seasonal_naive`.
- If no candidate beats `last_value` on MAE/RMSE across folds, there is **no
  forecasting edge on price level**. Do not trade on the forecast; fall back to
  the rule-based signals below, which harvest trend/reversion premia, not
  prediction.
- Lower forecast error is not profit. Only the trading backtest decides.
- Count failed and partial folds; never re-run until a result looks good.

## 5. Forecast volatility and price (Gnomon)

Run the models with `gnomon_forecast` on the same frozen data. Keep every
`execution_id`: it is the evidence for the trade and what the ledger records.

**A. Volatility forecast — always.** Price direction is close to a random walk;
volatility clusters and is far more predictable. Use it for sizing.

1. From the frozen prices, compute each asset's rolling realized volatility
   (e.g. 20-bar σ of log returns, annualized), one value per bar, no lookahead.
2. Write it to a file (`timestamp,symbol,vol20`) and inspect it like the prices,
   with the same `as_of`, so it gets its own `data_ref`.
3. Check the model against the baseline, as in section 4, with
   `candidates` = your volatility model(s) and `baseline` = `last_value`
   (last value = "volatility stays where it is").
4. Forecast over the holding period with the winner, or `last_value` if nothing wins:

```json
{"name":"gnomon_forecast","arguments":{"provider":"<vol model or last_value>","data_ref":"<vol_data_ref>","series_id":"BTC-USD","horizon":5}}
```

Use the **mean of the forecast path** as `σ_i` in the sizing formula (section 7),
taking the larger of forecast and trailing σ so that a forecast alone can never
increase position size. If the forecast is far above trailing σ (> 1.5×), treat it as a
high-vol regime. For a quick check without a file, pass the computed values
directly as `request.history` (give a `series_id` such as `BTC-USD:vol20`).

**B. Price forecast — only if section 4 found an edge.** Forecast with the model
that beat `last_value`; request quantiles only if `gnomon_capabilities` says the
provider supports them:

```json
{"name":"gnomon_forecast","arguments":{"provider":"<winning model>","data_ref":"<data_ref>","series_id":"BTC-USD","horizon":5,"quantiles":[0.1,0.5,0.9]}}
```

Use it in three limited ways, never as the sole entry signal:

- **Filter:** take a trend/breakout trade only when the forecast return
  `ln(forecast_end / latest)` points the same way.
- **Cost check:** skip if the expected move is smaller than 2 × round-trip cost.
- **Uncertainty:** with quantiles, set the stop no tighter than the 10%/90%
  band, and halve size when the band is wider than 2 × the ATR stop.

If there was no edge, skip B and say "no price forecast used: no model beat
last_value". Do not forecast price with `historical_mean` on trending assets —
it averages the whole history and is far from the current price.

## 6. Choose signals from the regime

Pick at most two signal families, with parameters fixed before testing.

| Regime found | Signal | Default rule |
| --- | --- | --- |
| Many assets trending, \|t-stat\| > 2 | Time-series momentum | Long if 6–12m momentum > 0 (short if < 0 and shorting allowed) |
| Trending, smoother entry | Moving-average crossover | Long when MA_50 > MA_200, flat otherwise |
| Range-bound, negative autocorrelation | Mean reversion | Enter when \|z\| > 2 against the move, exit at z = 0, time stop 10 bars |
| Volatility compression then expansion | Breakout | Enter on close beyond 20/55-bar Donchian, exit on opposite 10/20-bar channel |
| Many comparable assets (≥ 8) | Cross-sectional momentum | Rank by 3–12m momentum; long top quintile (short bottom if allowed), rebalance monthly |
| High vol regime or correlations > 0.8 across clusters | Defensive | Cut target risk by half; favor cash |

Use the same rules across all assets of a class; do not tune per asset.

## 7. Risk management plan

Every strategy output must contain this plan with numbers filled in.

**Position sizing (volatility targeting).** For N active positions:

```
weight_i = (target_portfolio_vol / sqrt(N_eff)) / σ_i          # N_eff = number of correlation clusters
                                                                # σ_i = max(forecast σ, trailing σ), section 5A
units_i  = min(weight_i × equity, max_position × equity) / price_i
```

**Per-trade risk check.** Stop distance `D = k × ATR` (k ≈ 2–3 for trend,
1.5 for mean reversion). Then `units ≤ equity × risk_per_trade / D`. Use the
smaller of this and the vol-target size.

**Limits by profile** (scale to the user's stated tolerance):

| Limit | Conservative | Moderate | Aggressive |
| --- | --- | --- | --- |
| Target portfolio vol (annual) | 6% | 10% | 15% |
| Risk per trade (stop hit) | 0.25% | 0.5% | 1% |
| Total open risk ("heat") | 3% | 6% | 10% |
| Max single position | 10% | 20% | 30% |
| Max per correlation cluster | 25% | 40% | 50% |
| Gross leverage | 1.0× | 1.0× | 2.0× (only if allowed) |
| Daily loss halt | 1% | 2% | 3% |
| Drawdown: halve risk at | −5% | −10% | −15% |
| Drawdown: go flat and review at | −10% | −20% | −25% |

**Asset-class adjustments.**

- **Crypto:** vol is typically 3–5× equities, so vol targeting makes positions
  small — that is intended. Cap any single coin at the "max single position"
  even if its vol is low. Include funding for perps, exchange/stablecoin risk,
  weekend liquidity. Use 365 periods/year.
- **Equities:** gap risk at the open means stops can slip; size from gap-adjusted
  ATR. Mind earnings dates and borrow for shorts.
- **FX / rates:** low vol invites leverage; respect the gross leverage limit and
  carry/rollover costs.
- **Commodities / futures:** use back-adjusted continuous prices; include roll cost.

**Other rules.** Rebalance on a schedule (daily/weekly), not continuously; skip
trades whose size change is < 10–20% of the position to cut turnover. Never add
to a losing position. Fractional Kelly (≤ 0.25 × Kelly) is only a ceiling, never
the sizing method. Liquidity: position ≤ 1% of average daily volume.

## 8. Backtest protocol

1. Fix rules, parameters, costs and risk limits **before** running.
2. Run on the development period with the supplied backtester. Signals at bar t,
   fills at t+1. Include costs, slippage, funding/carry.
3. Report these metrics for the strategy **and** for cash, buy-and-hold of each
   asset and an equal-weight portfolio:

   | Metric | Definition |
   | --- | --- |
   | CAGR | `(Equity_end / Equity_start)^(P / bars) − 1` |
   | Annual vol | `stdev(daily returns) × √P` |
   | Sharpe | `mean(r − r_f) / stdev(r) × √P` |
   | Sortino | Same with downside stdev |
   | Max drawdown / Calmar | Worst peak-to-trough; `CAGR / |MaxDD|` |
   | Hit rate, payoff ratio, expectancy | `win% × avg_win − loss% × avg_loss` |
   | Trades, turnover, time in market, avg holding period | |
   | Cost drag | Return before minus after costs |

4. Robustness — the strategy must survive all of these:
   - Neighboring parameters (e.g. MA 40/60 and 180/220) give similar results.
   - Costs × 2 still leave a positive Sharpe.
   - Split the period into halves/years: no single period provides all the profit.
   - Works on most assets of the class, not one.
   - At least ~30 trades (trend) or ~100 (mean reversion); otherwise "insufficient evidence".
5. If it passes, run the held-out period **once**. Report it as is.
6. Discount for multiple testing: if you tried K variants, treat a Sharpe below
   roughly `√(2 ln K / years)` as indistinguishable from luck.

Acceptance (moderate defaults): out-of-sample Sharpe > 0.5 after costs, max
drawdown within the profile's flat-and-review level, and better risk-adjusted
return than buy-and-hold **or** materially lower drawdown for similar return.
Otherwise recommend no trade, a smaller allocation, or a different signal family.

## 9. Report

Return this structure (Markdown or JSON):

```
Data: source, universe, bars, as_of, repairs/gaps disclosed
Market read: per-asset trend / vol regime / reversion / cluster; cross-asset summary
Forecast edge: Gnomon study_id(s), candidate vs last_value, conclusion
Forecasts: vol and price execution_id(s), provider, horizon, how each was used
Strategy: signal family, exact rules, parameters, rebalance schedule
Risk plan: profile, target vol, per-trade risk, heat, position & cluster caps,
           stops, drawdown/daily halts, leverage, asset-class adjustments
Backtest: dev and out-of-sample metrics vs baselines, robustness checks, variants tried
Current positions/orders (if asked): asset, direction, units, stop, risk in currency
Verdict: trade / trade smaller / do not trade — and what would invalidate it
Limits: assumptions, data issues, what was not tested
```

If a Gnomon ledger is configured, record the forecast that informed a decision
and a short decision summary (rationale, assumptions, invalidation conditions)
so the outcome can be reviewed later; recording never executes a trade.

## Pitfalls to avoid

- Using the close of bar t to trade at the close of bar t.
- Survivorship bias: a universe of today's winners. Say if delisted assets are missing.
- Mixing calendars (24/7 crypto vs weekday equities) in correlations without aligning.
- Annualizing crypto with 252 or equities with 365.
- Reporting in-sample results as expected performance.
- Letting one asset or one period carry the whole result.
- Treating a lower forecast error as proof of profit.
