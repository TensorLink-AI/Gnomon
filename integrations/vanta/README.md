# Vanta trading agent

An hourly trading agent for the [Vanta Network](https://github.com/taoshidev/vanta-network)
(Bittensor subnet 8). It forecasts with Gnomon and the Ephemeris TSFM panel, records
every decision in a Gnomon ledger before acting, and sends orders to a Vanta miner.

It follows the [trade-with-gnomon](../../skills/trade-with-gnomon/SKILL.md) workflow
and reuses that skill's `volatility.py` and `trade_decisions.py` directly.

## What one cycle does

Each cycle runs shortly after the hourly close, once per pair:

1. **Data.** Fetch 5-minute candles from Hyperliquid's public API. Vanta prices its crypto
   pairs (`BTCUSDC`, `ETHUSDC`, ...) from Hyperliquid USDC perps. The candles are
   aggregated into closed hours: a close price and a realised volatility for each hour.
   A gap restarts the series, and stale data blocks new exposure.
2. **Volatility (sizing).** Gnomon forecasts next-hour realised vol with `vol/har` (or
   `vol/ewma`), alongside the `vol/last` baseline. The size is
   `target_vol / forecast_vol`, capped.
3. **Direction.** The configured provider forecasts the price `horizon_hours` ahead with
   quantiles, alongside a random-walk baseline. Set the provider to `ephemeris` for the
   TSFM panel. The median return must clear the round-trip cost hurdle (fees, slippage
   and funding). Strength scales with the median's z-score.
4. **Risk.** Targets are scaled to fit under a gross-leverage cap. The agent halts and goes
   flat at 3% intraday drawdown (until the next UTC day) or at 5% drawdown from peak EOD
   equity (until an operator resumes). Vanta itself eliminates a miner at 5% intraday or 8% EOD.
5. **Record.** `record_trade_decision` claims an exclusive journal file for the event
   (`account:pair:hour`). It then writes the decision to the Gnomon ledger, bound to the
   direction forecast's `execution_id` and to the other three forecasts. If the journal
   already exists, the event was already decided, so the agent never decides it twice.
6. **Execute.** Orders are planned under Vanta's rules:
   - Positions are uni-directional, so a flip is `FLAT` and then a new position.
   - An order against a position reduces it.
   - The minimum order is 0.001 leverage.
   - Orders on the same pair are spaced 6 s apart (Vanta enforces a 5 s cooldown).

   Each order gets a deterministic `order_uuid` and is journaled as `pending` before it
   is submitted.
7. **Outcomes.** Each cycle appends realised closes and realised vol as ledger actuals.
   Gnomon can then score every model against its baseline (`review_decision`,
   `compare_history`).

If a submission times out, the outcome is unknown. The agent blocks that pair, and the
next cycle asks `/api/order-status/<uuid>` whether that exact order was processed. It
never resends the order under a new id.

## Setup

```bash
# Same Python environment as Gnomon (3.11+). No other dependencies.
pip install gnomon-forecast            # or `pip install .` from the Gnomon checkout
cd integrations/vanta
cp agent.example.toml agent.toml       # edit pairs, risk, mode
```

**Ephemeris (optional).** Copy `providers.example.toml` to `providers.toml` and export
`EPHEMERIS_BASE_URL` and `EPHEMERIS_API_TOKEN`. Then set
`forecast.providers_config = "providers.toml"` and `forecast.direction_provider = "ephemeris"`.
Each pair costs one billable forecast per hour.

**Vanta miner (live only).** Run `neurons/miner.py` from vanta-network. It serves the REST
API on `:8088`. Put a key in `vanta_api/api_keys.json` and export it as `VANTA_API_KEY`.
Before going live, check these Vanta requirements:
- The miner is registered.
- It has deposited collateral (300 to 1000 Theta).
- It has selected the crypto asset class.

## Run

```bash
python -m vanta_agent backtest --config agent.toml --hours 168   # local models: free
python -m vanta_agent backtest --config agent.toml --hours 72 --max-remote-calls 216  # with Ephemeris
python -m vanta_agent run --config agent.toml --once              # one paper cycle
python -m vanta_agent run --config agent.toml                     # hourly loop
python -m vanta_agent status --config agent.toml
python -m vanta_agent flatten --config agent.toml [--yes]         # kill switch
python -m vanta_agent resume --config agent.toml --eod-halt 0.06  # after an EOD halt
```

Run the loop under a supervisor such as pm2 or systemd, alongside the miner. Each cycle
prints one JSON line.

**Backtest data.** Hyperliquid serves only the most recent 5000 candles, which is about
17 days at 5 minutes. A 168-hour history plus a 240-hour test therefore fits in one
fetch. For longer backtests, set `market_data.source = "csv"` and supply
`<PAIR>.csv` files with `open_time,close` columns at 5-minute bars.

## Backtest → paper → live

The modes follow the trade-with-gnomon progression, with one ledger per mode
(`state/<mode>.db`):

1. **Backtest** with local models, then with Ephemeris on a budget. The report compares
   the strategy with `flat` and with `vol_sized_long` (same sizing, no direction view),
   net of costs. It also says whether the run would have been eliminated by Vanta. If
   direction does not beat `vol_sized_long` after costs, there is no edge yet.
   `"rw"` stays a valid direction provider in that case.
2. **Paper** with the hourly loop. Agree on a minimum number of complete decisions and on
   the promotion criteria before paper trading starts.
3. **Live** requires a promotion record. The user writes and approves it from
   `promotion_review` over the paper ledger, and `record_trade_decision` re-verifies it on
   every live decision. Live mode refuses to start without it.

## Limits

- The equity figure is the agent's own estimate. It marks positions at hourly closes and
  subtracts modelled costs. Vanta's validators hold the authoritative ledger, so the
  drawdown halts sit well inside Vanta's lines to leave room for that difference.
- A stale pair keeps its existing position. The agent cannot price a position on stale
  data, so the pair gets no new exposure until fresh data arrives.
- Vanta eliminates a miner after 60 days without an order. A strategy that never clears
  its cost hurdle will therefore be eliminated. That is still a truer answer than trading
  without an edge.
- The agent supports only crypto pairs priced from Hyperliquid. Forex and equities need
  another data source and Vanta's market-hours rules.
