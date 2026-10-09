"""Walk-forward backtest with the live code path's signals, policy and Vanta costs.

At each hour t the forecasts see only bars closed by t; orders fill at t's close
(plus fee and slippage) and are marked at t+1's close (minus funding). This is
simulation evidence only, and it is compared against two references: flat, and a
volatility-sized always-long position (same sizing, no direction view).
Remote providers (Ephemeris) are billable per call, so they need an explicit budget.
"""
from datetime import datetime, timezone
import math

from . import policy
from .market_data import MarketData
from .signals import forecast_signal, open_session

LOCAL_PROVIDERS = {"rw", "local/momentum"}


class _Track:
    def __init__(self, equity):
        self.equity = self.day_open = self.peak_eod = self.last_mark = equity
        self.day, self.max_intraday, self.max_eod, self.turnover, self.trades, self.rets = None, 0.0, 0.0, 0.0, 0, []
        self.positions = {}

    def rebalance(self, pair, target, cost_bps, min_rebalance):
        current = self.positions.get(pair, 0.0)
        for order in policy.plan_orders(pair, current, target, min_rebalance=min_rebalance):
            after = policy.apply_order(current, order)
            traded = abs(after - current)
            self.equity -= self.equity * traded * cost_bps / 1e4
            self.turnover += traded
            self.trades += 1
            current = after
        self.positions[pair] = current

    def mark(self, moves, funding_bps, when):
        pnl = sum(self.positions.get(p, 0.0) * r for p, r in moves.items())
        self.equity *= 1 + pnl - sum(abs(v) for v in self.positions.values()) * funding_bps / 1e4
        self.rets.append(self.equity / self.last_mark - 1)  # net of this hour's trading costs
        self.last_mark = self.equity
        if when.date() != self.day:
            if self.day is not None:
                self.peak_eod = max(self.peak_eod, self.equity)
            self.day, self.day_open = when.date(), self.equity
        self.max_intraday = max(self.max_intraday, 1 - self.equity / self.day_open)
        self.max_eod = max(self.max_eod, 1 - self.equity / self.peak_eod)

    def summary(self, start_equity):
        n = len(self.rets)
        mean = sum(self.rets) / n if n else 0.0
        sd = math.sqrt(sum((r - mean) ** 2 for r in self.rets) / (n - 1)) if n > 1 else 0.0
        return dict(net_return=round(self.equity / start_equity - 1, 5),
                    sharpe_annualised=round(mean / sd * math.sqrt(24 * 365), 2) if sd else None,
                    max_intraday_drawdown=round(self.max_intraday, 4), max_eod_drawdown=round(self.max_eod, 4),
                    vanta_eliminated=self.max_intraday >= 0.05 or self.max_eod >= 0.08,
                    turnover_leverage=round(self.turnover, 3), orders=self.trades)


def run(config, *, test_hours, now=None, market=None, max_remote_calls=0):
    cfg = config
    f, r, c = cfg.forecast, cfg.risk, cfg.costs
    remote = {f.direction_provider, f.direction_baseline} - LOCAL_PROVIDERS
    calls_needed = test_hours * len(cfg.universe.pairs) * len(remote)
    if remote and calls_needed > max_remote_calls:
        raise ValueError(f"Backtest needs ~{calls_needed} billable calls to {sorted(remote)}; "
                         f"pass --max-remote-calls {calls_needed} to allow them")
    now = now or datetime.now(timezone.utc)
    market = market or MarketData(cfg)
    series = {p: market.series(p, hours=f.history_hours + test_hours + 1, now=now) for p in cfg.universe.pairs}
    length = min(len(s) for s in series.values())
    if length < 96 + test_hours:
        raise ValueError(f"Only {length} contiguous hours available; need >= {96 + test_hours}")
    if len({s.times[-1] for s in series.values()}) != 1:
        raise ValueError("Pairs end at different hours; align the data before backtesting")
    start = length - test_hours - 1
    cost_bps = c.fee_bps + c.slippage_bps
    strat, vol_long, flat = _Track(r.initial_equity), _Track(r.initial_equity), _Track(r.initial_equity)
    hits = calls = active = 0
    with open_session(cfg) as session:
        for k in range(start, length - 1):
            raw, sizes, moves, signals = {}, {}, {}, {}
            for p, s in series.items():
                offset = len(s) - length  # align pairs on the common tail
                signals[p] = sig = forecast_signal(session, p, s.upto(offset + k + 1), cfg)
                calls += len(remote)
                raw[p], _ = policy.target_leverage(sig, r, c)
                sizes[p] = policy.vol_size(sig.sigma_1h_bps, r)
                moves[p] = s.closes[offset + k + 1] / s.closes[offset + k] - 1
            for track, targets in ((strat, policy.apply_portfolio_cap(raw, r)),
                                   (vol_long, policy.apply_portfolio_cap(sizes, r)), (flat, {})):
                for p in series:
                    track.rebalance(p, targets.get(p, 0.0), cost_bps, r.min_rebalance_leverage)
            first = next(iter(series.values()))
            when = first.times[len(first) - length + k + 1]
            for track in (strat, vol_long, flat):
                track.mark(moves, c.funding_bps_per_hour, when)
            hits += sum(1 for p in signals if raw[p] and (raw[p] > 0) == (moves[p] > 0))
            active += sum(1 for p in signals if raw[p])
    return dict(
        mode="backtest", hours=test_hours, pairs=list(cfg.universe.pairs), policy=cfg.agent.policy_revision,
        direction_provider=f.direction_provider, vol_model=f.vol_model, remote_calls=calls,
        strategy=strat.summary(r.initial_equity), vol_sized_long=vol_long.summary(r.initial_equity),
        flat=flat.summary(r.initial_equity),
        directional_hit_rate=round(hits / active, 3) if active else None, directional_pair_hours=active,
        caveat="Simulation evidence only. Fills at hourly closes; equity is an estimate of Vanta's ledger.")
