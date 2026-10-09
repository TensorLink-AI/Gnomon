"""Walk-forward simulation with the live code path's signals, policy and Vanta costs.

Two stages, so that iterating on a strategy is cheap:
1. `forecast_panel`: at each hour t, forecasts see only bars closed by t. This is the
   expensive (and for Ephemeris, billable) part, and it goes through a persistent
   forecast cache: the same request to the same provider revision is never repeated.
2. `simulate`: the policy over the panel. Orders fill at t's close (plus fee and
   slippage) and are marked at t+1's close (minus funding). Changing only [risk]
   parameters re-runs this stage alone.
Simulation evidence only; the references are flat and a volatility-sized
always-long position (same sizing, no direction view).
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import sqlite3

from . import policy
from .market_data import MarketData
from .signals import _quantile, forecast_signal, open_session

LOCAL_PROVIDERS = {"rw", "local/momentum"}


class RemoteBudgetExceeded(RuntimeError):
    pass


class ForecastCache:
    """Wraps a Gnomon session; persists replies keyed by provider revision + request."""

    def __init__(self, session, path=None, *, max_remote_calls=None):
        self.session, self.remote_calls, self.max_remote_calls = session, 0, max_remote_calls
        self.revisions = {name: info.get("revision") for name, info in session.capabilities()["providers"].items()}
        self.db = sqlite3.connect(str(path) if path else ":memory:")
        self.db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, reply TEXT NOT NULL)")

    def forecast(self, provider, request):
        revision = self.revisions.get(provider) or "unattested"
        key = sha256(json.dumps([provider, revision, request], sort_keys=True).encode()).hexdigest()
        row = self.db.execute("SELECT reply FROM replies WHERE key=?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        if provider not in LOCAL_PROVIDERS and not provider.startswith("vol/"):
            if self.max_remote_calls is not None and self.remote_calls >= self.max_remote_calls:
                raise RemoteBudgetExceeded(f"remote forecast budget of {self.max_remote_calls} calls spent "
                                           "(results so far are cached; rerun with a larger budget)")
            self.remote_calls += 1
        reply = self.session.forecast(provider, request)
        if reply.get("status") != "ok":
            return reply
        slim = json.loads(json.dumps({k: reply.get(k) for k in ("status", "result", "provider", "revision",
                                                                "execution_id")}, default=str))
        self.db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?)", (key, json.dumps(slim)))
        self.db.commit()
        return slim


@dataclass(frozen=True)
class Point:
    """One pair at one decision hour: what the policy sees, and what happened next."""
    time: str
    price: float
    next_price: float
    horizon: int
    sigma_1h_bps: float
    sigma_baseline_bps: float
    median_ret_bps: float
    baseline_median_ret_bps: float
    q_lo_ret_bps: float | None
    q_hi_ret_bps: float | None
    rv_next_bps: float
    realised_ret_h_bps: float | None  # t -> t+horizon; None near the end of the data


def _median_bps(reply, last):
    result = reply["result"]
    row = (result.get("quantiles") or [None])[-1]
    return 1e4 * math.log((_quantile(row, 0.5) or result["point"][-1]) / last)


def aligned_length(series):
    if len({s.times[-1] for s in series.values()}) != 1:
        raise ValueError("Pairs end at different hours; align the data first")
    return min(len(s) for s in series.values())


def forecast_panel(cache, config, series, decision_indices):
    """{pair: [Point]} at decision hours, given as indices into the pairs' common tail."""
    length, h, panel = aligned_length(series), config.forecast.horizon_hours, {}
    for pair, s in series.items():
        offset, points = len(s) - length, []
        for k in decision_indices:
            i = offset + k
            sig = forecast_signal(cache, pair, s.upto(i + 1), config)
            realised = 1e4 * math.log(s.closes[i + h] / s.closes[i]) if i + h < len(s) else None
            points.append(Point(time=sig.time, price=sig.price, next_price=s.closes[i + 1], horizon=h,
                                sigma_1h_bps=sig.sigma_1h_bps,
                                sigma_baseline_bps=float(sig.vol_baseline["result"]["point"][0]),
                                median_ret_bps=sig.median_ret_bps,
                                baseline_median_ret_bps=_median_bps(sig.direction_baseline, sig.price),
                                q_lo_ret_bps=sig.q_lo_ret_bps, q_hi_ret_bps=sig.q_hi_ret_bps,
                                rv_next_bps=s.rv_bps[i + 1], realised_ret_h_bps=realised))
        panel[pair] = points
    return panel


class Track:
    def __init__(self, equity):
        self.equity = self.day_open = self.peak_eod = self.last_mark = self.start = equity
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

    def summary(self):
        n = len(self.rets)
        mean = sum(self.rets) / n if n else 0.0
        sd = math.sqrt(sum((r - mean) ** 2 for r in self.rets) / (n - 1)) if n > 1 else 0.0
        return dict(net_return=round(self.equity / self.start - 1, 5),
                    sharpe_annualised=round(mean / sd * math.sqrt(24 * 365), 2) if sd else None,
                    max_intraday_drawdown=round(self.max_intraday, 4), max_eod_drawdown=round(self.max_eod, 4),
                    vanta_eliminated=self.max_intraday >= 0.05 or self.max_eod >= 0.08,
                    turnover_leverage=round(self.turnover, 3), orders=self.trades)


def simulate(panel, config, *, reference=None):
    """Run the policy over a panel; returns (summary, hourly net returns).

    `reference`: None (the strategy), "vol_long" or "flat". The agent's drawdown
    halts apply as in the live loop: flat for the rest of the UTC day after an
    intraday halt, flat to the end after an EOD halt.
    """
    r, c = config.risk, config.costs
    cost_bps = c.fee_bps + c.slippage_bps
    track, hits, active, halted_day, halted_all = Track(r.initial_equity), 0, 0, None, False
    pairs = list(panel)
    for k in range(len(panel[pairs[0]])):
        points = {p: panel[p][k] for p in pairs}
        when = datetime.fromisoformat(points[pairs[0]].time)
        if reference == "flat":
            raw = {p: 0.0 for p in pairs}
        elif reference == "vol_long":
            raw = {p: policy.vol_size(pt.sigma_1h_bps, r) for p, pt in points.items()}
        else:
            raw = {p: policy.target_leverage(pt, r, c)[0] for p, pt in points.items()}
        halted = halted_all or halted_day == when.date()
        targets = {p: 0.0 for p in pairs} if halted else policy.apply_portfolio_cap(raw, r)
        for p in pairs:
            track.rebalance(p, targets[p], cost_bps, r.min_rebalance_leverage)
        moves = {p: pt.next_price / pt.price - 1 for p, pt in points.items()}
        track.mark(moves, c.funding_bps_per_hour, when)
        if 1 - track.equity / track.peak_eod >= r.eod_drawdown_halt:
            halted_all = True
        elif 1 - track.equity / track.day_open >= r.intraday_drawdown_halt:
            halted_day = when.date()
        hits += sum(1 for p in pairs if targets[p] and (targets[p] > 0) == (moves[p] > 0))
        active += sum(1 for p in pairs if targets[p])
    out = track.summary()
    out.update(directional_hit_rate=round(hits / active, 3) if active else None, pair_hours_exposed=active)
    return out, track.rets


def forecast_quality(panel):
    """Forecast accuracy against each baseline on the same hours (ratio < 1 beats the baseline)."""
    vol_m = vol_b = dir_m = dir_b = covered = signs = 0.0
    n_vol = n_dir = n_band = n_sign = 0
    for points in panel.values():
        for pt in points:
            n_vol += 1
            vol_m += abs(pt.sigma_1h_bps - pt.rv_next_bps)
            vol_b += abs(pt.sigma_baseline_bps - pt.rv_next_bps)
            if pt.realised_ret_h_bps is None:
                continue
            n_dir += 1
            dir_m += abs(pt.median_ret_bps - pt.realised_ret_h_bps)
            dir_b += abs(pt.baseline_median_ret_bps - pt.realised_ret_h_bps)
            if pt.q_lo_ret_bps is not None and pt.q_hi_ret_bps is not None:
                n_band += 1
                covered += pt.q_lo_ret_bps <= pt.realised_ret_h_bps <= pt.q_hi_ret_bps
            if pt.median_ret_bps:
                n_sign += 1
                signs += (pt.median_ret_bps > 0) == (pt.realised_ret_h_bps > 0)
    ratio = lambda a, b: round(a / b, 4) if b else None
    return dict(vol_mae_vs_baseline=ratio(vol_m, vol_b), vol_mae_bps=round(vol_m / n_vol, 2) if n_vol else None,
                direction_mae_vs_baseline=ratio(dir_m, dir_b),
                direction_sign_accuracy=round(signs / n_sign, 3) if n_sign else None,
                band_coverage=round(covered / n_band, 3) if n_band else None, scored_hours=n_dir)


def run(config, *, test_hours, now=None, market=None, max_remote_calls=0, cache_path=None):
    """Backtest the last `test_hours` of the configured market data (CLI `backtest`)."""
    f = config.forecast
    remote = {f.direction_provider, f.direction_baseline} - LOCAL_PROVIDERS
    calls_needed = test_hours * len(config.universe.pairs) * len(remote)
    if remote and calls_needed > max_remote_calls:
        raise ValueError(f"Backtest needs ~{calls_needed} billable calls to {sorted(remote)}; "
                         f"pass --max-remote-calls {calls_needed} to allow them")
    now = now or datetime.now(timezone.utc)
    market = market or MarketData(config)
    series = {p: market.series(p, hours=f.history_hours + test_hours + 1, now=now) for p in config.universe.pairs}
    length = aligned_length(series)
    if length < 96 + test_hours:
        raise ValueError(f"Only {length} contiguous hours available; need >= {96 + test_hours}")
    with open_session(config) as session:
        cache = ForecastCache(session, cache_path, max_remote_calls=max_remote_calls)
        panel = forecast_panel(cache, config, series, range(length - test_hours - 1, length - 1))
    return dict(mode="backtest", hours=test_hours, pairs=list(config.universe.pairs), policy=config.revision,
                direction_provider=f.direction_provider, vol_model=f.vol_model, remote_calls=cache.remote_calls,
                strategy=simulate(panel, config)[0], vol_sized_long=simulate(panel, config, reference="vol_long")[0],
                flat=simulate(panel, config, reference="flat")[0], forecast_quality=forecast_quality(panel),
                caveat="Simulation evidence only. Fills at hourly closes; equity is an estimate of Vanta's ledger.")
