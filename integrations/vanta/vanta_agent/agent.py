"""One decision cycle: data -> forecasts -> ledger decision -> Vanta orders -> outcomes.

Ordering is what makes it recoverable:
1. reconcile any order whose outcome was unknown last cycle (live);
2. append realised actuals so earlier forecasts can be scored;
3. forecast, size and cap every pair;
4. per pair, claim the event's journal and record the decision in the Gnomon
   ledger BEFORE any order (an existing journal means the event was decided);
5. write each order as `pending` with its deterministic uuid, then submit.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

from gnomon import TemporalLedger

from . import policy
from .market_data import MarketData
from .signals import forecast_signal, open_session, requests_for
from .state import Book
from .vanta_client import Order, VantaClient, VantaError, VantaUncertain, order_uuid

_SCRIPTS = Path(__file__).resolve().parents[3] / "skills" / "trade-with-gnomon" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from trade_decisions import record_trade_decision  # noqa: E402

RECONCILE_GRACE = timedelta(minutes=10)


def utcnow():
    return datetime.now(timezone.utc)


class Agent:
    def __init__(self, config, *, market=None, client=None, clock=utcnow, sleep=time.sleep):
        if config.agent.mode == "backtest":
            raise ValueError("Use `python -m vanta_agent backtest` for backtest mode")
        self.config, self.clock, self.sleep = config, clock, sleep
        self.market = market or MarketData(config)
        self.live = config.agent.mode == "live"
        self.client = client or (VantaClient.from_config(config) if self.live else None)
        self.state_dir = config.path(config.agent.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.book_path = self.state_dir / f"book-{config.agent.mode}.json"
        self.ledger_path = self.state_dir / f"{config.agent.mode}.db"

    # -- outcomes -----------------------------------------------------------------
    def _append_actuals(self, ledger, book, pair, series):
        """Append realised closes and realised vol after the first decision on each series."""
        for series_id, unit, values in ((f"vanta/{pair}/close-1h", "USDC", series.closes),
                                        (f"vanta/{pair}/rv-1h", "bps", series.rv_bps)):
            since = book.last_actual.get(series_id)
            if since is None:
                continue
            for t, value in zip(series.times, values):
                if t.isoformat() > since:
                    ledger.append_actual(series_id=series_id, unit=unit, valid_time=t.isoformat(),
                                         value=float(value), source_available_at=t.isoformat(),
                                         source_ref=f"{self.config.market_data.source}:5m-candles")
                    book.last_actual[series_id] = t.isoformat()

    def _shadow_forecasts(self, session, pair, series):
        """Record shadow models' forecasts for `guide`; they never affect trading."""
        f = self.config.forecast
        vol_req, price_req = requests_for(pair, series, self.config)
        for provider, request in ([(m, vol_req) for m in f.shadow_vol_models] +
                                  [(m, price_req) for m in f.shadow_direction_providers]):
            try:
                session.forecast(provider, request)
            except Exception:  # a failing shadow is just missing evidence
                pass

    # -- execution ----------------------------------------------------------------
    def _reconcile(self, book, now):
        for pair, pending in list(book.needs_reconcile.items()):
            status = self.client.status(pending["uuid"])  # VantaUncertain propagates: stay blocked
            if status == "completed":
                book.fill(Order(pair, pending["order_type"], pending["leverage"]), self._cost_bps)
            elif status == "not_found" and now - datetime.fromisoformat(pending["since"]) < RECONCILE_GRACE:
                continue  # possibly still being relayed to validators
            self._journal_update(pending["journal"], pending["uuid"], status)
            del book.needs_reconcile[pair]

    @property
    def _cost_bps(self):
        return self.config.costs.fee_bps + self.config.costs.slippage_bps

    @staticmethod
    def _journal_update(path, uuid_, status, response=None):
        path = Path(path)
        entries = json.loads(path.read_text())
        for entry in entries:
            if entry["uuid"] == uuid_:
                entry["status"] = status
                if response is not None:
                    entry["response"] = response
        path.write_text(json.dumps(entries, indent=2))

    def _execute(self, book, pair, orders, client_id, journal_path, now):
        entries = [dict(uuid=order_uuid(client_id, i), order_type=o.order_type, leverage=o.leverage,
                        status="pending") for i, o in enumerate(orders)]
        journal_path.write_text(json.dumps(entries, indent=2))
        results = []
        for i, (order, entry) in enumerate(zip(orders, entries)):
            if not self.live:
                book.fill(order, self._cost_bps)
                self._journal_update(journal_path, entry["uuid"], "paper_filled")
                results.append("paper_filled")
                continue
            if i:
                self.sleep(self.config.vanta.same_pair_cooldown_seconds)
            try:
                response = self.client.submit(order, entry["uuid"])
            except VantaUncertain:
                book.needs_reconcile[pair] = dict(entry, journal=str(journal_path), since=now.isoformat())
                results.append("uncertain")
                break
            except VantaError as error:
                self._journal_update(journal_path, entry["uuid"], "rejected", str(error))
                results.append("rejected")
                break
            book.fill(order, self._cost_bps)
            self._journal_update(journal_path, entry["uuid"], "completed", response)
            results.append("completed")
        return results

    # -- the cycle ----------------------------------------------------------------
    def cycle(self):
        cfg, now = self.config, self.clock()
        book = Book.load(self.book_path, cfg.risk.initial_equity)
        report = dict(mode=cfg.agent.mode, time=now.isoformat(), pairs={})
        if self.live:
            if not self.client.health():
                raise RuntimeError(f"Vanta miner REST server unreachable at {cfg.vanta.url}")
            self._reconcile(book, now)

        ledger = TemporalLedger(self.ledger_path)
        paper_ledger = TemporalLedger(cfg.path(cfg.agent.paper_ledger), create=False) if self.live else None
        with open_session(cfg, ledger=ledger) as session:
            history = cfg.forecast.history_hours + cfg.forecast.horizon_hours + 2
            series = {p: self.market.series(p, hours=history, now=now) for p in cfg.universe.pairs}
            fresh = {p: s for p, s in series.items() if len(s) >= 96 and
                     now - s.times[-1] <= timedelta(minutes=cfg.market_data.stale_after_minutes)}
            book.mark({p: s.closes[-1] for p, s in fresh.items()}, now, cfg.costs.funding_bps_per_hour)
            halted = book.check_halts(now, cfg.risk)
            for p, s in series.items():
                self._append_actuals(ledger, book, p, s)

            signals, raw, why = {}, {}, {}
            for pair in cfg.universe.pairs:
                if pair not in fresh:
                    report["pairs"][pair] = dict(skipped="stale or insufficient data: no new exposure")
                    continue
                try:
                    signals[pair] = forecast_signal(session, pair, fresh[pair], cfg)
                except Exception as error:  # e.g. Ephemeris unavailable: hold, no new exposure
                    report["pairs"][pair] = dict(skipped=f"forecast failed: {error}")
                    continue
                raw[pair], why[pair] = policy.target_leverage(signals[pair], cfg.risk, cfg.costs)
                self._shadow_forecasts(session, pair, fresh[pair])
            targets = policy.apply_portfolio_cap(raw, cfg.risk)

            for pair, signal in signals.items():
                current = book.positions.get(pair, 0.0)
                target = 0.0 if halted else targets[pair]
                if pair in book.needs_reconcile:
                    report["pairs"][pair] = dict(skipped="order outcome unknown; reconciling", current=current)
                    continue
                orders = policy.plan_orders(pair, current, target, min_rebalance=cfg.risk.min_rebalance_leverage)
                day_dir = self.state_dir / "journal" / cfg.agent.mode / signal.time[:10]
                day_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{pair}-{signal.time[11:13]}h"
                try:
                    intent = record_trade_decision(
                        ledger, day_dir / f"{stem}.intent.json", mode=cfg.agent.mode,
                        forecast=signal.direction, account=cfg.agent.account, leg="rebalance",
                        event_id=f"{cfg.agent.account}:{pair}:{signal.time}",
                        policy_revision=cfg.revision,
                        action=dict(current_leverage=current, target_leverage=target,
                                    orders=[o.payload("") for o in orders]),
                        rationale=json.dumps(dict(policy=cfg.revision, halted=book.halt_reason or None,
                                                  sigma_1h_bps=round(signal.sigma_1h_bps, 2), **why[pair])),
                        assumptions=[f"Price {signal.price} is the last closed hourly bar at {signal.time}.",
                                     f"Costs: {cfg.costs.fee_bps} bps fee + {cfg.costs.slippage_bps} bps slippage "
                                     f"per order, {cfg.costs.funding_bps_per_hour} bps/h funding."],
                        invalidation_conditions=[
                            "Hourly data stale beyond the configured limit: no new exposure.",
                            "Estimated drawdown crosses a halt line: flatten (risk policy, no forecast needed)."],
                        evidence_refs=[{"kind": "execution", "id": r["execution_id"]} for r in
                                       (signal.vol, signal.vol_baseline, signal.direction_baseline)],
                        promotion_record=(str(cfg.path(cfg.agent.promotion_record)) if self.live else None),
                        paper_ledger=paper_ledger,
                    )
                except FileExistsError:
                    report["pairs"][pair] = dict(skipped="event already decided (journal exists)")
                    continue
                for sid in (f"vanta/{pair}/close-1h", f"vanta/{pair}/rv-1h"):
                    book.last_actual.setdefault(sid, signal.time)
                results = self._execute(book, pair, orders, intent["client_order_id"],
                                        day_dir / f"{stem}.orders.json", now) if orders else []
                report["pairs"][pair] = dict(
                    price=signal.price, sigma_1h_bps=round(signal.sigma_1h_bps, 2),
                    median_ret_bps=round(signal.median_ret_bps, 2), current=current, target=target,
                    orders=[f"{o.order_type} {o.leverage or ''}".strip() for o in orders], results=results,
                    decision_id=intent["decision_id"], execution_id=signal.direction["execution_id"],
                    **{k: v for k, v in why[pair].items() if k == "reason"})
        book.save(self.book_path)
        report.update(equity_estimate=round(book.equity, 2), positions=book.positions,
                      intraday_drawdown=round(book.intraday_drawdown, 4), eod_drawdown=round(book.eod_drawdown, 4),
                      halted=book.halt_reason or None, needs_reconcile=sorted(book.needs_reconcile))
        return report

    def run_forever(self, *, delay_seconds=45, log=print):
        """Run one cycle shortly after each hourly close."""
        while True:
            try:
                log(json.dumps(self.cycle()))
            except Exception as error:  # keep the loop alive; the next cycle reconciles
                log(json.dumps(dict(error=repr(error), time=self.clock().isoformat())))
            now = self.clock()
            nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, seconds=delay_seconds)
            self.sleep(max(1.0, (nxt - now).total_seconds()))

    def flatten_all(self):
        """Kill switch: FLAT every open position (live: sent to Vanta)."""
        now, book = self.clock(), Book.load(self.book_path, self.config.risk.initial_equity)
        out = {}
        for pair in list(book.positions):
            client_id = f"flatten:{self.config.agent.account}:{pair}:{now.isoformat()}"
            journal = self.state_dir / "journal" / f"flatten-{pair}-{int(now.timestamp())}.orders.json"
            journal.parent.mkdir(parents=True, exist_ok=True)
            out[pair] = self._execute(book, pair, [Order(pair, "FLAT", None)], client_id, journal, now)
        book.save(self.book_path)
        return out


def resume(config, *, eod_halt):
    """Clear a manual (EOD drawdown) halt. Operator action.

    The peak stays: Vanta measures 8% from the highest-ever EOD equity, so resuming
    means accepting a new, explicit halt line between today's drawdown and 8%.
    """
    path = config.path(config.agent.state_dir) / f"book-{config.agent.mode}.json"
    book = Book.load(path, config.risk.initial_equity)
    if not book.eod_drawdown < eod_halt < 0.08:
        raise ValueError(f"eod_halt must lie between the current drawdown {book.eod_drawdown:.2%} and 8%")
    book.halted_until, book.halt_reason, book.eod_halt_override = "", "", eod_halt
    book.save(path)
    return dict(resumed=True, equity_estimate=book.equity, eod_drawdown=book.eod_drawdown, eod_halt=eod_halt)

