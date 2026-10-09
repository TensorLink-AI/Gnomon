"""Persistent agent state: positions, an equity ESTIMATE and drawdown halts.

Vanta computes the authoritative equity. The agent's estimate (mark-to-market on
hourly closes, minus modelled fees and funding) exists to halt well before Vanta's
elimination lines: 5% intraday from the day-open, 8% EOD from the highest EOD.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

from .policy import apply_order


@dataclass
class Book:
    equity: float
    day: str = ""
    day_open_equity: float = 0.0
    peak_eod_equity: float = 0.0
    marked_at: str = ""
    positions: dict = field(default_factory=dict)  # pair -> signed leverage
    marks: dict = field(default_factory=dict)  # pair -> last mark price
    halted_until: str = ""  # ISO time, "manual", or ""
    halt_reason: str = ""
    needs_reconcile: dict = field(default_factory=dict)  # pair -> pending order record
    last_actual: dict = field(default_factory=dict)  # series_id -> last appended valid_time
    eod_halt_override: float = 0.0  # set by an operator resume; 0 = use the configured halt

    @classmethod
    def load(cls, path, initial_equity):
        path = Path(path)
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls(equity=initial_equity, day_open_equity=initial_equity, peak_eod_equity=initial_equity)

    def save(self, path):
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        os.replace(tmp, path)

    def mark(self, prices, now, funding_bps_per_hour):
        """Mark positions to `prices`, charge funding for the elapsed time, roll the UTC day."""
        hours = 0.0
        if self.marked_at:
            hours = max(0.0, (now - datetime.fromisoformat(self.marked_at)).total_seconds() / 3600)
        pnl = sum(lev * (prices[p] / self.marks[p] - 1) for p, lev in self.positions.items()
                  if p in prices and p in self.marks)
        funding = sum(abs(lev) for lev in self.positions.values()) * funding_bps_per_hour / 1e4 * hours
        self.equity *= 1 + pnl - funding
        self.marks.update(prices)
        self.marked_at = now.isoformat()
        today = now.date().isoformat()
        if today != self.day:
            if self.day:  # yesterday's close is today's open
                self.peak_eod_equity = max(self.peak_eod_equity, self.equity)
            self.day, self.day_open_equity = today, self.equity

    def fill(self, order, cost_bps):
        current = self.positions.get(order.pair, 0.0)
        after = apply_order(current, order)
        self.equity -= self.equity * abs(after - current) * cost_bps / 1e4
        if after == 0:
            self.positions.pop(order.pair, None)
        else:
            self.positions[order.pair] = after

    @property
    def intraday_drawdown(self):
        return max(0.0, 1 - self.equity / self.day_open_equity) if self.day_open_equity else 0.0

    @property
    def eod_drawdown(self):
        return max(0.0, 1 - self.equity / self.peak_eod_equity) if self.peak_eod_equity else 0.0

    def check_halts(self, now, risk):
        """Set a halt when a drawdown line is crossed; clear an expired daily halt."""
        if self.halted_until and self.halted_until != "manual" and now >= datetime.fromisoformat(self.halted_until):
            self.halted_until, self.halt_reason = "", ""
        if self.eod_drawdown >= (self.eod_halt_override or risk.eod_drawdown_halt):
            self.halted_until = "manual"
            self.halt_reason = f"drawdown from peak EOD equity {self.eod_drawdown:.2%}; operator must resume"
        elif self.intraday_drawdown >= risk.intraday_drawdown_halt and not self.halted_until:
            midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
            self.halted_until = midnight.isoformat()
            self.halt_reason = f"intraday drawdown {self.intraday_drawdown:.2%}; flat until next UTC day"
        return bool(self.halted_until)
