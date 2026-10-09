"""Price data for Vanta crypto pairs.

Vanta's crypto pairs (BTCUSDC, ETHUSDC, ...) are priced from Hyperliquid USDC perps,
so Hyperliquid's public candle endpoint is the matching source. Only CLOSED candles
are returned: a candle is usable once its close time is at or before `now`.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import json
import math
from pathlib import Path
import urllib.request

FIVE_MIN_MS = 5 * 60 * 1000
HOUR = timedelta(hours=1)


@dataclass(frozen=True)
class Candle:
    open_time: datetime  # UTC
    close: float

    @property
    def close_time(self) -> datetime:
        return self.open_time + timedelta(minutes=5)


def coin_for_pair(pair: str) -> str:
    """Vanta pair id -> Hyperliquid coin (BTCUSDC -> BTC, kPEPEUSDC -> kPEPE)."""
    if not pair.endswith("USDC"):
        raise ValueError(f"{pair}: only Hyperliquid-sourced USDC pairs are supported")
    return pair[: -len("USDC")]


def fetch_hyperliquid(pair, *, hours, now, url, timeout=20.0, opener=urllib.request.urlopen):
    end = int(now.timestamp() * 1000)
    body = json.dumps({"type": "candleSnapshot", "req": {
        "coin": coin_for_pair(pair), "interval": "5m",
        "startTime": end - (hours + 1) * 3600 * 1000, "endTime": end}}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with opener(request, timeout=timeout) as response:
        rows = json.loads(response.read())
    candles = [Candle(datetime.fromtimestamp(int(r["t"]) / 1000, timezone.utc), float(r["c"])) for r in rows]
    return [c for c in sorted(candles, key=lambda c: c.open_time) if c.close_time <= now]


def load_csv(path, *, now=None):
    """CSV with `open_time` (ISO-8601 with timezone, or epoch ms) and `close` columns, 5m bars."""
    candles = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row["open_time"]
            t = (datetime.fromtimestamp(int(raw) / 1000, timezone.utc) if raw.isdigit()
                 else datetime.fromisoformat(raw).astimezone(timezone.utc))
            candles.append(Candle(t, float(row["close"])))
    candles.sort(key=lambda c: c.open_time)
    return [c for c in candles if now is None or c.close_time <= now]


@dataclass(frozen=True)
class HourlySeries:
    """Hourly series built from 5m candles, stamped at each hour's CLOSE time."""
    times: tuple[datetime, ...]
    closes: tuple[float, ...]
    rv_bps: tuple[float, ...]  # realised vol of the hour's 5m log returns, bps

    def __len__(self):
        return len(self.times)

    def upto(self, n):
        return HourlySeries(self.times[:n], self.closes[:n], self.rv_bps[:n])


def hourly(candles, *, min_bars=10):
    """Aggregate complete hours. An hour with fewer than `min_bars` 5m bars is a gap:
    the series restarts after it, so no forecast ever spans missing data."""
    buckets = {}
    for c in candles:
        hour = c.open_time.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(hour, []).append(c)
    last_close = max((c.close_time for c in candles), default=None)
    times, closes, rvs, prev_close = [], [], [], None
    for hour in sorted(buckets):
        if hour + HOUR > last_close:  # the hour is still in progress
            break
        bars = buckets[hour]
        if len(bars) < min_bars or (times and hour + HOUR - times[-1] != HOUR):
            times, closes, rvs, prev_close = [], [], [], None
            if len(bars) < min_bars:
                continue
        prices = ([prev_close] if prev_close else []) + [b.close for b in bars]
        rets = [math.log(b / a) for a, b in zip(prices, prices[1:])]
        times.append(hour + HOUR)
        closes.append(bars[-1].close)
        rvs.append(1e4 * math.sqrt(sum(r * r for r in rets)))
        prev_close = bars[-1].close
    return HourlySeries(tuple(times), tuple(closes), tuple(rvs))


class MarketData:
    def __init__(self, config, opener=urllib.request.urlopen):
        self.config, self.opener = config, opener

    def candles(self, pair, *, hours, now):
        md = self.config.market_data
        if md.source == "csv":
            return load_csv(Path(self.config.path(md.csv_dir)) / f"{pair}.csv", now=now)
        return fetch_hyperliquid(pair, hours=hours, now=now, url=md.url,
                                 timeout=md.timeout_seconds, opener=self.opener)

    def series(self, pair, *, hours, now):
        return hourly(self.candles(pair, hours=hours, now=now))
