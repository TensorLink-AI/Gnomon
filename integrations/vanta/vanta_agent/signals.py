"""Forecasts through Gnomon: volatility for sizing, price quantiles for direction.

Every number the policy uses comes from a Gnomon execution (with an execution_id
recorded in the ledger when one is attached), alongside a baseline execution, so
the ledger can later compare the model against the baseline on realised outcomes.
"""
from dataclasses import dataclass
from datetime import timedelta
import math
from pathlib import Path
import statistics
import sys

from gnomon import AdapterCapabilities, ForecastResult, GnomonSession

_SCRIPTS = Path(__file__).resolve().parents[3] / "skills" / "trade-with-gnomon" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from volatility import register_volatility_models  # noqa: E402  (Gnomon's own helpers)

HOUR = timedelta(hours=1)


def _empirical_band(history, horizon, probs):
    """Quantiles of past `horizon`-hour log returns, centred on a zero median."""
    rets = [math.log(b / a) for a, b in zip(history, history[horizon:])]
    if len(rets) < 20:
        return {p: 0.0 for p in probs}
    percentiles = statistics.quantiles(rets, n=100, method="inclusive")  # 1st..99th
    centre = statistics.median(rets)
    return {p: percentiles[min(98, max(0, round(p * 100) - 1))] - centre for p in probs}


def _price_result(request, drift_per_step):
    last = request.history[-1]
    rows, points = [], []
    for step in range(1, request.horizon + 1):
        band = _empirical_band(request.history, step, request.quantiles or (0.5,))
        mid = drift_per_step * step
        points.append(last * math.exp(mid))
        rows.append({p: last * math.exp(mid + d) for p, d in band.items()})
    return ForecastResult(point=tuple(points), quantiles=tuple(rows) if request.quantiles else None,
                          timestamps=request.future_timestamps, series_id=request.series_id, unit=request.unit)


def random_walk(request):
    """Baseline: zero expected return, empirical band of past returns."""
    return _price_result(request, 0.0)


def momentum(lookback=24):
    """Strawman candidate: extrapolate the mean hourly log return of the last `lookback` hours."""
    def forecast(request):
        h = request.history[-(lookback + 1):]
        return _price_result(request, math.log(h[-1] / h[0]) / (len(h) - 1))
    return forecast


def open_session(config, *, ledger=None):
    path = config.forecast.providers_config
    session = GnomonSession.from_config(config.path(path) if path else None, ledger=ledger)
    register_volatility_models(session.engine)
    cap = AdapterCapabilities(quantiles=True, min_history=48)
    session.engine.register("rw", random_walk, capabilities=cap, revision="rw-empirical-band-v1")
    session.engine.register("local/momentum", momentum(), capabilities=cap, revision="momentum-24h-v1")
    f = config.forecast
    missing = {f.direction_provider, f.direction_baseline, f.vol_model, f.vol_baseline,
               *f.shadow_vol_models, *f.shadow_direction_providers} - set(session.capabilities()["providers"])
    if missing:
        session.close()
        raise RuntimeError(f"Providers not available in this session: {sorted(missing)}. "
                           "Check forecast.providers_config (e.g. the Ephemeris provider name).")
    return session


def _quantile(row, p):
    for key, value in (row or {}).items():
        if abs(float(key) - p) < 1e-9:
            return float(value)
    return None


@dataclass(frozen=True)
class Signal:
    pair: str
    time: str
    price: float
    horizon: int
    sigma_1h_bps: float
    median_ret_bps: float
    q_lo_ret_bps: float | None
    q_hi_ret_bps: float | None
    vol: dict
    vol_baseline: dict
    direction: dict
    direction_baseline: dict


def requests_for(pair, series, config):
    """Build the volatility and price requests at the series' last closed hour."""
    f = config.forecast
    n = min(len(series), f.history_hours)
    times = [t.isoformat() for t in series.times[-n:]]
    t = series.times[-1]
    vol = dict(history=list(series.rv_bps[-n:]), horizon=1, timestamps=times, cutoff=times[-1],
               future_timestamps=[(t + HOUR).isoformat()], frequency="h",
               series_id=f"vanta/{pair}/rv-1h", unit="bps")
    price = dict(history=list(series.closes[-n:]), horizon=f.horizon_hours, timestamps=times,
                 cutoff=times[-1], frequency="h", quantiles=list(f.quantiles),
                 future_timestamps=[(t + HOUR * (i + 1)).isoformat() for i in range(f.horizon_hours)],
                 series_id=f"vanta/{pair}/close-1h", unit="USDC")
    return vol, price


def _ok(reply, what):
    if reply.get("status") != "ok":
        raise RuntimeError(f"{what} forecast failed: {reply.get('error') or reply}")
    return reply


def forecast_signal(session, pair, series, config) -> Signal:
    f = config.forecast
    vol_req, price_req = requests_for(pair, series, config)
    vol = _ok(session.forecast(f.vol_model, vol_req), f.vol_model)
    vol_base = _ok(session.forecast(f.vol_baseline, vol_req), f.vol_baseline)
    direction = _ok(session.forecast(f.direction_provider, price_req), f.direction_provider)
    dir_base = (direction if f.direction_baseline == f.direction_provider else
                _ok(session.forecast(f.direction_baseline, price_req), f.direction_baseline))
    last = series.closes[-1]
    result = direction["result"]
    row = (result.get("quantiles") or [None])[-1]
    median = _quantile(row, 0.5) or result["point"][-1]
    to_bps = lambda price: None if price is None else 1e4 * math.log(price / last)
    sigma = max(float(vol["result"]["point"][0]), 1e-6)
    return Signal(pair=pair, time=series.times[-1].isoformat(), price=last, horizon=f.horizon_hours,
                  sigma_1h_bps=sigma, median_ret_bps=to_bps(median),
                  q_lo_ret_bps=to_bps(_quantile(row, min(f.quantiles))),
                  q_hi_ret_bps=to_bps(_quantile(row, max(f.quantiles))),
                  vol=vol, vol_baseline=vol_base, direction=direction, direction_baseline=dir_base)
