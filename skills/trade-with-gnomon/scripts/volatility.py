"""Reusable volatility helpers for trading with Gnomon. No trading policy.

Short-horizon direction is rarely forecastable, but volatility clusters and is. These
helpers turn fine-grained returns into realised volatility, register volatility
forecasters as Gnomon providers (so forecasts carry execution IDs and can be recorded,
compared and routed), and convert a volatility forecast into position sizes and return
quantiles. Validate every choice on the user's instruments before relying on it.
"""
import math

from gnomon import AdapterCapabilities, ForecastResult


def realised_volatility(returns, per):
    """sqrt(sum of squared returns) over consecutive blocks of `per`, aligned to the end.

    For 5-minute returns and per=12, each value is one hour's realised volatility in
    the returns' unit. A leading partial block is dropped.
    """
    start = len(returns) % per
    return [math.sqrt(sum(r * r for r in returns[i:i + per])) for i in range(start, len(returns), per)]


def _result(request, value):
    return ForecastResult(point=(max(float(value), 0.0),) * request.horizon, timestamps=request.future_timestamps,
                          series_id=request.series_id, unit=request.unit)


def last_value_volatility(request):
    """Baseline: next period's volatility equals the last period's."""
    return _result(request, request.history[-1])


def ewma_volatility(lam=0.94, window=168):
    """RiskMetrics-style exponentially weighted volatility over the latest `window` values."""
    def forecast(request):
        values = list(request.history[-window:])
        variance = values[0] ** 2
        for v in values[1:]:
            variance = lam * variance + (1 - lam) * v * v
        return _result(request, math.sqrt(variance))
    return forecast


def har_volatility(components=(1, 6, 24), window=336):
    """HAR-RV: regress next volatility on means over `components` lags (least squares)."""
    longest = max(components)

    def features(values, t):
        return [1.0] + [sum(values[t - c + 1:t + 1]) / c for c in components]

    def solve(rows, targets):
        n = len(rows[0])  # normal equations with a small ridge for stability
        a = [[sum(r[i] * r[j] for r in rows) + (1e-9 if i == j else 0.0) for j in range(n)] for i in range(n)]
        b = [sum(r[i] * y for r, y in zip(rows, targets)) for i in range(n)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda k: abs(a[k][col]))
            a[col], a[pivot], b[col], b[pivot] = a[pivot], a[col], b[pivot], b[col]
            for k in range(col + 1, n):
                f = a[k][col] / a[col][col]
                a[k] = [x - f * y for x, y in zip(a[k], a[col])]
                b[k] -= f * b[col]
        beta = [0.0] * n
        for i in reversed(range(n)):
            beta[i] = (b[i] - sum(a[i][j] * beta[j] for j in range(i + 1, n))) / a[i][i]
        return beta

    def forecast(request):
        values = [float(v) for v in request.history[-(window + longest + 1):]]
        rows = [features(values, t) for t in range(longest - 1, len(values) - 1)]
        beta = solve(rows, values[longest:])
        return _result(request, sum(b * x for b, x in zip(beta, features(values, len(values) - 1))))
    return forecast


def register_volatility_models(engine, *, min_history=72):
    """Register `vol/last` (baseline), `vol/ewma` and `vol/har` on a Gnomon engine."""
    capabilities = AdapterCapabilities(min_history=min_history)
    engine.register("vol/last", last_value_volatility, capabilities=capabilities, revision="vol/last-v1")
    engine.register("vol/ewma", ewma_volatility(), capabilities=capabilities, revision="vol/ewma-0.94-v1")
    engine.register("vol/har", har_volatility(), capabilities=capabilities, revision="vol/har-1-6-24-v1")
    return ["vol/last", "vol/ewma", "vol/har"]


def vol_target_position(forecast_vol, target_vol, cap=1.0):
    """Position size that aims for `target_vol` of risk per period, capped at `cap`."""
    return min(cap, target_vol / forecast_vol) if forecast_vol > 0 else 0.0


def return_quantiles(sigma, standardised, probs=(0.05, 0.25, 0.5, 0.75, 0.95), min_history=48):
    """Return quantiles for the next period: past standardised returns (return / its
    volatility forecast, from periods already closed) scaled by `sigma`; normal until
    `min_history` exist."""
    z_normal = {0.05: -1.6449, 0.25: -0.6745, 0.5: 0.0, 0.75: 0.6745, 0.95: 1.6449}
    if len(standardised) < min_history:
        return {p: z_normal.get(p, 0.0) * sigma for p in probs}
    zs = sorted(standardised)
    return {p: zs[min(len(zs) - 1, int(p * len(zs)))] * sigma for p in probs}


def exceedance_probabilities(sigma, standardised, threshold, min_history=48):
    """P(return > threshold) and P(return < -threshold) from the same distribution."""
    if len(standardised) < min_history or sigma <= 0:
        cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
        return 1 - cdf(threshold / sigma) if sigma > 0 else 0.0, cdf(-threshold / sigma) if sigma > 0 else 0.0
    n = len(standardised)
    return (sum(1 for z in standardised if z * sigma > threshold) / n,
            sum(1 for z in standardised if z * sigma < -threshold) / n)
