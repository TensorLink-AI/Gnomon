"""Policy: forecasts -> target leverage -> Vanta orders. Pure functions, no I/O.

Sizing comes from the volatility forecast (hold target_vol / forecast_vol);
direction comes from the price forecast's median H-hour return, which must clear
the round-trip cost hurdle. Strength scales with the median's z-score.
"""
import math

from .vanta_client import Order

MIN_ORDER_LEVERAGE = 0.001  # Vanta's minimum order leverage


def cost_hurdle_bps(costs, horizon_hours):
    """Round trip (enter + exit) fees and slippage, plus funding for the holding horizon."""
    return 2 * (costs.fee_bps + costs.slippage_bps) + costs.funding_bps_per_hour * horizon_hours


def vol_size(sigma_1h_bps, risk):
    return min(risk.max_position_leverage, risk.target_vol_bps_per_hour / sigma_1h_bps) if sigma_1h_bps > 0 else 0.0


def target_leverage(signal, risk, costs):
    """Signed target leverage for one pair, with the reasoning behind it."""
    size = vol_size(signal.sigma_1h_bps, risk)
    hurdle = cost_hurdle_bps(costs, signal.horizon)
    edge = signal.median_ret_bps
    z = edge / (signal.sigma_1h_bps * math.sqrt(signal.horizon))
    why = dict(vol_size=round(size, 4), hurdle_bps=round(hurdle, 2), median_ret_bps=round(edge, 2), z=round(z, 3))
    if abs(edge) <= hurdle:
        return 0.0, {**why, "reason": "median return inside cost hurdle"}
    if edge < 0 and not risk.allow_short:
        return 0.0, {**why, "reason": "short signal, shorting disabled"}
    strength = min(1.0, abs(z) / risk.z_full)
    return math.copysign(size * strength, edge), {**why, "reason": "edge clears hurdle", "strength": round(strength, 3)}


def apply_portfolio_cap(targets, risk):
    """Scale all targets down together so gross leverage stays under the cap."""
    gross = sum(abs(v) for v in targets.values())
    scale = min(1.0, risk.max_portfolio_leverage / gross) if gross > 0 else 1.0
    return {pair: _round(v * scale) for pair, v in targets.items()}


def _round(leverage):
    # Floor to Vanta's 0.001 grid (never round exposure up), after snapping float noise.
    value = math.copysign(math.floor(round(abs(leverage) * 1000, 6)) / 1000, leverage)
    return 0.0 if abs(value) < MIN_ORDER_LEVERAGE else value


def plan_orders(pair, current, target, *, min_rebalance):
    """Orders moving `current` signed leverage to `target` under Vanta's rules.

    Positions are uni-directional: an order against a position reduces it, and one
    larger than the position closes it. A flip is therefore FLAT then a new position.
    """
    target = _round(target)
    if target == current:
        return []
    if target == 0:
        return [Order(pair, "FLAT", None)]
    side = "LONG" if target > 0 else "SHORT"
    if current == 0:
        return [Order(pair, side, abs(target))]
    if (current > 0) != (target > 0):
        return [Order(pair, "FLAT", None), Order(pair, side, abs(target))]
    delta = _round(target - current)
    if abs(delta) < max(min_rebalance, MIN_ORDER_LEVERAGE):
        return []
    return [Order(pair, "LONG" if delta > 0 else "SHORT", abs(delta))]


def apply_order(current, order):
    """Signed leverage after an order fills."""
    if order.order_type == "FLAT":
        return 0.0
    signed = order.leverage if order.order_type == "LONG" else -order.leverage
    after = round(current + signed, 6)
    if current != 0 and (after == 0 or (after > 0) != (current > 0)):
        return 0.0  # an opposing order larger than the position closes it
    return after
