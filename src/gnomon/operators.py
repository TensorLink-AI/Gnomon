"""The deterministic operator substrate.

Every operator here: is pure and deterministic (no randomness, no wall
clock), reads data only from values already materialised through a
Snapshot, declares the claim classes its output can support, and owns
honest abstention criteria — an operator that cannot say "inconclusive"
does not ship. The registry in ``registry.py`` carries each operator's
schemas and policies; this module carries only the mathematics.

The causal family (granger, interrupted_time_series, synthetic_control)
is deliberately absent: nothing here can support a causal claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

from .contracts import SupportAssessment, SupportReason


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sse(values: list[float]) -> float:
    if not values:
        return 0.0
    centre = _mean(values)
    return sum((value - centre) ** 2 for value in values)


def _mad(values: list[float]) -> float:
    centre = median(values)
    return median([abs(value - centre) for value in values])


def _robust_scale(values: list[float]) -> float:
    """1.4826·MAD, falling back to a small epsilon for constant series."""
    scale = 1.4826 * _mad(values)
    return scale if scale > 1e-12 else 1e-12


def inconclusive(reason_code: str, message: str, recovery: str) -> SupportAssessment:
    return SupportAssessment(
        "inconclusive",
        [SupportReason(reason_code, message)],
        recovery_actions=[SupportReason("recovery", recovery)],
    )


# ---------------------------------------------------------------------------
# select_window / resample / difference
# ---------------------------------------------------------------------------

def select_window(
    timestamps: list[Any], values: list[float],
    *, start: Any | None = None, end: Any | None = None,
) -> dict[str, Any]:
    pairs = [
        (moment, value) for moment, value in zip(timestamps, values)
        if (start is None or moment >= start) and (end is None or moment <= end)
    ]
    return {
        "timestamps": [moment for moment, _ in pairs],
        "values": [value for _, value in pairs],
        "dropped": len(values) - len(pairs),
    }


def resample_aggregate(
    timestamps: list[Any], values: list[float],
    *, factor: int, statistic: str = "mean",
) -> dict[str, Any]:
    if factor < 1:
        raise ValueError("factor must be >= 1")
    reducers = {"mean": _mean, "sum": sum, "min": min, "max": max, "last": lambda block: block[-1]}
    if statistic not in reducers:
        raise ValueError(f"unknown statistic {statistic!r}")
    out_timestamps, out_values = [], []
    for index in range(0, len(values) - factor + 1, factor):
        block = values[index : index + factor]
        out_timestamps.append(timestamps[index + factor - 1])
        out_values.append(reducers[statistic](block))
    return {"timestamps": out_timestamps, "values": out_values,
            "factor": factor, "statistic": statistic,
            "truncated_tail": len(values) % factor}


def difference(values: list[float], *, lag: int = 1) -> list[float]:
    if lag < 1 or lag >= len(values):
        raise ValueError("lag must be >= 1 and shorter than the series")
    return [values[index] - values[index - lag] for index in range(lag, len(values))]


# ---------------------------------------------------------------------------
# changepoint_detection — binary segmentation on mean shift
# ---------------------------------------------------------------------------

MIN_SEGMENT = 4
MIN_CHANGEPOINT_HISTORY = 12


@dataclass
class Changepoint:
    index: int
    timestamp: Any
    before_mean: float
    after_mean: float
    shift: float
    relative_gain: float  # cost reduction relative to segment cost


def _best_split(values: list[float], offset: int) -> tuple[int, float] | None:
    """Best single split of a segment: index (absolute) and cost reduction.

    Prefix sums make each candidate split O(1); the previous
    recompute-from-scratch scan was O(n²) per segment, which priced a
    23,594-row daily series out of any interactive budget (~5.6e8 inner
    operations — the 2026-08 MCP evaluation's DGS10 investigate was
    killed at 300 s). Values are centred on the segment mean first: SSE
    is shift-invariant, and the sum-of-squares identity is numerically
    fragile when |mean| dwarfs the spread.
    """
    n = len(values)
    if n < 2 * MIN_SEGMENT:
        return None
    centre = _mean(values)
    prefix = [0.0] * (n + 1)
    prefix_squares = [0.0] * (n + 1)
    for index, value in enumerate(values):
        shifted = value - centre
        prefix[index + 1] = prefix[index] + shifted
        prefix_squares[index + 1] = prefix_squares[index] + shifted * shifted

    def segment_sse(lo: int, hi: int) -> float:
        count = hi - lo
        if count <= 0:
            return 0.0
        segment_sum = prefix[hi] - prefix[lo]
        return max(0.0, (prefix_squares[hi] - prefix_squares[lo])
                   - segment_sum * segment_sum / count)

    total = segment_sse(0, n)
    best_index, best_gain = None, 0.0
    for split in range(MIN_SEGMENT, n - MIN_SEGMENT + 1):
        gain = total - segment_sse(0, split) - segment_sse(split, n)
        if gain > best_gain:
            best_index, best_gain = split, gain
    if best_index is None:
        return None
    return offset + best_index, best_gain


def changepoint_detection(
    timestamps: list[Any], values: list[float],
    *, max_changepoints: int = 3, sensitivity: float = 1.0,
) -> dict[str, Any]:
    """Binary segmentation for mean shifts with a BIC-style penalty.

    Abstains (inconclusive) below MIN_CHANGEPOINT_HISTORY observations, and
    reports "no change detected" as a supported descriptive result — absence
    of change is a conclusion, not a failure."""
    if len(values) < MIN_CHANGEPOINT_HISTORY:
        return {
            "changepoints": [],
            "support": inconclusive(
                "insufficient_history",
                f"Changepoint detection needs at least {MIN_CHANGEPOINT_HISTORY} "
                f"observations (have {len(values)}).",
                "Supply a longer window of history.",
            ).to_dict(),
        }
    noise = _robust_scale(difference(values)) if len(values) > 2 else _robust_scale(values)
    penalty = sensitivity * 2.0 * (noise ** 2) * (len(values) ** 0.5)
    segments: list[tuple[int, int]] = [(0, len(values))]
    found: list[tuple[int, float]] = []
    while len(found) < max_changepoints:
        best = None
        for segment_index, (lo, hi) in enumerate(segments):
            candidate = _best_split(values[lo:hi], lo)
            if candidate and candidate[1] > penalty:
                if best is None or candidate[1] > best[1][1]:
                    best = (segment_index, candidate)
        if best is None:
            break
        segment_index, (split, gain) = best
        lo, hi = segments.pop(segment_index)
        segments.extend([(lo, split), (split, hi)])
        found.append((split, gain))
    found.sort()
    changepoints = []
    for split, gain in found:
        before = values[max(0, split - 12):split]
        after = values[split:split + 12]
        changepoints.append(Changepoint(
            split, timestamps[split], _mean(before), _mean(after),
            _mean(after) - _mean(before),
            gain / (_sse(values) + 1e-12),
        ).__dict__)
    support = SupportAssessment(
        "supported",
        [] if changepoints else [SupportReason(
            "no_change_detected",
            "No mean shift exceeded the noise-scaled penalty; the series is "
            "consistent with a stable level plus noise.",
        )],
        sensitivity={"penalty": penalty, "noise_scale": noise},
    )
    return {"changepoints": changepoints, "support": support.to_dict()}


# ---------------------------------------------------------------------------
# regime_detection — classify the latest changepoint as shift vs transient
# ---------------------------------------------------------------------------

MIN_PERSIST = 5


def regime_detection(
    timestamps: list[Any], values: list[float],
    *, max_changepoints: int = 3,
) -> dict[str, Any]:
    """Classify detected changes: sustained regime shift versus transient
    excursion that reverted. Abstains when the post-change window is too
    short to distinguish the two — that is exactly the honest answer."""
    detection = changepoint_detection(timestamps, values, max_changepoints=max_changepoints)
    if not detection["changepoints"]:
        return {**detection, "regimes": [], "classification": None}
    regimes = []
    for change in detection["changepoints"]:
        index = change["index"]
        pre = values[max(0, index - 24):index]
        post = values[index:]
        scale = _robust_scale(pre)
        shift_size = abs(change["shift"]) / scale
        if len(post) < MIN_PERSIST:
            classification = "undetermined"
            status: SupportAssessment = inconclusive(
                "post_window_too_short",
                f"Only {len(post)} observations follow the change at "
                f"{change['timestamp']}; at least {MIN_PERSIST} are needed to "
                "distinguish a regime shift from a transient.",
                "Wait for more observations or widen the window.",
            )
        else:
            tail = post[MIN_PERSIST - 1:] or post
            tail_shift = abs(_mean(tail) - _mean(pre)) / scale
            reverted = tail_shift < max(1.0, shift_size / 2)
            classification = "transient_anomaly" if reverted else "regime_shift"
            status = SupportAssessment("supported", sensitivity={
                "standardised_shift": shift_size,
                "standardised_tail_shift": tail_shift,
            })
        regimes.append({
            **change,
            "classification": classification,
            "support": status.to_dict(),
        })
    # A temporary excursion normally appears as two opposing changepoints.
    # Judging the return edge in isolation compares the restored baseline with
    # a pre-window dominated by the excursion and falsely calls a new regime.
    # Reconcile adjacent edges against the level before the first edge; this is
    # structural interpretation of detected changes, not a new detector.
    for index in range(1, len(regimes)):
        opening, closing = regimes[index - 1], regimes[index]
        if float(opening["shift"]) * float(closing["shift"]) >= 0:
            continue
        before = values[max(0, int(opening["index"]) - 24):
                        int(opening["index"])]
        after = values[int(closing["index"]):]
        if len(before) < MIN_PERSIST or len(after) < MIN_PERSIST:
            continue
        scale = max(_robust_scale(before), 1e-12)
        return_distance = abs(_mean(after) - _mean(before)) / scale
        opening_size = abs(float(opening["shift"])) / scale
        if return_distance < max(1.0, opening_size / 2):
            sensitivity = dict(
                (closing.get("support") or {}).get("sensitivity") or {})
            sensitivity.update({
                "paired_reversal": True,
                "opening_index": int(opening["index"]),
                "return_distance": return_distance,
            })
            closing["classification"] = "transient_anomaly"
            closing["support"] = SupportAssessment(
                "supported", sensitivity=sensitivity).to_dict()
    overall = regimes[-1]["classification"]
    return {**detection, "regimes": regimes, "classification": overall}


# ---------------------------------------------------------------------------
# anomaly_score — robust z-scores, optionally seasonally adjusted
# ---------------------------------------------------------------------------

def anomaly_score(
    timestamps: list[Any], values: list[float],
    *, season: int = 1, threshold: float = 3.5,
) -> dict[str, Any]:
    if len(values) < 8:
        return {"anomalies": [], "scores": [], "support": inconclusive(
            "insufficient_history",
            f"Anomaly scoring needs at least 8 observations (have {len(values)}).",
            "Supply a longer window of history.",
        ).to_dict()}
    if season > 1 and len(values) >= 2 * season:
        adjusted = []
        for index, value in enumerate(values):
            phase = [values[j] for j in range(index % season, len(values), season)]
            adjusted.append(value - median(phase))
    else:
        centre = median(values)
        adjusted = [value - centre for value in values]
    scale = _robust_scale(adjusted)
    scores = [round(value / scale, 4) for value in adjusted]
    anomalies = [
        {"timestamp": timestamps[index], "value": values[index], "score": score}
        for index, score in enumerate(scores) if abs(score) >= threshold
    ]
    return {"anomalies": anomalies, "scores": scores,
            "support": SupportAssessment("supported", sensitivity={
                "robust_scale": scale, "threshold": threshold,
            }).to_dict()}


# ---------------------------------------------------------------------------
# seasonality_analysis
# ---------------------------------------------------------------------------

def seasonality_analysis(values: list[float], frequency: str) -> dict[str, Any]:
    from .temporal import detect_season
    period, strength, source = detect_season(values, frequency)
    if source == "frequency_default" and len(values) < 2 * period:
        return {"period": None, "strength": 0.0, "profile": [], "support": inconclusive(
            "insufficient_history",
            f"Fewer than two cycles of the default period {period} are "
            "observed; seasonality cannot be established.",
            "Supply at least two full seasonal cycles.",
        ).to_dict()}
    profile = []
    if period > 1 and len(values) >= 2 * period:
        overall = median(values)
        for phase in range(period):
            phase_values = [values[index] for index in range(phase, len(values), period)]
            profile.append(round(median(phase_values) - overall, 6))
    return {"period": period, "strength": strength, "source": source,
            "profile": profile,
            "support": SupportAssessment(
                "supported" if source == "autocorrelation" else "conditionally_supported",
                [] if source == "autocorrelation" else [SupportReason(
                    "default_period",
                    "No autocorrelation peak was found; the period is the "
                    "frequency default, not a measured cycle.",
                )],
            ).to_dict()}


# ---------------------------------------------------------------------------
# cross_correlation — lead/lag on differenced series (associational only)
# ---------------------------------------------------------------------------

MIN_CROSS_CORRELATION = 12


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    mean_a, mean_b = _mean(a), _mean(b)
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    if var_a <= 1e-12 or var_b <= 1e-12:
        return None
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    return cov / (var_a * var_b) ** 0.5


def cross_correlation(
    values_a: list[float], values_b: list[float],
    *, max_lag: int = 8,
) -> dict[str, Any]:
    """Lead/lag correlation on first differences. A significant lead is
    *predictive precedence* — claim class associational, never causal."""
    n = min(len(values_a), len(values_b))
    if n < MIN_CROSS_CORRELATION:
        return {"correlations": {}, "best": None, "support": inconclusive(
            "insufficient_history",
            f"Cross-correlation needs at least {MIN_CROSS_CORRELATION} paired "
            f"observations (have {n}).",
            "Supply longer overlapping histories.",
        ).to_dict()}
    diff_a = difference(values_a[-n:])
    diff_b = difference(values_b[-n:])
    correlations: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = diff_a[: len(diff_a) - lag or None], diff_b[lag:]
        else:
            a, b = diff_a[-lag:], diff_b[: len(diff_b) + lag or None]
        paired = min(len(a), len(b))
        if paired < 6:
            continue
        value = _pearson(a[:paired], b[:paired])
        if value is not None:
            correlations[lag] = round(value, 4)
    if not correlations:
        return {"correlations": {}, "best": None, "support": inconclusive(
            "degenerate_series",
            "At least one series is constant after differencing.",
            "Check the inputs are genuine time-varying series.",
        ).to_dict()}
    significance = 2.0 / (n - 1) ** 0.5
    best_lag = max(correlations, key=lambda lag: abs(correlations[lag]))
    best = {"lag": best_lag, "correlation": correlations[best_lag],
            "significant": abs(correlations[best_lag]) >= significance}
    status = SupportAssessment(
        "supported" if best["significant"] else "unsupported",
        [] if best["significant"] else [SupportReason(
            "below_significance",
            f"No lead/lag correlation reached the significance bound "
            f"{significance:.3f}.",
        )],
        assumptions=["Correlations are computed on first differences and are "
                     "associational: a lead is predictive precedence, not causation."],
        sensitivity={"significance_bound": significance},
    )
    return {"correlations": {str(k): v for k, v in sorted(correlations.items())},
            "best": best, "support": status.to_dict()}


# ---------------------------------------------------------------------------
# event_study — pre/post windows around events (associational only)
# ---------------------------------------------------------------------------

def event_study(
    timestamps: list[Any], values: list[float], event_times: list[Any],
    *, window: int = 7,
) -> dict[str, Any]:
    index_of = {moment: index for index, moment in enumerate(timestamps)}
    studies = []
    for event_time in event_times:
        index = index_of.get(event_time)
        if index is None:
            nearest = [i for i, m in enumerate(timestamps) if m <= event_time]
            index = nearest[-1] + 1 if nearest else None
        if index is None or index < window or index + window > len(values):
            studies.append({"event_time": event_time, "effect": None,
                            "support": inconclusive(
                                "window_out_of_range",
                                f"Fewer than {window} observations exist on one "
                                f"side of {event_time}.",
                                "Reduce the window or supply more history.",
                            ).to_dict()})
            continue
        pre = values[index - window:index]
        post = values[index:index + window]
        scale = _robust_scale(pre)
        effect = _mean(post) - _mean(pre)
        studies.append({
            "event_time": event_time,
            "effect": round(effect, 6),
            "standardised_effect": round(effect / scale, 4),
            "pre_mean": _mean(pre), "post_mean": _mean(post),
            "support": SupportAssessment(
                "supported",
                assumptions=["Pre/post contrast is associational; concurrent "
                             "influences are not controlled for."],
                sensitivity={"window": window, "pre_scale": scale},
            ).to_dict(),
        })
    return {"studies": studies}


# ---------------------------------------------------------------------------
# simulate_scenario — deterministic transformations of a published forecast
# ---------------------------------------------------------------------------

def simulate_scenario(
    rows: list[dict[str, Any]],
    adjustments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply named deterministic adjustments (shift/scale, optionally from a
    step onward) to a published forecast's point and interval columns.
    Scenarios inherit the base forecast's calibration; they do not invent
    new uncertainty."""
    scenario_rows = [dict(row) for row in rows]
    applied = []
    for adjustment in adjustments:
        kind = adjustment.get("type")
        value = float(adjustment.get("value", 0.0))
        from_step = int(adjustment.get("from_step", 1))
        if kind not in ("shift", "scale"):
            raise ValueError(f"unknown adjustment type {kind!r}")
        for step, row in enumerate(scenario_rows, 1):
            if step < from_step:
                continue
            for column in ("point", "q10", "q50", "q90"):
                row[column] = row[column] + value if kind == "shift" else row[column] * value
        applied.append({"type": kind, "value": value, "from_step": from_step})
    return {"rows": scenario_rows, "applied": applied,
            "support": SupportAssessment(
                "conditionally_supported",
                [SupportReason("scenario_assumption",
                               "Scenario values hold only under the stated "
                               "adjustments; probabilities are inherited from "
                               "the base forecast's calibration.")],
            ).to_dict()}


# ---------------------------------------------------------------------------
# evaluate_threshold_risk
# ---------------------------------------------------------------------------

def evaluate_threshold_risk(
    rows: list[dict[str, Any]], points: list[float], residuals: list[float],
    threshold: float,
) -> dict[str, Any]:
    if not residuals:
        return {"risk": None, "support": inconclusive(
            "no_calibration_residuals",
            "No backtest residuals exist to calibrate exceedance probabilities.",
            "Run an evaluated forecast first.",
        ).to_dict()}
    from .pipeline import threshold_analysis_stage
    # Per-lead spreads read off the rows themselves, so the probabilities
    # describe the quantiles printed beside them under whatever recentring
    # policy produced those rows. (This call previously passed quantiles
    # keyed by probability where step-keyed spreads were expected and
    # raised KeyError on every invocation.)
    spreads = {
        step: (
            float(row["q50"]) - float(row["q10"]),
            float(row["q50"]) - float(row["point"]),
            float(row["q90"]) - float(row["q50"]),
        )
        for step, row in enumerate(rows, 1)
    }
    analysis = threshold_analysis_stage(threshold, rows, points, residuals, spreads)
    peak = max(analysis["probability_above"]) if analysis["probability_above"] else 0.0
    return {"risk": analysis, "peak_step_probability": peak,
            "support": SupportAssessment(
                "supported",
                assumptions=[
                    "Probabilities are per-step marginals from pooled backtest "
                    "residuals; the peak marginal is a lower bound on the "
                    "probability of any exceedance over the horizon.",
                ],
            ).to_dict()}


# ---------------------------------------------------------------------------
# evaluate_actions — with the mandatory degraded mode
# ---------------------------------------------------------------------------

@dataclass
class ActionEvaluation:
    name: str
    feasible: bool
    constraint_results: dict[str, Any] = field(default_factory=dict)
    expected_utility: float | None = None
    downside: float | None = None


def evaluate_actions(
    actions: list[dict[str, Any]],
    scenario_probabilities: dict[str, float],
    *,
    utilities: dict[str, dict[str, float]] | None = None,
    max_acceptable_risk: float | None = None,
) -> dict[str, Any]:
    """Compare feasible actions under scenario probabilities.

    With utilities (action → scenario → payoff): expected utility and a
    choice. Without them: the feasible-action comparison plus the scenario
    probabilities, ``conditionally_supported: missing utility inputs`` —
    never a silent guess, never a hard failure."""
    probability_mass = sum(scenario_probabilities.values())
    if not actions:
        return {"evaluations": [], "selected": None, "support": inconclusive(
            "no_actions", "No candidate actions were supplied.",
            "Supply at least one action to evaluate.",
        ).to_dict()}
    if abs(probability_mass - 1.0) > 0.05:
        return {"evaluations": [], "selected": None, "support": SupportAssessment(
            "invalid",
            [SupportReason("invalid_probabilities",
                           f"Scenario probabilities sum to {probability_mass:.3f}, not 1.")],
        ).to_dict()}
    evaluations: list[ActionEvaluation] = []
    for action in actions:
        name = str(action["name"])
        constraint_results: dict[str, Any] = {}
        feasible = bool(action.get("feasible", True))
        if max_acceptable_risk is not None and "residual_risk" in action:
            ok = float(action["residual_risk"]) <= max_acceptable_risk
            constraint_results["max_acceptable_risk"] = {
                "limit": max_acceptable_risk,
                "value": float(action["residual_risk"]),
                "satisfied": ok,
            }
            feasible = feasible and ok
        item = ActionEvaluation(name, feasible, constraint_results)
        if utilities is not None and name in utilities:
            payoffs = utilities[name]
            item.expected_utility = sum(
                scenario_probabilities.get(scenario, 0.0) * payoff
                for scenario, payoff in payoffs.items()
            )
            item.downside = min(payoffs.values())
        evaluations.append(item)
    feasible_items = [item for item in evaluations if item.feasible]
    if not feasible_items:
        return {
            "evaluations": [item.__dict__ for item in evaluations],
            "selected": None,
            "support": SupportAssessment(
                "unsupported",
                [SupportReason("no_feasible_action",
                               "Every candidate action violates its constraints.")],
            ).to_dict(),
        }
    if utilities is None or any(item.expected_utility is None for item in feasible_items):
        return {
            "evaluations": [item.__dict__ for item in evaluations],
            "selected": None,
            "scenario_probabilities": scenario_probabilities,
            "support": SupportAssessment(
                "conditionally_supported",
                [SupportReason("missing_utility_inputs",
                               "No utilities were supplied for some feasible "
                               "actions; returning the feasible-action "
                               "comparison and scenario probabilities instead "
                               "of a choice.")],
                recovery_actions=[SupportReason(
                    "provide_utilities",
                    "Supply a payoff per action per scenario to enable an "
                    "expected-utility choice.")],
            ).to_dict(),
        }
    ranked = sorted(feasible_items, key=lambda item: (-item.expected_utility, item.name))
    selected = ranked[0]
    margin = (selected.expected_utility - ranked[1].expected_utility) if len(ranked) > 1 else None
    return {
        "evaluations": [item.__dict__ for item in evaluations],
        "selected": selected.name,
        "decision_rule": "maximum expected utility over feasible actions",
        "selection_margin": margin,
        "scenario_probabilities": scenario_probabilities,
        "support": SupportAssessment(
            "supported",
            sensitivity={"selection_margin": margin},
            assumptions=["Utilities are taken as supplied; no preference "
                         "elicitation was performed."],
        ).to_dict(),
    }
