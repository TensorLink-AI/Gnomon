"""Provider-neutral helpers for governed model-authored context priors.

Gnomon does not call an LLM. Host integrations may use these helpers to ask
their own model for independent point paths, then deterministically validate,
aggregate, and seal those paths through :mod:`gnomon.llm_dossier`.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from datetime import datetime
from typing import Any


def _json_objects(text: str) -> list[dict[str, Any]]:
    """Extract JSON objects without trusting markdown or surrounding prose."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    offset = 0
    while offset < len(text):
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        offset = start + consumed
    return objects


def empirical_quantile(values: list[float], probability: float) -> float:
    """Dependency-free linear empirical quantile for bounded path draws."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a quantile from no values")
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    weight = position - left
    return ordered[left] + (ordered[right] - ordered[left]) * weight


def sample_path_stability(
    paths: list[list[float]], history_values: list[float] | None,
) -> dict[str, Any]:
    """Return scale-free draw dispersion, explicitly not forecast skill."""
    if not paths or not paths[0]:
        raise ValueError("at least one non-empty sampled path is required")
    history = [float(value) for value in history_values or []
               if math.isfinite(float(value))]
    increments = [abs(right - left) for left, right in zip(history, history[1:])
                  if abs(right - left) > 0]
    if increments:
        scale = statistics.median(increments)
        scale_basis = "median_nonzero_history_increment"
    elif len(history) >= 2 and max(history) > min(history):
        scale = max(history) - min(history)
        scale_basis = "history_range"
    else:
        scale = max(1.0, abs(statistics.median(history)) * .01) if history else 1.0
        scale_basis = "level_floor"
    scale = max(float(scale), 1e-12)

    widths = []
    for index in range(len(paths[0])):
        point_values = [path[index] for path in paths]
        widths.append((empirical_quantile(point_values, .9) -
                       empirical_quantile(point_values, .1)) / scale)
    pairwise = [
        statistics.mean(abs(a - b) for a, b in zip(left, right)) / scale
        for left_index, left in enumerate(paths)
        for right in paths[left_index + 1:]
    ]
    tolerance = scale * 1e-6
    direction_agreement = []
    for index in range(1, len(paths[0])):
        signs = []
        for path in paths:
            change = path[index] - path[index - 1]
            signs.append(1 if change > tolerance else
                         -1 if change < -tolerance else 0)
        direction_agreement.append(max(signs.count(-1), signs.count(0),
                                       signs.count(1)) / len(signs))
    return {
        "version": "0.1",
        "interpretation": "stability_not_historical_skill",
        "scale_basis": scale_basis,
        "path_count": len(paths),
        "horizon": len(paths[0]),
        "median_pointwise_q80_width_scaled": statistics.median(widths),
        "p90_pointwise_q80_width_scaled": empirical_quantile(widths, .9),
        "median_pairwise_mae_scaled": (
            statistics.median(pairwise) if pairwise else 0.0),
        "max_pairwise_mae_scaled": max(pairwise) if pairwise else 0.0,
        "mean_direction_agreement": (
            statistics.mean(direction_agreement)
            if direction_agreement else 1.0),
        "unanimous_direction_fraction": (
            sum(value == 1.0 for value in direction_agreement) /
            len(direction_agreement) if direction_agreement else 1.0),
    }


def candidate_from_sampled_paths(
    outputs: list[str], future_timestamps: list[str],
    *, history_values: list[float] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate independent model paths on a host-owned grid and aggregate."""
    accepted: list[list[float]] = []
    rejection_reasons: list[str] = []
    rationales: list[str] = []
    expected = len(future_timestamps)
    display_timestamps = []
    for timestamp in future_timestamps:
        try:
            display_timestamps.append(datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")).strftime(
                    "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            display_timestamps.append(timestamp)
    for output in outputs:
        objects = _json_objects(output)
        first = objects[0] if objects else {}
        raw = first.get("forecast_path") if isinstance(first, dict) else None
        values = raw.get("values") if isinstance(raw, dict) else None
        if not isinstance(values, list):
            match = re.search(
                r"<forecast>\s*(.*?)\s*</forecast>", output,
                flags=re.IGNORECASE | re.DOTALL)
            parsed_values: list[float] = []
            parsed_timestamps: list[str] = []
            if match:
                for line in match.group(1).splitlines():
                    line = line.strip().strip("(), ")
                    if not line:
                        continue
                    parts = line.rsplit(",", 1)
                    if len(parts) != 2:
                        parsed_values = []
                        break
                    parsed_timestamps.append(parts[0].strip(" '\""))
                    try:
                        parsed_values.append(float(parts[1].strip()))
                    except ValueError:
                        parsed_values = []
                        break
            if parsed_values and parsed_timestamps == display_timestamps:
                values = parsed_values
        if not isinstance(values, list) or len(values) != expected:
            rejection_reasons.append(
                f"forecast response requires {expected} host-grid-bound values")
            continue
        try:
            path = [float(value) for value in values]
        except (TypeError, ValueError):
            rejection_reasons.append("forecast_path contains a non-number")
            continue
        if not all(math.isfinite(value) for value in path):
            rejection_reasons.append("forecast_path contains a non-finite value")
            continue
        accepted.append(path)
        rationale = " ".join(str(
            raw.get("rationale") or "" if isinstance(raw, dict) else ""
        ).split())
        if rationale:
            rationales.append(rationale[:300])
    diagnostics = {
        "requested": len(outputs), "accepted": len(accepted),
        "rejected": len(outputs) - len(accepted),
        "rejection_reasons": rejection_reasons[:8],
        "aggregation": "linear_empirical_marginal_q10_q50_q90",
        "timestamp_binding": "host_grid_order",
        "request_mode": "concurrent_single_sample_requests",
    }
    if not accepted:
        return None, diagnostics
    diagnostics["stability"] = sample_path_stability(accepted, history_values)
    rows = []
    for index, timestamp in enumerate(future_timestamps):
        point_values = [path[index] for path in accepted]
        rows.append({
            "timestamp": timestamp,
            "q10": empirical_quantile(point_values, .1),
            "q50": empirical_quantile(point_values, .5),
            "q90": empirical_quantile(point_values, .9),
        })
    candidate = {
        "quantiles": rows,
        "_validated_sample_paths": accepted,
        "rationale": (
            f"Host-aggregated {len(accepted)} sampled model-authored point "
            "paths into empirical marginal quantiles. "
            + ("Sample rationales: " + " | ".join(rationales[:2])
               if rationales else "")),
    }
    return candidate, diagnostics


def build_sampled_context_prior_prompt(
    *, timestamps: list[str], values: list[float],
    future_timestamps: list[str], context: str,
) -> str:
    """Build a compact numeric prompt for a host's own model provider."""
    history_values = ",".join(f"{float(value):.12g}" for value in values)

    def grid_summary(grid: list[str]) -> str:
        parsed = [datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")) for timestamp in grid]
        if len(parsed) < 2:
            return f"timestamps={grid!r}"
        steps = [(right - left).total_seconds()
                 for left, right in zip(parsed, parsed[1:])]
        if steps and max(steps) == min(steps):
            return (f"start={grid[0]}, end={grid[-1]}, "
                    f"step_seconds={steps[0]:.12g}, count={len(grid)}")
        return "timestamps=" + json.dumps(grid, separators=(",", ":"))

    return f"""\
I have a time series forecasting task for you.

Here is context known at the forecast cutoff. Factor in relevant background
knowledge, satisfy any stated constraints, and respect any stated scenarios.
<context>
{context}
</context>

The host owns the historical grid ({grid_summary(timestamps)}). Values below
are in exact grid order:
<history>
[{history_values}]
</history>

Predict the future grid ({grid_summary(future_timestamps)}).

Return only compact JSON with exactly one finite value per future grid point,
in order. Do not echo timestamps:

{{"forecast_path":{{"values":[1.0,2.0],"rationale":"brief basis"}}}}

Use no observations after the cutoff.
"""


def recommended_sample_count(horizon: int) -> int:
    """Bound host inference while tolerating one rejected long path."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return 4 if horizon >= 96 else 5
