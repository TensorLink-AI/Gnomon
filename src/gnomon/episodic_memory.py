"""Episodic memory for adaptive routing: retrieve the most similar matured past forecasts.

An episode is one past routed origin: a feature vector describing the series as it was
known at that origin (recorded when the router forecast, never recomputed from later
data) and each provider's matched per-origin loss once every target was observed. At a
new forecast the router describes the current request the same way, retrieves the k
nearest matured episodes from its own series and pool, and scores providers by their
similarity-weighted losses relative to the baseline.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math

from .forecast_adapter import ForecastAdapterError

FEATURES = ("volatility_ratio", "trend", "seasonality", "zero_share", "missing_share", "length_cycles",
            "level_shift", "cv", "covariate_share")
DEFAULT_FEATURES = ["volatility_ratio", "trend", "seasonality", "zero_share", "missing_share", "length_cycles",
                    "level_shift", "cv"]
_SPEC_KEYS = ("features", "short_window", "long_window", "season", "mask_covariate", "future_covariate")


def _int(value, field, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ForecastAdapterError(f"memory.{field} must be an integer in [{minimum}, {maximum or 'inf'}]",
                                   details={"rejected_fields": [f"memory.{field}"]})
    return value


def _num(value, field, minimum):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ForecastAdapterError(f"memory.{field} must be a finite number >= {minimum}",
                                   details={"rejected_fields": [f"memory.{field}"]})
    return float(value)


def validate_memory(memory):
    """Validated memory policy with a spec_id over the fields that define the feature space."""
    if memory is None:
        return None
    allowed = {"features", "short_window", "long_window", "season", "k", "min_effective_n", "own_weight",
               "recency_half_life_days", "mask_covariate", "future_covariate"}
    if not isinstance(memory, dict) or set(memory) - allowed:
        raise ForecastAdapterError(f"memory allows only {sorted(allowed)}", details={"rejected_fields": ["memory"]})
    features = memory.get("features", DEFAULT_FEATURES)
    if not isinstance(features, list) or not features or len(set(features)) != len(features) \
            or any(f not in FEATURES for f in features):
        raise ForecastAdapterError(f"memory.features must be distinct names from {list(FEATURES)}",
                                   details={"rejected_fields": ["memory.features"]})
    short = _int(memory.get("short_window", 14), "short_window", 2)
    long = _int(memory.get("long_window", 112), "long_window", 3)
    if long <= short:
        raise ForecastAdapterError("memory.long_window must exceed short_window",
                                   details={"rejected_fields": ["memory.long_window"]})
    season = _int(memory.get("season", 1), "season", 1)
    for name in ("mask_covariate", "future_covariate"):
        if memory.get(name) is not None and (not isinstance(memory[name], str) or not memory[name]):
            raise ForecastAdapterError(f"memory.{name} must be a covariate name",
                                       details={"rejected_fields": [f"memory.{name}"]})
    if "covariate_share" in features and not memory.get("future_covariate"):
        raise ForecastAdapterError("covariate_share needs memory.future_covariate",
                                   details={"rejected_fields": ["memory.future_covariate"]})
    spec = {"features": list(features), "short_window": short, "long_window": long, "season": season,
            "mask_covariate": memory.get("mask_covariate"), "future_covariate": memory.get("future_covariate"),
            "k": _int(memory.get("k", 32), "k", 1, 1000),
            "min_effective_n": _num(memory.get("min_effective_n", 8), "min_effective_n", 1),
            "own_weight": _num(memory.get("own_weight", 1.0), "own_weight", 1),
            "recency_half_life_days": _num(memory.get("recency_half_life_days", 0), "recency_half_life_days", 0)}
    spec["spec_id"] = sha256(json.dumps({k: spec[k] for k in _SPEC_KEYS}, sort_keys=True).encode()).hexdigest()[:12]
    return spec


def _std(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _clip(x, bound=5.0):
    return max(-bound, min(bound, x))


def _column(rows, names, name):
    if not name or not rows or name not in names:
        return None
    j = list(names).index(name)
    return [float(row[j]) for row in rows]


def compute_features(spec, history, timestamps=(), *, past_covariates=(), past_covariate_names=(),
                     future_covariates=(), future_covariate_names=()):
    """Features from what the request carries at its origin; None where a feature is undefined.

    Undefined (too little history, a flat window, an absent covariate) is recorded as None and
    excluded from distances, never imputed.
    """
    values = [float(v) for v in history]
    n = len(values)
    short_n, long_n, season = spec["short_window"], spec["long_window"], spec["season"]
    mask = _column(past_covariates, past_covariate_names, spec.get("mask_covariate"))
    out = {name: None for name in spec["features"]}
    if n >= long_n:
        long, short = values[-long_n:], values[-short_n:]
        long_mask = mask[-long_n:] if mask is not None else None
        observed = [v for v, m in zip(long, long_mask) if m > 0] if long_mask is not None else long
        s_long, m_long = _std(long), sum(long) / long_n
        s_short = _std(short)
        if "volatility_ratio" in out and s_long > 0:
            out["volatility_ratio"] = _clip(math.log(max(s_short, 1e-3 * s_long) / s_long))
        if "trend" in out and s_long > 0:
            diffs = [b - a for a, b in zip(short, short[1:])]
            out["trend"] = _clip((sum(diffs) / len(diffs)) / s_long)
        if "seasonality" in out and season > 1 and long_n >= 2 * season and s_long > 0:
            dev = [v - m_long for v in long]
            out["seasonality"] = sum(a * b for a, b in zip(dev, dev[season:])) / (s_long ** 2 * (long_n - season))
        if "zero_share" in out and observed:
            out["zero_share"] = sum(1 for v in observed if v == 0) / len(observed)
        if "level_shift" in out and s_long > 0:
            out["level_shift"] = _clip((sum(short) / short_n - m_long) / s_long)
        if "cv" in out and s_long > 0 and m_long != 0:
            out["cv"] = _clip(math.log(s_long / abs(m_long)))
        if "missing_share" in out:
            if long_mask is not None:
                out["missing_share"] = sum(1 for m in long_mask if m <= 0) / long_n
            elif len(timestamps) >= long_n:
                from datetime import datetime
                stamps = [datetime.fromisoformat(t) for t in list(timestamps)[-long_n:]]
                steps = sorted(b - a for a, b in zip(stamps, stamps[1:]))
                step = steps[len(steps) // 2]
                if step.total_seconds() > 0:
                    expected = (stamps[-1] - stamps[0]) / step + 1
                    out["missing_share"] = max(0.0, 1 - long_n / expected)
    if "length_cycles" in out and n:
        out["length_cycles"] = math.log(min(n, 4 * long_n) / max(season, 1))
    if "covariate_share" in out:
        future = _column(future_covariates, future_covariate_names, spec.get("future_covariate"))
        if future:
            out["covariate_share"] = sum(1 for v in future if v != 0) / len(future)
    return out


def request_features(spec, request: dict):
    return compute_features(spec, request.get("history", []), request.get("timestamps", ()),
                            past_covariates=request.get("past_covariates", ()),
                            past_covariate_names=request.get("past_covariate_names", ()),
                            future_covariates=request.get("future_covariates", ()),
                            future_covariate_names=request.get("future_covariate_names", ()))


def _robust_scales(vectors, names):
    """Median and MAD per feature over the episodes visible now (point-in-time standardisation)."""
    scales = {}
    for name in names:
        xs = sorted(v[name] for v in vectors if v.get(name) is not None)
        if not xs:
            continue
        med = xs[len(xs) // 2]
        mad = sorted(abs(x - med) for x in xs)[len(xs) // 2]
        scales[name] = (med, 1.4826 * mad if mad > 0 else (_std(xs) or 1.0))
    return scales


def _distance(a, b, names, scales):
    shared = [n for n in names if a.get(n) is not None and b.get(n) is not None and n in scales]
    if len(shared) * 2 < len(names):  # compare only on at least half the feature space
        return None
    total = sum(((a[n] - b[n]) / scales[n][1]) ** 2 for n in shared)
    return math.sqrt(total * len(names) / len(shared))


def episodic_scores(rows, policy, own_series, query, origin_time=None):
    """Similarity-weighted provider scores relative to the baseline (baseline = 1.0).

    rows: matured evidence rows with "features" (recorded at routing time under this spec).
    Losses are normalised by each series' mean baseline loss over its visible episodes, then
    combined as weighted sums, so an episode with a zero baseline loss cannot explode a ratio.
    Returns (scores, effective_n, neighbours).
    """
    from datetime import datetime
    memory = policy["memory"]
    providers, baseline = [policy["baseline"], *policy["candidates"]], policy["baseline"]
    pool = policy.get("pool")
    allowed = {own_series, *(pool["series"] if pool else [])}
    episodes = [r for r in rows if r["series_id"] in allowed and r.get("features") is not None
                and all(p in r["losses"] for p in providers)]
    if not episodes or query is None:
        return {}, 0.0, []
    names = memory["features"]
    scales = _robust_scales([e["features"] for e in episodes], names)
    scored = []
    for e in episodes:
        d = _distance(query, e["features"], names, scales)
        if d is not None:
            scored.append((d, e))
    scored.sort(key=lambda x: (x[0], x[1]["origin"], x[1]["series_id"]))
    nearest = scored[:memory["k"]]
    if not nearest:
        return {}, 0.0, []
    distances = sorted(d for d, _ in nearest)
    bandwidth = distances[len(distances) // 2] or 1e-9
    series_scale = {}
    for e in episodes:
        series_scale.setdefault(e["series_id"], []).append(e["losses"][baseline])
    series_scale = {s: sum(v) / len(v) for s, v in series_scale.items()}
    half_life = memory["recency_half_life_days"]
    now = datetime.fromisoformat(origin_time) if origin_time and half_life else None
    weighted, weights, neighbours = {p: 0.0 for p in providers}, [], []
    for d, e in nearest:
        scale = series_scale.get(e["series_id"]) or 0.0
        if scale <= 0:
            continue
        w = math.exp(-0.5 * (d / bandwidth) ** 2)
        if e["series_id"] == own_series:
            w *= memory["own_weight"]
        if now is not None:
            age_days = (now - datetime.fromisoformat(e["origin"])).total_seconds() / 86400
            w *= 0.5 ** (max(age_days, 0.0) / half_life)
        if w <= 0:
            continue
        for p in providers:
            weighted[p] += w * e["losses"][p] / scale
        weights.append(w)
        neighbours.append({"series_id": e["series_id"], "origin": e["origin"], "distance": round(d, 4),
                           "weight": round(w, 4), "best_provider": min(providers, key=lambda p: e["losses"][p]),
                           "features": {name: e["features"].get(name) for name in names},
                           "losses": {p: e["losses"][p] for p in providers},
                           "baseline_loss_scale": scale,
                           "normalized_losses": {p: e["losses"][p] / scale for p in providers}})
    total = sum(weights)
    if total <= 0 or weighted[baseline] <= 0:
        return {}, 0.0, []
    effective_n = total ** 2 / sum(w * w for w in weights)
    scores = {p: weighted[p] / weighted[baseline] for p in providers}
    neighbours.sort(key=lambda x: -x["weight"])
    return scores, effective_n, neighbours
