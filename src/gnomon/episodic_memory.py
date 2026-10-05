"""Episodic memory for adaptive routing: retrieve the most similar matured past forecasts.

An episode is one past routed origin: a feature vector describing the series as it was
known at that origin (recorded when the router forecast, never recomputed from later
data) and each provider's matched per-origin loss once every target was observed. At a
new forecast the router describes the current request the same way, retrieves the k
nearest matured episodes from its own series and pool, and scores providers by their
similarity-weighted losses relative to the baseline.

Optional, off by default (a policy without them behaves exactly as before): feature
`profile` presets for level / return / intermittent series and four extra features;
`context` features (the latest value of named past covariates, e.g. a market-wide state);
`dedupe_seconds` (one neighbour per series per window, so overlapping outcomes are not
counted as independent); `shrinkage` toward "no difference"; `confidence_z` (rank
candidates by a pessimistic score); `novelty_threshold` (abstain when the current state is
unlike anything remembered); and `diagnostics` (standard errors, intervals, neighbour loss
quantiles, novelty) in every memory decision.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math

from .forecast_adapter import ForecastAdapterError
from . import fase_memory

FEATURES = ("volatility_ratio", "trend", "seasonality", "zero_share", "missing_share", "length_cycles",
            "level_shift", "cv", "covariate_share", "autocorrelation", "skewness", "vol_of_vol",
            "demand_interval") + fase_memory.FEATURES
DEFAULT_FEATURES = ["volatility_ratio", "trend", "seasonality", "zero_share", "missing_share", "length_cycles",
                    "level_shift", "cv"]
# Feature sets that suit the kind of series. "levels" is the default set. For return-like
# series (changes, log returns, P&L) level features such as cv are unstable; for
# intermittent demand, the interval between nonzero values matters.
PROFILES = {
    "fase": list(fase_memory.FEATURES),
    "levels": DEFAULT_FEATURES,
    "returns": ["volatility_ratio", "level_shift", "trend", "autocorrelation", "skewness", "vol_of_vol"],
    "intermittent": ["zero_share", "demand_interval", "missing_share", "cv", "level_shift", "volatility_ratio"],
}
CONTEXT_PREFIX = "context:"
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
               "recency_half_life_days", "mask_covariate", "future_covariate", "profile", "context",
               "dedupe_seconds", "shrinkage", "confidence_z", "novelty_threshold", "diagnostics", "distance", "retention", "recent_capacity", "long_term_capacity"}
    if not isinstance(memory, dict) or set(memory) - allowed:
        raise ForecastAdapterError(f"memory allows only {sorted(allowed)}", details={"rejected_fields": ["memory"]})
    for field, choices in (("distance", ("robust", "fase")), ("retention", ("window", "fase"))):
        if memory.get(field, choices[0]) not in choices:
            raise ForecastAdapterError(f"memory.{field} must be one of {choices}",
                                       details={"rejected_fields": [f"memory.{field}"]})
    if memory.get("retention") == "fase" and (memory.get("distance", "robust") != "fase" or memory.get("dedupe_seconds", 0)):
        raise ForecastAdapterError("memory.retention=fase requires distance=fase and dedupe_seconds=0",
                                   details={"rejected_fields": ["memory.retention"]})
    profile = memory.get("profile")
    if profile is not None:
        if profile not in PROFILES:
            raise ForecastAdapterError(f"memory.profile must be one of {sorted(PROFILES)}",
                                       details={"rejected_fields": ["memory.profile"]})
        if "features" in memory:
            raise ForecastAdapterError("give memory.features or memory.profile, not both",
                                       details={"rejected_fields": ["memory.profile"]})
    features = memory.get("features", PROFILES[profile] if profile else DEFAULT_FEATURES)
    if not isinstance(features, list) or not features or len(set(features)) != len(features) \
            or any(f not in FEATURES for f in features):
        raise ForecastAdapterError(f"memory.features must be distinct names from {list(FEATURES)}",
                                   details={"rejected_fields": ["memory.features"]})
    short = _int(memory.get("short_window", 14), "short_window", 2)
    long = _int(memory.get("long_window", 15360 if profile == "fase" else 112), "long_window", 3)
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
    context = memory.get("context", [])
    if not isinstance(context, list) or len(context) > 16 or len(set(context)) != len(context) \
            or any(not isinstance(c, str) or not c for c in context):
        raise ForecastAdapterError("memory.context must be up to 16 distinct past-covariate names",
                                   details={"rejected_fields": ["memory.context"]})
    threshold = memory.get("novelty_threshold")
    if threshold is not None:
        threshold = _num(threshold, "novelty_threshold", 0)
        if threshold <= 0:
            raise ForecastAdapterError("memory.novelty_threshold must be > 0",
                                       details={"rejected_fields": ["memory.novelty_threshold"]})
    diagnostics = memory.get("diagnostics", False)
    if not isinstance(diagnostics, bool):
        raise ForecastAdapterError("memory.diagnostics must be true or false",
                                   details={"rejected_fields": ["memory.diagnostics"]})
    spec = {"features": list(features), "short_window": short, "long_window": long, "season": season,
            "mask_covariate": memory.get("mask_covariate"), "future_covariate": memory.get("future_covariate"),
            "k": _int(memory.get("k", 10 if profile == "fase" else 32), "k", 1, 1000),
            "min_effective_n": _num(memory.get("min_effective_n", 8), "min_effective_n", 1),
            "own_weight": _num(memory.get("own_weight", 1.0), "own_weight", 1),
            "recency_half_life_days": _num(memory.get("recency_half_life_days", 0), "recency_half_life_days", 0),
            "context": list(context),
            "dedupe_seconds": _num(memory.get("dedupe_seconds", 0), "dedupe_seconds", 0),
            "shrinkage": _num(memory.get("shrinkage", 0), "shrinkage", 0),
            "confidence_z": _num(memory.get("confidence_z", 0), "confidence_z", 0),
            "novelty_threshold": threshold, "diagnostics": diagnostics}
    spec.update(distance=memory.get("distance", "robust"), retention=memory.get("retention", "window"),
                recent_capacity=_int(memory.get("recent_capacity", 100), "recent_capacity", 1, 10000),
                long_term_capacity=_int(memory.get("long_term_capacity", 900), "long_term_capacity", 0, 10000))
    identity = {k: spec[k] for k in _SPEC_KEYS}
    if spec["context"]:  # only when used, so existing spec_ids (and recorded episodes) stay valid
        identity["context"] = spec["context"]
    spec["spec_id"] = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return spec


def feature_names(spec):
    """Names compared by the distance: the series features, then the context features."""
    return list(spec["features"]) + [CONTEXT_PREFIX + c for c in spec.get("context", [])]


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
    values = [float(v) if v is not None else float("nan") for v in history]
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
        if "autocorrelation" in out and s_long > 0 and long_n > 2:
            dev = [v - m_long for v in long]
            out["autocorrelation"] = sum(a * b for a, b in zip(dev, dev[1:])) / (s_long ** 2 * (long_n - 1))
        if "skewness" in out and s_long > 0:
            out["skewness"] = _clip(sum((v - m_long) ** 3 for v in long) / long_n / s_long ** 3)
        if "vol_of_vol" in out:
            blocks = [long[i:i + short_n] for i in range(long_n % short_n, long_n, short_n)]
            sds = [_std(b) for b in blocks if len(b) == short_n]
            mean_sd = sum(sds) / len(sds) if len(sds) >= 2 else 0.0
            if mean_sd > 0:
                out["vol_of_vol"] = _clip(_std(sds) / mean_sd)
        if "demand_interval" in out and observed:
            nonzero = sum(1 for v in observed if v != 0)
            if nonzero:
                out["demand_interval"] = math.log(len(observed) / nonzero)
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
    if any(name in fase_memory.FEATURES for name in out):
        computed = fase_memory.features(history, window=spec["long_window"], season=season, mask=mask,
                                        covariates=past_covariates, covariate_names=past_covariate_names,
                                        mask_name=spec.get("mask_covariate"))
        out.update({name: computed[name] for name in out if name in computed})
    for name in spec.get("context", []):
        # The latest value of a named past covariate at the origin: caller-supplied state such
        # as a market-wide volatility, a promotion flag or a peer-group aggregate.
        column = _column(past_covariates, past_covariate_names, name)
        value = column[-1] if column else None
        out[CONTEXT_PREFIX + name] = value if value is not None and math.isfinite(value) else None
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
    Returns (scores, effective_n, neighbours). See `episodic_evidence` for the optional extras.
    """
    evidence = episodic_evidence(rows, policy, own_series, query, origin_time)
    return evidence["scores"], evidence["effective_n"], evidence["neighbours"]


def _neighbours(scored, k, dedupe_seconds):
    """The k nearest; with dedupe_seconds, at most one episode per series within that window,
    so overlapping multi-step outcomes and near-identical adjacent origins count once."""
    if not dedupe_seconds:
        return scored[:k]
    from datetime import datetime
    chosen, seen = [], {}
    for d, e in scored:
        t = datetime.fromisoformat(e["origin"]).timestamp()
        times = seen.setdefault(e["series_id"], [])
        if any(abs(t - u) < dedupe_seconds for u in times):
            continue
        times.append(t)
        chosen.append((d, e))
        if len(chosen) == k:
            break
    return chosen


def _weighted_quantile(pairs, q):
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for value, w in pairs:
        acc += w
        if acc >= q * total:
            return value
    return pairs[-1][0]


def _novelty(query, vectors, names, scales, k, reference_size=512, sample_size=32, distance=_distance):
    """Median distance from the query to its k nearest episodes, divided by the same quantity
    for a deterministic sample of remembered episodes: about 1 for a familiar state, well above
    1 for a state unlike anything remembered. Both are measured against the same evenly spaced
    reference subset (at most `reference_size` episodes), so the cost stays bounded in long
    replays and the two distances are comparable."""
    def median_k(target, pool):
        ds = sorted(d for d in (distance(target, v, names, scales) for v in pool) if d is not None)[:k]
        return ds[len(ds) // 2] if ds else None
    reference = vectors[::max(1, len(vectors) // reference_size)]
    own = median_k(query, reference)
    step = max(1, len(reference) // sample_size)
    typical = [m for i in range(0, len(reference), step)
               if (m := median_k(reference[i], reference[:i] + reference[i + 1:])) is not None]
    if own is None or not typical:
        return None
    base = sorted(typical)[len(typical) // 2]
    return round(own / base, 4) if base > 0 else None


def episodic_evidence(rows, policy, own_series, query, origin_time=None, *, retained_pool=None, accelerate=False):
    """Everything memory knows about this query.

    Always: scores, effective_n, neighbours (as `episodic_scores`). With memory options:
    `dedupe_seconds` changes which neighbours count; `shrinkage` pulls candidate scores toward
    1.0 (no difference) by effective_n / (effective_n + shrinkage); `confidence_z` adds
    `selection_scores` = score + z * standard error, which selection uses instead of scores;
    `novelty_threshold` / `diagnostics` add `novelty`; `diagnostics` adds standard errors, 90%
    intervals, neighbour loss-ratio quantiles and the unshrunk scores.
    """
    from datetime import datetime
    memory = policy["memory"]
    empty = {"scores": {}, "effective_n": 0.0, "neighbours": []}
    providers, baseline = [policy["baseline"], *policy["candidates"]], policy["baseline"]
    pool = policy.get("pool")
    allowed = {own_series, *(pool["series"] if pool else [])}
    episodes = [r for r in rows if r["series_id"] in allowed and r.get("features") is not None
                and all(p in r["losses"] for p in providers)]
    if not episodes or query is None:
        return empty
    names = feature_names(memory)
    retention_info = {}
    if memory.get("retention") == "fase":
        retained = retained_pool
        if retained is None:
            retained = fase_memory.Pool(memory["recent_capacity"], memory["long_term_capacity"])
            for row in sorted(episodes, key=lambda r: (r.get("completed_at", r["origin"]), r["origin"], r["series_id"])):
                retained.complete(row)
        episodes = retained.rows
        retention_info = {"retention_pool": {"recent": len(retained.recent), "long_term": len(retained.long_term)}}
    bounded = memory.get("distance") == "fase"
    distance = fase_memory.distance if bounded else _distance
    if accelerate and bounded:
        from .episodic_memory_fast import bounded_distances
        ds = bounded_distances([[e['features'].get(n) for n in names] for e in episodes],
                               [query.get(n) for n in names], names)
        scored = [(float(d), e) for d, e in zip(ds, episodes) if math.isfinite(d)]
    else:
        scales = (fase_memory.scales if bounded else _robust_scales)([e["features"] for e in episodes], names)
        scored = []
        for e in episodes:
            d = distance(query, e["features"], names, scales)
            if d is not None:
                scored.append((d, e))
    scored.sort(key=lambda x: (x[0], x[1]["origin"], x[1]["series_id"]))
    if retention_info:
        retention_info["retention_receipt"] = retained.receipt(query, names, memory["k"], providers, scored=scored)
    nearest = _neighbours(scored, memory["k"], memory.get("dedupe_seconds", 0))
    if not nearest:
        return empty
    distances = sorted(d for d, _ in nearest)
    bandwidth = distances[len(distances) // 2] or 1e-9
    series_scale = {}
    for e in episodes:
        series_scale.setdefault(e["series_id"], []).append(e["losses"][baseline])
    series_scale = {s: sum(v) / len(v) for s, v in series_scale.items()}
    half_life = memory["recency_half_life_days"]
    now = datetime.fromisoformat(origin_time) if origin_time and half_life else None
    weighted, weights, neighbours, used = {p: 0.0 for p in providers}, [], [], []
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
        used.append((w, scale, e))
        neighbours.append({"series_id": e["series_id"], "origin": e["origin"], "distance": round(d, 4),
                           "weight": round(w, 4), "best_provider": min(providers, key=lambda p: e["losses"][p]),
                           "features": {name: e["features"].get(name) for name in names},
                           "losses": {p: e["losses"][p] for p in providers},
                           "baseline_loss_scale": scale,
                           "normalized_losses": {p: e["losses"][p] / scale for p in providers}})
    total = sum(weights)
    if total <= 0 or weighted[baseline] <= 0:
        return empty
    effective_n = total ** 2 / sum(w * w for w in weights)
    scores = {p: weighted[p] / weighted[baseline] for p in providers}
    neighbours.sort(key=lambda x: -x["weight"])
    out = {"scores": scores, "effective_n": effective_n, "neighbours": neighbours, **retention_info}
    shrink, z = memory.get("shrinkage", 0), memory.get("confidence_z", 0)
    diagnostics = memory.get("diagnostics", False)
    if not (shrink or z or diagnostics or memory.get("novelty_threshold")):
        return out  # default path: exactly the original scores
    # Ratio estimator standard error (delta method): score = A/B with A = sum w*a, B = sum w*b.
    base_sum = weighted[baseline]
    standard_errors = {}
    for p in providers:
        resid = sum((w * (e["losses"][p] / scale - scores[p] * e["losses"][baseline] / scale)) ** 2
                    for w, scale, e in used)
        standard_errors[p] = math.sqrt(resid) / base_sum
    raw = dict(scores)
    if shrink:
        factor = effective_n / (effective_n + shrink)
        scores = {p: (s if p == baseline else 1.0 + (s - 1.0) * factor) for p, s in scores.items()}
        standard_errors = {p: se * factor for p, se in standard_errors.items()}
        out["scores"] = scores
    if z:
        out["selection_scores"] = {p: (s if p == baseline else s + z * standard_errors[p]) for p, s in scores.items()}
    if diagnostics or memory.get("novelty_threshold"):
        if accelerate and bounded:
            scales = fase_memory.scales([e["features"] for e in episodes], names)
        out["novelty"] = _novelty(query, [e["features"] for e in episodes], names, scales, memory["k"], distance=distance)
    if diagnostics:
        out["diagnostics"] = {
            "unshrunk_scores": {p: round(v, 6) for p, v in raw.items()},
            "standard_errors": {p: round(v, 6) for p, v in standard_errors.items()},
            "interval90": {p: [round(scores[p] - 1.645 * standard_errors[p], 6),
                               round(scores[p] + 1.645 * standard_errors[p], 6)] for p in providers},
            "loss_ratio_quantiles": {
                p: {q: round(_weighted_quantile([(e["losses"][p] / e["losses"][baseline], w) for w, _, e in used
                                                 if e["losses"][baseline] > 0], q), 6)
                    for q in (0.1, 0.5, 0.9)}
                for p in providers if any(e["losses"][baseline] > 0 for _, _, e in used)},
            "novelty": out["novelty"]}
    return out
