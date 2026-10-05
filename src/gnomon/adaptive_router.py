"""Cost-aware adaptive routing over a user-defined set of forecasting providers.

A router serves one provider per forecast and learns which to serve from the
ledger's matched, prospectively recorded history (``compare_history``): each
provider's recent error relative to an explicit baseline, plus declared per-call
dollar cost and latency penalties, under optional hard limits. Shadow runs of the
other providers keep evidence accumulating. Every routed forecast records a
decision explaining the choice. The selection rule is a pure function shared by
live routing and offline replay, so a replayed evaluation tests the same logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from time import monotonic

from .episodic_memory import (FEATURES as MEMORY_FEATURES, PROFILES as MEMORY_PROFILES, compute_features,
                              episodic_evidence, request_features, validate_memory)
from .forecast_adapter import ForecastAdapterError

ROUTE_KIND = "adaptive_route/1"
_METRICS = ("mae", "rmsle")
_IDENTITY_POLICIES = ("attested", "prospective_unattested")

ROUTER_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["candidates", "baseline"],
    "properties": {
        "candidates": {"type": "array", "minItems": 1, "maxItems": 7, "items": {"type": "string"}},
        "baseline": {"type": "string"},
        "metric": {"enum": list(_METRICS), "default": "mae"},
        "recent_origins": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 20},
        "min_origins": {"type": "integer", "minimum": 1, "default": 5},
        "min_improvement": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.02},
        "lookback_seconds": {"type": "integer", "minimum": 1, "default": 604800},
        "shadow_every": {"type": "integer", "minimum": 0, "default": 1},
        "switch_penalty": {"type": "number", "minimum": 0, "default": 0,
                           "description": "Added to the utility of every provider except the one this router last "
                                          "served for the series, so it switches only when the evidence clears "
                                          "min_improvement + switch_penalty. 0 = no hysteresis."},
        "identity_policy": {"enum": list(_IDENTITY_POLICIES), "default": "attested"},
        "costs": {"type": "object", "additionalProperties": {
            "type": "object", "additionalProperties": False,
            "properties": {"usd_per_call": {"type": "number", "minimum": 0},
                           "latency_seconds": {"type": "number", "minimum": 0}}}},
        "cost_weights": {"type": "object", "additionalProperties": False,
                         "properties": {"usd": {"type": "number", "minimum": 0},
                                        "latency_seconds": {"type": "number", "minimum": 0}}},
        "limits": {"type": "object", "additionalProperties": False,
                   "properties": {"max_usd_per_call": {"type": "number", "minimum": 0},
                                  "max_latency_seconds": {"type": "number", "minimum": 0}}},
        "pool": {"type": "object", "additionalProperties": False, "required": ["series"],
                 "properties": {"series": {"type": "array", "minItems": 1, "maxItems": 128, "items": {"type": "string"}},
                                "own_weight": {"type": "number", "minimum": 1, "default": 1}},
                 "description": "Also learn from these series (same unit and horizon). Each series is scored relative "
                                "to its own baseline, then combined weighted by matched origins; the forecast series' "
                                "origins are multiplied by own_weight."},
        "context": {"type": "object", "additionalProperties": False, "required": ["feature", "thresholds"],
                    "properties": {"feature": {"enum": ["volatility_ratio", "trend"]},
                                   "on": {"enum": ["values", "differences"], "default": "differences"},
                                   "short_window": {"type": "integer", "minimum": 2, "default": 12},
                                   "long_window": {"type": "integer", "minimum": 3, "default": 96},
                                   "thresholds": {"type": "array", "minItems": 1, "maxItems": 5,
                                                  "items": {"type": "number"}}},
                    "description": "Condition evidence on a regime computed only from each forecast's own history: "
                                   "volatility_ratio = std(short)/std(long); trend = mean(short)/std(long). "
                                   "Ascending thresholds bin it into labels. Same-regime evidence is used when it "
                                   "has min_origins matched origins, otherwise all evidence."},
        "memory": {"type": "object", "additionalProperties": False,
                   "properties": {"distance": {"enum": ["robust", "fase"], "default": "robust"},
                                  "retention": {"enum": ["window", "fase"], "default": "window"},
                                  "recent_capacity": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 100},
                                  "long_term_capacity": {"type": "integer", "minimum": 0, "maximum": 10000, "default": 900},
                                  "features": {"type": "array", "items": {"enum": list(MEMORY_FEATURES)}},
                                  "short_window": {"type": "integer", "minimum": 2, "default": 14},
                                  "long_window": {"type": "integer", "minimum": 3, "default": 112},
                                  "season": {"type": "integer", "minimum": 1, "default": 1},
                                  "k": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 32},
                                  "min_effective_n": {"type": "number", "minimum": 1, "default": 8},
                                  "own_weight": {"type": "number", "minimum": 1, "default": 1},
                                  "recency_half_life_days": {"type": "number", "minimum": 0, "default": 0},
                                  "mask_covariate": {"type": "string"}, "future_covariate": {"type": "string"},
                                  "profile": {"enum": list(MEMORY_PROFILES),
                                              "description": "Feature preset instead of `features`: levels "
                                                             "(default set), returns, intermittent."},
                                  "context": {"type": "array", "maxItems": 16, "items": {"type": "string"},
                                              "description": "Past-covariate names whose latest value at the "
                                                             "origin joins the distance (e.g. a market-wide "
                                                             "state, a promotion flag, a peer aggregate)."},
                                  "dedupe_seconds": {"type": "number", "minimum": 0, "default": 0,
                                                     "description": "At most one neighbour per series within "
                                                                    "this window (set about the horizon)."},
                                  "shrinkage": {"type": "number", "minimum": 0, "default": 0,
                                                "description": "Pull candidate scores toward 1.0 by "
                                                               "effective_n/(effective_n+shrinkage)."},
                                  "confidence_z": {"type": "number", "minimum": 0, "default": 0,
                                                   "description": "Select on score + z * standard error."},
                                  "novelty_threshold": {"type": "number", "exclusiveMinimum": 0,
                                                        "description": "Abstain from memory when the state's "
                                                                       "neighbour distance exceeds this multiple "
                                                                       "of the typical one."},
                                  "diagnostics": {"type": "boolean", "default": False,
                                                  "description": "Report standard errors, intervals, neighbour "
                                                                 "loss quantiles and novelty."}},
                   "description": "Episodic memory: describe each forecast by features of its own request (recorded "
                                  "at routing time), retrieve the k most similar matured past origins from this "
                                  "series and its pool, and score providers by similarity-weighted losses relative "
                                  "to the baseline. Used when the effective sample size reaches min_effective_n; "
                                  "otherwise context, then all evidence."},
    },
    "description": (
        "Serve one of `candidates` or `baseline` per forecast, chosen from matched recent ledger evidence. "
        "utility = provider_score / baseline_score + cost_weights.usd * usd_per_call "
        "+ cost_weights.latency_seconds * latency_seconds; lower is better. A candidate is served only if its "
        "utility beats the baseline's by at least min_improvement, it meets `limits`, and at least min_origins "
        "matched origins exist. shadow_every=k runs every provider on origins whose hash is divisible by k "
        "(1 = always, 0 = never; evidence stops accruing). identity_policy=prospective_unattested admits "
        "providers without an attested revision (e.g. Ephemeris), only from forecasts recorded before their "
        "first target, and discloses it."),
}


def _number(value, field, *, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (upper is not None and value > upper):
        raise ForecastAdapterError(f"{field} must be a finite number >= 0" + (f" and <= {upper}" if upper is not None else ""),
                                   details={"rejected_fields": [field]})
    return float(value)


def _integer(value, field, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ForecastAdapterError(f"{field} must be an integer >= {minimum}" + (f" and <= {maximum}" if maximum else ""),
                                   details={"rejected_fields": [field]})
    return value


def validate_policy(policy: dict, *, replay: bool = False) -> dict:
    """Normalise a policy; offline replay permits 32 providers, live routing eight."""
    if not isinstance(policy, dict):
        raise ForecastAdapterError("router policy must be an object")
    unknown = set(policy) - set(ROUTER_SCHEMA["properties"])
    if unknown:
        raise ForecastAdapterError(f"unknown router fields: {sorted(unknown)}", details={"rejected_fields": sorted(unknown)})
    candidates, baseline = policy.get("candidates"), policy.get("baseline")
    max_candidates = 31 if replay else 7
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= max_candidates or not isinstance(baseline, str) or not baseline:
        raise ForecastAdapterError(f"router requires 1-{max_candidates} candidates and an explicit baseline",
                                   details={"rejected_fields": ["candidates", "baseline"]})
    providers = [baseline, *candidates]
    if any(not isinstance(p, str) or not p for p in providers) or len(set(providers)) != len(providers):
        raise ForecastAdapterError("router providers must be distinct nonempty names, excluding the baseline from candidates",
                                   details={"rejected_fields": ["candidates"]})
    metric = policy.get("metric", "mae")
    if metric not in _METRICS:
        raise ForecastAdapterError("metric must be mae or rmsle", details={"rejected_fields": ["metric"]})
    identity_policy = policy.get("identity_policy", "attested")
    if identity_policy not in _IDENTITY_POLICIES:
        raise ForecastAdapterError("identity_policy must be attested or prospective_unattested",
                                   details={"rejected_fields": ["identity_policy"]})
    costs = {}
    for name, cost in (policy.get("costs") or {}).items():
        if name not in providers:
            raise ForecastAdapterError(f"costs names a provider outside the router: {name}", details={"rejected_fields": ["costs"]})
        if not isinstance(cost, dict) or set(cost) - {"usd_per_call", "latency_seconds"}:
            raise ForecastAdapterError("each cost allows only usd_per_call and latency_seconds", details={"rejected_fields": ["costs"]})
        costs[name] = {k: _number(v, f"costs.{name}.{k}") for k, v in cost.items()}
    weights = policy.get("cost_weights") or {}
    if not isinstance(weights, dict) or set(weights) - {"usd", "latency_seconds"}:
        raise ForecastAdapterError("cost_weights allows only usd and latency_seconds", details={"rejected_fields": ["cost_weights"]})
    limits = policy.get("limits") or {}
    if not isinstance(limits, dict) or set(limits) - {"max_usd_per_call", "max_latency_seconds"}:
        raise ForecastAdapterError("limits allows only max_usd_per_call and max_latency_seconds", details={"rejected_fields": ["limits"]})
    normalised = {
        "candidates": list(candidates), "baseline": baseline, "metric": metric,
        "recent_origins": _integer(policy.get("recent_origins", 20), "recent_origins", 1, 1000),
        "min_origins": _integer(policy.get("min_origins", 5), "min_origins", 1),
        "min_improvement": _number(policy.get("min_improvement", 0.02), "min_improvement", upper=1),
        "lookback_seconds": _integer(policy.get("lookback_seconds", 604800), "lookback_seconds", 1),
        "shadow_every": _integer(policy.get("shadow_every", 1), "shadow_every", 0),
        "identity_policy": identity_policy, "costs": costs,
        **({"switch_penalty": _number(policy["switch_penalty"], "switch_penalty")}
           if policy.get("switch_penalty") else {}),
        "cost_weights": {k: _number(v, f"cost_weights.{k}") for k, v in weights.items()},
        "limits": {k: _number(v, f"limits.{k}") for k, v in limits.items()},
    }
    normalised["pool"] = _validate_pool(policy.get("pool"))
    normalised["context"] = _validate_context(policy.get("context"))
    normalised["memory"] = validate_memory(policy.get("memory"))
    if normalised["min_origins"] > normalised["recent_origins"]:
        raise ForecastAdapterError("min_origins cannot exceed recent_origins", details={"rejected_fields": ["min_origins"]})
    violation = _limit_violation(baseline, normalised)
    if violation:
        raise ForecastAdapterError(f"the baseline itself violates {violation}; choose a baseline within limits",
                                   details={"rejected_fields": ["baseline", "limits"]})
    return normalised


def _validate_pool(pool):
    if pool is None:
        return None
    if not isinstance(pool, dict) or set(pool) - {"series", "own_weight"} or not isinstance(pool.get("series"), list) \
            or not 1 <= len(pool["series"]) <= 128 or any(not isinstance(x, str) or not x for x in pool["series"]) \
            or len(set(pool["series"])) != len(pool["series"]):
        raise ForecastAdapterError("pool requires 1-128 distinct series names and optional own_weight",
                                   details={"rejected_fields": ["pool"]})
    own_weight = _number(pool.get("own_weight", 1.0), "pool.own_weight")
    if own_weight < 1:
        raise ForecastAdapterError("pool.own_weight must be >= 1", details={"rejected_fields": ["pool.own_weight"]})
    return {"series": list(pool["series"]), "own_weight": own_weight}


def _validate_context(context):
    if context is None:
        return None
    allowed = {"feature", "on", "short_window", "long_window", "thresholds"}
    if not isinstance(context, dict) or set(context) - allowed:
        raise ForecastAdapterError(f"context allows only {sorted(allowed)}", details={"rejected_fields": ["context"]})
    if context.get("feature") not in ("volatility_ratio", "trend"):
        raise ForecastAdapterError("context.feature must be volatility_ratio or trend",
                                   details={"rejected_fields": ["context.feature"]})
    on = context.get("on", "differences")
    if on not in ("values", "differences"):
        raise ForecastAdapterError("context.on must be values or differences", details={"rejected_fields": ["context.on"]})
    short = _integer(context.get("short_window", 12), "context.short_window", 2)
    long = _integer(context.get("long_window", 96), "context.long_window", 3)
    thresholds = context.get("thresholds")
    if not isinstance(thresholds, list) or not 1 <= len(thresholds) <= 5 or any(
            type(t) not in (int, float) or not math.isfinite(t) for t in thresholds) or thresholds != sorted(thresholds) \
            or len(set(thresholds)) != len(thresholds):
        raise ForecastAdapterError("context.thresholds must be 1-5 strictly ascending finite numbers",
                                   details={"rejected_fields": ["context.thresholds"]})
    if long <= short:
        raise ForecastAdapterError("context.long_window must exceed short_window",
                                   details={"rejected_fields": ["context.long_window"]})
    spec = {"feature": context["feature"], "on": on, "short_window": short, "long_window": long,
            "thresholds": [float(t) for t in thresholds]}
    spec["spec_id"] = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    return spec


def context_label(history, spec):
    """Regime label from pre-origin history only, or None when history is too short or flat."""
    if spec is None:
        return None
    values = [float(v) for v in history if v is not None and math.isfinite(float(v))]
    if spec["on"] == "differences":
        values = [b - a for a, b in zip(values, values[1:])]
    if len(values) < spec["long_window"]:
        return None
    short, long = values[-spec["short_window"]:], values[-spec["long_window"]:]

    def std(xs):
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    scale = std(long)
    if scale <= 0:
        return None
    value = std(short) / scale if spec["feature"] == "volatility_ratio" else (sum(short) / len(short)) / scale
    return f"{spec['feature']}:bin{sum(1 for t in spec['thresholds'] if value >= t)}"


def pooled_scores(rows, policy, own_series, label=None):
    """Scores relative to the baseline (baseline = 1.0) from matured, fully matched evidence rows.

    rows: [{"series_id", "origin", "label", "losses": {provider: per-origin metric}}]. Each series
    contributes its latest recent_origins rows (matching `label` when given); series are combined
    weighted by row count, with the forecast series multiplied by pool.own_weight.
    """
    providers, baseline = [policy["baseline"], *policy["candidates"]], policy["baseline"]
    pool = policy.get("pool")
    allowed = {own_series, *(pool["series"] if pool else [])}
    by_series = {}
    for row in sorted(rows, key=lambda r: r["origin"]):
        if row["series_id"] in allowed and (label is None or row.get("label") == label) \
                and all(p in row["losses"] for p in providers):
            by_series.setdefault(row["series_id"], []).append(row)
    weighted, total_weight, matched, used = {p: 0.0 for p in providers}, 0.0, 0, []
    for series, series_rows in by_series.items():
        recent = series_rows[-policy["recent_origins"]:]
        base = sum(r["losses"][baseline] for r in recent) / len(recent)
        if base <= 0:
            continue
        weight = len(recent) * (pool["own_weight"] if pool and series == own_series else 1.0)
        for p in providers:
            weighted[p] += weight * (sum(r["losses"][p] for r in recent) / len(recent)) / base
        total_weight += weight
        matched += len(recent)
        used.append(series)
    scores = {p: v / total_weight for p, v in weighted.items()} if total_weight else {}
    return scores, matched, sorted(used)


def choose_from_rows(rows, policy, own_series, label, features=None, origin=None, incumbent=None, scorer=None):
    """Most specific evidence first: similar episodes (memory), same regime, then all evidence.

    `incumbent` is the provider this router last served for the series; it only matters
    when the policy sets switch_penalty. `scorer(own_series, features, origin)` replaces the
    memory scorer (replay acceleration); it must return what `episodic_evidence` returns.
    """
    abstained = None
    if policy.get("memory") and features is not None:
        memory = policy["memory"]
        evidence = scorer(own_series, features, origin) if scorer else \
            episodic_evidence(rows, policy, own_series, features, origin)
        scores, effective_n, neighbours = evidence["scores"], evidence["effective_n"], evidence["neighbours"]
        novelty, threshold = evidence.get("novelty"), memory.get("novelty_threshold")
        if scores and threshold and novelty is not None and novelty > threshold:
            abstained = {"reason": "unfamiliar_state", "novelty": novelty, "novelty_threshold": threshold}
        elif scores and effective_n >= memory["min_effective_n"]:
            decision = select_provider(evidence.get("selection_scores", scores),
                                       max(len(neighbours), policy["min_origins"]), policy, incumbent)
            decision.update(evidence_level="memory", context_label=label,
                            series_used=sorted({n["series_id"] for n in neighbours}),
                            effective_n=round(effective_n, 2), neighbours=neighbours[:5])
            if "retention_receipt" in evidence:
                decision["retention_receipt"] = evidence["retention_receipt"]
                decision["retention_pool"] = evidence["retention_pool"]
            if "selection_scores" in evidence:
                decision["memory_selection"] = "score_plus_z_standard_error"
            if "diagnostics" in evidence:
                decision["memory_diagnostics"] = evidence["diagnostics"]
            elif novelty is not None:
                decision["memory_novelty"] = novelty
            return decision
    levels = ([("context", label)] if policy.get("context") and label is not None else []) + [("all", None)]
    for level, value in levels:
        scores, matched, used = pooled_scores(rows, policy, own_series, value)
        if matched >= policy["min_origins"] or level == "all":
            decision = select_provider(scores, matched, policy, incumbent)
            decision.update(evidence_level=level, context_label=label, series_used=used)
            if abstained:
                decision["memory_abstained"] = abstained
            return decision


def policy_revision(policy: dict, provider_revisions: dict) -> str:
    """Stable identity for a router: its normalised policy plus every provider's revision."""
    body = json.dumps({"policy": policy, "revisions": provider_revisions}, sort_keys=True, separators=(",", ":"))
    return "router/" + sha256(body.encode()).hexdigest()[:16]


def _limit_violation(provider, policy):
    cost, limits = policy["costs"].get(provider, {}), policy["limits"]
    if "max_usd_per_call" in limits and cost.get("usd_per_call", 0.0) > limits["max_usd_per_call"]:
        return "max_usd_per_call"
    if "max_latency_seconds" in limits and cost.get("latency_seconds", 0.0) > limits["max_latency_seconds"]:
        return "max_latency_seconds"
    return None


def select_provider(scores: dict, matched_origins: int, policy: dict, incumbent: str | None = None) -> dict:
    """Pure selection rule. `scores` maps provider -> mean metric over the same matched origins.

    With policy switch_penalty > 0 and a known incumbent (the provider last served for this
    series), every other provider's utility is raised by switch_penalty (hysteresis).
    """
    baseline, providers = policy["baseline"], [policy["baseline"], *policy["candidates"]]
    weights = policy["cost_weights"]
    table = []
    for p in providers:
        cost = policy["costs"].get(p, {})
        penalty = weights.get("usd", 0.0) * cost.get("usd_per_call", 0.0) + \
            weights.get("latency_seconds", 0.0) * cost.get("latency_seconds", 0.0)
        table.append({"provider": p, "score": scores.get(p), "cost_penalty": penalty,
                      "usd_per_call": cost.get("usd_per_call"), "latency_seconds": cost.get("latency_seconds"),
                      "excluded": _limit_violation(p, policy) or (None if p in scores else "no_matched_evidence")})

    def choice(reason, provider=baseline):
        out = {"provider": provider, "reason": reason, "evidence_based": reason.startswith("evidence"),
               "matched_origins": matched_origins, "table": table}
        if policy.get("switch_penalty") and incumbent is not None:
            out["incumbent"] = incumbent
        return out

    if matched_origins < policy["min_origins"] or scores.get(baseline) is None:
        return choice("insufficient_evidence")
    base_score = scores[baseline]
    if base_score <= 0:
        return choice("baseline_error_zero")
    penalty = policy.get("switch_penalty", 0.0) if incumbent is not None else 0.0
    for row in table:
        if row["score"] is not None:
            row["relative_score"] = row["score"] / base_score
            row["utility"] = row["relative_score"] + row["cost_penalty"]
            if penalty and row["provider"] != incumbent:
                row["switch_penalty"] = penalty
                row["utility"] += penalty
    base_utility = table[0]["utility"]
    eligible = [r for r in table[1:] if r["excluded"] is None and "utility" in r]
    best = min(eligible, key=lambda r: r["utility"], default=None)
    if best is None:
        return choice("evidence_no_eligible_candidate")
    if best["utility"] <= base_utility - policy["min_improvement"]:
        return choice("evidence_and_cost_favour_candidate", best["provider"])
    return choice("evidence_improvement_below_threshold")


def runs_shadow(series_id: str, origin: str, shadow_every: int) -> bool:
    """Deterministic, restart-safe shadow sampling keyed by series and origin."""
    if shadow_every == 0:
        return False
    if shadow_every == 1:
        return True
    digest = sha256(f"{series_id}|{origin}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % shadow_every == 0


def _evidence_start(policy, origin_time, timestamps):
    """Window start: enough recent steps for recent_origins, capped by lookback_seconds.

    Slack covers shadow sampling (1 in shadow_every origins is fully matched) and regime
    conditioning (roughly 1 in len(thresholds)+1 origins shares the current label).
    """
    from .ledger import _time
    lookback = timedelta(seconds=policy["lookback_seconds"])
    if policy.get("memory"):
        return origin_time - lookback  # memory retrieves from the whole lookback, not just recent origins
    recent = [datetime.fromisoformat(_time(t)) for t in timestamps[-11:]]
    gaps = sorted((b - a) for a, b in zip(recent, recent[1:]) if b > a)
    if gaps:
        regimes = len(policy["context"]["thresholds"]) + 1 if policy.get("context") else 1
        steps = policy["recent_origins"] * 3 * max(1, policy["shadow_every"]) * regimes
        lookback = min(lookback, gaps[len(gaps) // 2] * steps)
    return origin_time - lookback


def _compare_window(ledger, policy, revisions, series_id, unit, horizon, start_time, end_time, source_as_of,
                    recorded_as_of, depth=0):
    """compare_history over [start, end]; past the ledger's 1000-execution cap, read both halves.

    Each origin is scored independently, so disjoint origin windows combine exactly.
    """
    from .ledger_history import compare_history
    providers = [policy["baseline"], *policy["candidates"]]
    start_time = min(start_time, end_time)
    try:
        return compare_history(ledger, series_id=series_id, horizon=horizon, unit=unit,
                               providers={p: revisions[p] for p in providers},
                               start=start_time.isoformat(), end=end_time.isoformat(),
                               source_as_of=source_as_of, recorded_as_of=recorded_as_of, metric=policy["metric"],
                               recent_origins=policy["recent_origins"], identity_policy=policy["identity_policy"])
    except ForecastAdapterError as error:
        if "exceeds 1000 executions" not in str(error) or depth >= 12 or end_time - start_time < timedelta(seconds=2):
            raise
    middle = start_time + (end_time - start_time) / 2
    older = _compare_window(ledger, policy, revisions, series_id, unit, horizon, start_time, middle,
                            source_as_of, recorded_as_of, depth + 1)
    newer = _compare_window(ledger, policy, revisions, series_id, unit, horizon, middle + timedelta(microseconds=1),
                            end_time, source_as_of, recorded_as_of, depth + 1)
    incompatible = next((a for a in (older, newer) if a.get("status") == "incompatible_evidence"), None)
    return {**(incompatible or newer), "origins": older.get("origins", []) + newer.get("origins", []),
            "excluded": older.get("excluded", []) + newer.get("excluded", []),
            "unattested_providers": sorted(set(older.get("unattested_providers", []))
                                           | set(newer.get("unattested_providers", [])))}


def _series_rows(ledger, policy, revisions, series_id, unit, horizon, start_time, end_time, source_as_of,
                 recorded_as_of):
    """Per-origin matched losses for one series."""
    answer = _compare_window(ledger, policy, revisions, series_id, unit, horizon, start_time, end_time,
                             source_as_of, recorded_as_of)
    if answer.get("status") == "incompatible_evidence":
        # The ledger refused to rank this cohort (task shape or provider identity changed in the
        # window); its per-origin rows must not be ranked here either.
        raise ForecastAdapterError(f"evidence for {series_id} is incompatible: {answer.get('reason')}",
                                   details={"reason": "incompatible_evidence", "series_id": series_id,
                                            "ledger_reason": answer.get("reason")})
    rows = [{"series_id": series_id, "origin": o["origin"],
             "losses": {m["provider"]: m[policy["metric"]] for m in o["models"] if m.get(policy["metric"]) is not None}}
            for o in answer.get("origins", [])]
    if (policy.get("memory") or {}).get("retention") == "fase":
        by_origin = {o["origin"]: o for o in answer.get("origins", [])}
        with ledger._connect() as conn:
            for row in rows:
                times = []
                for actual_id in by_origin[row["origin"]].get("actual_ids", []):
                    actual = conn.execute("SELECT valid_time, source_available_at, recorded_at FROM actuals WHERE actual_id=?",
                                          (actual_id,)).fetchone()
                    if actual:
                        times.extend(actual)
                row["completed_at"] = max(times) if times else row["origin"]
    return rows, answer


def _recorded_inputs(ledger, series_ids, context_spec_id, memory_spec_id, since, now):
    """Regime labels and memory features this router recorded at routing time, by (series, origin).

    A routing decision is recorded at or after its origin, so decisions recorded before the
    evidence window starts cannot describe an origin inside it; the decisions index bounds the
    read. Only values recorded under the current spec are returned: old forecasts are never
    reinterpreted with a newer rule.
    """
    from .ledger import _time
    labels, features, since = {}, {}, _time(since)
    with ledger._connect() as conn:
        for sid in series_ids:
            for origin, label, label_spec, vector, vector_spec in conn.execute(
                    "SELECT json_extract(payload_json, '$.inputs.origin'), "
                    "json_extract(payload_json, '$.inputs.context_label'), "
                    "json_extract(payload_json, '$.inputs.context_spec_id'), "
                    "json_extract(payload_json, '$.inputs.memory_features'), "
                    "json_extract(payload_json, '$.inputs.memory_spec_id') "
                    "FROM decisions WHERE json_extract(payload_json, '$.inputs.kind') = ? AND "
                    "json_extract(payload_json, '$.inputs.series_id') = ? AND recorded_at BETWEEN ? AND ?",
                    (ROUTE_KIND, sid, since, _time(now))):
                key = (sid, _time(origin))
                if label is not None and context_spec_id is not None and label_spec == context_spec_id:
                    labels[key] = label
                if vector is not None and memory_spec_id is not None and vector_spec == memory_spec_id:
                    features.setdefault(key, json.loads(vector))  # first recorded description of that origin
    return labels, features


def _recorded_retention(ledger, rows, policy, now):
    """Attach immutable decision-time receipts; never reconstruct past retrievals."""
    from .ledger import _time
    by_key = {(r["series_id"], r["origin"]): r for r in rows}
    with ledger._connect() as conn:
        records = conn.execute("SELECT payload_json FROM decisions WHERE "
                               "json_extract(payload_json, '$.inputs.kind')=? AND recorded_at<=? "
                               "ORDER BY recorded_at, decision_id", (ROUTE_KIND, _time(now)))
        seen = set()
        for record in records:
            payload = json.loads(record[0])
            inputs = payload["inputs"]
            key = (inputs.get("series_id"), inputs.get("origin"))
            if key not in by_key or key in seen or inputs.get("memory_spec_id") != policy["memory"]["spec_id"]:
                continue
            # A receipt belongs to its complete memory policy, not just its feature schema.
            if inputs.get("retention_policy") != policy:
                continue
            row = by_key[key]
            row["served_provider"] = payload.get("action", {}).get("served_provider")
            row["retention_receipt"] = inputs.get("selection", {}).get("retention_receipt", [])
            seen.add(key)


def last_served(ledger, router: str, series_id: str, before) -> str | None:
    """The provider this router last served for the series, from its recorded decisions."""
    from .ledger import _time
    with ledger._connect() as conn:
        row = conn.execute(
            "SELECT json_extract(payload_json, '$.action.served_provider') FROM decisions "
            "WHERE json_extract(payload_json, '$.inputs.kind') = ? AND json_extract(payload_json, '$.policy.router') = ? "
            "AND json_extract(payload_json, '$.inputs.series_id') = ? AND recorded_at <= ? "
            "ORDER BY recorded_at DESC LIMIT 1", (ROUTE_KIND, router, series_id, _time(before))).fetchone()
    return row[0] if row else None


def ledger_evidence(ledger, policy: dict, revisions: dict, *, series_id, unit, horizon, origin, now,
                    timestamps=(), source_as_of=None) -> dict:
    """Matched prospective evidence visible now, for the forecast series and its pool.

    Outcomes count only if available by `source_as_of` (default `now`); routing passes the
    request's point-in-time cutoff so a historical request never sees later outcomes.
    Bounded windows keep each call at O(recent_origins) rows per series. Rows are labelled
    with the regime this router recorded when it routed each origin, so conditioning never
    reinterprets old forecasts with a newer rule.
    """
    from .ledger import _time
    origin_time = datetime.fromisoformat(_time(origin))
    source_as_of = _time(now if source_as_of is None else min(_time(source_as_of), _time(now)))
    end_time = min(origin_time, datetime.fromisoformat(source_as_of))
    start_time = _evidence_start(policy, origin_time, timestamps)
    if (policy.get("memory") or {}).get("retention") == "fase":
        with ledger._connect() as conn:
            first = conn.execute("SELECT MIN(json_extract(payload_json, '$.inputs.origin')) FROM decisions "
                                 "WHERE json_extract(payload_json, '$.inputs.kind') = ?", (ROUTE_KIND,)).fetchone()[0]
        if first:
            start_time = min(start_time, datetime.fromisoformat(_time(first)))
    series = [series_id, *[s for s in (policy["pool"]["series"] if policy.get("pool") else []) if s != series_id]]
    rows, excluded, unattested, failed = [], [], set(), []
    for sid in series:
        try:
            series_rows, answer = _series_rows(ledger, policy, revisions, sid, unit, horizon, start_time, end_time,
                                               source_as_of, now)
        except ForecastAdapterError as error:
            if sid == series_id:
                raise
            failed.append({"series_id": sid, "error": str(error)[:200]})  # A pool member must not block routing.
            continue
        rows += series_rows
        excluded += answer.get("excluded", [])[:5]
        unattested.update(answer.get("unattested_providers", []))
    if policy.get("context") or policy.get("memory"):
        labels, features = _recorded_inputs(ledger, series, (policy.get("context") or {}).get("spec_id"),
                                             (policy.get("memory") or {}).get("spec_id"),
                                             min(start_time, end_time).isoformat(), now)
        for row in rows:
            key = (row["series_id"], _time(row["origin"]))
            row["label"], row["features"] = labels.get(key), features.get(key)
    if (policy.get("memory") or {}).get("retention") == "fase":
        _recorded_retention(ledger, rows, policy, now)
    return {"rows": rows, "window": {"start": min(start_time, end_time).isoformat(), "end": end_time.isoformat(),
                                     "source_as_of": source_as_of},
            "excluded": excluded[:20], "unattested_providers": sorted(unattested), "pool_failures": failed,
            "status": "ok"}


def route_forecast(session, name: str, policy: dict, request: dict) -> dict:
    """Choose, serve and shadow; record the routing decision; return the served forecast."""
    ledger, engine = session.ledger, session.engine
    if ledger is None:
        raise ForecastAdapterError("adaptive routing requires a configured ledger")
    if not isinstance(request, dict):
        raise ForecastAdapterError("routed forecasts take a request object")
    for field in ("series_id", "timestamps", "future_timestamps"):
        if not request.get(field):
            raise ForecastAdapterError(f"routed forecasts require {field} so outcomes can be matched later",
                                       details={"rejected_fields": [field]})
    identities = engine.capabilities()
    providers = [policy["baseline"], *policy["candidates"]]
    missing = [p for p in providers if p not in identities]
    if missing:
        raise ForecastAdapterError(f"router {name} names unregistered providers: {missing}")
    revisions = {p: identities[p]["revision"] for p in providers}
    unversioned = [p for p, r in revisions.items() if r in (None, "latest", "unversioned")]
    if unversioned and policy["identity_policy"] != "prospective_unattested":
        raise ForecastAdapterError(
            f"providers without an attested revision cannot be ranked: {unversioned}. Set identity_policy = "
            "\"prospective_unattested\" to admit them from prospectively recorded forecasts, disclosed.",
            details={"rejected_fields": ["identity_policy"]})
    origin = request.get("cutoff") or request["timestamps"][-1]
    now = ledger._now()
    from .ledger import _time
    # Point in time: only outcomes knowable at the request's origin (or its explicit
    # known_time_cutoff) count, so repeating a historical request cannot use later outcomes.
    evidence_as_of = min(_time(request.get("known_time_cutoff") or origin), _time(now))
    label = context_label(request.get("history", []), policy.get("context"))
    features = request_features(policy["memory"], request) if policy.get("memory") else None
    try:
        evidence = ledger_evidence(ledger, policy, revisions, series_id=request["series_id"], unit=request.get("unit"),
                                   horizon=request.get("horizon", len(request["future_timestamps"])), origin=origin,
                                   now=now, timestamps=request["timestamps"], source_as_of=evidence_as_of)
        incumbent = last_served(ledger, name, request["series_id"], now) if policy.get("switch_penalty") else None
        decision = choose_from_rows(evidence["rows"], policy, request["series_id"], label, features, _time(origin),
                                    incumbent)
    except ForecastAdapterError as error:  # Serving must not fail because evidence is unreadable or refused.
        status = "evidence_incompatible" if error.details.get("reason") == "incompatible_evidence" \
            else "evidence_unavailable"
        evidence = {"rows": [], "window": None, "excluded": [], "unattested_providers": [], "pool_failures": [],
                    "status": status, "error": str(error)[:300]}
        decision = select_provider({}, 0, policy)
        decision.update(reason=status, evidence_level=None, context_label=label, series_used=[])
    evidence_summary = {k: v for k, v in evidence.items() if k != "rows"}
    evidence_summary["rows_read"] = len(evidence["rows"])
    served_name = decision["provider"]
    started = monotonic()
    served = session.forecast(served_name, request)
    latencies = {served_name: monotonic() - started}
    shadow_ids, shadow_errors = [], []
    if runs_shadow(request["series_id"], _time(origin), policy["shadow_every"]):
        for p in providers:
            if p == served_name:
                continue
            started = monotonic()
            try:
                shadow_ids.append(engine.forecast(p, request).execution_id)
            except Exception as error:  # A failed shadow must not fail the served forecast.
                shadow_errors.append({"provider": p, "error": type(error).__name__, "message": str(error)[:300]})
            latencies[p] = monotonic() - started
    router_revision = policy_revision(policy, revisions)
    decision_id = ledger.record_decision(
        execution_ids=[served["execution_id"], *shadow_ids],
        policy={"kind": ROUTE_KIND, "router": name, "router_revision": router_revision, "policy": policy},
        inputs={"kind": ROUTE_KIND, "series_id": request["series_id"], "origin": origin,
                "evidence_as_of": evidence_as_of, "recorded_as_of": now,
                "context_label": label, "context_spec_id": (policy.get("context") or {}).get("spec_id"),
                **({"retention_policy": policy} if (policy.get("memory") or {}).get("retention") == "fase" else {}),
                "memory_features": features, "memory_spec_id": (policy.get("memory") or {}).get("spec_id"),
                "evidence": evidence_summary, "selection": {k: v for k, v in decision.items() if k != "provider"},
                "provider_revisions": revisions, "measured_latency_seconds": latencies,
                "shadow_failures": shadow_errors},
        action={"served_provider": served_name, "served_execution_id": served["execution_id"],
                "shadow_execution_ids": shadow_ids})
    declared_usd = sum(policy["costs"].get(p, {}).get("usd_per_call", 0.0) for p in latencies)
    routing = {"router": name, "router_revision": router_revision, "served_provider": served_name,
               "reason": decision["reason"], "evidence_based": decision["evidence_based"],
               "matched_origins": decision["matched_origins"], "table": decision["table"],
               "evidence_level": decision["evidence_level"], "context_label": label,
               **({"effective_n": decision.get("effective_n"), "memory_neighbours": decision.get("neighbours", [])}
                  if policy.get("memory") else {}),
               **{k: decision[k] for k in ("memory_diagnostics", "memory_novelty", "memory_abstained",
                                           "memory_selection", "incumbent") if k in decision},
               "series_used": decision["series_used"], "pool_failures": evidence["pool_failures"],
               "evidence_window": evidence["window"], "evidence_as_of": evidence_as_of,
               "unattested_providers": evidence["unattested_providers"],
               **({"evidence_error": evidence["error"]} if "error" in evidence else {}),
               "shadow_execution_ids": shadow_ids, "shadow_failures": shadow_errors,
               "measured_latency_seconds": latencies, "declared_usd_this_call": declared_usd,
               "routing_decision_id": decision_id}
    return {**served, "routing": routing}


def replay_router(folds: list[dict], policy: dict, histories: dict | None = None, *,
                  return_decisions: bool = False, covariates: dict | None = None,
                  decision_losses: bool = False, accelerate: bool = False) -> dict:
    """Offline replay of the live selection rule over saved one-step folds, on one shared clock.

    Each fold: {"series_id", "origin": iso, "target_time": iso, "actual": float or list,
    "points": {provider: float or list}}. For multi-step folds, target_time is the last
    target (the origin matures when all its outcomes are known) and the loss is the
    per-origin metric over the horizon, as in live comparison. Folds from every series are processed in origin order. A fold
    becomes evidence only once its target_time <= the current origin, and only if it was
    shadowed. Context labels are computed at routing time from the series' history as known
    then, as live routing does: either `histories[series]` as a list of (time, value) pairs,
    of which values observed at or before each origin are used, or a plain list of values
    before the first fold, extended with each matured fold's first actual (one-step folds).
    Memory features are computed with the live function from the same known history (which
    must then be timestamped), plus `covariates[series][name]` as (time, value) pairs: the
    mask covariate up to the origin, the future covariate over (origin, target_time].

    Memory `context` covariates are read from the same `covariates` timelines (their latest
    value at or before the origin). With switch_penalty, the incumbent is the provider this
    replay last served for the series. `decision_losses=True` adds each decision's served and
    per-provider losses (with return_decisions). `accelerate=True` scores memory with numpy
    (episodic_memory_fast; needs numpy; dedupe, novelty and diagnostics not supported).

    Times must be timezone-aware and are compared as UTC instants. Scores are the mean
    per-fold loss (for one-step folds, the live per-origin metric averaged over origins), not
    an aggregate RMSLE over all folds. As live comparison does by default, a fold where any
    provider's RMSLE is undefined (negative prediction or actual) is excluded, not clipped.
    """
    import heapq
    from .evidence_summary import rmsle
    from .ledger import _time
    providers = [policy["baseline"], *policy["candidates"]]
    import bisect
    stamped, history = {}, {}
    for key, values in (histories or {}).items():
        values = list(values)
        if values and isinstance(values[0], (list, tuple)):
            pairs = sorted((_time(t), float(v)) for t, v in values)
            stamped[key] = ([t for t, _ in pairs], [v for _, v in pairs])
        else:
            history[key] = values

    def known_history(sid, origin):
        if sid in stamped:
            times, values = stamped[sid]
            return values[:bisect.bisect_right(times, origin)]
        return history.get(sid, [])

    def memory_features(sid, origin, target_time):
        spec = policy.get("memory")
        if spec is None or sid not in stamped:
            return None
        times, values = stamped[sid]
        cut = bisect.bisect_right(times, origin)
        timeline = (covariates or {}).get(sid, {})
        past_names = [spec["mask_covariate"]] if spec.get("mask_covariate") in timeline else []
        past_names += [c for c in spec.get("context", []) if c in timeline and c not in past_names]
        fase_features = any(n.startswith("fase:") for n in spec["features"])
        if fase_features:
            past_names += [n for n in timeline if n not in past_names and n != spec.get("future_covariate")]
        future_names = [spec["future_covariate"]] if spec.get("future_covariate") in timeline else []
        if past_names:
            mask = spec.get("mask_covariate")
            # Context features read only the latest value at the origin, so their column is that
            # value repeated; the mask keeps its per-step values (0 where unstamped).
            latest = {n: _latest(timeline[n], origin) for n in past_names if n != mask}
            past = [[timeline[n].get(t, 0.0) if n == mask else
                     timeline[n].get(t, float("nan")) if fase_features else latest[n]
                     for n in past_names] for t in times[:cut]]
        else:
            past = ()
        future = [[v] for t, v in sorted(timeline[future_names[0]].items()) if origin < t <= target_time] \
            if future_names else ()
        return compute_features(spec, values[:cut], times[:cut], past_covariates=past,
                                past_covariate_names=past_names, future_covariates=future,
                                future_covariate_names=future_names)

    covariates = {sid: {name: {_time(t): float(v) for t, v in pairs} for name, pairs in named.items()}
                  for sid, named in (covariates or {}).items()}
    sorted_times = {}

    def _latest(series_values, t):
        """A context covariate's latest value at or before t (NaN = unknown when none yet)."""
        if t in series_values:
            return series_values[t]
        key = id(series_values)
        if key not in sorted_times:
            sorted_times[key] = sorted(series_values)
        times_ = sorted_times[key]
        i = bisect.bisect_right(times_, t)
        return series_values[times_[i - 1]] if i else float("nan")

    def steps(value):
        return list(value) if isinstance(value, (list, tuple)) else [value]

    def loss(point, actual):
        pairs = list(zip(steps(point), steps(actual), strict=True))
        if policy["metric"] == "rmsle":
            return rmsle(pairs, "reject")[0]
        return sum(abs(p - a) for p, a in pairs) / len(pairs)

    folds = sorted(({**f, "origin": _time(f["origin"], "origin"), "target_time": _time(f["target_time"], "target_time")}
                    for f in folds), key=lambda f: (f["origin"], f.get("series_id", "")))
    excluded = {}
    pending, rows, serial = [], [], 0
    served_losses, per_series, counts = [], {}, {p: 0 for p in providers}
    levels, labels, usd, decisions = {}, {}, 0.0, []
    index = None
    if (policy.get("memory") or {}).get("retention") == "fase":
        if accelerate:
            from .episodic_memory_fast import check_supported
            check_supported(policy["memory"])
        from .fase_memory import ReplayIndex
        index = ReplayIndex(policy, accelerate=accelerate)
    elif accelerate and policy.get("memory"):
        from .episodic_memory_fast import MemoryIndex
        index = MemoryIndex(policy)
    last_served_by_series = {}
    for fold in folds:
        sid = fold.get("series_id", "")
        while pending and pending[0][0] <= fold["origin"]:
            _, _, matured = heapq.heappop(pending)
            if matured["series_id"] not in stamped:
                history.setdefault(matured["series_id"], []).append(steps(matured["actual"])[0])
            if matured["shadowed"]:
                rows.append(matured["row"])
                if index is not None:
                    index.add(matured["row"])
        oldest = _time(datetime.fromisoformat(fold["origin"]) - timedelta(seconds=policy["lookback_seconds"]))
        visible = rows if (policy.get("memory") or {}).get("retention") == "fase" else [r for r in rows if r["origin"] >= oldest]
        label = context_label(known_history(sid, fold["origin"]), policy.get("context"))
        labels[label] = labels.get(label, 0) + 1
        features = memory_features(sid, fold["origin"], fold["target_time"])
        scorer = (lambda own, q, origin, oldest=oldest: index.evidence(own, q, origin, oldest)) if index else None
        decision = choose_from_rows(visible, policy, sid, label, features, fold["origin"],
                                    last_served_by_series.get(sid) if policy.get("switch_penalty") else None, scorer)
        served = decision["provider"]
        last_served_by_series[sid] = served
        counts[served] += 1
        levels[decision["evidence_level"]] = levels.get(decision["evidence_level"], 0) + 1
        shadowed = runs_shadow(sid, fold["origin"], policy["shadow_every"])
        usd += sum(policy["costs"].get(p, {}).get("usd_per_call", 0.0) for p in (providers if shadowed else [served]))
        try:
            losses = {p: loss(fold["points"][p], fold["actual"]) for p in providers}
        except ForecastAdapterError as error:  # Unscoreable: neither evidence nor part of any score.
            excluded[str(error)] = excluded.get(str(error), 0) + 1
            serial += 1
            heapq.heappush(pending, (fold["target_time"], serial, {
                "series_id": sid, "actual": fold["actual"], "shadowed": False, "row": None}))
            continue
        served_losses.append(losses[served])
        if return_decisions:
            decisions.append({"series_id": sid, "origin": fold["origin"], "served": served, "shadowed": shadowed,
                              "reason": decision["reason"], "evidence_level": decision["evidence_level"],
                              "context_label": label,
                              **{k: decision[k] for k in ("memory_novelty", "memory_abstained", "memory_diagnostics",
                                                          "incumbent") if k in decision},
                              **({"served_loss": losses[served], "losses": losses} if decision_losses else {})})
        entry = per_series.setdefault(sid, {"router": [], **{p: [] for p in providers}})
        entry["router"].append(losses[served])
        for p in providers:
            entry[p].append(losses[p])
        serial += 1
        heapq.heappush(pending, (fold["target_time"], serial, {
            "series_id": sid, "actual": fold["actual"], "shadowed": shadowed,
            "row": {"series_id": sid, "origin": fold["origin"], "label": label, "features": features,
                    "losses": losses, "served_provider": served, "completed_at": fold["target_time"],
                    "retention_receipt": decision.get("retention_receipt", [])}}))

    def mean(values):
        return sum(values) / len(values) if values else None

    return {"folds": len(folds), "metric": policy["metric"], "aggregation": f"mean_{policy['metric']}_over_folds",
            "excluded_folds": excluded, "router_score": mean(served_losses),
            "fixed_provider_scores": {p: mean([x for e in per_series.values() for x in e[p]]) for p in providers},
            "per_series": {sid: {k: mean(v) for k, v in e.items()} for sid, e in per_series.items()},
            "served_counts": counts, "evidence_levels": levels,
            "context_labels": {k: v for k, v in labels.items() if k is not None},
            **({"decisions": decisions} if return_decisions else {}), "declared_usd_total": usd}


def memory_ablation(folds: list[dict], policy: dict, histories: dict | None = None, *,
                    covariates: dict | None = None, warmup_origins: int = 0, n_boot: int = 2000,
                    seed: int = 0, accelerate: bool = False) -> dict:
    """Does memory earn its place? Replay `policy` with and without its memory on the same folds.

    Both arms share everything else (pool, context, costs, switch_penalty). Compares per-fold
    losses of the served forecasts after the first `warmup_origins` distinct origins, with a
    bootstrap over origins (all series at an origin resampled together, since series that
    share a clock are not independent). Also reports the best fixed provider and how often
    each arm switched provider, so a memory router that only adds churn is visible.
    """
    import random
    if not policy.get("memory"):
        raise ForecastAdapterError("memory_ablation needs a policy with memory", details={"rejected_fields": ["memory"]})
    _integer(warmup_origins, "warmup_origins", 0)
    _integer(n_boot, "n_boot", 1)
    arms = {}
    for name, arm_policy in (("memory", policy), ("no_memory", {**policy, "memory": None})):
        arms[name] = replay_router(folds, arm_policy, histories, return_decisions=True, covariates=covariates,
                                   decision_losses=True, accelerate=accelerate and name == "memory")
    origins = sorted({d["origin"] for d in arms["memory"]["decisions"]})
    scored_from = origins[warmup_origins] if warmup_origins < len(origins) else None
    by_key = {name: {(d["series_id"], d["origin"]): d for d in arm["decisions"]} for name, arm in arms.items()}
    paired = [(k[1], m["served_loss"] - by_key["no_memory"][k]["served_loss"], m["served"] != by_key["no_memory"][k]["served"])
              for k, m in by_key["memory"].items()
              if k in by_key["no_memory"] and scored_from is not None and k[1] >= scored_from]

    def switch_rate(decisions):
        by_series = {}
        for d in sorted(decisions, key=lambda d: d["origin"]):
            by_series.setdefault(d["series_id"], []).append(d["served"])
        changes = sum(sum(a != b for a, b in zip(v, v[1:])) for v in by_series.values())
        steps = sum(max(len(v) - 1, 0) for v in by_series.values())
        return round(changes / steps, 4) if steps else None

    keys = [k for k in by_key['memory'] if k in by_key['no_memory']
            and scored_from is not None and k[1] >= scored_from]
    scored = {name: [rows[k] for k in keys] for name, rows in by_key.items()}
    def average(values):
        return sum(values) / len(values) if values else None
    fixed = {p: average([d['losses'][p] for d in scored['memory']])
             for p in [policy['baseline'], *policy['candidates']]}
    result = {"folds_compared": len(paired), "warmup_origins": warmup_origins,
              "scored_from": scored_from,
              "memory_score": average([d['served_loss'] for d in scored['memory']]),
              "no_memory_score": average([d['served_loss'] for d in scored['no_memory']]),
              "fixed_provider_scores": fixed,
              "memory_evidence_levels": {level: sum(d['evidence_level'] == level for d in scored['memory'])
                                         for level in sorted({d['evidence_level'] for d in scored['memory']})},
              "served_differently_share": round(sum(c for _, _, c in paired) / len(paired), 4) if paired else None,
              "switch_rate": {name: switch_rate(ds) for name, ds in scored.items()}}
    if not paired:
        return {**result, "verdict": "no_comparable_folds"}
    by_origin = {}
    for origin, diff, _ in paired:
        by_origin.setdefault(origin, []).append(diff)
    groups = list(by_origin.values())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [x for _ in groups for x in groups[rng.randrange(len(groups))]]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo, hi = boots[int(0.05 * n_boot)], boots[min(n_boot - 1, int(0.95 * n_boot))]
    mean_diff = sum(d for _, d, _ in paired) / len(paired)
    best_fixed = min((v for v in result["fixed_provider_scores"].values() if v is not None), default=None)
    verdict = "memory_better" if hi < 0 else "memory_worse" if lo > 0 else "not_distinguishable"
    return {**result, "mean_loss_difference": mean_diff, "difference_ci90": [lo, hi],
            "memory_beats_best_fixed": (result["memory_score"] is not None and best_fixed is not None
                                        and result["memory_score"] < best_fixed),
            "verdict": verdict}
