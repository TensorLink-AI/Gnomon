"""Deterministic comparison of canonical and competing temporal evidence.

The adjudicator does not forecast and does not infer causality.  It ranks
already-computed receipts, preserves their provenance, and states whether an
agent synthesis has enough independent opposition to be publication-eligible.
Probability is exposed only when supplied by a fitted calibrated executable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


ADJUDICATION_VERSION = "0.2"
_SUPPORT_WEIGHT = {"supported": 1.0, "weak": .5, "abstained": 0.0}
_NON_ANSWERS = {None, "", "uncertain"}


def _probabilities(result: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    answer = result.get("answer") or {}
    distribution = answer.get("property_distribution") or {}
    values = distribution.get("probabilities") or answer.get(
        "direction_probabilities") or {}
    if not isinstance(values, dict) or not values:
        return {}, {"available": False, "reason": "no_fitted_probability_distribution"}
    probabilities = {str(key): float(value) for key, value in values.items()}
    return probabilities, {
        "available": True,
        "source": "fitted_property_distribution",
        "folds": distribution.get("folds"),
        "balanced_accuracy": distribution.get("balanced_accuracy"),
        "brier_skill": distribution.get("brier_skill"),
    }


def _effect_direction(canonical: Any, location: float, shape: str) -> str | None:
    """Project an effect into the canonical property's public vocabulary."""
    label = str(canonical)
    positive = location > 0
    if label in {"higher", "similar", "lower"}:
        return "higher" if positive else "lower"
    if label in {"upward", "constant", "downward"}:
        return "upward" if positive else "downward"
    if label in {"increased", "stable", "decreased"}:
        return "increased" if positive else "decreased"
    if label in {"more_frequent", "less_frequent"}:
        return "more_frequent" if positive else "less_frequent"
    if label in {"stronger", "weaker"}:
        return "stronger" if positive else "weaker"
    if label in {"shift", "no_shift", "reversal"}:
        return "shift"
    if label in {"strengthened", "weakened"} \
            or shape == "seasonal_regime_change":
        return "strengthened" if positive else "weakened"
    return None


def _context_candidate(effect: dict[str, Any] | None,
                       canonical: Any) -> dict[str, Any] | None:
    if not isinstance(effect, dict):
        return None
    # MCP forecast receipts wrap the canonical effect contract so they can
    # carry scenario/outcome counts beside it.  Direct describe calls may pass
    # the contract itself.
    wrapper = effect
    effect = effect.get("measured_effect") or effect.get("effect") or effect
    distribution = effect.get("distribution") or {}
    provenance = effect.get("provenance") or wrapper.get("provenance") or {}
    if not isinstance(provenance, dict):
        provenance = {"source_reference": str(provenance)}
    location = distribution.get("location")
    if not isinstance(location, (int, float)) or float(location) == 0:
        return None
    samples = int(distribution.get("sample_count")
                  or distribution.get("sample_size") or 0)
    provenance_class = provenance.get("provenance_class")
    validated = bool(provenance.get("observed") and samples > 0 and
                     provenance_class != "human_assumption")
    direction = _effect_direction(
        canonical, float(location), str(effect.get("shape") or "unknown"))
    if direction is None:
        return None
    return {
        "value": direction,
        "source": "conditional_effect",
        "support": "supported" if validated and samples > 0 else "weak",
        "evidence_weight": 1.0 if validated and samples > 0 else .25,
        "provenance": {
            key: provenance.get(key) for key in (
                "provenance_class", "known_at", "source_reference",
                "observed", "reliability") if provenance.get(key) is not None
        },
        "conditional_only": not (validated and samples > 0),
    }


def adjudicate_temporal_evidence(
    result: dict[str, Any], *, evidence: list[dict[str, Any]],
    analogues: dict[str, Any] | None = None,
    conditional_effect: dict[str, Any] | None = None,
    missing: list[str] | None = None,
) -> dict[str, Any]:
    """Compare evidence without replacing the immutable canonical answer."""
    best = result.get("best_estimate") or {}
    canonical = best.get("value")
    canonical_support = str(best.get("support") or "abstained")
    probabilities, calibration = _probabilities(result)
    candidates: dict[str, dict[str, Any]] = {}

    def add(value: Any, source: str, support: str, weight: float,
            provenance: dict[str, Any] | None = None,
            conditional_only: bool = False) -> None:
        if value in _NON_ANSWERS:
            return
        label = str(value)
        item = candidates.setdefault(label, {
            "value": label, "evidence_weight": 0.0,
            "sources": [], "support": "abstained",
            "conditional_only": True,
        })
        item["evidence_weight"] += float(weight)
        item["sources"].append({"kind": source, "support": support,
                                **({"provenance": provenance}
                                   if provenance else {})})
        if _SUPPORT_WEIGHT.get(support, 0) > _SUPPORT_WEIGHT.get(item["support"], 0):
            item["support"] = support
        item["conditional_only"] = item["conditional_only"] and conditional_only

    add(canonical, "canonical_fitted_executable", canonical_support,
        _SUPPORT_WEIGHT.get(canonical_support, 0))
    for row in evidence:
        source = str(row.get("kind") or "computed_evidence")
        # The canonical executable is already represented above. Counting its
        # projection again would give it two votes merely because the planner
        # serialized the same fitted receipt in two places.
        if (row.get("direction") == canonical
                and source in {"fitted_executable", "fitted_volatility",
                               "fitted_temporal_property",
                               "fitted_dependence",
                               "published_forecast_projection"}):
            continue
        add(row.get("direction"), source,
            str(row.get("support") or "weak"),
            _SUPPORT_WEIGHT.get(str(row.get("support") or "weak"), .25))
    if analogues and analogues.get("available"):
        agreement = float(analogues.get("agreement") or 0.0)
        add(analogues.get("consensus_direction"), "historical_analogues",
            "supported" if agreement >= .8 else "weak", agreement,
            {"matches": len(analogues.get("matches") or []),
             "outcomes_are_historical_only": True})
    context = _context_candidate(conditional_effect, canonical)
    if context:
        add(context["value"], context["source"], context["support"],
            context["evidence_weight"], context["provenance"],
            context["conditional_only"])

    for item in candidates.values():
        item["evidence_weight"] = round(item["evidence_weight"], 6)
        if item["value"] in probabilities:
            item["fitted_probability"] = probabilities[item["value"]]
    ranked = sorted(candidates.values(), key=lambda item: (
        -float(item["evidence_weight"]), -len(item["sources"]),
        item["value"] != canonical, item["value"]))
    winner = ranked[0] if ranked else None
    alternative = next((item for item in ranked if item["value"] != canonical), None)
    margin = ((winner["evidence_weight"] - ranked[1]["evidence_weight"])
              if len(ranked) > 1 else winner["evidence_weight"] if winner else 0.0)
    opposing_sources = (alternative.get("sources") or []) if alternative else []
    independent_opposition = len({row["kind"] for row in opposing_sources})
    supported_opposition = len({
        row["kind"] for row in opposing_sources
        if row.get("support") == "supported"
    })
    alternative_preferred = bool(
        alternative and winner and winner["value"] == alternative["value"])
    synthesis_eligible = bool(
        canonical_support != "supported" and alternative_preferred
        and not alternative.get("conditional_only")
        and supported_opposition >= 2)

    if not ranked:
        relationship = "unresolved"
    elif alternative is None:
        relationship = "canonical_only"
    elif alternative_preferred:
        relationship = "alternative_preferred"
    else:
        relationship = "canonical_preferred"
    flips = list(missing or [])
    if alternative and supported_opposition < 2:
        flips.append("independent_evidence_for_the_alternative")
    if not calibration["available"]:
        flips.append("fitted_probability_calibration")
    return {
        "version": ADJUDICATION_VERSION,
        "canonical": {"value": canonical, "support": canonical_support,
                      "immutable": True},
        "alternative": alternative,
        "relationship": relationship,
        "evidence_balance": {
            "winner": winner["value"] if winner else None,
            "margin": round(float(margin), 6),
            "quantity": "receipt_evidence_weight_not_probability",
        },
        "candidates": ranked[:3],
        "conditional_candidate": ({
            "value": context["value"],
            "support": context["support"],
            "provenance": context["provenance"],
            "conditional_only": context["conditional_only"],
        } if context else None),
        "calibration": calibration,
        "synthesis_eligibility": {
            "eligible": synthesis_eligible,
            "independent_opposing_sources": independent_opposition,
            "supported_independent_opposing_sources": supported_opposition,
            "requires_separate_label": True,
            "primary_forecast_unchanged": True,
        },
        "what_would_flip": list(dict.fromkeys(flips))[:3],
        "primary_forecast_unchanged": True,
    }
