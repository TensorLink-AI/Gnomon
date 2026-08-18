"""Deterministic evidence ladder for context-effect scenarios."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .effect_registry import EffectPrior, resolve_effect_prior
from .effects import EffectDistribution, EffectProvenance, effect_contract
from .tracking import TrackingStore


def resolve_effect_evidence(
    store: TrackingStore, *, project: str, event_type: str, series: str,
    as_of: str, external_priors: list[EffectPrior] | None = None,
    target: str = "*", domain: str = "*", population: str = "*",
    unit: str = "*", human_assumption: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one effect without allowing a weaker source to override one.

    Ladder: validated organizational memory, versioned external evidence,
    explicit human assumption, then no numeric effect. Conflicts are disclosed
    but never averaged across provenance classes.
    """
    cutoff = datetime.fromisoformat(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("effect resolution as_of requires a timezone offset")
    local = store.pooled_effect_prior(
        project, event_type, series, as_of=as_of, require_validation=True)
    candidates: list[dict[str, Any]] = []
    if local.get("eligible_for_scenario"):
        candidates.append({"lane": "organizational_memory", **local})

    eligible_external = [
        prior for prior in (external_priors or [])
        if prior.known_at is not None
        and datetime.fromisoformat(prior.known_at) <= cutoff
    ]
    external = resolve_effect_prior(
        eligible_external, event_type=event_type, target=target,
        domain=domain, population=population, unit=unit,
    ) if eligible_external else None
    if external and external.get("status") == "scenario_prior":
        scale = float(external["standard_error"])
        location = float(external["mean"])
        contract = effect_contract(
            EffectDistribution(
                "normal", location, scale,
                location - 1.2815515655446004 * scale,
                location + 1.2815515655446004 * scale,
                0.8, int(external["sample_size"]), unit,
            ),
            EffectProvenance(
                "external_prior", True,
                max(prior.known_at for prior in eligible_external
                    if prior.prior_id in external["prior_ids"]),
                ", ".join(external["sources"]), None,
                min(1.0, int(external["sample_size"]) / 100.0),
                "precision-pooled equally specific governed priors",
            ),
            shape=str(external["effect_family"]),
        )
        candidates.append({"lane": "external_prior", "status": "validated",
                           "effect": contract, "details": external})

    if human_assumption is not None:
        location = float(human_assumption["location"])
        known_at = str(human_assumption["known_at"])
        if datetime.fromisoformat(known_at).tzinfo is None:
            raise ValueError("human assumption known_at requires a timezone offset")
        if datetime.fromisoformat(known_at) <= cutoff:
            contract = effect_contract(
                EffectDistribution(
                    "assumption", location, 0.0, location, location,
                    None, 0, str(human_assumption.get("unit") or unit),
                ),
                EffectProvenance(
                    "human_assumption", False, known_at,
                    str(human_assumption.get("source") or "operator"),
                    None, None, "explicit caller-supplied assumption",
                ),
                shape=str(human_assumption.get("shape") or "unknown"),
            )
            candidates.append({"lane": "human_assumption",
                               "status": "assumption", "effect": contract})

    if not candidates:
        return {
            "status": "insufficient_evidence", "selected_lane": None,
            "effect": None, "considered": [],
            "may_affect_primary_forecast": False,
            "recovery_actions": [
                "confirm and score more comparable event outcomes",
                "supply a versioned external prior with known_at",
                "supply an explicit human assumption for sensitivity only",
            ],
        }
    selected = candidates[0]
    if selected["lane"] == "organizational_memory":
        selected_effect = {
            "distribution": selected["distribution"],
            "provenance": selected["provenance"],
            "shape": "unknown",
        }
    else:
        selected_effect = selected["effect"]
    selected_location = selected_effect["distribution"]["location"]
    conflicts = []
    for candidate in candidates[1:]:
        effect = candidate.get("effect") or {
            "distribution": candidate.get("distribution")}
        location = effect["distribution"]["location"]
        if selected_location and location and selected_location * location < 0:
            conflicts.append({"selected": selected["lane"],
                              "conflicts_with": candidate["lane"],
                              "reason": "effect directions disagree"})
    return {
        "status": selected["status"], "selected_lane": selected["lane"],
        "effect": selected_effect,
        "may_affect_primary_forecast": False,
        "considered": [candidate["lane"] for candidate in candidates],
        "conflicts": conflicts,
        "rule": "first eligible lane wins; provenance classes are never averaged",
    }
