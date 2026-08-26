"""Governed publication projections over immutable forecast artifacts.

Publication chooses what a human sees first.  It never changes a forecast,
support assessment, or candidate seal, and it never grants automation rights.
Those invariants make best-effort interpretation useful without weakening the
history-only artifact that it interprets.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from .llm_dossier import verify_temporal_dossier_seal
from .llm_dossier import validate_temporal_dossier
from .effect_proposals import assess_composed_effect, compose_effect
from .temporal_state import build_temporal_state
from .context_intelligence import candidate_evidence_score

PublicationMode = Literal["strict", "best_effort", "scenario"]
PUBLICATION_VERSION = "0.1"
MODES = frozenset({"strict", "best_effort", "scenario"})
MAX_SCENARIOS = 8
SELECTION_LABEL = "hypothesis_ranking"


def dominant_scenario_id(scenarios: list[dict[str, Any]]) -> str | None:
    """Return the path that evidence makes non-discretionary, if any."""
    trusted = [item for item in scenarios
               if item.get("role") == "context_conditioned"
               and item.get("support") == "context_trusted"]
    if trusted:
        return str(trusted[0]["scenario_id"])
    admitted = [item for item in scenarios
                if item.get("role") == "fitted_context_candidate"
                and ((item.get("effect") or {}).get("evidence") or {}).get("decisive")]
    if not admitted:
        return None
    admitted.sort(key=lambda item: item["effect"]["evidence"]["score"],
                  reverse=True)
    if len(admitted) == 1 or (
            admitted[0]["effect"]["evidence"]["score"]
            - admitted[1]["effect"]["evidence"]["score"] >= .05):
        return str(admitted[0]["scenario_id"])
    return None


def compile_dossier_for_result(raw: Any, *, context_text: str, known_at: str,
                               result: dict[str, Any], compiler_model: str
                               ) -> tuple[dict[str, Any], list[str]]:
    """Seal an agent proposal against the exact path it will accompany.

    This is the one-call product seam: hosts may submit extracted context and
    a typed proposal with the forecast request; they need not reproduce
    Gnomon's sealing protocol or manufacture a forecast array.
    """
    primary = _rows(result.get("primary_forecast")) or _rows(result.get("forecast"))
    timestamps = [str(row.get("timestamp")) for row in primary]
    history_proxy = [float(row.get("q50", row.get("point"))) for row in primary
                     if row.get("q50", row.get("point")) is not None]
    return validate_temporal_dossier(
        raw, context_text=context_text, cutoff=known_at,
        future_timestamps=timestamps, history=history_proxy,
        compiler_model=compiler_model)


def _seal(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def _rows(value: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in value] if isinstance(value, list) else []


def _scenario(identifier: str, role: str, rows: list[dict[str, Any]], *,
              support: str, automation_eligible: bool,
              selection_eligible: bool = True,
              claim_ids: list[str] | None = None,
              assumptions: list[str] | None = None,
              source_seal: str | None = None,
              effect: dict[str, Any] | None = None) -> dict[str, Any]:
    item = {
        "scenario_id": identifier, "role": role, "forecast": rows,
        "support": support, "automation_eligible": automation_eligible,
        "selection_eligible": selection_eligible,
        "claim_ids": list(claim_ids or []),
        "assumptions": list(assumptions or []),
    }
    if source_seal:
        item["source_seal_sha256"] = source_seal
    if effect is not None:
        item["effect"] = effect
    item["scenario_seal_sha256"] = _seal(item)
    return item


def _same_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _path_support(rows: list[dict[str, Any]], fallback: str) -> str:
    tiers = [str(row.get("tier")) for row in rows if row.get("tier")]
    if not tiers:
        return fallback
    order = {"best_effort": 0, "conditionally_supported": 1,
             "supported": 2, "context_trusted": 2}
    return min(tiers, key=lambda item: order.get(item, -1))


def build_scenario_catalog(result: dict[str, Any], *,
                           dossiers: list[dict[str, Any]] | None = None
                           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build bounded immutable paths and typed dispositions for all context."""
    published = _rows(result.get("forecast"))
    primary = _rows(result.get("primary_forecast")) or published
    support = str(result.get("support") or "unsupported")
    verified_context_claim_ids = [
        str(claim.get("claim_id")) for dossier in dossiers or []
        if verify_temporal_dossier_seal(dossier)
        for claim in dossier.get("claims") or [] if claim.get("claim_id")]
    verified_context_claim_ids.extend(
        str(event_id) for event_id in
        ((result.get("context_outcome") or {}).get("events") or []))
    scenarios = [_scenario(
        "primary", "immutable_primary", primary,
        support=_path_support(primary, support),
        automation_eligible=_path_support(primary, support) in {
            "supported", "context_trusted"},
    )]
    dispositions: list[dict[str, Any]] = []
    dispositions.extend({
        "context_id": str(item.get("transformation_id") or
                          f"transformation-rejection-{index}"),
        "disposition": "rejected",
        "reason_code": str(item.get("reason_code") or "invalid_transformation"),
        "reason": str(item.get("reason") or "Transformation validation failed."),
        "violations": list(item.get("violations") or []),
    } for index, item in enumerate(
        result.get("transformation_rejections") or [], 1))
    context_outcome = result.get("context_outcome") or {}
    historically_admitted = (
        context_outcome.get("admission_basis") == "historical_fold_ablation")
    if published and not _same_rows(primary, published):
        scenarios.append(_scenario(
            ("historically_admitted" if historically_admitted
             else "context_conditioned"),
            ("historically_admitted" if historically_admitted
             else "context_conditioned"), published,
            support=support,
            automation_eligible=(historically_admitted
                                 and support in {"supported", "context_trusted"}),
            claim_ids=verified_context_claim_ids,
        ))
    for index, raw in enumerate(result.get("sensitivity_scenarios") or [], 1):
        rows = _rows(raw.get("forecast"))
        if rows:
            scenarios.append(_scenario(
                f"sensitivity-{index}", "conditional_sensitivity", rows,
                support=str(raw.get("support") or "hypothetical_sensitivity"),
                automation_eligible=False,
                assumptions=[str(item) for item in raw.get("assumptions") or []],
            ))

    # Fitted context executables may nominate a sealed conditional path.  They
    # are ranked only by disclosed out-of-sample evidence and can never inherit
    # automation authority from the immutable primary.
    for index, raw in enumerate(result.get("context_candidates") or [], 1):
        rows = _rows(raw.get("forecast"))
        if not rows or len(rows) != len(primary):
            dispositions.append({
                "context_id": str(raw.get("hypothesis_id") or f"candidate-{index}"),
                "disposition": "rejected", "reason_code": "invalid_candidate_horizon",
                "reason": "A fitted context candidate must match the primary horizon.",
            })
            continue
        evidence = candidate_evidence_score(raw)
        identifier = f"fitted-context-{index}"
        scenarios.append(_scenario(
            identifier, "fitted_context_candidate", rows,
            support="conditionally_supported" if evidence["decisive"] else "weak",
            automation_eligible=False,
            claim_ids=[str(raw.get("hypothesis_id"))],
            assumptions=[str(raw.get("kind") or "fitted context executable")],
            effect={"evidence": evidence, "validation": raw.get("validation") or {}},
        ))
        dispositions.append({
            "context_id": str(raw.get("hypothesis_id") or identifier),
            "disposition": "scenario", "reason_code": (
                "out_of_sample_candidate_admitted" if evidence["decisive"]
                else "candidate_retained_but_not_admitted"),
            "scenario_ids": [identifier], "evidence": evidence,
        })

    # Safe declarative transformations are executed before this seam and
    # arrive with a validator seal.  Publication never evaluates expressions;
    # it only authenticates the sealed numeric candidate and assigns authority
    # from its evidence lane.
    for index, raw in enumerate(result.get("transformation_candidates") or [], 1):
        identifier = f"transformation-{index}"
        rows = _rows(raw.get("forecast"))
        source_seal = str(raw.get("source_seal_sha256") or "")
        candidate_id = str(raw.get("transformation_id") or identifier)
        lane = str(raw.get("lane") or "scenario_only")
        validation = raw.get("validation") or {}
        is_recurrence = "recurrence_plausibility_passed" in validation
        selection_eligible = (
            validation.get("recurrence_plausibility_passed", True) is True
            and (not is_recurrence
                 or validation.get("recurrence_replay_admitted") is True))
        valid = bool(
            rows and len(rows) == len(primary) and source_seal
            and raw.get("primary_forecast_unchanged") is True
            and lane in {"historically_testable", "prior_assisted", "scenario_only"})
        if not valid:
            dispositions.append({
                "context_id": candidate_id, "disposition": "rejected",
                "reason_code": "invalid_transformation_candidate",
                "reason": "A transformation requires a seal, a matching horizon, and a known lane.",
            })
            continue
        evidence = candidate_evidence_score(raw)
        admitted = lane == "historically_testable" and evidence["decisive"]
        support = ("conditionally_supported" if admitted else
                   "prior_assisted" if lane == "prior_assisted" else
                   "hypothetical_sensitivity")
        scenarios.append(_scenario(
            identifier, "historically_admitted" if admitted
            else "model_authored_transformation", rows,
            support=support, automation_eligible=False,
            selection_eligible=selection_eligible,
            claim_ids=[str(item) for item in raw.get("claim_ids") or []],
            assumptions=[f"declarative transformation lane={lane}"],
            source_seal=source_seal,
            effect={"evidence": evidence, "validation": validation,
                    "transformation_id": candidate_id, "lane": lane},
        ))
        dispositions.append({
            "context_id": candidate_id,
            "disposition": "used" if admitted else "scenario",
            "reason_code": ("historically_tested_transformation_admitted"
                            if admitted else
                            "transformation_retained_plausibility_failed"
                            if not selection_eligible else
                            "prior_assisted_transformation"
                            if lane == "prior_assisted" else
                            "scenario_only_transformation"),
            "scenario_ids": [identifier], "evidence": evidence,
        })

    transformation_claim_sets = [
        set(item.get("claim_ids") or []) for item in scenarios
        if item.get("role") in {"historically_admitted",
                                "model_authored_transformation"}
    ]

    context_outcome = result.get("context_outcome")
    if isinstance(context_outcome, dict):
        status = str(context_outcome.get("status") or "rejected")
        event_ids = context_outcome.get("events") or ["engine-context"]
        dispositions.extend({
            "context_id": str(event_id), "disposition": (
                "used" if status == "applied" else
                "scenario" if status == "scenario_only" else "rejected"),
            "reason_code": status,
            "reason": str(context_outcome.get("reason") or
                          "See the immutable context outcome receipt."),
        } for event_id in event_ids)
    for index, dossier in enumerate(dossiers or [], 1):
        if not verify_temporal_dossier_seal(dossier):
            dispositions.append({
                "context_id": f"dossier-{index}", "disposition": "rejected",
                "reason_code": "invalid_candidate_seal",
                "reason": "The dossier seal does not authenticate its body.",
            })
            continue
        proposal = dossier.get("effect_proposal")
        candidate = dossier.get("forecast_candidate")
        claims = dossier.get("claims") or []
        if not candidate and not proposal:
            critique = dossier.get("effect_proposal_critique") or {}
            if critique.get("status") == "rejected":
                violations = [violation for attempt in critique.get("attempts") or []
                              for violation in attempt.get("violations") or []]
                dispositions.append({
                    "context_id": f"dossier-{index}:effect-proposal",
                    "disposition": "rejected",
                    "reason_code": "effect_proposal_rejected",
                    "reason": "; ".join(str(item.get("message")) for item in violations)[:1000],
                    "violations": violations,
                })
            candidate_critique = dossier.get("candidate_critique") or {}
            if candidate_critique.get("status") == "rejected":
                dispositions.append({
                    "context_id": f"dossier-{index}:forecast-candidate",
                    "disposition": "rejected",
                    "reason_code": "forecast_candidate_rejected",
                    "reason": "; ".join(
                        str(item) for item in
                        candidate_critique.get("reasons") or [])[:1000],
                    "recovery_action": candidate_critique.get("recovery_action"),
                })
            dispositions.extend({
                "context_id": f"dossier-{index}:{item.get('claim_id')}",
                "disposition": "used", "reason_code": "claims_only",
                "reason": "Verified claim informs interpretation but supplied no numeric path.",
                "claim_id": item.get("claim_id"),
            } for item in claims)
            continue
        emitted: list[str] = []
        if proposal:
            identifier = f"effect-composed-{index}"
            requested_series = set(proposal.get("scope", {}).get("series") or [])
            actual_series = str(result.get("series") or "*")
            if (proposal.get("scope", {}).get("kind") != "single_series"
                    and actual_series != "*" and "*" not in requested_series
                    and actual_series not in requested_series):
                dispositions.append({
                    "context_id": f"dossier-{index}", "disposition": "rejected",
                    "reason_code": "effect_scope_mismatch",
                    "reason": f"Effect scope does not include series {actual_series!r}.",
                })
            else:
                assessment = assess_composed_effect(primary, proposal)
                if not assessment["accepted"]:
                    dispositions.append({
                        "context_id": f"dossier-{index}:effect-proposal",
                        "disposition": "rejected",
                        "reason_code": "effect_composition_implausible",
                        "reason": "; ".join(item["message"] for item in
                                             assessment["violations"]),
                        "violations": assessment["violations"],
                        "composition_assessment": assessment,
                    })
                else:
                    scenarios.append(_scenario(
                        identifier, "effect_composed", compose_effect(primary, proposal),
                        support="prior_assisted", automation_eligible=False,
                        claim_ids=[str(item) for item in proposal.get("claim_ids") or []],
                        assumptions=[str(proposal.get("rationale") or ""),
                                     str(proposal.get("uncertainty_basis") or "")],
                        source_seal=str(dossier["seal_sha256"]),
                        effect={**proposal, "composition_assessment": assessment},
                    ))
                    emitted.append(identifier)
        if candidate:
            # Preserve the v0.1 public identifier while making the less
            # authoritative origin explicit in the typed role. A model may
            # supply this alongside a typed effect; the selector sees both.
            identifier = f"prior-assisted-{index}"
            candidate_critique = dossier.get("candidate_critique") or {}
            selection_eligible = candidate_critique.get(
                "selection_eligible", True) is True
            candidate_claims = {str(item) for item in
                                candidate.get("claim_ids") or []}
            governed_by_transformation = any(
                candidate_claims and candidate_claims.intersection(claims)
                for claims in transformation_claim_sets)
            if governed_by_transformation:
                # A model cannot bypass a failed replay/admission check by
                # restating its own forecast under the same cited claims. The
                # executable path owns numeric authority; the model path stays
                # visible for explanation, comparison, and outcome scoring.
                selection_eligible = False
            scenarios.append(_scenario(
                identifier, "model_authored", _rows(candidate.get("quantiles")),
                support="prior_assisted", automation_eligible=False,
                selection_eligible=selection_eligible,
                claim_ids=[str(item) for item in candidate.get("claim_ids") or []],
                assumptions=[str(candidate.get("rationale") or ""), *[
                    str(item) for item in
                    (candidate.get("plausibility") or {}).get("warnings") or []],
                    *(["uncertainty normalization: " + json.dumps(
                        (candidate.get("plausibility") or {}).get(
                            "uncertainty_normalization"), sort_keys=True)]
                      if (candidate.get("plausibility") or {}).get(
                          "uncertainty_normalization") else []),
                    *([str(candidate_critique.get("selection_reason"))]
                      if not candidate_critique.get(
                          "selection_eligible", True) else []),
                    *(["A governed transformation over the same cited claims "
                        "owns recommendation authority."]
                      if governed_by_transformation else [])],
                source_seal=str(dossier["seal_sha256"]),
            ))
            emitted.append(identifier)
        dispositions.extend({
            "context_id": f"dossier-{index}:{item.get('claim_id')}",
            "disposition": "scenario",
            "reason_code": "prior_assisted_not_historically_admitted",
            "scenario_ids": emitted, "claim_id": item.get("claim_id"),
        } for item in claims)
    if len(scenarios) > MAX_SCENARIOS:
        role_priority = {
            "immutable_primary": 100,
            "historically_admitted": 95,
            "context_conditioned": 90,
            "fitted_context_candidate": 80,
            "effect_composed": 70,
            "model_authored": 60,
            "model_authored_transformation": 60,
            "conditional_sensitivity": 50,
        }
        ranked = sorted(
            scenarios,
            key=lambda item: (
                role_priority.get(str(item.get("role")), 0),
                float((((item.get("effect") or {}).get("evidence") or {}).get(
                    "score") or 0.0)),
                str(item.get("scenario_id"))),
            reverse=True,
        )
        kept_ids = {item["scenario_id"] for item in ranked[:MAX_SCENARIOS]}
        dropped = [item for item in scenarios
                   if item["scenario_id"] not in kept_ids]
        scenarios = [item for item in scenarios
                     if item["scenario_id"] in kept_ids]
        dispositions.extend({
            "context_id": str(item["scenario_id"]),
            "disposition": "rejected",
            "reason_code": "bounded_portfolio_overflow",
            "reason": (
                f"Retained the {MAX_SCENARIOS} higher-priority sealed paths; "
                "this alternative remains recoverable from its source receipt."),
        } for item in dropped)
    return scenarios, dispositions


def validate_scenario_selection(raw: Any, *, scenarios: list[dict[str, Any]],
                                dossiers: list[dict[str, Any]] | None = None
                                ) -> dict[str, Any] | None:
    """Validate an LLM ranking without accepting any model-authored number."""
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("scenario_selection must be an object")
    ids = {item["scenario_id"] for item in scenarios}
    selected = str(raw.get("selected_scenario_id") or "")
    ranking = [str(item) for item in raw.get("ranking") or []]
    if selected not in ids or not ranking or set(ranking) != ids \
            or len(ranking) != len(set(ranking)) or ranking[0] != selected:
        raise ValueError("scenario selection must rank every known scenario id once with the selected id first")
    selected_scenario = next(item for item in scenarios
                             if item["scenario_id"] == selected)
    if selected_scenario.get("selection_eligible", True) is not True:
        raise ValueError("scenario selection cannot promote a candidate with an invalid derivation")
    dominant = dominant_scenario_id(scenarios)
    if dominant is not None and selected != dominant:
        raise ValueError(
            "scenario selection cannot override the evidence-dominant path")
    claim_ids = {str(claim.get("claim_id")) for dossier in dossiers or []
                 for claim in dossier.get("claims") or []}
    claim_ids.update(str(item) for scenario in scenarios
                     for item in scenario.get("claim_ids") or [])
    cited = [str(item) for item in raw.get("cited_claim_ids") or []]
    counter = [str(item) for item in raw.get("counterevidence_claim_ids") or []]
    if set(cited + counter) - claim_ids:
        raise ValueError("scenario selection cites an unknown claim id")
    if not cited and not counter:
        raise ValueError("scenario selection requires cited evidence or counterevidence")
    if set(cited) & set(counter):
        raise ValueError("a claim cannot be both supporting evidence and counterevidence")
    selected_claims = set(next(item for item in scenarios
                               if item["scenario_id"] == selected)["claim_ids"])
    if selected_claims and not selected_claims.intersection(cited):
        raise ValueError("selected conditional scenario requires one of its claims to be cited")
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        confidence = math.nan
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("scenario selection confidence must be between 0 and 1")
    flip = str(raw.get("what_would_change_selection") or "").strip()
    rationale = str(raw.get("rationale") or "").strip()
    if not rationale or not flip:
        raise ValueError("scenario selection requires rationale and what_would_change_selection")
    return {
        "label": SELECTION_LABEL, "selected_scenario_id": selected,
        "channel": "governed_scenario_selection",
        "ranking": ranking, "cited_claim_ids": cited,
        "counterevidence_claim_ids": counter, "confidence": confidence,
        "rationale": rationale[:1000],
        "what_would_change_selection": flip[:1000],
        "primary_forecast_unchanged": True,
        "support_unchanged": True, "automation_authorized": False,
    }


def scenario_selection_contract(*, scenarios: list[dict[str, Any]],
                                dossiers: list[dict[str, Any]] | None = None,
                                temporal_state: dict[str, Any] | None = None,
                                ) -> dict[str, Any]:
    """Compact prompt packet for the governed, number-free LLM channel."""
    claims = [claim for dossier in dossiers or []
              if verify_temporal_dossier_seal(dossier)
              for claim in dossier.get("claims") or []]
    known_claims = {str(claim.get("claim_id")) for claim in claims}
    claims.extend({
        "claim_id": str(item), "relation": "validated_context_event",
        "mechanism": "A deterministic context event was validated and applied by Gnomon.",
    } for scenario in scenarios for item in scenario.get("claim_ids") or []
                  if str(item) not in known_claims)
    dominant = dominant_scenario_id(scenarios)
    return {
        "selection_required": dominant is None,
        "deterministic_scenario_id": dominant,
        "selection_basis": ("governed_evidence_dominance" if dominant
                            else "ambiguous_evidence_requires_bounded_ranking"),
        "instruction": (
            "Rank only the supplied scenario_ids. Explain the ranking using "
            "claim_ids, name counterevidence, give confidence, and state what "
            "would change the selection. Do not output forecast numbers, "
            "support labels, or automation advice."),
        "scenarios": [{
            "scenario_id": item["scenario_id"], "role": item["role"],
            "support": item["support"], "claim_ids": item["claim_ids"],
            "forecast_seal": item["scenario_seal_sha256"],
            "summary": {
                "first_q50": ((item["forecast"][0].get("q50",
                                item["forecast"][0].get("point")))
                               if item["forecast"] else None),
                "last_q50": ((item["forecast"][-1].get("q50",
                               item["forecast"][-1].get("point")))
                              if item["forecast"] else None),
                "steps": len(item["forecast"]),
            },
        } for item in scenarios],
        "claims": claims,
        "temporal_state": temporal_state,
        "response_schema": {
            "selected_scenario_id": "string", "ranking": ["scenario_id"],
            "cited_claim_ids": ["claim_id"],
            "counterevidence_claim_ids": ["claim_id"],
            "confidence": "number 0..1", "rationale": "string",
            "what_would_change_selection": "string",
        },
    }


def publish_result(result: dict[str, Any], *, mode: PublicationMode = "strict",
                   dossiers: list[dict[str, Any]] | None = None,
                   scenario_selection: dict[str, Any] | None = None,
                   automation_policy: dict[str, Any] | None = None,
                   artifact_id: str | None = None) -> dict[str, Any]:
    """Return a compact, sealed human-facing projection over frozen paths."""
    if mode not in MODES:
        raise ValueError(f"unknown publication mode {mode!r}")
    scenarios, dispositions = build_scenario_catalog(result, dossiers=dossiers)
    selection = validate_scenario_selection(
        scenario_selection, scenarios=scenarios, dossiers=dossiers)
    by_id = {item["scenario_id"]: item for item in scenarios}
    if mode == "strict":
        eligible = next((item for item in scenarios
                         if item["role"] == "historically_admitted"), scenarios[0])
        selected_id = eligible["scenario_id"]
        selection = None  # model advice cannot govern strict publication
    elif selection is not None:
        selected_id = selection["selected_scenario_id"]
    elif mode == "best_effort":
        selected_id = next((item["scenario_id"] for item in scenarios
                            if item["role"] == "context_conditioned"), None)
        admitted = [item for item in scenarios
                    if item["role"] == "fitted_context_candidate"
                    and ((item.get("effect") or {}).get("evidence") or {}).get("decisive")]
        if selected_id is None and admitted:
            selected_id = max(
                admitted,
                key=lambda item: item["effect"]["evidence"]["score"]
            )["scenario_id"]
        selected_id = selected_id or next((item["scenario_id"] for item in scenarios
                            if item["role"] in {"effect_composed", "model_authored",
                                                "model_authored_transformation"}
                            and item.get("selection_eligible", True) is True),
                           "primary")
    else:
        selected_id = "primary"
    selected = by_id[selected_id]
    explicit_automation = bool((automation_policy or {}).get("authorize"))
    policy_complete = bool(
        isinstance(automation_policy, dict)
        and str(automation_policy.get("policy_id") or "").strip()
        and automation_policy.get("minimum_support") in {
            "supported", "context_trusted"})
    automation = bool(explicit_automation and policy_complete
                      and selected["automation_eligible"])
    payload = {
        "schema_version": PUBLICATION_VERSION, "artifact_id": artifact_id,
        "mode": mode, "recommended_scenario_id": selected_id,
        "recommended_forecast": selected["forecast"],
        "recommended_support": selected["support"],
        "primary_scenario_id": "primary", "primary_forecast": by_id["primary"]["forecast"],
        "primary_forecast_unchanged": True,
        "scenario_count": len(scenarios),
        # Full sealed portfolio is retained for outcome scoring. ``scenarios``
        # is the compact human-facing projection.
        "candidate_portfolio": scenarios,
        "scenarios": scenarios if mode == "scenario" else [by_id["primary"], selected]
                     if selected_id != "primary" else [by_id["primary"]],
        "context_dispositions": dispositions,
        "temporal_state": build_temporal_state(result, dossiers=dossiers),
        "scenario_selection": selection,
        "automation": {
            "eligible": automation,
            "explicit_policy_supplied": bool(automation_policy),
            "policy_complete": policy_complete,
            "requested": explicit_automation,
            "reason": ("explicit policy and scenario evidence permit automation"
                       if automation else "human recommendation is separate from automation eligibility"),
        },
    }
    payload["selection_contract"] = scenario_selection_contract(
        scenarios=scenarios, dossiers=dossiers,
        temporal_state=payload["temporal_state"])
    payload["candidate_admission"] = {
        "status": "cold_start",
        "rule": "candidate outcomes are scored separately; only fold-safe historical admission may upgrade authority",
        "hierarchy": ["same_series", "related_series", "organization", "external_prior"],
        "benchmark_or_model_confidence_cannot_upgrade_support": True,
    }
    payload["publication_seal_sha256"] = _seal(payload)
    return payload


def verify_publication(payload: dict[str, Any]) -> bool:
    """Verify seals, immutable primary visibility, and authority separation."""
    if not isinstance(payload, dict) or not payload.get("publication_seal_sha256"):
        return False
    body = {key: value for key, value in payload.items()
            if key != "publication_seal_sha256"}
    if _seal(body) != payload["publication_seal_sha256"]:
        return False
    scenarios = payload.get("scenarios") or []
    portfolio = payload.get("candidate_portfolio") or scenarios
    if not scenarios or not portfolio or len(portfolio) > MAX_SCENARIOS:
        return False
    ids = [item.get("scenario_id") for item in portfolio]
    if len(ids) != len(set(ids)):
        return False
    primary = next((item for item in portfolio
                    if item.get("scenario_id") == "primary"), None)
    if not primary or primary.get("forecast") != payload.get("primary_forecast"):
        return False
    if payload.get("primary_forecast_unchanged") is not True:
        return False
    for item in portfolio:
        seal = item.get("scenario_seal_sha256")
        body = {key: value for key, value in item.items()
                if key != "scenario_seal_sha256"}
        if not seal or _seal(body) != seal:
            return False
    selected = next((item for item in portfolio if item.get("scenario_id") ==
                     payload.get("recommended_scenario_id")), None)
    if (not selected
            or selected.get("forecast") != payload.get("recommended_forecast")
            or selected.get("support") != payload.get("recommended_support")):
        return False
    automation = payload.get("automation") or {}
    if automation.get("eligible") and not selected.get("automation_eligible"):
        return False
    if payload.get("mode") == "strict" and selected.get("role") not in {
            "immutable_primary", "historically_admitted"}:
        return False
    if payload.get("scenario_selection") and \
            payload["scenario_selection"].get("automation_authorized") is not False:
        return False
    return True


def record_publication(store: Any, *, project: str, forecast_id: str,
                       series: str, payload: dict[str, Any]) -> str:
    """Reuse synthesis receipts to score recommendation uplift later."""
    if not verify_publication(payload):
        raise ValueError("refusing to record an invalid publication")
    portfolio = payload.get("candidate_portfolio") or payload["scenarios"]
    selected = next(item for item in portfolio
                    if item["scenario_id"] == payload["recommended_scenario_id"])
    primary = next(item for item in portfolio
                   if item["scenario_id"] == "primary")
    synthesis_id = f"publication:{payload['publication_seal_sha256'][:20]}"
    store.record_temporal_synthesis(
        project=project, forecast_id=forecast_id, series=series,
        question_id="publication", synthesis_id=synthesis_id,
        canonical={"value": "primary", "forecast": primary["forecast"]},
        synthesis={
            "label": "hypothesis_ranking",
            "channel": "governed_scenario_selection",
            "value": selected["scenario_id"], "forecast": selected["forecast"],
            "support": selected["support"],
            "primary_forecast_unchanged": True,
            "automation_eligible": payload["automation"]["eligible"],
        },
        evidence_refs=[item["scenario_seal_sha256"] for item in portfolio],
    )
    # Record every alternative, not only the displayed winner. This yields
    # honest candidate evidence as actuals arrive.
    for candidate in portfolio:
        if candidate["scenario_id"] in {"primary", selected["scenario_id"]}:
            continue
        store.record_temporal_synthesis(
            project=project, forecast_id=forecast_id, series=series,
            question_id="publication_candidate",
            synthesis_id=f"{synthesis_id}:{candidate['scenario_id']}",
            canonical={"value": "primary", "forecast": primary["forecast"]},
            synthesis={
                "label": "candidate_portfolio", "value": candidate["scenario_id"],
                "forecast": candidate["forecast"], "support": candidate["support"],
                "primary_forecast_unchanged": True, "automation_eligible": False,
            }, evidence_refs=[candidate["scenario_seal_sha256"]],
        )
    return synthesis_id


def write_publication(path: str | Path, payload: dict[str, Any]) -> Path:
    if not verify_publication(payload):
        raise ValueError("refusing to persist an invalid publication")
    artifact = Path(path)
    # Forecast directories are already integrity sealed.  A sibling sidecar
    # keeps that immutable identity intact while carrying its own body seal.
    seal = payload["publication_seal_sha256"]
    destination = artifact.parent / f"{artifact.name}.{seal[:16]}.publication.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise ValueError("conflicting content-addressed publication sidecar")
        return destination
    destination.write_text(encoded, encoding="utf-8")
    return destination
