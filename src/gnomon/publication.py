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

from .llm_dossier import (deterministic_events_from_claims,
                          validate_temporal_dossier,
                          verify_temporal_dossier_seal)
from .effect_proposals import assess_composed_effect, compose_effect
from .temporal_state import build_temporal_state
from .context_intelligence import candidate_evidence_score

PublicationMode = Literal["strict", "best_effort", "scenario"]
PUBLICATION_VERSION = "0.1"
MODES = frozenset({"strict", "best_effort", "scenario"})
MAX_SCENARIOS = 8
SELECTION_LABEL = "hypothesis_ranking"


def _context_summary(dispositions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one authoritative publication-level context disposition.

    Individual parsers may reject one representation while another governed
    lane successfully executes the same source (for example, a numeric event
    parser rejects an exogenous value while a fold-tested transformation uses
    the relationship).  Expose both details, but never make a human reconcile
    contradictory top-level statuses themselves.
    """
    counts = {kind: sum(item.get("disposition") == kind
                        for item in dispositions)
              for kind in ("used", "scenario", "rejected")}
    if not dispositions:
        status = "not_supplied"
        message = "No context was supplied to the publication contract."
    elif counts["used"] and counts["rejected"]:
        status = "partially_used"
        message = (
            "At least one governed context lane affected the human-facing "
            "recommendation; other representations were rejected. See typed "
            "per-lane dispositions.")
    elif counts["used"]:
        status = "used"
        message = "Governed context affected the human-facing recommendation."
    elif counts["scenario"] and counts["rejected"]:
        status = "partially_represented"
        message = (
            "Context was retained in labelled scenarios while other "
            "representations were rejected; it did not earn governed use.")
    elif counts["scenario"]:
        status = "scenario_only"
        message = (
            "Context was retained only in labelled scenarios and did not "
            "earn governed use.")
    else:
        status = "rejected"
        message = "No supplied context representation passed its governed lane."
    return {
        "status": status,
        "authoritative_for_publication": True,
        "counts": counts,
        "message": message,
        "follow_up_required_for_current_recommendation": status == "rejected",
        "further_calls_add_nothing_for_current_recommendation": bool(
            counts["used"]),
    }


def _scope_recovery_actions(
        dispositions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark rejected side-lane repairs optional once context was used."""
    context_used = any(item.get("disposition") == "used"
                       for item in dispositions)
    if not context_used:
        return dispositions
    scoped = []
    for item in dispositions:
        action = item.get("recovery_action")
        if item.get("disposition") == "rejected" and isinstance(action, dict):
            item = {**item, "recovery_action": {
                **action,
                "required_for_current_recommendation": False,
                "scope": "optional_rejected_lane_only",
            }}
        scoped.append(item)
    return scoped


def dominant_scenario_id(scenarios: list[dict[str, Any]]) -> str | None:
    """Return the path that evidence makes non-discretionary, if any."""
    historically_admitted = [item for item in scenarios
                             if item.get("role") == "historically_admitted"]
    if historically_admitted:
        # These paths already won their disclosed out-of-sample contest. An
        # LLM may explain that result but cannot demote it in favour of an
        # untested interpretation.
        return str(historically_admitted[0]["scenario_id"])
    retrospective = [item for item in scenarios
                     if item.get("role") == "retrospectively_validated"]
    if retrospective:
        # A fixed source-supplied specification beat the baseline on
        # per-origin observations, but the specification itself was not
        # known at those historical origins. It may govern today's human
        # recommendation, never claim bitemporal historical admission.
        return str(retrospective[0]["scenario_id"])
    trusted = [item for item in scenarios
               if item.get("role") == "context_conditioned"
               and item.get("support") == "context_trusted"]
    if trusted:
        return str(trusted[0]["scenario_id"])
    observation = [item for item in scenarios
                   if item.get("role") == "observation_counterfactual"
                   and ((item.get("effect") or {}).get(
                       "conditional_replay") or {}).get(
                           "selection_eligible") is True]
    if observation:
        # A fixed executable that cleared conditional replay outranks a
        # number-free model ranking. The model may explain it, not silently
        # replace it with an unsupported sealed path.
        return str(observation[0]["scenario_id"])
    source_determined_scenarios = [item for item in scenarios
                       if item.get("role") == "effect_composed"
                       and item.get("support") == "hypothetical_sensitivity"
                       and any(normalization.get("code") in {
                                   "EXACT_CITED_LEVEL_MULTIPLIER",
                                   "APPROXIMATE_CITED_LEVEL_MULTIPLIER",
                               }
                               for normalization in
                               (item.get("effect") or {}).get(
                                   "semantic_normalizations") or [])]
    if len(source_determined_scenarios) == 1:
        # The caller supplied one operative numeric scenario. Approximate
        # language widens the scenario's uncertainty; it does not make the
        # context-free primary answer the answer to the caller's conditional
        # question. A model may explain this path but cannot silently demote
        # it. Support stays hypothetical and automation stays disabled.
        return str(source_determined_scenarios[0]["scenario_id"])
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


def _candidate_rows(candidate: dict[str, Any], primary: list[dict[str, Any]]) \
        -> list[dict[str, Any]]:
    """Resolve partial model anchors against, never instead of, the primary."""
    if candidate.get("requires_primary_completion") is not True:
        return _rows(candidate.get("quantiles"))
    anchors = _rows(candidate.get("quantile_anchors"))
    if not anchors or not primary:
        return []
    index_by_timestamp = {str(row.get("timestamp")): index
                          for index, row in enumerate(primary)}
    resolved = []
    for anchor in anchors:
        index = index_by_timestamp.get(str(anchor.get("timestamp")))
        if index is None:
            return []
        resolved.append((index, anchor))
    resolved.sort(key=lambda item: item[0])
    rows = [{"timestamp": row.get("timestamp"),
             "q10": row.get("q10", row.get("point")),
             "q50": row.get("q50", row.get("point")),
             "q90": row.get("q90", row.get("point"))}
            for row in primary]
    for (left_index, left), (right_index, right) in zip(
            resolved, resolved[1:]):
        width = right_index - left_index
        for index in range(left_index, right_index + 1):
            weight = (index - left_index) / max(1, width)
            for key in ("q10", "q50", "q90"):
                rows[index][key] = (float(left[key])
                                    + (float(right[key]) - float(left[key]))
                                    * weight)
    for index, anchor in resolved:
        rows[index].update({key: float(anchor[key])
                            for key in ("q10", "q50", "q90")})
    return rows


def _scenario(identifier: str, role: str, rows: list[dict[str, Any]], *,
              support: str, automation_eligible: bool,
              selection_eligible: bool = True,
              human_selection_eligible: bool | None = None,
              claim_ids: list[str] | None = None,
              assumptions: list[str] | None = None,
              source_seal: str | None = None,
              effect: dict[str, Any] | None = None) -> dict[str, Any]:
    item = {
        "scenario_id": identifier, "role": role, "forecast": rows,
        "support": support, "automation_eligible": automation_eligible,
        "selection_eligible": selection_eligible,
        "human_selection_eligible": (
            selection_eligible if human_selection_eligible is None
            else human_selection_eligible),
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


def _context_recovery(disposition: dict[str, Any]) -> dict[str, Any]:
    """Return the shortest honest path from rejection to admissibility."""
    code = str(disposition.get("reason_code") or "context_unresolved")
    if code == "context_unresolved":
        return {
            "code": "provide_grounded_context",
            "message": (
                "Keep the primary answer, or resubmit context containing an "
                "explicit dated schedule, numeric relationship, or bounded "
                "scenario assumption that can be cited verbatim."),
            "required_evidence": [
                "effective dates", "target or driver identity",
                "verbatim numeric rule or explicitly labelled scenario assumption",
            ],
            "automation_eligible": False,
        }
    if code == "unsafe_wildcard_numeric_event":
        return {
            "code": "bind_numeric_event_target",
            "message": (
                "Keep the verified result. If this additional numeric event "
                "is intended to apply, resubmit it with the exact target "
                "series named in the cited source."),
            "required_evidence": [
                "verbatim target identity", "verbatim numeric rule",
            ],
            "automation_eligible": False,
        }
    if code == "event_outside_forecast_window":
        return {
            "code": "retain_as_historical_claim",
            "message": (
                "Keep the event as historical evidence; provide a future "
                "effective window only if the source explicitly states one."),
            "required_evidence": ["source-stated future effective window"],
            "automation_eligible": False,
        }
    if code in {
            "transformation_validation_failed", "effect_proposal_rejected",
            "forecast_candidate_rejected"}:
        return {
            "code": "correct_context_proposal",
            "message": (
                "Keep the primary answer and correct only the rejected typed "
                "proposal using values and series identities present in its "
                "cited source."),
            "required_evidence": [
                "verbatim source span", "resolved series identity",
                "unit-consistent finite values",
            ],
            "automation_eligible": False,
        }
    if code == "invalid_candidate_seal":
        return {
            "code": "recompile_from_source",
            "message": (
                "Discard the altered dossier and compile a new sealed proposal "
                "from the original source document."),
            "required_evidence": ["original source document"],
            "automation_eligible": False,
        }
    if code == "bounded_portfolio_overflow":
        return {
            "code": "inspect_sealed_receipt",
            "message": (
                "Inspect the sealed source receipt for this lower-ranked path; "
                "it was omitted only from the compact response."),
            "required_evidence": ["source receipt"],
            "automation_eligible": False,
        }
    return {
        "code": "correct_rejected_context",
        "message": (
            "Keep the primary answer and correct the cited field identified by "
            "the rejection before resubmitting context."),
        "required_evidence": ["rejection reason", "cited source"],
        "automation_eligible": False,
    }


def _claim_disposition(
        claim: dict[str, Any], *, dossier_index: int,
        disposition: str, reason_code: str,
        reason: str | None = None,
        scenario_ids: list[str] | None = None) -> dict[str, Any]:
    """Project a verified claim, preserving unresolved-trigger recovery."""
    if claim.get("timing_status") == "unresolved_trigger":
        return {
            "context_id": f"dossier-{dossier_index}:{claim.get('claim_id')}",
            "disposition": "scenario",
            "reason_code": "trigger_timing_unresolved",
            "reason": (
                "The source states a relevant temporal rule but does not "
                "establish whether or when its trigger occurs in the "
                "forecast horizon. It was not applied numerically."),
            "claim_id": claim.get("claim_id"),
            "scenario_ids": list(scenario_ids or []),
            "recovery_action": {
                "code": "provide_dated_trigger",
                "message": (
                    "Provide the trigger identity and its effective date or "
                    "window; Gnomon will recompile the same cited rule."),
                "required_evidence": [
                    "trigger identity", "effective date or window",
                    "target or entity scope",
                ],
                "automation_eligible": False,
                "required_for_current_recommendation": False,
            },
        }
    if claim.get("timing_status") == "atemporal_context":
        return {
            "context_id": f"dossier-{dossier_index}:{claim.get('claim_id')}",
            "disposition": "scenario",
            "reason_code": "background_context_not_conditioned",
            "reason": (
                "The source states background evidence or a relationship, "
                "not a dated event. It remains available to interpretation "
                "but was not treated as a deterministic forecast adjustment."),
            "claim_id": claim.get("claim_id"),
            "scenario_ids": list(scenario_ids or []),
            "recovery_action": {
                "code": "provide_applicability_evidence",
                "message": (
                    "Provide the current driver observations, comparison "
                    "period, or an explicit bounded scenario assumption "
                    "needed to apply this background evidence."),
                "required_evidence": [
                    "applicable driver observations or comparison period",
                    "target and entity scope",
                    "bounded scenario assumption when historical validation "
                    "is unavailable",
                ],
                "automation_eligible": False,
                "required_for_current_recommendation": False,
            },
        }
    return {
        "context_id": f"dossier-{dossier_index}:{claim.get('claim_id')}",
        "disposition": disposition, "reason_code": reason_code,
        "reason": reason, "claim_id": claim.get("claim_id"),
        **({"scenario_ids": list(scenario_ids)} if scenario_ids is not None
           else {}),
    }


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
    transformation_dispositions = [{
        "context_id": str(item.get("transformation_id") or
                          f"transformation-rejection-{index}"),
        "disposition": "rejected",
        "reason_code": str(item.get("reason_code") or "invalid_transformation"),
        "reason": str(item.get("reason") or "Transformation validation failed."),
        "violations": list(item.get("violations") or []),
    } for index, item in enumerate(
        result.get("transformation_rejections") or [], 1)]
    dispositions.extend(transformation_dispositions)
    represented_transform_codes = {
        str(code) for item in transformation_dispositions
        for code in [item.get("reason_code"), *[
            violation.get("code") for violation in item.get("violations") or []
            if isinstance(violation, dict)]] if code
    }
    for index, item in enumerate(result.get("context_rejections") or [], 1):
        context_id = str(item.get("context_id") or
                         f"context-submission-{index}")
        reason_code = str(item.get("reason_code") or "context_unresolved")
        reason = str(item.get("reason") or
                     "Supplied context could not be grounded or executed.")
        repeated_codes: set[str] = set()
        if reason_code == "transformation_preflight_rejected":
            try:
                parsed = json.loads(reason)
            except (TypeError, ValueError):
                parsed = []
            repeated_codes = {
                str(violation.get("code")) for failure in parsed
                if isinstance(failure, dict)
                for violation in failure.get("violations") or []
                if isinstance(violation, dict) and violation.get("code")
            }
        duplicate_summary = bool(
            repeated_codes
            and repeated_codes <= represented_transform_codes)
        dispositions.append({
            "context_id": context_id,
            "disposition": "superseded" if duplicate_summary else "rejected",
            "reason_code": (
                "duplicate_transformation_preflight_summary"
                if duplicate_summary else reason_code),
            "reason": (
                "Submission-level preflight repeats the typed transformation "
                "rejection already listed above."
                if duplicate_summary else reason),
            **({"supersedes_reason_code": reason_code,
                "represented_violation_codes": sorted(repeated_codes)}
               if duplicate_summary else {}),
        })
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
        retrospective = bool(
            admitted
            and validation.get("specification_known_at_each_origin") is False)
        admitted_role = ("retrospectively_validated" if retrospective
                         else "historically_admitted")
        support = ("conditionally_supported" if admitted else
                   "prior_assisted" if lane == "prior_assisted" else
                   "hypothetical_sensitivity")
        scenarios.append(_scenario(
            identifier, admitted_role if admitted
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
            "reason_code": (
                            "retrospectively_tested_transformation_selected"
                            if retrospective else
                            "historically_tested_transformation_admitted"
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
                                "retrospectively_validated",
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
        deterministic_claim_ids = {
            str(event.get("derived_from_claim_id"))
            for event in deterministic_events_from_claims(dossier)
            if event.get("derived_from_claim_id")}
        if proposal and deterministic_claim_ids.intersection(
                str(item) for item in proposal.get("claim_ids") or []):
            # An exact absolute/range claim belongs to the deterministic
            # context contract. A model-authored additive effect over the same
            # words is a different operation (notably, +0 is not "set to
            # zero") and must not become a competing scenario.
            dispositions.append({
                "context_id": f"dossier-{index}:effect-proposal",
                # This is not rejected caller context.  It is a redundant,
                # weaker internal representation of context that the exact
                # deterministic lane already used.  Keeping it visible is
                # useful provenance; counting it as rejected makes a fully
                # resolved instruction look only partially handled.
                "disposition": "superseded",
                "reason_code": "superseded_by_deterministic_context_contract",
                "reason": (
                    "The cited claim states an absolute value or range; its "
                    "deterministic context representation owns numeric "
                    "authority, so a model-authored additive effect was not "
                    "published."),
            })
            proposal = None
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
            deterministic_used = bool(
                isinstance(context_outcome, dict)
                and context_outcome.get("status") == "applied")
            dispositions.extend(_claim_disposition(
                item, dossier_index=index,
                disposition=(
                    "used" if deterministic_used
                    and str(item.get("claim_id")) in deterministic_claim_ids
                    else "scenario"),
                reason_code=(
                    "deterministic_claim_applied" if deterministic_used
                    and str(item.get("claim_id")) in deterministic_claim_ids
                    else "interpretation_only_no_numeric_path"),
                reason=(
                    "Verified claim was applied through the deterministic "
                    "context contract." if deterministic_used
                    and str(item.get("claim_id")) in deterministic_claim_ids
                    else "Verified claim is retained for interpretation but "
                    "did not alter the selected numeric forecast."),
            ) for item in claims)
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
                        support="hypothetical_sensitivity",
                        automation_eligible=False,
                        claim_ids=[str(item) for item in proposal.get("claim_ids") or []],
                        assumptions=[str(proposal.get("rationale") or ""),
                                     str(proposal.get("uncertainty_basis") or "")],
                        source_seal=str(dossier["seal_sha256"]),
                        effect={**proposal, "composition_assessment": assessment},
                    ))
                    emitted.append(identifier)
                    dispositions.append({
                        "context_id": f"dossier-{index}:effect-proposal",
                        "disposition": "scenario",
                        "reason_code": "effect_proposal_composed",
                        "reason": (
                            "The cited effect was composed into a sealed, "
                            "conditional scenario; it does not alter the "
                            "immutable primary or authorize automation."
                        ),
                        "scenario_ids": [identifier],
                        "claim_ids": [str(item) for item in
                                      proposal.get("claim_ids") or []],
                    })
        replay_admitted = False
        if candidate:
            # Preserve the v0.1 public identifier while making the less
            # authoritative origin explicit in the typed role. A model may
            # supply this alongside a typed effect; the selector sees both.
            identifier = f"prior-assisted-{index}"
            candidate_critique = dossier.get("candidate_critique") or {}
            candidate_origin = str(
                candidate_critique.get("candidate_origin") or "model_authored")
            conditional_replay = candidate.get("conditional_replay") or {}
            calibration_replay = candidate.get("calibration_replay") or {}
            replay_admitted = (
                candidate_origin == "observation_interpretation_counterfactual"
                and conditional_replay.get("selection_eligible") is True)
            selection_eligible = candidate_critique.get(
                "selection_eligible", True) is True
            candidate_claims = {str(item) for item in
                                candidate.get("claim_ids") or []}
            governed_by_transformation = any(
                candidate_claims and candidate_claims.intersection(claims)
                for claims in transformation_claim_sets)
            governed_by_deterministic_claim = bool(
                candidate_origin == "model_authored"
                and candidate_claims.intersection(deterministic_claim_ids))
            relevant_observation_replays = [
                item.get("conditional_replay") or {}
                for item in dossier.get("observation_interpretations") or []
                if candidate_claims.intersection({
                    str(claim_id) for claim_id in item.get("claim_ids") or []
                })
            ]
            replay_insufficient_only = bool(relevant_observation_replays) and all(
                str(replay.get("status") or "").startswith("insufficient")
                for replay in relevant_observation_replays)
            human_selection_eligible = bool(
                selection_eligible
                or (candidate_origin == "model_authored"
                    and candidate_critique.get("status") == "accepted"
                    and replay_insufficient_only
                    and not governed_by_transformation
                    and not governed_by_deterministic_claim))
            if governed_by_transformation or governed_by_deterministic_claim:
                # A model cannot bypass a failed replay/admission check by
                # restating its own forecast under the same cited claims. The
                # executable path owns numeric authority; the model path stays
                # visible for explanation, comparison, and outcome scoring.
                selection_eligible = False
            scenarios.append(_scenario(
                identifier,
                ("calibration_counterfactual" if
                 candidate_origin == "calibration_counterfactual" else
                 "observation_counterfactual" if candidate_origin ==
                 "observation_interpretation_counterfactual" else
                 "model_authored"),
                _candidate_rows(candidate, primary),
                support=("conditionally_supported" if replay_admitted
                         else "prior_assisted"), automation_eligible=False,
                selection_eligible=selection_eligible,
                human_selection_eligible=human_selection_eligible,
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
                      if governed_by_transformation else []),
                    *(["A deterministic absolute/range context contract over "
                        "the same cited claims owns recommendation authority; "
                        "this model path is retained only for outcome scoring."]
                      if governed_by_deterministic_claim else [])],
                source_seal=str(dossier["seal_sha256"]),
                effect={
                    "candidate_origin": candidate_origin,
                    "conditional_replay": conditional_replay,
                    "calibration_replay": calibration_replay,
                },
            ))
            emitted.append(identifier)
        dispositions.extend(_claim_disposition(
            item, dossier_index=index, disposition="scenario",
            reason_code=("conditional_replay_admitted" if replay_admitted
                         else "prior_assisted_not_historically_admitted"),
            scenario_ids=emitted,
        ) for item in claims)
    if len(scenarios) > MAX_SCENARIOS:
        role_priority = {
            "immutable_primary": 100,
            "historically_admitted": 95,
            "retrospectively_validated": 92,
            "context_conditioned": 90,
            "fitted_context_candidate": 80,
            "effect_composed": 70,
            "model_authored": 60,
            "observation_counterfactual": 75,
            "calibration_counterfactual": 76,
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
    for disposition in dispositions:
        if (disposition.get("disposition") == "rejected"
                and not isinstance(disposition.get("recovery_action"), dict)):
            # Older/external dossiers may carry a prose recovery string. A
            # malformed teaching signal must not crash an agent adapter or
            # masquerade as a ready-to-issue action.
            disposition["recovery_action"] = _context_recovery(disposition)
    return scenarios, dispositions


def validate_scenario_selection(raw: Any, *, scenarios: list[dict[str, Any]],
                                dossiers: list[dict[str, Any]] | None = None,
                                known_evidence_ids: set[str] | None = None,
                                required_counterevidence_ids: set[str] | None = None,
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
    if selected_scenario.get("human_selection_eligible",
                             selected_scenario.get(
                                 "selection_eligible", True)) is not True:
        raise ValueError("scenario selection cannot promote a candidate with an invalid derivation")
    dominant = dominant_scenario_id(scenarios)
    if dominant is not None and selected != dominant:
        raise ValueError(
            "scenario selection cannot override the evidence-dominant path")
    claim_ids = {str(claim.get("claim_id")) for dossier in dossiers or []
                 for claim in dossier.get("claims") or []}
    hypothesis_ids = {
        str(hypothesis.get("hypothesis_id")) for dossier in dossiers or []
        for hypothesis in dossier.get("hypotheses") or []
        if hypothesis.get("hypothesis_id")
    }
    claim_ids.update(hypothesis_ids)
    claim_ids.update(known_evidence_ids or set())
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
    counter_hypotheses = {
        str(hypothesis.get("hypothesis_id")) for dossier in dossiers or []
        for hypothesis in dossier.get("hypotheses") or []
        if hypothesis.get("kind") == "unsupported"
        and hypothesis.get("hypothesis_id")
    }
    counter_hypotheses.update(required_counterevidence_ids or set())
    if (selected_scenario.get("role") == "model_authored"
            and counter_hypotheses
            and not counter_hypotheses.intersection(counter)):
        raise ValueError(
            "prior-assisted selection must cite compiled counterevidence")
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
    claims.extend({
        "claim_id": str(hypothesis.get("hypothesis_id")),
        "relation": ("counterevidence"
                     if hypothesis.get("kind") == "unsupported"
                     else f"hypothesis:{hypothesis.get('kind', 'unknown')}"),
        "mechanism": str(hypothesis.get("rationale") or "")[:1000],
        "direction": hypothesis.get("direction"),
        "validation": hypothesis.get("validation"),
    } for dossier in dossiers or [] if verify_temporal_dossier_seal(dossier)
      for hypothesis in dossier.get("hypotheses") or []
      if hypothesis.get("hypothesis_id"))
    observation_evidence = [{
        "interpretation_id": item.get("interpretation_id"),
        "claim_ids": list(item.get("claim_ids") or []),
        "kind": item.get("kind"),
        "excluded_observations": item.get("excluded_observations"),
        "retained_observations": item.get("retained_observations"),
        "input_mutated": item.get("input_mutated"),
        "conditional_replay": {
            key: (item.get("conditional_replay") or {}).get(key)
            for key in ("status", "origins", "minimum_origins",
                        "selection_eligible")
            if (item.get("conditional_replay") or {}).get(key) is not None
        },
    } for dossier in dossiers or [] if verify_temporal_dossier_seal(dossier)
      for item in dossier.get("observation_interpretations") or []]
    dominant = dominant_scenario_id(scenarios)
    return {
        "selection_required": dominant is None,
        "deterministic_scenario_id": dominant,
        "selection_basis": ("governed_evidence_dominance" if dominant
                            else "ambiguous_evidence_requires_bounded_ranking"),
        "instruction": (
            "Rank only the supplied scenario_ids. Explain the ranking using "
            "claim_ids (including compiled hypothesis ids), name all material "
            "counterevidence, and weigh any accepted historical-contamination "
            "evidence against the replay strength of alternatives. Give "
            "confidence and state what "
            "would change the selection. Do not output forecast numbers, "
            "support labels, or automation advice."),
        "scenarios": [{
            "scenario_id": item["scenario_id"], "role": item["role"],
            "support": item["support"], "claim_ids": item["claim_ids"],
            "human_selection_eligible": item.get(
                "human_selection_eligible",
                item.get("selection_eligible", True)),
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
            "derivation": {
                "assumptions": list(item.get("assumptions") or [])[:2],
                "conditional_replay_status": str(
                    (((item.get("effect") or {}).get(
                        "conditional_replay") or {}).get("status") or
                     "not_applicable")),
                "historically_admitted": bool(
                    ((item.get("effect") or {}).get(
                        "conditional_replay") or {}).get(
                            "selection_eligible") is True),
                "human_recommendation_eligible": bool(
                    ((item.get("effect") or {}).get(
                        "conditional_replay") or {}).get(
                            "human_recommendation_eligible") is True),
                "point_relative_improvement": (
                    ((item.get("effect") or {}).get(
                        "conditional_replay") or {}).get(
                            "relative_improvement")),
                "probabilistic_relative_improvement": (
                    ((item.get("effect") or {}).get(
                        "conditional_replay") or {}).get(
                            "probabilistic_relative_improvement")),
                "chronological_block_wins": (
                    ((item.get("effect") or {}).get(
                        "conditional_replay") or {}).get(
                            "chronological_block_wins")),
                "required_block_wins": (
                    ((item.get("effect") or {}).get(
                        "conditional_replay") or {}).get(
                            "required_block_wins")),
                "calibration_replay": ({
                    key: ((item.get("effect") or {}).get(
                        "calibration_replay") or {}).get(key)
                    for key in (
                        "status", "correction", "stated_rate_per_hour",
                        "drift_start", "repair_boundary", "family",
                        "expanding_origins", "candidate_mae",
                        "human_recommendation_eligible")
                    if ((item.get("effect") or {}).get(
                        "calibration_replay") or {}).get(key) is not None
                } or None),
                "admission_withheld_reason": (
                    ((item.get("effect") or {}).get(
                        "conditional_replay") or {}).get(
                            "admission_withheld_reason")),
                "history_contamination_claimed": bool(observation_evidence),
                "primary_retains_claimed_contamination": bool(
                    observation_evidence
                    and item.get("role") == "immutable_primary"),
                "conditional_path_addresses_claimed_contamination": bool(
                    item.get("role") != "immutable_primary"
                    and set(item.get("claim_ids") or []).intersection({
                        str(claim_id)
                        for evidence in observation_evidence
                        for claim_id in evidence.get("claim_ids") or []
                    })),
            },
        } for item in scenarios],
        "claims": claims,
        "observation_evidence": observation_evidence,
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
                            if item["role"] == "historically_admitted"), None)
        selected_id = selected_id or next((
            item["scenario_id"] for item in scenarios
            if item["role"] == "retrospectively_validated"), None)
        selected_id = selected_id or next((item["scenario_id"] for item in scenarios
                            if item["role"] == "context_conditioned"), None)
        admitted = [item for item in scenarios
                    if item["role"] == "fitted_context_candidate"
                    and ((item.get("effect") or {}).get("evidence") or {}).get("decisive")]
        if selected_id is None and admitted:
            selected_id = max(
                admitted,
                key=lambda item: item["effect"]["evidence"]["score"]
            )["scenario_id"]
        selected_id = selected_id or next((
            item["scenario_id"] for item in scenarios
            if item["role"] == "observation_counterfactual"
            and ((item.get("effect") or {}).get(
                "conditional_replay") or {}).get(
                    "selection_eligible") is True), None)
        selected_id = selected_id or next((
            item["scenario_id"] for item in scenarios
            if item["role"] == "observation_counterfactual"
            and item.get("selection_eligible", True) is True
            and ((item.get("effect") or {}).get(
                "conditional_replay") or {}).get(
                    "human_recommendation_eligible") is True), None)
        selected_id = selected_id or next((
            item["scenario_id"] for item in scenarios
            if item["role"] == "calibration_counterfactual"
            and item.get("selection_eligible", True) is True
            and ((item.get("effect") or {}).get(
                "calibration_replay") or {}).get(
                    "human_recommendation_eligible") is True), None)
        selected_id = selected_id or next((item["scenario_id"] for item in scenarios
                            if item["role"] in {"effect_composed",
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
    selected_role = str(selected.get("role") or "unknown")
    if selection is not None:
        selection_method = "governed_scenario_selection"
    elif selected_role == "historically_admitted":
        selection_method = "historical_evidence_dominance"
    elif selected_role == "retrospectively_validated":
        selection_method = "retrospective_fixed_specification_evidence"
    elif selected_role == "context_conditioned":
        selection_method = "verified_context_contract"
    elif selected_role == "fitted_context_candidate":
        selection_method = "out_of_sample_evidence_dominance"
    elif selected_role == "calibration_counterfactual":
        selection_method = "source_determined_calibration_best_effort"
    elif selected_role == "observation_counterfactual":
        replay = ((selected.get("effect") or {}).get(
            "conditional_replay") or {})
        selection_method = (
            "conditional_replay_evidence" if replay.get(
                "selection_eligible") is True
            else "conditional_replay_best_effort")
    elif selected_role in {
            "model_authored", "model_authored_transformation",
            "effect_composed"}:
        selection_method = (
            "default_prior_assisted_lane"
            if selected.get("support") == "prior_assisted"
            else "default_conditional_scenario_lane")
    else:
        selection_method = "immutable_primary_default"
    prior_assisted_default = selection_method == "default_prior_assisted_lane"
    dispositions = _scope_recovery_actions([{
        **item,
        "disposition": (
            "used" if selected_id in (item.get("scenario_ids") or [])
            else item.get("disposition")),
        **({"selection_role": "human_facing_recommendation"}
           if selected_id in (item.get("scenario_ids") or []) else {}),
    } for item in dispositions])
    recommendation_authority = {
        "selected_role": selected_role,
        "selection_method": selection_method,
        "independent_selection_performed": selection is not None,
        "historically_admitted": selected_role == "historically_admitted",
        "conditional_replay_admitted": (
            selected_role == "observation_counterfactual"
            and ((selected.get("effect") or {}).get(
                "conditional_replay") or {}).get("selection_eligible") is True),
        "prior_assisted": selected.get("support") == "prior_assisted",
        "human_review_required": bool(
            prior_assisted_default or not selected.get("automation_eligible")),
        "reason": (
            "A sealed prior-assisted path is the human-facing best estimate, "
            "but it was not independently ranked or historically admitted."
            if prior_assisted_default else
            "A sealed conditional scenario is the human-facing best estimate, "
            "but it is hypothetical, was not historically admitted, and "
            "cannot authorize automation."
            if selection_method == "default_conditional_scenario_lane" else
            "A fixed observation counterfactual beat the strongest raw "
            "comparator under expanding-origin conditional replay; it remains "
            "non-automatable and requires human review."
            if selection_method == "conditional_replay_evidence" else
            "A fixed observation counterfactual improved both point and "
            "probabilistic replay in two chronological blocks, but missed "
            "the strict admission margin. It is a non-automatable, "
            "human-facing best effort only."
            if selection_method == "conditional_replay_best_effort" else
            "A source-stated additive measurement drift was removed from a "
            "copy of history before a fold-tested forecast was fit. This is "
            "a prior-assisted human recommendation and cannot authorize automation."
            if selection_method == "source_determined_calibration_best_effort" else
            "A fixed source-supplied specification beat the baseline on "
            "per-origin historical observations. The specification itself "
            "was not known at those origins, so this is retrospective "
            "validation for human use, not bitemporal historical admission."
            if selection_method ==
            "retrospective_fixed_specification_evidence" else
            "Recommendation authority follows the disclosed selection method."
        ),
    }
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
        "context_summary": _context_summary(dispositions),
        "temporal_state": build_temporal_state(result, dossiers=dossiers),
        "scenario_selection": selection,
        "recommendation_authority": recommendation_authority,
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


def select_publication(payload: dict[str, Any], raw_selection: dict[str, Any]
                       ) -> dict[str, Any]:
    """Apply a number-free governed ranking to an existing sealed portfolio.

    Forecast paths and their seals are reused byte-for-byte. The operation may
    only change which path is shown first; it cannot upgrade support or grant
    automation authority.
    """
    if not verify_publication(payload):
        raise ValueError("refusing to select from an invalid publication")
    if payload.get("mode") == "strict":
        raise ValueError("strict publications cannot be reranked")
    portfolio = [dict(item) for item in payload.get("candidate_portfolio") or []]
    evidence = (payload.get("selection_contract") or {}).get("claims") or []
    known_evidence_ids = {
        str(item.get("claim_id")) for item in evidence
        if isinstance(item, dict) and item.get("claim_id")
    }
    required_counterevidence_ids = {
        str(item.get("claim_id")) for item in evidence
        if isinstance(item, dict) and item.get("claim_id")
        and item.get("relation") == "counterevidence"
    }
    selection = validate_scenario_selection(
        raw_selection, scenarios=portfolio, dossiers=None,
        known_evidence_ids=known_evidence_ids,
        required_counterevidence_ids=required_counterevidence_ids)
    if selection is None:
        raise ValueError("scenario selection is required")
    selected = next(item for item in portfolio
                    if item["scenario_id"] == selection["selected_scenario_id"])
    primary = next(item for item in portfolio
                   if item["scenario_id"] == "primary")
    result = {key: value for key, value in payload.items()
              if key != "publication_seal_sha256"}
    dispositions = _scope_recovery_actions([{
        **item,
        "disposition": (
            "used" if selected["scenario_id"] in
            (item.get("scenario_ids") or []) else item.get("disposition")),
        **({"selection_role": "human_facing_recommendation"}
           if selected["scenario_id"] in (item.get("scenario_ids") or [])
           else {}),
    } for item in payload.get("context_dispositions") or []])
    result.update({
        "recommended_scenario_id": selected["scenario_id"],
        "recommended_forecast": selected["forecast"],
        "recommended_support": selected["support"],
        "primary_forecast": primary["forecast"],
        "primary_forecast_unchanged": True,
        "scenario_selection": selection,
        "context_dispositions": dispositions,
        "context_summary": _context_summary(dispositions),
        "scenarios": (portfolio if payload.get("mode") == "scenario" else
                      [primary, selected] if selected is not primary else [primary]),
        "recommendation_authority": {
            "selected_role": str(selected.get("role") or "unknown"),
            "selection_method": "governed_scenario_selection",
            "independent_selection_performed": True,
            "historically_admitted": (
                selected.get("role") == "historically_admitted"),
            "prior_assisted": selected.get("support") == "prior_assisted",
            "human_review_required": not bool(
                selected.get("automation_eligible")),
            "reason": (
                "A bounded number-free ranking selected one existing sealed "
                "path; forecast values and support were unchanged."),
        },
        "automation": {
            "eligible": False,
            "explicit_policy_supplied": False,
            "policy_complete": False,
            "requested": False,
            "reason": "scenario selection cannot authorize automation",
        },
        "supersedes_publication_seal_sha256": payload[
            "publication_seal_sha256"],
    })
    result["publication_seal_sha256"] = _seal(result)
    if not verify_publication(result):
        raise ValueError("scenario selection produced an invalid publication")
    return result


def write_selected_publication(source: str | Path,
                               payload: dict[str, Any]) -> Path:
    """Persist a content-addressed selection beside its source publication."""
    if not verify_publication(payload):
        raise ValueError("refusing to persist an invalid selected publication")
    source_path = Path(source)
    seal = payload["publication_seal_sha256"]
    destination = source_path.parent / f"selection_{seal[:16]}.publication.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise ValueError("conflicting content-addressed selected publication")
        return destination
    destination.write_text(encoded, encoding="utf-8")
    return destination


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
                # Candidate paths are conditional answers awaiting outcomes;
                # use the tracking store's existing typed synthesis channel
                # instead of inventing a publication-only label.
                "label": "conditional_answer", "value": candidate["scenario_id"],
                "channel": "candidate_portfolio",
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
