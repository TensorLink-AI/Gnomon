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
import statistics
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


def _covariate_input_evaluation(result: dict[str, Any]) -> dict[str, Any] | None:
    evidence = result.get("covariates")
    if not isinstance(evidence, dict):
        return None
    considered = bool(evidence.get("considered"))
    admitted = bool(evidence.get("admitted"))
    status = ("admitted" if admitted else "evaluated_not_admitted"
              if considered else "received_not_evaluable")
    return {
        "status": status,
        "received": True,
        "evaluated": considered,
        "admitted": admitted,
        "retained": list(evidence.get("retained") or []),
        "rejected": list(evidence.get("rejected") or []),
        "reason": (
            "Covariate input was admitted by fold-safe evaluation."
            if admitted else
            "Covariate input was evaluated but did not beat the governed baseline."
            if considered else
            "Covariate input passed ingestion but the base evaluation could not "
            "support an admission test."
        ),
    }


def _context_summary(
        dispositions: list[dict[str, Any]],
        input_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    if not dispositions and input_evaluation:
        status = str(input_evaluation["status"])
        message = str(input_evaluation["reason"])
    elif not dispositions:
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
        "follow_up_required_for_current_recommendation": status in {
            "rejected", "received_not_evaluable"},
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
    primary = next((item for item in scenarios
                    if item.get("role") == "immutable_primary"), None)
    alternatives = [item for item in scenarios
                    if item.get("role") != "immutable_primary"]
    if primary is not None and alternatives and not any(
            item.get("human_selection_eligible",
                     item.get("selection_eligible", True)) is True
            for item in alternatives):
        # Failed admission is already a deterministic result. Asking an LLM
        # to rank a path it is forbidden to select adds latency and creates a
        # misleading appearance of discretion where none exists.
        return str(primary["scenario_id"])
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
    seasonal_assisted = [
        item for item in scenarios
        if item.get("role") == "model_assisted"
        and item.get("human_selection_eligible") is True
        and ((item.get("effect") or {}).get("validation") or {}).get(
            "basis") == "full_cycle_prequential"
        and ((item.get("effect") or {}).get("validation") or {}).get(
            "complete_phase_coverage") is True]
    if len(seasonal_assisted) == 1:
        # This is a predeclared structured baseline, not a winner selected
        # from an expanding model tournament. A complete, fold-safe phase
        # sweep already made the numeric choice; an LLM may explain it but
        # cannot inconsistently demote it in best-effort publication.
        return str(seasonal_assisted[0]["scenario_id"])
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
    declarative = [item for item in scenarios
                   if item.get("role") == "model_authored_transformation"
                   and item.get("selection_eligible", True) is True
                   and item.get("support") == "prior_assisted"
                   and all(((item.get("effect") or {}).get("validation") or {}).get(
                               key) is True
                           for key in ("approved_ast", "constants_entailed",
                                       "known_at_cutoff", "units_checked"))]
    if len(declarative) == 1:
        # In best-effort mode a single source-grounded executable answers the
        # caller's stated conditional question. Asking a model to choose
        # between that path and a context-free primary adds no evidence and
        # has allowed the model to ignore its own cited equation. Support
        # remains prior-assisted and automation remains forbidden.
        return str(declarative[0]["scenario_id"])
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


def _primary_disagreement(
    candidate: list[dict[str, Any]], primary: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe candidate distance from primary without implying either wins."""
    if not candidate or len(candidate) != len(primary):
        return {"available": False, "reason": "unaligned_paths"}
    candidate_points = [float(row.get("q50", row.get("point")))
                        for row in candidate]
    primary_points = [float(row.get("q50", row.get("point")))
                      for row in primary]
    widths = [abs(float(row.get("q90", point)) -
                  float(row.get("q10", point)))
              for row, point in zip(primary, primary_points)]
    positive_widths = [value for value in widths if value > 0]
    if positive_widths:
        scale = statistics.median(positive_widths)
        scale_basis = "median_primary_q80_width"
    else:
        increments = [abs(right - left) for left, right in
                      zip(primary_points, primary_points[1:])
                      if right != left]
        if increments:
            scale = statistics.median(increments)
            scale_basis = "median_nonzero_primary_increment"
        else:
            scale = max(1.0, abs(statistics.median(primary_points)) * .01)
            scale_basis = "primary_level_floor"
    scaled = [abs(candidate_point - primary_point) / max(scale, 1e-12)
              for candidate_point, primary_point in
              zip(candidate_points, primary_points)]
    direction_disagreements = []
    tolerance = max(scale, 1e-12) * 1e-6
    for cp0, cp1, pp0, pp1 in zip(
            candidate_points, candidate_points[1:],
            primary_points, primary_points[1:]):
        candidate_sign = 1 if cp1 - cp0 > tolerance else \
            -1 if cp1 - cp0 < -tolerance else 0
        primary_sign = 1 if pp1 - pp0 > tolerance else \
            -1 if pp1 - pp0 < -tolerance else 0
        direction_disagreements.append(candidate_sign != primary_sign)
    return {
        "available": True,
        "interpretation": "difference_not_skill",
        "scale_basis": scale_basis,
        "median_absolute_difference_scaled": statistics.median(scaled),
        "max_absolute_difference_scaled": max(scaled),
        "direction_disagreement_fraction": (
            sum(direction_disagreements) / len(direction_disagreements)
            if direction_disagreements else 0.0),
    }


def _normalize_sampled_prior_uncertainty(
    candidate: list[dict[str, Any]], primary: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prevent a handful of LLM draws from masquerading as calibrated tails.

    The candidate median remains untouched.  Five stochastic paths cannot
    estimate calibrated 10th/90th percentiles, so their dispersion remains a
    diagnostic while the conditional path inherits the immutable primary's
    calibrated offsets. The derived scenario is resealed and carries this rule
    in its effect metadata; neither source path is mutated.
    """
    if not candidate or len(candidate) != len(primary):
        return candidate, {"applied": False, "reason": "unaligned_paths"}
    rows = []
    adjusted = 0
    for candidate_row, primary_row in zip(candidate, primary):
        if str(candidate_row.get("timestamp")) != str(primary_row.get("timestamp")):
            return candidate, {"applied": False, "reason": "unaligned_timestamps"}
        centre = float(candidate_row.get("q50", candidate_row.get("point")))
        lower = float(candidate_row.get("q10", centre))
        upper = float(candidate_row.get("q90", centre))
        primary_centre = float(primary_row.get(
            "q50", primary_row.get("point")))
        primary_lower = float(primary_row.get("q10", primary_centre))
        primary_upper = float(primary_row.get("q90", primary_centre))
        resolved_lower = centre - (primary_centre - primary_lower)
        resolved_upper = centre + (primary_upper - primary_centre)
        adjusted += int(resolved_lower != lower or resolved_upper != upper)
        rows.append({**candidate_row, "q10": resolved_lower,
                     "q50": centre, "q90": resolved_upper})
    return rows, {
        "applied": True,
        "basis": "immutable_primary_offsets_around_sampled_median",
        "candidate_centre_unchanged": True,
        "primary_forecast_unchanged": True,
        "rows_adjusted": adjusted,
        "interpretation": "calibrated_offsets_not_sampling_dispersion",
    }


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
    if code == "INSUFFICIENT_RELATIONSHIP_HISTORY":
        return {
            "code": "collect_relationship_history",
            "message": (
                "Keep the immutable primary and collect the aligned target "
                "and driver observations named in the rejection. Rerun the "
                "same sealed lag structure once the stated minimum is met."),
            "required_evidence": [
                "additional aligned target observations",
                "additional aligned driver observations",
            ],
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
    cited_fact = {
        "source_span": claim.get("source_span"),
        "relation": claim.get("relation"),
        "confidence": claim.get("confidence"),
    }
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
            "cited_fact": cited_fact,
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
        range_constraint = claim.get("relation") == "constrains_range"
        return {
            "context_id": f"dossier-{dossier_index}:{claim.get('claim_id')}",
            "disposition": "scenario",
            "reason_code": "background_context_not_conditioned",
            "reason": (
                "The source states background evidence or a relationship, "
                "not a dated event. It remains available to interpretation "
                "but was not treated as a deterministic forecast adjustment."),
            "claim_id": claim.get("claim_id"),
            "cited_fact": cited_fact,
            "scenario_ids": list(scenario_ids or []),
            "recovery_action": {
                "code": "provide_applicability_evidence",
                "message": (
                    "Provide the aligned reference path and evidence for how "
                    "its level and timing transfer to this target; a single "
                    "peer bound cannot identify a forecast path."
                    if range_constraint else
                    "Provide the current driver observations, comparison "
                    "period, or an explicit bounded scenario assumption "
                    "needed to apply this background evidence."),
                "required_evidence": ([
                    "reference observations over the forecast grid",
                    "target-to-reference scale or historical overlap",
                    "target and entity scope",
                ] if range_constraint else [
                    "applicable driver observations or comparison period",
                    "target and entity scope",
                    "bounded scenario assumption when historical validation "
                    "is unavailable",
                ]),
                "automation_eligible": False,
                "required_for_current_recommendation": False,
            },
        }
    return {
        "context_id": f"dossier-{dossier_index}:{claim.get('claim_id')}",
        "disposition": disposition, "reason_code": reason_code,
        "reason": reason, "claim_id": claim.get("claim_id"),
        "cited_fact": cited_fact,
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
    model_assisted = result.get("model_assisted") or {}
    assisted_points = model_assisted.get("points") or []
    if (isinstance(model_assisted, dict)
            and len(assisted_points) == len(primary) and primary):
        assisted_validation = model_assisted.get("validation") or {}
        assisted_human_eligible = bool(
            model_assisted.get("support") == "conditionally_supported"
            or (assisted_validation.get("basis") == "full_cycle_prequential"
                and assisted_validation.get(
                    "complete_phase_coverage") is True))
        assisted_rows: list[dict[str, Any]] = []
        for row, raw_point in zip(primary, assisted_points):
            point = float(raw_point)
            centre = float(row.get("q50", row.get("point", point)))
            assisted_rows.append({
                "timestamp": row["timestamp"], "point": point,
                # The point lane deliberately owns no calibrated intervals.
                # For a bounded human comparison, preserve the immutable
                # primary's uncertainty offsets rather than inventing a new
                # spread or presenting a zero-width distribution.
                "q10": point + float(row.get("q10", centre)) - centre,
                "q50": point,
                "q90": point + float(row.get("q90", centre)) - centre,
            })
        scenarios.append(_scenario(
            "model-assisted", "model_assisted", assisted_rows,
            support=str(model_assisted.get("support") or "prior_assisted"),
            automation_eligible=False,
            selection_eligible=assisted_human_eligible,
            human_selection_eligible=assisted_human_eligible,
            assumptions=[
                "The point path won only the disclosed reduced-rigor "
                "out-of-sample comparison.",
                "Scenario interval offsets are inherited from the immutable "
                "primary and are not independently calibrated for this path.",
            ],
            effect={
                "candidate_origin": "model_assisted",
                "validation": assisted_validation,
                "plausibility": model_assisted.get("plausibility") or {},
                "selected_model": model_assisted.get("selected_model"),
                "interval_basis": "immutable_primary_offsets",
            },
        ))
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
            **({"source_span": str(item["source_span"])}
               if item.get("source_span") else {}),
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
                 or validation.get("recurrence_replay_admitted") is True)
            and not (lane == "historically_testable"
                     and validation.get("beats_baseline") is False))
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
        weak_historical_fit = bool(
            lane == "historically_testable" and not admitted
            and validation.get("beats_baseline") is False)
        dispositions.append({
            "context_id": candidate_id,
            "disposition": "used" if admitted else "scenario",
            "reason_code": (
                            "retrospectively_tested_transformation_selected"
                            if retrospective else
                            "historically_tested_transformation_admitted"
                            if admitted else
                            "historical_relationship_did_not_beat_baseline"
                            if weak_historical_fit else
                            "transformation_retained_plausibility_failed"
                            if not selection_eligible else
                            "prior_assisted_transformation"
                            if lane == "prior_assisted" else
                            "scenario_only_transformation"),
            "reason": (
                "The cited relationship was fitted and tested on aligned "
                "historical origins but did not beat last-value. The immutable "
                "primary remains the recommendation; no context correction is "
                "required."
                if weak_historical_fit else None),
            "scenario_ids": [identifier], "evidence": evidence,
        })

    transformation_claim_sets = [
        set(item.get("claim_ids") or []) for item in scenarios
        if item.get("role") in {"historically_admitted",
                                "retrospectively_validated",
                                "model_authored_transformation"}
    ]
    transformation_claim_ids = set().union(*transformation_claim_sets) \
        if transformation_claim_sets else set()

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
    # A model-authored interpretation may coexist with an independently
    # replayed companion executable. Match them by authenticated source spans,
    # not dossier-local claim ids. This may make the model path selectable for
    # human review, never evidence-supported or automation-eligible.
    governed_companion_sources: list[set[str]] = []
    for source_dossier in dossiers or []:
        if not verify_temporal_dossier_seal(source_dossier):
            continue
        source_critique = source_dossier.get("candidate_critique") or {}
        source_candidate = source_dossier.get("forecast_candidate") or {}
        if (source_critique.get("candidate_origin") not in {
                "governed_companion_mapping",
                "governed_categorical_state_mapping"}
                or (source_candidate.get("validation") or {}).get(
                    "beats_baseline") is not True):
            continue
        source_ids = {str(item) for item in
                      source_candidate.get("claim_ids") or []}
        source_spans = {
            " ".join(str(claim.get("source_span") or "").split())
            for claim in source_dossier.get("claims") or []
            if str(claim.get("claim_id")) in source_ids
            and str(claim.get("source_span") or "").strip()
        }
        if source_spans:
            governed_companion_sources.append(source_spans)
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
        deterministic_events = deterministic_events_from_claims(dossier)
        deterministic_claim_ids = {
            str(event.get("derived_from_claim_id"))
            for event in deterministic_events
            if event.get("derived_from_claim_id")}
        absolute_override_claim_ids = {
            str(event.get("derived_from_claim_id"))
            for event in deterministic_events
            if str(event.get("event_type") or "").startswith("override:")
            and event.get("derived_from_claim_id")
        }
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
            # A transformation scenario already owns the complete typed
            # disposition and cites its source claims. Emitting those claims
            # again as "interpretation only" contradicts the numeric path and
            # makes a fully handled instruction look partially unresolved.
            standalone_claims = [
                item for item in claims
                if str(item.get("claim_id")) not in transformation_claim_ids]
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
            ) for item in standalone_claims)
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
                and candidate_claims.intersection(
                    absolute_override_claim_ids
                    if not (isinstance(context_outcome, dict)
                            and context_outcome.get("status") == "applied")
                    else deterministic_claim_ids))
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
            candidate_source_spans = {
                " ".join(str(claim.get("source_span") or "").split())
                for claim in claims
                if str(claim.get("claim_id")) in candidate_claims
                and str(claim.get("source_span") or "").strip()
            }
            governed_companion_evidence = bool(
                candidate_origin == "model_authored"
                and candidate_source_spans
                and any(candidate_source_spans <= source_spans
                        for source_spans in governed_companion_sources))
            if governed_by_transformation or governed_by_deterministic_claim:
                # A model cannot bypass a failed replay/admission check by
                # restating its own forecast under the same cited claims. The
                # executable path owns numeric authority; the model path stays
                # visible for explanation, comparison, and outcome scoring.
                selection_eligible = False
            # Compute human eligibility only after applying numeric-authority
            # exclusions. Previously an initially eligible model path kept a
            # stale ``True`` here even after the governed executable over the
            # same claims made it outcome-scoring-only. That let a selector
            # choose model-authored intervals over the engine's executable.
            elicitation = candidate.get("elicitation") or {}
            sampled_prior = bool(
                candidate_origin == "model_authored"
                and elicitation.get("kind") == "sampled_point_paths")
            sampled_prior_sufficient = bool(
                not sampled_prior
                or (isinstance(elicitation.get("accepted_paths"), int)
                    and int(elicitation["accepted_paths"]) >= 3))
            human_selection_eligible = bool(
                sampled_prior_sufficient
                and (selection_eligible
                     or governed_companion_evidence
                     or (candidate_origin == "model_authored"
                         and candidate_critique.get("status") == "accepted"
                         and replay_insufficient_only
                         and not governed_by_transformation
                         and not governed_by_deterministic_claim)))
            candidate_rows = _candidate_rows(candidate, primary)
            uncertainty_normalization = None
            if (candidate_origin == "model_authored"
                    and (candidate.get("elicitation") or {}).get(
                        "kind") == "sampled_point_paths"):
                candidate_rows, uncertainty_normalization = (
                    _normalize_sampled_prior_uncertainty(
                        candidate_rows, primary))
            scenarios.append(_scenario(
                identifier,
                ("calibration_counterfactual" if
                 candidate_origin == "calibration_counterfactual" else
                 "observation_counterfactual" if candidate_origin ==
                 "observation_interpretation_counterfactual" else
                 candidate_origin if candidate_origin in {
                     "governed_companion_mapping",
                     "governed_categorical_state_mapping"} else
                 "model_authored"),
                candidate_rows,
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
                    *(["Fewer than three independent sampled paths survived; "
                       "the candidate remains visible but is insufficient "
                       "for human-facing recommendation selection."]
                      if sampled_prior and not sampled_prior_sufficient else []),
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
                    "elicitation": candidate.get("elicitation") or {},
                    # Every model-authored path exposes one typed uncertainty
                    # contract. Empirical paths are the richer representation;
                    # a compiler-authored quantile path remains a real sealed
                    # distribution, but must say plainly that repeated-sample
                    # stability was not measured. This keeps downstream agents
                    # from treating a missing metadata object as either a point
                    # forecast or implicit confidence evidence.
                    "distribution": ({
                        "kind": "sealed_empirical_model_paths",
                        "sample_count": len(candidate.get("sample_paths") or []),
                        "horizon": len(candidate_rows),
                        "quantile_levels": [0.1, 0.5, 0.9],
                        "source": "sealed_context_receipt",
                        "probabilistic_consumers_should_use": "sample_paths",
                        "stability_evidence": "host_observed",
                        "compact_human_summary": "recommended_forecast",
                        "automation_eligible": False,
                    } if candidate_origin == "model_authored"
                         and candidate.get("sample_paths") else {
                        "kind": "sealed_model_quantiles",
                        "horizon": len(candidate_rows),
                        "quantile_levels": [0.1, 0.5, 0.9],
                        "source": "sealed_context_receipt",
                        "probabilistic_consumers_should_use": "quantiles",
                        "stability_evidence": "not_measured",
                        "historical_skill_evidence": False,
                        "compact_human_summary": "recommended_forecast",
                        "automation_eligible": False,
                    } if candidate_origin == "model_authored" else {
                        "kind": "sealed_governed_quantiles",
                        "candidate_origin": candidate_origin,
                        "horizon": len(candidate_rows),
                        "quantile_levels": [0.1, 0.5, 0.9],
                        "source": "sealed_context_receipt",
                        "probabilistic_consumers_should_use": "quantiles",
                        "historical_skill_evidence": bool(replay_admitted),
                        "validation_status": (
                            str(conditional_replay.get("status"))
                            if conditional_replay else "not_available"),
                        "compact_human_summary": "recommended_forecast",
                        "automation_eligible": False,
                    }),
                    "primary_disagreement": _primary_disagreement(
                        candidate_rows, primary),
                    "uncertainty_normalization": uncertainty_normalization,
                    "governed_companion_evidence": governed_companion_evidence,
                    "conditional_replay": conditional_replay,
                    "calibration_replay": calibration_replay,
                    "validation": candidate.get("validation") or {},
                    "executable": candidate.get("executable") or {},
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
            "governed_companion_mapping": 80,
            "governed_categorical_state_mapping": 80,
            "model_assisted": 78,
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
    required_known = {str(item) for item in
                      required_counterevidence_ids or set()}
    # A sealed publication carries hypotheses in its compact evidence table.
    # Preserve their type when re-ranking without the original dossier.
    hypothesis_ids.update(required_known)
    claim_ids.update(str(item) for item in known_evidence_ids or set()
                     if str(item) not in required_known)
    claim_ids.update(str(item) for scenario in scenarios
                     for item in scenario.get("claim_ids") or [])
    cited = [str(item) for item in raw.get("cited_claim_ids") or []]
    legacy_counter = [str(item) for item in
                      raw.get("counterevidence_claim_ids") or []]
    counter_hypotheses = [str(item) for item in
                          raw.get("counterevidence_hypothesis_ids") or []]
    # Backward-compatible migration for callers using the original, overly
    # broad field. Public output is canonical: claim IDs and hypothesis IDs
    # never share one misleading slot.
    counter_hypotheses = list(dict.fromkeys([
        *counter_hypotheses,
        *[item for item in legacy_counter if item in hypothesis_ids],
    ]))
    counter = list(dict.fromkeys(
        item for item in legacy_counter if item not in hypothesis_ids))
    if set(cited + counter) - claim_ids:
        raise ValueError("scenario selection cites an unknown claim id")
    if set(counter_hypotheses) - hypothesis_ids:
        raise ValueError(
            "scenario selection cites an unknown counterevidence hypothesis id")
    if not cited and not counter and not counter_hypotheses:
        raise ValueError("scenario selection requires cited evidence or counterevidence")
    if set(cited) & set(counter):
        raise ValueError("a claim cannot be both supporting evidence and counterevidence")
    selected_claims = set(next(item for item in scenarios
                               if item["scenario_id"] == selected)["claim_ids"])
    if selected_claims and not selected_claims.intersection(cited):
        raise ValueError("selected conditional scenario requires one of its claims to be cited")
    required_counter_hypotheses = {
        str(hypothesis.get("hypothesis_id")) for dossier in dossiers or []
        for hypothesis in dossier.get("hypotheses") or []
        if hypothesis.get("kind") == "unsupported"
        and hypothesis.get("hypothesis_id")
    }
    required_counter_hypotheses.update(required_known)
    if (selected_scenario.get("role") == "model_authored"
            and required_counter_hypotheses
            and not required_counter_hypotheses.intersection(
                counter_hypotheses)):
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
        "counterevidence_claim_ids": counter,
        "counterevidence_hypothesis_ids": counter_hypotheses,
        "confidence": confidence,
        "rationale": rationale[:1000],
        "what_would_change_selection": flip[:1000],
        "primary_forecast_unchanged": True,
        "support_unchanged": True, "automation_authorized": False,
    }


def best_effort_prior_selection(
    *, scenarios: list[dict[str, Any]],
    dossiers: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Apply the explicit best-effort policy to one sampled model prior.

    This is a publication policy, not an evidence upgrade. It applies only
    when no historically/evidence-dominant path exists and exactly one
    human-eligible model candidate carries a host-aggregated distribution from
    at least three independently elicited paths. Strict and scenario publication
    remain unchanged, and automation remains categorically unavailable.
    """
    if dominant_scenario_id(scenarios) is not None:
        return None
    sampled = []
    for item in scenarios:
        elicitation = ((item.get("effect") or {}).get("elicitation") or {})
        if (item.get("role") == "model_authored"
                and item.get("human_selection_eligible") is True
                and elicitation.get("host_observed") is True
                and elicitation.get("historical_skill_evidence") is False
                and elicitation.get("automation_eligible") is False
                and isinstance(elicitation.get("accepted_paths"), int)
                and int(elicitation["accepted_paths"]) >= 3):
            sampled.append(item)
    if len(sampled) != 1:
        return None
    selected = sampled[0]
    eligible = [item for item in scenarios
                if item.get("human_selection_eligible") is True
                and item is not selected]
    ineligible = [item for item in scenarios
                  if item.get("human_selection_eligible") is not True]
    ranking = [selected["scenario_id"], *[
        item["scenario_id"] for item in eligible], *[
        item["scenario_id"] for item in ineligible]]
    cited = list(dict.fromkeys(str(item) for item in
                              selected.get("claim_ids") or []))
    counter = list(dict.fromkeys(
        str(hypothesis.get("hypothesis_id"))
        for dossier in dossiers or []
        for hypothesis in dossier.get("hypotheses") or []
        if hypothesis.get("kind") == "unsupported"
        and hypothesis.get("hypothesis_id")))
    raw = {
        "selected_scenario_id": selected["scenario_id"],
        "ranking": ranking,
        "cited_claim_ids": cited,
        "counterevidence_claim_ids": [],
        "counterevidence_hypothesis_ids": counter,
        "confidence": .5,
        "rationale": (
            "The caller selected best_effort publication, and one bounded "
            "host-sampled model prior directly conditions on the supplied "
            "claims. Sampling agreement is stability context, not historical "
            "skill; the immutable primary and counterevidence remain visible."),
        "what_would_change_selection": (
            "Historical replay, resolved outcomes, or a supported executable "
            "that contradicts this prior would change the recommendation."),
    }
    selection = validate_scenario_selection(
        raw, scenarios=scenarios, dossiers=dossiers)
    if selection is not None:
        selection["channel"] = "best_effort_sampled_prior_policy"
    return selection


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
    primary_rows = next((item.get("forecast") or [] for item in scenarios
                         if item.get("role") == "immutable_primary"), [])

    def shape_summary(item: dict[str, Any]) -> dict[str, Any]:
        rows = list(item.get("forecast") or [])
        centres = [float(row.get("q50", row.get("point"))) for row in rows]
        if not centres:
            return {"first_q50": None, "last_q50": None, "steps": 0}
        minimum = min(range(len(centres)), key=centres.__getitem__)
        maximum = max(range(len(centres)), key=centres.__getitem__)
        directions = [0 if math.isclose(centres[index], centres[index - 1])
                      else 1 if centres[index] > centres[index - 1] else -1
                      for index in range(1, len(centres))]
        nonzero = [value for value in directions if value]
        turning_points = sum(left != right for left, right in
                             zip(nonzero, nonzero[1:]))
        summary = {
            "first_q50": centres[0], "last_q50": centres[-1],
            "minimum_q50": centres[minimum],
            "minimum_timestamp": rows[minimum].get("timestamp"),
            "maximum_q50": centres[maximum],
            "maximum_timestamp": rows[maximum].get("timestamp"),
            "turning_points": turning_points, "steps": len(rows),
        }
        if len(primary_rows) == len(rows) and rows:
            deviations = [centre - float(primary_rows[index].get(
                "q50", primary_rows[index].get("point")))
                          for index, centre in enumerate(centres)]
            extreme = max(range(len(deviations)),
                          key=lambda index: abs(deviations[index]))
            summary.update({
                "largest_primary_deviation": deviations[extreme],
                "largest_primary_deviation_timestamp": rows[extreme].get(
                    "timestamp"),
            })
        return summary

    def candidate_validation_summary(item: dict[str, Any]) -> dict[str, Any] | None:
        validation = ((item.get("effect") or {}).get("validation") or {})
        if not validation:
            return None
        points = int(validation.get("validation_points") or 0)
        beats = validation.get("beats_baseline") is True
        threshold = validation.get("multiplicity_adjusted_threshold")
        skill = validation.get("skill")
        summary = {
            key: validation.get(key) for key in (
                "scheme", "validation_points", "skill", "beats_baseline",
                "baseline", "candidate_tables",
                "multiplicity_adjusted_threshold",
                "publication_evidence_weight",
                "publication_shrunk_to_baseline",
                "relationship_known_at_each_origin")
            if validation.get(key) is not None
        }
        summary["evidence_sufficiency"] = (
            "supported_replay" if beats and points >= 8 else
            "preliminary_short_replay" if beats else "not_admitted")
        if isinstance(skill, (int, float)) and isinstance(
                threshold, (int, float)):
            summary["skill_margin_over_adjusted_threshold"] = (
                float(skill) - float(threshold))
        return summary

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
            "evidence against the replay strength of alternatives. A role "
            "name is provenance, not proof: compare candidate_validation, "
            "evidence volume, shrinkage, assumptions and path shape. Treat "
            "preliminary_short_replay as useful but insufficient evidence, "
            "not automatic dominance over another bounded human-only path. "
            "Only a scenario whose human_selection_eligible field is true may "
            "be selected; every ineligible scenario must rank below every "
            "eligible scenario, while remaining visible as counterevidence. Give "
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
            "summary": shape_summary(item),
            "derivation": {
                "assumptions": list(item.get("assumptions") or [])[:2],
                "candidate_origin": ((item.get("effect") or {}).get(
                    "candidate_origin")),
                "elicitation": ((item.get("effect") or {}).get(
                    "elicitation") or None),
                "candidate_validation": candidate_validation_summary(item),
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
            "counterevidence_hypothesis_ids": ["hypothesis_id"],
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
    if selection is None and mode == "best_effort":
        selection = best_effort_prior_selection(
            scenarios=scenarios, dossiers=dossiers)
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
    policy_selected = bool(
        selection is not None
        and selection.get("channel") == "best_effort_sampled_prior_policy")
    if policy_selected:
        selection_method = "best_effort_sampled_prior_policy"
    elif selection is not None:
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
        "independent_selection_performed": (
            selection is not None and not policy_selected),
        "historically_admitted": selected_role == "historically_admitted",
        "conditional_replay_admitted": (
            selected_role == "observation_counterfactual"
            and ((selected.get("effect") or {}).get(
                "conditional_replay") or {}).get("selection_eligible") is True),
        "prior_assisted": selected.get("support") == "prior_assisted",
        "human_review_required": bool(
            prior_assisted_default or not selected.get("automation_eligible")),
        "reason": (
            "The caller explicitly requested best_effort publication. One "
            "host-aggregated prior distribution became the human-facing estimate "
            "under that policy; sampling stability is not historical skill, "
            "the immutable primary remains visible, and automation is forbidden."
            if policy_selected else
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
    input_evaluation = _covariate_input_evaluation(result)
    selected_distribution = ((selected.get("effect") or {}).get(
        "distribution"))
    payload = {
        "schema_version": PUBLICATION_VERSION, "artifact_id": artifact_id,
        "mode": mode, "recommended_scenario_id": selected_id,
        "recommended_forecast": selected["forecast"],
        "recommended_support": selected["support"],
        **({
            "recommended_distribution": selected_distribution,
            "recommended_forecast_semantics": "compact_distribution_summary",
        } if selected_distribution else {}),
        "primary_scenario_id": "primary", "primary_forecast": by_id["primary"]["forecast"],
        "primary_forecast_unchanged": True,
        "scenario_count": len(scenarios),
        # Full sealed portfolio is retained for outcome scoring. ``scenarios``
        # is the compact human-facing projection.
        "candidate_portfolio": scenarios,
        "scenarios": scenarios if mode == "scenario" else [by_id["primary"], selected]
                     if selected_id != "primary" else [by_id["primary"]],
        "context_dispositions": dispositions,
        **({"context_input_evaluation": input_evaluation}
           if input_evaluation else {}),
        "context_summary": _context_summary(dispositions, input_evaluation),
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
        "context_summary": _context_summary(
            dispositions, payload.get("context_input_evaluation")),
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
            "scenario_role": selected.get("role"),
            "candidate_origin": ((selected.get("effect") or {}).get(
                "candidate_origin")),
            "scenario_seal_sha256": selected.get("scenario_seal_sha256"),
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
                "scenario_role": candidate.get("role"),
                "candidate_origin": ((candidate.get("effect") or {}).get(
                    "candidate_origin")),
                "scenario_seal_sha256": candidate.get(
                    "scenario_seal_sha256"),
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
