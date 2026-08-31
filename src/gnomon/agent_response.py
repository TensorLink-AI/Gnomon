"""Compact, sealed synthesis obligations for agent-facing tool results.

The contract is a projection of facts already published by Gnomon.  It does
not infer a new forecast, context disposition, support tier, or automation
right.  Agent hosts can use it as a bounded checklist after executing a typed
and fully bound product call.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEMA_VERSION = "0.2"
_MAX_ITEMS = 8


def _seal(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique(values: list[str], limit: int = _MAX_ITEMS) -> tuple[list[str], int]:
    ordered = list(dict.fromkeys(value for value in values if value))
    return ordered[:limit], max(0, len(ordered) - limit)


def _source_references(context: dict[str, Any]) -> list[str]:
    references: list[str] = []
    for evidence in context.get("context_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        source = evidence.get("source") or {}
        reference = source.get("reference") if isinstance(source, dict) else None
        if reference:
            references.append(str(reference))
    return references


def _interval_disclosure_count(result: dict[str, Any]) -> int:
    count = 0
    for warning in result.get("warnings") or []:
        if isinstance(warning, dict):
            message = str(warning.get("message") or "")
        else:
            message = str(warning)
        lowered = message.casefold()
        if any(token in lowered for token in (
                "interval", "coverage", "calibrat", "quantile")):
            count += 1
    return count


def _decision_interpretations(packet: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    for raw in packet.get("interpretations") or []:
        if not isinstance(raw, dict) or raw.get("value") is None:
            continue
        supporting, supporting_omitted = _unique([
            str(value) for value in raw.get("supporting") or []], limit=4)
        conflicting, conflicting_omitted = _unique([
            str(value) for value in raw.get("conflicting") or []], limit=4)
        decision_eligible = raw.get("decision_eligible")
        if decision_eligible is None:
            # Reasoning-packet v0.1 already defined compatible,
            # non-conditional interpretations as selectable through its
            # verifier; preserve that exact legacy meaning when replaying an
            # older sealed answer receipt.
            decision_eligible = bool(
                raw.get("compatible") and not raw.get("conditional_only"))
        row = {
            "value": raw.get("value"),
            "support": str(raw.get("support") or "abstained"),
            "compatible": bool(raw.get("compatible")),
            "decision_eligible": bool(decision_eligible),
            "supporting_evidence": supporting,
            "conflicting_evidence": conflicting,
            **({"conditional_only": True}
               if raw.get("conditional_only") else {}),
            **({"omitted": {
                **({"supporting_evidence": supporting_omitted}
                   if supporting_omitted else {}),
                **({"conflicting_evidence": conflicting_omitted}
                   if conflicting_omitted else {}),
            }} if supporting_omitted or conflicting_omitted else {}),
        }
        rows.append(row)
    return rows[:_MAX_ITEMS], max(0, len(rows) - _MAX_ITEMS)


def _uncertainty_contract(answer: dict[str, Any]) -> dict[str, Any]:
    embedded = answer.get("answer") or {}
    interval = embedded.get("interval")
    status = answer.get("calibration_status")
    if not isinstance(status, dict):
        calibration = answer.get("calibration")
        status = {
            "available": bool(isinstance(calibration, dict) and calibration),
            "applicable": bool(isinstance(calibration, dict) and calibration),
            **({"reason": "not_reported"}
               if not isinstance(calibration, dict) or not calibration else {}),
        }
    return {
        "interval_status": "present" if interval is not None else "withheld",
        **({"interval": interval} if interval is not None else {}),
        "calibration": {
            key: status[key] for key in (
                "available", "applicable", "reason", "folds",
                "requested_horizon") if key in status
        },
    }


def _decision_contract(answer: dict[str, Any], *,
                       answer_receipt: Any = None) -> dict[str, Any] | None:
    question = answer.get("question") or {}
    question_id = question.get("id")
    if not question_id:
        return None
    best = answer.get("best_estimate") or {}
    embedded = answer.get("answer") or {}
    support = str(best.get("support") or embedded.get("support")
                  or (answer.get("support") or {}).get("state") or "abstained")
    conclusion = best.get("value")
    reasoning = embedded.get("reasoning") or {}
    packet = reasoning.get("packet") if isinstance(reasoning, dict) else {}
    packet = packet if isinstance(packet, dict) else {}
    selection = packet.get("selection_contract") or {}
    sufficiency = packet.get("evidence_sufficiency") or {}
    selector = (selection.get("selector") or packet.get("selector")
                or ("gnomon_canonical" if support == "supported" else "model"))
    interpretations, interpretations_omitted = _decision_interpretations(packet)
    authority = ("binding" if support == "supported" else
                 "abstained" if support == "abstained" else "advisory")
    value_status = ("abstention" if support == "abstained"
                    or conclusion is None else "value")
    decision_eligible = value_status == "value" and (
        authority == "binding" or any(
            row["compatible"] and row["decision_eligible"]
            and row["value"] == conclusion
            for row in interpretations
        )
    )
    conditions: list[dict[str, Any]] = []
    conditional = answer.get("conditional_effect")
    if isinstance(conditional, dict):
        conditions.append({
            key: conditional[key] for key in (
                "role", "primary_forecast_unchanged", "provenance")
            if key in conditional
        })
    persistence = embedded.get("conditional_on_persistence")
    if persistence is not None:
        conditions.append({"conditional_on_persistence": persistence})
    context = answer.get("context_assessment") or {}
    primary_unchanged = reasoning.get("primary_forecast_unchanged")
    if primary_unchanged is None:
        primary_unchanged = context.get("canonical_primary_preserved")
    contract = {
        "question_id": str(question_id),
        "property": str(question.get("property") or "unknown"),
        "inference_mode": str(question.get("verb") or "unknown"),
        "conclusion": conclusion,
        "value_status": value_status,
        "support": support,
        "authority": authority,
        "selector": str(selector),
        "evidence_sufficiency": str(
            sufficiency.get("level") or packet.get("sufficiency") or "unknown"),
        "interpretations": interpretations,
        "conditions": conditions,
        "uncertainty": _uncertainty_contract(answer),
        "decision_eligible": decision_eligible,
        "automation_eligible": bool(
            best.get("automation_eligible", embedded.get(
                "automation_eligible")) is True),
        "primary_forecast_unchanged": primary_unchanged is True,
        "provenance": {
            "artifact_id": answer.get("artifact_id"),
            "question_id": str(question_id),
            **({"answer_receipt": str(answer_receipt)}
               if answer_receipt else {}),
        },
        "required": "all_emitted_fields",
        **({"relationship_to_primary": str(
            context["relationship_to_primary"])}
           if context.get("relationship_to_primary") else {}),
        **({"omitted": {"interpretations": interpretations_omitted}}
           if interpretations_omitted else {}),
    }
    return contract


def build_agent_response_contract(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Project exact response obligations from a forecast tool payload.

    ``None`` means the payload has no forecast results.  Bounded arrays retain
    explicit omitted counts and immutable artifact locations; truncation can
    therefore never look like completeness.
    """
    results = [item for item in payload.get("results") or []
               if isinstance(item, dict)]
    if not results:
        return None
    publication = payload.get("publication") or {}
    publication_by_series = {
        str(item.get("series")): item
        for item in payload.get("publications") or []
        if isinstance(item, dict) and item.get("series")
    }
    answers = [item for item in payload.get("answers") or []
               if isinstance(item, dict)]
    if (not answers
            and not any(item.get("context_outcome") for item in results)
            and not (publication.get("context_dispositions") or [])
            and not any(item.get("context_dispositions")
                        for item in publication_by_series.values())):
        return None

    series_contracts: list[dict[str, Any]] = []
    for result in results:
        series_name = str(result.get("series") or "__default__")
        series_publication = publication_by_series.get(
            series_name, publication)
        dispositions = [
            item for item in series_publication.get(
                "context_dispositions") or []
            if isinstance(item, dict)]
        publication_codes = [
            str(item.get("reason_code")) for item in dispositions
            if item.get("disposition") in {"rejected", "scenario"}
            and item.get("reason_code")]
        publication_sources = []
        for item in dispositions:
            evidence = item.get("source_evidence") or {}
            source = (evidence.get("source")
                      if isinstance(evidence, dict) else {})
            if isinstance(source, dict) and source.get("reference"):
                publication_sources.append(str(source["reference"]))
        context = result.get("context_outcome") or {}
        codes, codes_omitted = _unique([
            *[str(value) for value in context.get("failed_gate_codes") or []],
            *publication_codes,
        ])
        sources, sources_omitted = _unique([
            *_source_references(context), *publication_sources,
        ])
        consequence_count = sum(
            1
            for item in result.get("sensitivity_scenarios") or []
            if isinstance(item, dict) and item.get("consequence_summary")
        )
        interval_count = _interval_disclosure_count(result)
        relationship = context.get("relationship_to_primary")
        canonical_preserved = bool(
            context.get("canonical_primary_preserved",
                        series_publication.get(
                            "primary_forecast_unchanged", True)))
        context_automation = context.get("automation_eligible")
        if context_automation is None:
            context_automation = (series_publication.get(
                "context_summary") or {}).get(
                    "context_can_authorize_automation")
        from .support import result_support_tier
        contract = {
            "series": series_name,
            "support": result.get("support"),
            "tier_floor": result_support_tier(result),
            "context_status": context.get("status", "not_supplied"),
            "canonical_primary_preserved": canonical_preserved,
            "context_automation_eligible": bool(context_automation),
            "failed_gate_codes": codes,
            "source_references": sources,
            "scenario_consequence_count": consequence_count,
            "interval_disclosure_count": interval_count,
            **({"relationship_to_primary": str(relationship)}
               if relationship else {}),
            **({"selected_output_role": str(context["selected_output_role"])}
               if context.get("selected_output_role") else {}),
            # Every emitted series field is an obligation. A second list of
            # long aliases duplicated those same facts and pushed large
            # context responses over the wire budget; one explicit rule is
            # both smaller and harder for an agent to interpret selectively.
            "required": "all_emitted_fields",
            **({"omitted": {
                **({"failed_gate_codes": codes_omitted}
                   if codes_omitted else {}),
                **({"source_references": sources_omitted}
                   if sources_omitted else {}),
            }} if any((codes_omitted, sources_omitted)) else {}),
        }
        series_contracts.append(contract)

    decisions = [decision for answer in answers
                 if (decision := _decision_contract(
                     answer, answer_receipt=payload.get("answer_receipt")))
                 is not None]
    body = {
        "schema_version": SCHEMA_VERSION,
        "projection_only": True,
        "series": series_contracts,
        **({"decisions": decisions} if decisions else {}),
    }
    return {**body, "contract_seal_sha256": _seal(body)}


def verify_agent_response_contract(
        payload: dict[str, Any], contract: dict[str, Any]) -> bool:
    """Require byte-semantic equality with a freshly projected contract."""
    expected = build_agent_response_contract(payload)
    return expected is not None and contract == expected


def verify_agent_decision_selection(
    contract: dict[str, Any], question_id: str, selection: dict[str, Any],
) -> list[dict[str, str]]:
    """Verify that a proposed decision follows one sealed decision block."""
    decision = next((item for item in contract.get("decisions") or []
                     if isinstance(item, dict)
                     and item.get("question_id") == question_id), None)
    if decision is None:
        return [{"code": "DECISION_CONTRACT_MISSING",
                 "message": f"No decision contract exists for {question_id!r}."}]
    value = selection.get("value")
    if decision.get("value_status") == "abstention":
        if value in (None, "", "uncertain", "Uncertain", "abstain"):
            return []
        return [{"code": "DECISION_OVERRIDES_ABSTENTION",
                 "message": "An abstained conclusion cannot be upgraded by presentation."}]
    canonical = decision.get("conclusion")
    if decision.get("authority") == "binding" and value != canonical:
        return [{"code": "DECISION_OVERRIDES_BINDING",
                 "message": "A supported binding conclusion must be preserved exactly."}]
    interpretations = {
        str(item.get("value")): item
        for item in decision.get("interpretations") or []
        if isinstance(item, dict)
    }
    row = interpretations.get(str(value))
    if row is None:
        return [{"code": "DECISION_NOT_IN_CONTRACT",
                 "message": "The selected interpretation is not in the sealed contract."}]
    if (not row.get("compatible") or not row.get("decision_eligible")
            or row.get("conditional_only")):
        return [{"code": "DECISION_NOT_ELIGIBLE",
                 "message": "The selected interpretation is incompatible or conditional-only."}]
    if value == canonical:
        return []
    cited = [str(item) for item in selection.get("cited_evidence") or []]
    supporting = set(str(item) for item in row.get("supporting_evidence") or [])
    if not cited:
        return [{"code": "DECISION_EVIDENCE_REQUIRED",
                 "message": "A weak canonical override must cite supporting typed evidence."}]
    invalid = [item for item in cited if item not in supporting]
    if invalid:
        return [{"code": "DECISION_EVIDENCE_UNSUPPORTED",
                 "message": "Cited evidence does not support the selected interpretation."}]
    return []
