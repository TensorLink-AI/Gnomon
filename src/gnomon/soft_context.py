"""Typed contract between narrative compilation and numeric context gates.

The compiler may describe an effect, never size one.  This module keeps that
boundary explicit and gives every grounded event a public disposition even
when no numerical candidate can honestly be fitted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .context import ContextEvent, event_applies


RECEIPT_SCHEMA_VERSION = "0.1"
COMPILER_PROMPT_VERSION = "context-investigation-0.2"
EFFECT_FAMILIES = (
    "level_shift", "trend_change", "variance_change", "temporary_pulse",
    "saturation_bound", "seasonal_regime_change", "unknown",
)
EFFECT_DIRECTIONS = ("increase", "decrease", "unknown")
EFFECT_DURATIONS = ("temporary", "persistent", "unknown")
ENTITY_KINDS = (
    "service", "product", "medication", "procedure", "calendar",
    "capacity", "price", "environment", "unknown",
)
CONTEXT_OUTCOMES = ("not_considered", "rejected", "scenario_only", "applied")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def content_fingerprint(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_context_receipt(
    *, documents: list[dict[str, Any]], events: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]], rejected: list[dict[str, Any]],
    rejected_hypotheses: list[dict[str, Any]],
    proposer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an immutable, content-addressed compiler receipt.

    No wall-clock field participates: replaying the same validated compiler
    output over the same source documents produces the same ID.  Model and
    prompt versions do participate, so changing the compiler cannot silently
    reuse an old receipt.
    """
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "prompt_version": COMPILER_PROMPT_VERSION,
        "documents": documents,
        "events": events,
        "hypotheses": hypotheses,
        "rejected": rejected,
        "rejected_hypotheses": rejected_hypotheses,
        "compiler": dict(proposer or {}),
    }
    body["receipt_id"] = "context_receipt:" + hashlib.sha256(
        _canonical(body).encode("utf-8")
    ).hexdigest()
    return body


def write_context_receipt(receipt: dict[str, Any], directory: str | Path) -> Path:
    """Persist a receipt by ID without allowing an ID/content mismatch."""
    expected = make_context_receipt(
        documents=list(receipt.get("documents") or []),
        events=list(receipt.get("events") or []),
        hypotheses=list(receipt.get("hypotheses") or []),
        rejected=list(receipt.get("rejected") or []),
        rejected_hypotheses=list(receipt.get("rejected_hypotheses") or []),
        proposer=dict(receipt.get("compiler") or {}),
    )
    if receipt.get("receipt_id") != expected["receipt_id"]:
        raise ValueError("context receipt ID does not match its content")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (expected["receipt_id"].split(":", 1)[1] + ".json")
    rendered = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError("an immutable context receipt already exists with different content")
    path.write_text(rendered, encoding="utf-8")
    return path


def event_hypothesis(event: ContextEvent) -> dict[str, Any]:
    soft = event.attributes.get("soft_context") or {}
    return {
        "hypothesis_id": f"soft:{event.event_id}",
        "event_id": event.event_id,
        "entity_scope": list(event.entity_scope),
        "effect_family": soft.get("effect_family", "unknown"),
        "direction": soft.get("direction", "unknown"),
        "duration": soft.get("duration", "unknown"),
        "normalized_entity": soft.get("normalized_entity"),
        "entity_kind": soft.get("entity_kind", "unknown"),
        "delay_steps": soft.get("delay_steps"),
        "duration_steps": soft.get("duration_steps"),
        "magnitude": None,
        "grounding_status": "grounded",
        "numeric_status": "requires_numeric_estimation",
        "may_affect_numbers": False,
        **({"evidence_quote": event.attributes["evidence_quote"]}
           if event.attributes.get("evidence_quote") else {}),
    }


def context_outcome(
    events: list[ContextEvent], series_name: str, *,
    context_assessment: Any = None,
    future_context: dict[str, Any] | None = None,
    conditional_forecasts: list[dict[str, Any]] | None = None,
    sensitivity_scenarios: list[dict[str, Any]] | None = None,
    primary_forecast_changed: bool | None = None,
) -> dict[str, Any]:
    """Project all context lanes into one unambiguous public disposition."""
    applicable = [event for event in events if event_applies(event, series_name)]
    receipt_ids = sorted({
        str(event.attributes.get("context_receipt_id"))
        for event in applicable if event.attributes.get("context_receipt_id")
    })
    if not applicable:
        return {"status": "not_considered", "primary_forecast_changed": False,
                "canonical_primary_preserved": True,
                "automation_eligible": False, "events": []}
    admitted = list((future_context or {}).get("admitted") or [])
    if bool(getattr(context_assessment, "admitted", False)) or admitted:
        changed = (True if primary_forecast_changed is None
                   else bool(primary_forecast_changed))
        historical_admission = bool(getattr(context_assessment, "admitted", False))
        used = list(getattr(context_assessment, "events_used", []) or [])
        used.extend(str(item.get("event_id")) for item in admitted
                    if isinstance(item, dict) and item.get("event_id"))
        used_ids = sorted(set(used))
        dispositions = [{"context_id": event.event_id,
                         "disposition": ("used" if event.event_id in used_ids
                                         else "scenario" if not event.event_type.startswith(
                                             ("constraint:", "override:", "structural:"))
                                         else "rejected")}
                        for event in applicable]
        return {
            "status": "applied", "primary_forecast_changed": changed,
            # Numeric admission permits a labelled context-conditioned
            # recommendation. Automation remains a separate explicit policy
            # decision; this context outcome alone never grants it.
            "automation_eligible": False,
            "canonical_primary_preserved": True,
            "selected_output_role": (
                "context_conditioned_projection" if changed
                else "primary_forecast_no_numeric_context_change"),
            "canonical_primary_location": "artifact.results[].primary_forecast",
            "events": used_ids,
            "dispositions": dispositions,
            "admission_basis": ("historical_fold_ablation"
                                if historical_admission
                                else "future_context_contract"),
            **({"context_receipt_ids": receipt_ids} if receipt_ids else {}),
            "basis": (
                "a deterministic context candidate passed its numeric "
                "admission gate and changed this horizon" if changed else
                "a deterministic context candidate passed its numeric "
                "admission gate, but no admitted effect changed this horizon"
            ),
        }
    conditional = list(conditional_forecasts or [])
    sensitivity = list(sensitivity_scenarios or [])
    hypotheses = [event_hypothesis(event) for event in applicable]
    if conditional or sensitivity:
        measured_ids = {str(event_id) for item in conditional
                        for event_id in item.get("events", [])}
        sensitivity_ids = {str(event_id) for item in sensitivity
                           for event_id in item.get("events", [])}
        for hypothesis in hypotheses:
            event_id = str(hypothesis.get("event_id"))
            if event_id in measured_ids:
                hypothesis["numeric_status"] = "measured_for_conditional_scenario"
                hypothesis["may_affect_numbers"] = True
                hypothesis["may_affect_primary_forecast"] = False
            elif event_id in sensitivity_ids:
                hypothesis["numeric_status"] = "standardized_sensitivity_not_event_effect"
                hypothesis["may_affect_numbers"] = True
                hypothesis["may_affect_primary_forecast"] = False
    exclusions = list(getattr(context_assessment, "events_excluded", []) or [])
    gate_reasons = list(getattr(context_assessment, "reasons", []) or [])
    failed_gate_codes = [
        str(check.get("code"))
        for check in (getattr(context_assessment, "gate_checks", []) or [])
        if isinstance(check, dict) and check.get("code")
        and check.get("passed") is False
    ]
    future_failures = [
        check for check in ((future_context or {}).get("checks") or [])
        if isinstance(check, dict) and check.get("code")
        and check.get("passed") is False
    ]
    future_failure_ids = {str(check.get("event_id"))
                          for check in future_failures if check.get("event_id")}
    # When every applicable event is governed by the dedicated future lane,
    # its transformation-specific gate is the decisive explanation. The
    # generic historical-ablation lane may also have declined the same event,
    # but projecting that displaced gate instead caused agents to explain the
    # wrong decision.
    if future_failures and future_failure_ids >= {
            event.event_id for event in applicable}:
        failed_gate_codes = []
        gate_reasons = []
    for check in future_failures:
        code = str(check["code"])
        if code not in failed_gate_codes:
            failed_gate_codes.append(code)
        detail = str(check.get("detail") or code)
        if detail not in gate_reasons:
            gate_reasons.append(detail)
    # Generic, grounded events are useful scenarios even when their effect
    # size is not measurable. Namespaced deterministic claims have a stricter
    # contract: a failed parser/admission is a rejection, never a scenario.
    generic = [event for event in applicable if not event.event_type.startswith(
        ("constraint:", "override:", "structural:"))]
    scenario_ids = {str(event_id) for item in sensitivity
                    for event_id in item.get("events", [])}
    represented = [event for event in applicable
                   if event in generic or event.event_id in scenario_ids]
    if represented:
        generic_ids = {event.event_id for event in represented}
        dispositions = [{
            "context_id": event.event_id,
            "disposition": ("scenario" if event.event_id in generic_ids
                            else "rejected"),
        } for event in applicable]
        point_candidate = bool(
            getattr(context_assessment, "point_candidate", None))
        return {
            "status": ("partially_represented"
                       if len(generic) != len(applicable) else "scenario_only"),
            "primary_forecast_changed": False,
            "canonical_primary_preserved": True,
            "automation_eligible": False,
            "events": [event.event_id for event in applicable],
            "dispositions": dispositions,
            **({"context_receipt_ids": receipt_ids} if receipt_ids else {}),
            "hypotheses": hypotheses,
            "conditional_forecasts_produced": len(conditional),
            "sensitivity_scenarios_produced": len(sensitivity),
            **({
                "selected_output_role": "interval_weak_context_scenario",
                "scenario_support": getattr(
                    context_assessment, "point_support",
                    "point_supported_interval_weak"),
                "automation_eligible": False,
            } if point_candidate else {}),
            "basis": (
                ("the event effect improved historical point forecasts, but its "
                 "intervals failed the independent coverage gate; the fitted path "
                 "is available only as a non-automatable scenario"
                 if point_candidate else
                 "the event is grounded, but no admitted numerical effect may alter "
                 "the primary forecast; an event-effect magnitude remains null unless "
                 "measured from data, and any sensitivity path is explicitly standardized")
            ),
            "recovery_actions": [
                "provide prior occurrences of the same event type for effect estimation",
                "track post-event outcomes and rerun once an effect can be measured",
            ],
            **({"gate_reasons": gate_reasons} if gate_reasons else {}),
            **({"failed_gate_codes": failed_gate_codes}
               if failed_gate_codes else {}),
            **({"excluded": exclusions} if exclusions else {}),
        }
    return {
        "status": "rejected", "primary_forecast_changed": False,
        "canonical_primary_preserved": True,
        "automation_eligible": False,
        "events": [event.event_id for event in applicable],
        **({"context_receipt_ids": receipt_ids} if receipt_ids else {}),
        **({"failed_gate_codes": failed_gate_codes}
           if failed_gate_codes else {}),
        "reasons": ((future_context or {}).get("rejected") or exclusions
                    or gate_reasons or ["context did not pass numeric admission"]),
    }
