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


SCHEMA_VERSION = "0.1"
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
    if (not any(item.get("context_outcome") for item in results)
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
        contract = {
            "series": series_name,
            "support": result.get("support"),
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
            "required_facts": [
                name for name, required in (
                    ("primary_preservation", canonical_preserved),
                    ("context_automation_limit", not bool(context_automation)),
                    ("failed_gate_codes", bool(codes)),
                    ("source_references", bool(sources)),
                    ("scenario_consequences", consequence_count > 0),
                    ("typed_primary_relationship", bool(relationship)),
                    ("interval_limitations", interval_count > 0),
                ) if required],
            **({"omitted": {
                **({"failed_gate_codes": codes_omitted}
                   if codes_omitted else {}),
                **({"source_references": sources_omitted}
                   if sources_omitted else {}),
            }} if any((codes_omitted, sources_omitted)) else {}),
        }
        series_contracts.append(contract)

    body = {
        "schema_version": SCHEMA_VERSION,
        "projection_only": True,
        "series": series_contracts,
        "fact_locations": {
            "scenario_consequences":
                "results[].sensitivity_scenarios[].consequence_summary",
            "interval_limitations": "results[].warnings",
        },
    }
    return {**body, "contract_seal_sha256": _seal(body)}


def verify_agent_response_contract(
        payload: dict[str, Any], contract: dict[str, Any]) -> bool:
    """Require byte-semantic equality with a freshly projected contract."""
    expected = build_agent_response_contract(payload)
    return expected is not None and contract == expected
