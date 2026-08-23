"""Compact, deterministic reasoning affordances for every public response.

This module shapes computed results into arguments an agent can explain.  It
never computes or replaces a canonical answer: every fact is a JSON pointer
to a field already present in the response or immutable receipt.
"""

from __future__ import annotations

from typing import Any


BOUNDARY_VERSION = "0.1"


def _pointer_exists(payload: Any, pointer: str) -> bool:
    node = payload
    for token in pointer.strip("/").split("/") if pointer != "/" else []:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return False
    return True


def verify_fact_sources(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Return violations for boundary facts without a real source field."""
    envelope = payload.get("reasoning") or {}
    violations: list[dict[str, str]] = []
    names: set[str] = set()
    for fact in envelope.get("facts") or []:
        name, pointer = str(fact.get("name") or ""), str(fact.get("source") or "")
        if not name or name in names:
            violations.append({"code": "FACT_NAME_NOT_UNIQUE", "fact": name})
        names.add(name)
        if not pointer.startswith("/") or not _pointer_exists(payload, pointer):
            violations.append({"code": "FACT_SOURCE_MISSING", "fact": name,
                               "source": pointer})
    return violations


def actionable_rejection(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize an error/abstention into one terminal, repairable verdict."""
    error = payload.get("error") or {}
    code = str(error.get("code") or payload.get("code") or "UNSUPPORTED")
    message = str(error.get("message") or payload.get("message") or
                  "The requested operation could not be completed.")
    repairs = (error.get("repair_options") or payload.get("repair_options") or
               payload.get("recovery_actions") or payload.get("next_actions") or [])
    return {
        "code": code,
        "reason": message,
        "missing": error.get("details") or payload.get("missing_evidence") or {},
        "admissibility_path": repairs[:1],
        "terminal": True,
    }


def _canonical_pointer(payload: dict[str, Any]) -> str | None:
    for pointer in ("/best_estimate/value", "/headline", "/artifact_id",
                    "/results/0/forecast/0/q50"):
        if _pointer_exists(payload, pointer):
            return pointer
    return None


def build_argument_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Build one bounded, source-addressed argument for any response verb."""
    existing = payload.get("reasoning") or {}
    contrast = existing.get("contrast") or existing
    because = list(contrast.get("because") or [])[:2]
    against = list(contrast.get("against") or [])[:1]
    unknown = list(contrast.get("unknown") or [])[:2]
    flips = list(existing.get("what_would_flip") or
                 (existing.get("adjudication") or {}).get("what_would_flip") or [])[:1]

    if not because:
        if payload.get("support_assessment"):
            because = [{"source": "/support_assessment",
                        "meaning": "computed support assessment"}]
        elif payload.get("results"):
            because = [{"source": "/results",
                        "meaning": "computed per-series results"}]
    if not unknown:
        unknown = list(payload.get("limitations") or [])[:2]
    if not flips:
        actions = (payload.get("recovery_actions") or payload.get("next_actions") or [])
        flips = actions[:1]

    facts: list[dict[str, str]] = []
    canonical = _canonical_pointer(payload)
    if canonical:
        facts.append({"name": "canonical_answer", "source": canonical})
    for name, pointer in (("support", "/support"), ("tier_floor", "/tier_floor"),
                          ("artifact_identity", "/artifact_id"),
                          ("series_triage", "/series_triage")):
        if _pointer_exists(payload, pointer):
            facts.append({"name": name, "source": pointer})

    supported = bool(payload.get("headline") or payload.get("results") or
                     payload.get("answers"))
    sufficient_for = ["quote_canonical_answer", "explain_support"] if supported else []
    further = list(sufficient_for) if supported else []
    return {
        "version": BOUNDARY_VERSION,
        "canonical_immutable": True,
        "canonical_source": canonical,
        "because": because,
        "against": against,
        "unknown": unknown,
        "what_would_flip": flips,
        "facts": facts,
        "sufficiency": {
            "sufficient_for": sufficient_for,
            "further_calls_add_nothing_for": further,
            "requires_follow_up": not supported,
        },
    }


def apply_reasoning_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if result.get("status") == "error" or "error" in result:
        result["rejection"] = actionable_rejection(result)
        return result
    result["reasoning"] = build_argument_envelope(result)
    violations = verify_fact_sources(result)
    if violations:
        # This indicates a Gnomon contract bug, never something for the LLM to
        # guess around.
        result["status"] = "error"
        result["error"] = {"code": "UNTRACEABLE_RESPONSE_FACT",
                           "message": "A published fact has no source field.",
                           "details": violations}
    return result


def measure_redundant_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Attribute calls made after a response declared the task sufficient.

    Each row is ``{"tool": ..., "result": ...}``.  This deliberately
    measures host policy separately from the minimum calls a surface needs.
    """
    sufficient_at: int | None = None
    redundant: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        if sufficient_at is not None:
            redundant.append({"index": index, "tool": call.get("tool"),
                              "reason": "prior response declared task sufficient"})
        result = call.get("result") or {}
        sufficiency = ((result.get("reasoning") or {}).get("sufficiency") or {})
        if sufficient_at is None and sufficiency.get("requires_follow_up") is False \
                and sufficiency.get("sufficient_for"):
            sufficient_at = index
    return {
        "observed_calls": len(calls),
        "surface_required_calls": (sufficient_at + 1 if sufficient_at is not None
                                   else len(calls)),
        "redundant_calls": len(redundant),
        "redundant": redundant,
    }
