"""Compact, deterministic reasoning affordances for every public response.

This module shapes computed results into arguments an agent can explain.  It
never computes or replaces a canonical answer: every fact is a JSON pointer
to a field already present in the response or immutable receipt.
"""

from __future__ import annotations

import re
from typing import Any


BOUNDARY_VERSION = "0.1"
RECOVERY_AUTHORITY_LIMIT = (
    "No evidence/support/accuracy/automation upgrade."
)

_RECOVERY_SUMMARIES = {
    "lower_minimum_support": "Use the achieved tier.",
    "reduce_horizon": "Use the supportable horizon.",
    "retry_best_effort": "Publish best-effort orientation rows.",
    "provide_more_history": "Supply more observations.",
}


class BoundaryContractError(ValueError):
    """A public response contains a shape the boundary cannot safely project."""


def _items(value: Any, field: str) -> list[Any]:
    """Return a bounded-envelope collection or fail with a useful field name.

    Strings and mappings are deliberately not coerced.  Guessing that a
    malformed limitation or action is a one-item list would make the response
    look governed while silently changing its contract.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise BoundaryContractError(f"{field} must be a list, got {type(value).__name__}")
    return value


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BoundaryContractError(f"{field} must be an object, got {type(value).__name__}")
    return value


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
    if not isinstance(envelope, dict):
        return [{"code": "REASONING_NOT_OBJECT", "fact": "reasoning"}]
    violations: list[dict[str, str]] = []
    names: set[str] = set()
    facts = envelope.get("facts") or []
    if not isinstance(facts, list):
        return [{"code": "FACTS_NOT_LIST", "fact": "facts"}]
    for fact in facts:
        if not isinstance(fact, dict):
            violations.append({"code": "FACT_NOT_OBJECT", "fact": ""})
            continue
        name, pointer = str(fact.get("name") or ""), str(fact.get("source") or "")
        if not name or name in names:
            violations.append({"code": "FACT_NAME_NOT_UNIQUE", "fact": name})
        names.add(name)
        if not pointer.startswith("/") or not _pointer_exists(payload, pointer):
            violations.append({"code": "FACT_SOURCE_MISSING", "fact": name,
                               "source": pointer})
    return violations


def _action_fields(action: dict[str, Any]) -> tuple[str, str]:
    return (
        str(action.get("code") or action.get("action") or "recovery"),
        str(action.get("message") or action.get("description") or
            "Review the active blocker and retry with corrected input."),
    )


def _support_patch(code: str, description: str) -> tuple[dict[str, Any], str] | None:
    """Return only patches fully determined by an existing support action."""
    if code == "lower_minimum_support":
        match = re.search(r"minimum_support ['\"]([^'\"]+)['\"]", description)
        if match:
            tier = match.group(1)
            return ({"minimum_support": tier},
                    f"Publish unchanged result at earned tier {tier!r}.")
    if code == "reduce_horizon":
        match = re.search(r"(?:horizon|Request horizon) (\d+)", description)
        if match:
            horizon = int(match.group(1))
            return ({"horizon": horizon, "minimum_support": "best_effort"},
                    f"Re-run horizon {horizon}; publish only its earned tier.")
    if code == "retry_best_effort":
        return ({"minimum_support": "best_effort"},
                "Publish best-effort rows without measured accuracy.")
    return None


def build_recovery_plan(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Project authored repairs into ranked, honest execution contracts.

    Existing recovery fields remain canonical. This projection may expose an
    exact patch already entailed by them, but never guesses a missing path,
    frequency, schema choice, policy value, or external fact.
    """
    error = payload.get("error")
    if isinstance(error, dict):
        actions = _items(error.get("repair_options"), "error.repair_options")
        source_root = "/error/repair_options"
    else:
        actions = _items(
            payload.get("recovery_actions") or payload.get("next_actions"),
            "recovery_actions",
        )
        source_root = ("/recovery_actions" if payload.get("recovery_actions")
                       is not None else "/next_actions")

    plan: list[dict[str, Any]] = []
    for index, raw in enumerate(actions):
        if not isinstance(raw, dict):
            raise BoundaryContractError(
                f"{source_root.strip('/')}[{index}] must be an object, got "
                f"{type(raw).__name__}")
        code, description = _action_fields(raw)
        tool: str | None = None
        patch: dict[str, Any] | None = None
        expected_effect = "Retry after the required input or choice is supplied."

        tool_call = raw.get("tool_call")
        if isinstance(tool_call, dict) and isinstance(tool_call.get("name"), str) \
                and isinstance(tool_call.get("arguments"), dict):
            tool = tool_call["name"]
            patch = dict(tool_call["arguments"])
            expected_effect = "Run the authored retry; it must earn its own tier."
        elif isinstance(raw.get("tool"), str) and isinstance(
                raw.get("arguments"), dict):
            # Compatibility with the original synthetic BoundaryBench shape.
            tool = raw["tool"]
            patch = dict(raw["arguments"])
            expected_effect = "Run the authored retry; it must earn its own tier."
        elif source_root == "/recovery_actions":
            support = _support_patch(code, description)
            if support is not None:
                tool, (patch, expected_effect) = "gnomon_forecast", support

        execution: dict[str, Any] = {
            "mode": "tool" if tool is not None and patch is not None
                    else "user_input",
            "requires_user_input": not (tool is not None and patch is not None),
        }
        if tool is not None and patch is not None:
            execution.update({"tool": tool, "argument_patch": patch})
        plan.append({
            "rank": index + 1,
            "recommended": index == 0,
            "code": code,
            # The source pointer retains the complete authored prose. Keep
            # this routing projection compact enough for brief responses.
            "description": _RECOVERY_SUMMARIES.get(code, description),
            "source": f"{source_root}/{index}",
            "execution": execution,
            "expected_effect": expected_effect,
            "external_fact_required": execution["requires_user_input"],
            "authority_limit": RECOVERY_AUTHORITY_LIMIT,
        })
    return plan


def actionable_rejection(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize an error/abstention into one terminal, repairable verdict."""
    error = payload.get("error") or {}
    code = str(error.get("code") or payload.get("code") or "UNSUPPORTED")
    message = str(error.get("message") or payload.get("message") or
                  "The requested operation could not be completed.")
    repairs = _items(
        error.get("repair_options") or payload.get("repair_options") or
        payload.get("recovery_actions") or payload.get("next_actions"),
        "repair_options",
    )
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
    existing = _mapping(payload.get("reasoning"), "reasoning")
    contrast = _mapping(existing.get("contrast") or existing, "reasoning.contrast")
    because = _items(contrast.get("because"), "reasoning.because")[:2]
    against = _items(contrast.get("against"), "reasoning.against")[:1]
    unknown = _items(contrast.get("unknown"), "reasoning.unknown")[:2]
    adjudication = _mapping(existing.get("adjudication"), "reasoning.adjudication")
    flips = _items(
        existing.get("what_would_flip") or adjudication.get("what_would_flip"),
        "reasoning.what_would_flip",
    )[:1]

    if not because:
        if payload.get("support_assessment"):
            because = [{"source": "/support_assessment",
                        "meaning": "computed support assessment"}]
        elif payload.get("results"):
            because = [{"source": "/results",
                        "meaning": "computed per-series results"}]
    if not unknown:
        unknown = _items(payload.get("limitations"), "limitations")[:2]
    if not flips:
        actions = _items(
            payload.get("recovery_actions") or payload.get("next_actions"),
            "recovery_actions",
        )
        flips = actions[:1]

    facts: list[dict[str, str]] = []
    canonical = _canonical_pointer(payload)
    if canonical:
        facts.append({"name": "canonical_answer", "source": canonical})
    for name, pointer in (("support", "/support"), ("tier_floor", "/tier_floor"),
                          ("artifact_identity", "/artifact_id"),
                          ("series_triage", "/triage")):
        if _pointer_exists(payload, pointer):
            facts.append({"name": name, "source": pointer})

    task = payload.get("task") or {}
    task = task if isinstance(task, dict) else {}
    request_kind = str(
        task.get("task_type") or payload.get("verb") or payload.get("kind") or ""
    ).strip()
    support = payload.get("support")
    support_state = (support.get("state") if isinstance(support, dict) else support)
    assessment = payload.get("support_assessment") or {}
    assessment = assessment if isinstance(assessment, dict) else {}
    assessment_state = assessment.get("status")
    tier_floor = payload.get("tier_floor")
    unusable = {
        "invalid", "unsupported", "inconclusive", "abstained", "best_effort",
    }
    has_answer = canonical is not None
    recoverable_fallback = tier_floor == "best_effort" and bool(flips)
    supported = bool(
        request_kind and has_answer and support_state not in unusable
        and assessment_state not in unusable and tier_floor not in unusable
        and not recoverable_fallback
    )
    sufficient_for = ([f"{request_kind}:canonical_answer", "explain_support"]
                      if supported else [])
    further = list(sufficient_for)
    resolution = (
        {"kind": "complete", "reason": "The requested canonical answer is sufficient."}
        if supported else
        {"kind": "recovery", "action": flips[0]}
        if flips else
        {"kind": "terminal", "reason": (
            "No admissible automated follow-up is available for this response; "
            "review its typed limitations or change the requested operation."
        )}
    )
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
        "resolution": resolution,
    }


def apply_reasoning_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    try:
        recovery_plan = build_recovery_plan(result)
        if recovery_plan:
            result["recovery_plan"] = recovery_plan
        if result.get("status") == "error" or "error" in result:
            result["rejection"] = actionable_rejection(result)
            if recovery_plan:
                result["rejection"]["recovery_plan_ref"] = "/recovery_plan/0"
            return result
        result["reasoning"] = build_argument_envelope(result)
        if recovery_plan:
            result["reasoning"]["recovery_plan_ref"] = "/recovery_plan/0"
    except BoundaryContractError as exc:
        result["status"] = "error"
        result["error"] = {
            "code": "INVALID_RESPONSE_CONTRACT",
            "message": str(exc),
            "details": {"boundary_version": BOUNDARY_VERSION},
        }
        result["rejection"] = {
            "code": "INVALID_RESPONSE_CONTRACT",
            "reason": str(exc),
            "missing": {},
            "admissibility_path": [],
            "terminal": True,
        }
        return result
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
