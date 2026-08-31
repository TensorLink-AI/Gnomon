"""Bounded wire responses that never trim Gnomon's evidence contract."""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "CAPABILITIES_RESPONSE_BUDGET_BYTES",
    "DESCRIBE_RESPONSE_BUDGET_BYTES",
    "RESPONSE_BUDGET_BYTES",
    "enforce_response_budget",
]

#: Hard ceiling on a tool response's serialised size. Tuned to hold a
#: single-series brief forecast (~5KB with its full support assessment)
#: with headroom; anything past it is bulk, and bulk lives in the artifact.
#: The trim never touches the epistemic contract — see ``_PROTECTED_KEYS``.
#: Retuned as richer bounded decision and recovery projections were added.
RESPONSE_BUDGET_BYTES = 9728
DESCRIBE_RESPONSE_BUDGET_BYTES = 2400
CAPABILITIES_RESPONSE_BUDGET_BYTES = 6000

#: Subtrees that are the contract and are never trimmed, wherever they
#: appear: support state, warnings, abstention payloads, disclosed
#: assumptions, and structured error/repair options. A response may
#: therefore exceed the budget when its epistemics alone do — that is
#: deliberate; the budget disciplines bulk, not honesty.
_PROTECTED_KEYS = frozenset({
    "headline", "support", "support_assessment", "tier_floor",
    "limitations", "limitation_groups", "warnings", "assumptions", "reasons",
    "recovery_actions", "next_actions", "disclosures", "notes", "staleness",
    "artifact_id", "artifact_path", "data_ref", "error", "repair_options",
    "context_outcome", "admission", "question", "answer", "executable", "calibration",
    "direction_probabilities", "primary_forecast_unchanged",
    # The bounded multi-series disclosure contract: ranking rule, preserved
    # remainder, and artifact identity are copied deterministically and must
    # survive any budget trim.
    "triage",
    "reasoning", "sufficiency", "facts", "rejection",
    # The default wire publication is a bounded, seal-linked decision
    # projection. Its complete authenticated receipt and forecast arrays live
    # at publication_path; format=full explicitly requests that bulk inline.
    # Either projection must retain every decision/authority disclosure.
    "publication", "publication_path",
})

_TRIM_HEAD = 3
_TRIM_TAIL = 2


def _holds_protected(node: Any) -> bool:
    """Return whether a subtree contains protected contract content."""
    if isinstance(node, dict):
        return any(key in _PROTECTED_KEYS or _holds_protected(value)
                   for key, value in node.items())
    if isinstance(node, list):
        return any(_holds_protected(item) for item in node)
    return False


def _trim_bulk(node: Any, path: str, trimmed: list[dict[str, Any]]) -> Any:
    """Head/tail-trim long arrays outside the protected subtrees."""
    if isinstance(node, dict):
        return {
            key: (value if key in _PROTECTED_KEYS
                  else _trim_bulk(value, f"{path}.{key}" if path else str(key),
                                  trimmed))
            for key, value in node.items()
        }
    if isinstance(node, list) and len(node) > _TRIM_HEAD + _TRIM_TAIL \
            and any(_holds_protected(item) for item in node):
        # A long list whose entries carry the contract (per-channel results)
        # is descended, never cut. Bulk inside each entry remains trimmable.
        return [_trim_bulk(item, f"{path}[{index}]", trimmed)
                for index, item in enumerate(node)]
    if isinstance(node, list) and len(node) > _TRIM_HEAD + _TRIM_TAIL:
        record: dict[str, Any] = {
            "path": path, "total": len(node),
            "kept": f"first {_TRIM_HEAD}, last {_TRIM_TAIL}",
        }
        if all(isinstance(item, (int, float)) and not isinstance(item, bool)
               for item in node):
            record["summary"] = {
                "min": min(node), "max": max(node),
                "mean": round(sum(node) / len(node), 6),
            }
        trimmed.append(record)
        return node[:_TRIM_HEAD] + node[-_TRIM_TAIL:]
    if isinstance(node, list):
        return [_trim_bulk(item, f"{path}[{index}]", trimmed)
                for index, item in enumerate(node)]
    return node


def enforce_response_budget(
    payload: Any, budget_bytes: int = RESPONSE_BUDGET_BYTES,
) -> Any:
    """Trim bulk arrays while preserving support and recovery contracts."""
    if not isinstance(payload, dict) or payload.get("status") == "error" \
            or "error" in payload:
        return payload
    try:
        raw = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return payload
    if len(raw) <= budget_bytes:
        return payload
    trimmed: list[dict[str, Any]] = []
    result = _trim_bulk(payload, "", trimmed)
    # A forecast may already carry first/last preview metadata before the
    # global budget trims that preview again. Keep its counts truthful.
    for entry in result.get("results", []):
        if not isinstance(entry, dict) or not isinstance(
                entry.get("forecast"), list):
            continue
        preview = entry.get("forecast_preview")
        if not isinstance(preview, dict):
            continue
        returned = len(entry["forecast"])
        total = int(entry.get("forecast_rows") or returned)
        preview["returned_rows"] = returned
        preview["omitted_middle_rows"] = max(0, total - returned)
    artifact_path = payload.get("artifact_path")
    where = (f"the artifact at {artifact_path}" if artifact_path
             else "the on-disk artifact or store")
    result["truncated"] = True
    result["truncation"] = {
        "budget_bytes": budget_bytes,
        "trimmed": trimmed,
        "note": (
            f"Response exceeded the {budget_bytes}-byte budget; "
            f"long arrays keep their first {_TRIM_HEAD} and last "
            f"{_TRIM_TAIL} entries. Support assessments, warnings, and "
            f"assumptions are never trimmed. The complete data lives in "
            f"{where}."
        ),
    }
    return result
