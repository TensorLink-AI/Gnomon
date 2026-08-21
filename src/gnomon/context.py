"""Typed context events: the contract for admitting outside knowledge.

A context event is a claim that something happens (or happened) in the
world at a known time — a launch, a holiday, an outage — proposed by a
user or an LLM and considered as a covariate for forecasting. Events are
proposals only. They enter a forecast solely through the deterministic
admission gate defined in the system design: temporal-availability
validation, identical-fold ablation, and a minimum stable improvement.

The dangerous failure mode is future leakage during backtests. An LLM
proposing events for historical folds knows what actually happened after
those cutoffs, so ``known_at`` must be grounded in verifiable source
metadata (a dated file, a calendar entry), never taken from the model's
assertion alone. Events without a verifiable source may inform only the
future horizon and are never admissible for backtesting.

Events are consumed by the forecast pipeline's context stage, which
replays the selection folds with a context-adjusted candidate and admits
it only on measured, stable improvement (``gnomon.context_eval``). Every
gate decision — the conditions evaluated, what each measured, and which
one decided a rejection — is disclosed as a ``context_gate`` evidence
record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import GnomonError

# Source types whose references can be independently checked (a file path,
# a calendar export, a prior Gnomon artifact). Assertions by a model or a
# person are usable for the future horizon but never for backtests.
VERIFIABLE_SOURCE_TYPES = frozenset(
    {"planning_file", "calendar", "artifact", "url", "dataset"}
)
CONTEXT_STATUSES = frozenset({"confirmed", "tentative", "cancelled"})
UNVERIFIED_EXTERNAL_CREATOR = "unverified_external"


@dataclass(frozen=True)
class ContextSource:
    type: str
    reference: str


@dataclass(frozen=True)
class ContextEvent:
    event_id: str
    event_type: str
    entity_scope: tuple[str, ...]
    effective_start: str
    effective_end: str
    known_at: str
    status: str = "confirmed"
    confidence: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)
    source: ContextSource | None = None
    created_by: str = "user"


def _parse_timestamp(value: str, field_name: str, problems: list[str]) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        problems.append(f"{field_name} is not a valid ISO-8601 timestamp: {value!r}")
        return None
    if parsed.tzinfo is None:
        problems.append(f"{field_name} must carry an explicit timezone offset")
        return None
    return parsed


def validate_context_event(event: ContextEvent) -> list[str]:
    """Return every contract violation; an empty list means the event is valid.

    Validity here is structural. It does not admit the event into any
    forecast — admission additionally requires availability at each fold
    cutoff and a demonstrated, stable metric improvement.
    """
    problems: list[str] = []
    if not event.event_id:
        problems.append("event_id must be non-empty")
    if not event.event_type:
        problems.append("event_type must be non-empty")
    if not event.entity_scope:
        problems.append("entity_scope must name at least one series")
    if event.status not in CONTEXT_STATUSES:
        problems.append(
            "status must be one of confirmed, tentative, cancelled"
        )

    start = _parse_timestamp(event.effective_start, "effective_start", problems)
    end = _parse_timestamp(event.effective_end, "effective_end", problems)
    _parse_timestamp(event.known_at, "known_at", problems)
    if start is not None and end is not None and end < start:
        problems.append("effective_end precedes effective_start")

    if not 0.0 <= event.confidence <= 1.0:
        problems.append("confidence must be between 0 and 1")
    if event.source is not None and not event.source.reference:
        problems.append("source.reference must be non-empty when a source is given")
    expected_shape = (event.attributes or {}).get("expected_shape")
    if expected_shape is not None:
        from .context_model import EFFECT_SHAPES
        if str(expected_shape) not in EFFECT_SHAPES:
            problems.append(
                f"expected_shape must be one of {', '.join(EFFECT_SHAPES)}; "
                f"got {expected_shape!r}. A nomination narrows the shape "
                f"contest; it cannot invent a shape."
            )
    soft = (event.attributes or {}).get("soft_context")
    if soft is not None:
        if not isinstance(soft, dict):
            problems.append("soft_context must be an object")
        else:
            allowed = {
                "effect_family": {"level_shift", "trend_change",
                                  "variance_change", "temporary_pulse",
                                  "saturation_bound", "seasonal_regime_change",
                                  "unknown"},
                "direction": {"increase", "decrease", "unknown"},
                "duration": {"temporary", "persistent", "unknown"},
                "entity_kind": {"service", "product", "medication",
                                "procedure", "calendar", "capacity", "price",
                                "environment", "unknown"},
            }
            for field_name, choices in allowed.items():
                if soft.get(field_name, "unknown") not in choices:
                    problems.append(
                        f"soft_context.{field_name} is not in the closed vocabulary"
                    )
            forbidden = {"magnitude", "effect_size", "numeric_effect"}
            if forbidden.intersection(soft):
                problems.append("soft_context may classify an effect but cannot supply a magnitude")
            normalized = soft.get("normalized_entity")
            if normalized is not None and (not isinstance(normalized, str)
                                           or not normalized.strip()):
                problems.append("soft_context.normalized_entity must be a non-empty string or null")
            for range_name in ("delay_steps", "duration_steps"):
                value = soft.get(range_name)
                if value is not None and (
                    not isinstance(value, list) or len(value) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int)
                           for item in value)
                    or value[0] < 0 or value[1] < value[0]
                ):
                    problems.append(
                        f"soft_context.{range_name} must be [minimum, maximum] non-negative steps or null"
                    )
    return problems


def event_applies(event: ContextEvent, series_name: str) -> bool:
    """Whether the event's scope covers *series_name* (``"*"`` matches all)."""
    return "*" in event.entity_scope or series_name in event.entity_scope


def event_to_dict(event: ContextEvent) -> dict[str, Any]:
    payload = asdict(event)
    payload["entity_scope"] = list(event.entity_scope)
    return payload


def event_from_dict(raw: dict[str, Any], *,
                    trust_declared_creator: bool = False) -> ContextEvent:
    source = raw.get("source")
    return ContextEvent(
        event_id=str(raw.get("event_id", "")),
        event_type=str(raw.get("event_type", "")),
        entity_scope=tuple(str(item) for item in raw.get("entity_scope", ())),
        effective_start=str(raw.get("effective_start", "")),
        effective_end=str(raw.get("effective_end", "")),
        known_at=str(raw.get("known_at", "")),
        status=str(raw.get("status", "confirmed")),
        confidence=float(raw.get("confidence", 1.0)),
        attributes=dict(raw.get("attributes") or {}),
        source=ContextSource(str(source.get("type", "")), str(source.get("reference", "")))
        if isinstance(source, dict) else None,
        # Inline MCP/JSON callers cannot self-promote a source assertion into
        # verified historical knowledge by writing ``created_by: user``. File
        # loading is an explicit operator-controlled boundary and opts in.
        created_by=(str(raw.get("created_by", "user"))
                    if trust_declared_creator
                    else UNVERIFIED_EXTERNAL_CREATOR),
    )


def events_from_list(raw_events: list, *,
                     trust_declared_creator: bool = False) -> list[ContextEvent]:
    """Build and structurally validate events from raw dicts, loudly.

    Shared by the file loader and the inline ``context_events`` tool
    argument: an MCP client holds no filesystem, so the tool surface
    must accept events directly or the admission lanes are unreachable
    from it. Any invalid event fails the whole batch; silently dropping
    a proposed event would hide it from the admission record.
    """
    events = [event_from_dict(
        item, trust_declared_creator=trust_declared_creator)
        for item in raw_events]
    problems = {
        event.event_id or f"index {index}": event_problems
        for index, event in enumerate(events)
        if (event_problems := validate_context_event(event))
    }
    if problems:
        raise GnomonError(
            "INVALID_CONTEXT_EVENT", "One or more context events violate the contract.",
            {"problems": problems},
        )
    return events


def load_events_file(path: str) -> list[ContextEvent]:
    """Load and structurally validate a context-events JSON file.

    The file format is ``{"schema_version": "0.1", "events": [...]}`` — the
    exact shape ``gnomon context validate`` emits.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise GnomonError("CONTEXT_FILE_NOT_FOUND", f"Context events file does not exist: {file}")
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GnomonError("INVALID_CONTEXT_FILE", f"Context events file is not valid JSON: {exc}") from exc
    raw_events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(raw_events, list):
        raise GnomonError(
            "INVALID_CONTEXT_FILE",
            'Context events file must be an object with an "events" array.',
        )
    return events_from_list(raw_events, trust_declared_creator=True)


def backtest_admissible(event: ContextEvent) -> bool:
    """Whether the event may participate in historical folds at all.

    Requires a verifiable source: ``known_at`` claimed without checkable
    provenance cannot rule out future leakage, so such events are limited
    to the future horizon.
    """
    if validate_context_event(event):
        return False
    return (
        event.created_by != UNVERIFIED_EXTERNAL_CREATOR
        and
        event.source is not None
        and event.source.type in VERIFIABLE_SOURCE_TYPES
        and bool(event.source.reference)
    )
