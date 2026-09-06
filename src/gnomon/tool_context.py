"""Legacy context and covariate input preparation, independent of tool registration."""

from __future__ import annotations


from typing import Any

from .context import load_events_file
from .contracts import GnomonError


def _covariates_from(arguments: dict[str, Any]):
    """The covariate dataset from the file channel or the inline channel.

    Mutually exclusive rather than concatenated (unlike context events):
    two covariate datasets have no defined merge, and silently preferring
    one would hide the other from the admission record.
    """
    from .contracts import GnomonError

    file_path = arguments.get("covariates_file")
    inline = arguments.get("covariates")
    if not file_path and inline is None:
        if arguments.get("covariate_mapping"):
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "covariate_mapping was given without covariate rows: supply "
                "covariates (inline array) or covariates_file.",
            )
        return None
    if file_path and inline is not None:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "Provide covariates_file or inline covariates, not both.",
        )
    mapping = arguments.get("covariate_mapping")
    if not mapping:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "covariate_mapping (name:type:future_known entries) is required "
            "with covariates or covariates_file.",
        )
    from .covariates import covariates_from_rows, load_covariates

    kwargs = {
        "time_column": arguments.get("covariate_time_column", "timestamp"),
        "known_at_column": arguments.get("covariate_known_at_column", "known_at"),
        "series_column": arguments.get("covariate_series_column"),
    }
    if file_path:
        return load_covariates(file_path, mapping, **kwargs)
    if not isinstance(inline, list):
        raise GnomonError(
            "INVALID_ARGUMENTS", "covariates must be an array of row objects.",
        )
    return covariates_from_rows(inline, mapping, **kwargs)


def _context_events_from(arguments: dict[str, Any]):
    """Events from file, strict inline, and qualitative inline channels.

    The inline channel exists because an MCP client holds no
    filesystem: with a file-only parameter the admission lanes were
    unreachable from the published tool surface — a model could be
    told about future_events and still have no way to supply an event.
    All channels validate loudly through the same contract check. Qualitative
    events are structurally unable to enter numeric admission: they can only
    create labelled, non-automatable sensitivity scenarios.
    """
    import re

    from .context import events_from_list
    from .contracts import GnomonError

    def normalize_daily_times(item: dict[str, Any]) -> list[dict[str, str]]:
        """Resolve unambiguous date-only fields on a daily grid."""
        if str(arguments.get("frequency") or "") != "D":
            return []
        normalizations = []
        for field in ("effective_start", "effective_end", "known_at"):
            value = item.get(field)
            if not isinstance(value, str):
                continue
            try:
                from datetime import datetime
                parsed = datetime.fromisoformat(value)
            except ValueError:
                continue
            if parsed.tzinfo is not None:
                continue
            if len(value) == 10:
                suffix = "T23:59:59+00:00" if field == "effective_end" \
                    else "T00:00:00+00:00"
                item[field] = value + suffix
            else:
                item[field] = value + "+00:00"
            normalizations.append({
                "field": field, "from": value, "to": item[field],
                "basis": "daily_grid_calendar_boundary",
            })
        return normalizations

    events = None
    if arguments.get("context_events_file"):
        events = load_events_file(arguments["context_events_file"])
    inline = arguments.get("context_events")
    if inline is not None:
        if not isinstance(inline, list):
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "context_events must be an array of event objects",
            )
        normalized_inline = []
        for index, raw in enumerate(inline, 1):
            if not isinstance(raw, dict):
                raise GnomonError(
                    "INVALID_ARGUMENTS", "context_events items must be objects",
                    {"item_index": index})
            item = dict(raw)
            time_normalizations = normalize_daily_times(item)
            claim_kind = item.pop("claim_kind", None)
            if claim_kind is not None:
                legacy_type = str(item.pop("event_type", "") or "")
                item.pop("attributes", None)
                expected_prefix = ("override:" if claim_kind == "exact"
                                   else "constraint:")
                if legacy_type and not legacy_type.startswith(expected_prefix):
                    raise GnomonError(
                        "INVALID_ARGUMENTS",
                        "compact claim_kind conflicts with legacy event_type",
                        {"item_index": index, "claim_kind": claim_kind})
                if claim_kind not in {"min", "max", "exact"}:
                    raise GnomonError(
                        "INVALID_ARGUMENTS", "unknown context claim_kind",
                        {"item_index": index, "claim_kind": claim_kind})
                span = str(item.pop("source_span", "") or "").strip()
                if not span:
                    raise GnomonError(
                        "INVALID_ARGUMENTS",
                        "compact literal context requires verbatim source_span",
                        {"item_index": index})
                reference = str(item.pop("source_reference", "inline") or "inline")
                source_document = str(arguments.get(
                    "_trusted_context_source_text") or arguments.get(
                        "context_source_text") or "").strip()
                document_folded = source_document.casefold()
                span_offset = document_folded.find(span.casefold()) \
                    if source_document else -1
                if source_document and span_offset < 0:
                    arguments.setdefault("context_rejections", []).append({
                        "context_id": str(item.get("event_id") or
                                          f"context-event-{index}"),
                        "reason_code": "source_span_not_in_context_document",
                        "reason": (
                            "The claimed verbatim span is not present in the "
                            "host-bound source document."),
                        "source_span": span,
                    })
                    continue
                if span_offset >= 0:
                    # Classify the sentence that owns the quote, not an
                    # unrelated task instruction elsewhere in the message.
                    left = max(source_document.rfind(mark, 0, span_offset)
                               for mark in ".!?\n") + 1
                    span_end = span_offset + len(span)
                    endings = [source_document.find(mark, span_end)
                               for mark in ".!?\n"]
                    right = min((end for end in endings if end >= 0),
                                default=len(source_document))
                    semantic_text = source_document[left:right]
                else:
                    semantic_text = span
                from .future_context import literal_input_authority
                source_authority = literal_input_authority(semantic_text)
                if source_authority == "forecast":
                    arguments.setdefault("context_rejections", []).append({
                        "context_id": str(item.get("event_id") or
                                          f"context-event-{index}"),
                        "reason_code": "external_prediction_not_constraint",
                        "reason": (
                            "The quoted source predicts a value; it does not "
                            "state a binding constraint or observed outcome."),
                        "source_span": span,
                    })
                    continue
                if source_authority == "assumed":
                    arguments.setdefault("context_rejections", []).append({
                        "context_id": str(item.get("event_id") or
                                          f"context-event-{index}"),
                        "reason_code": "scenario_assumption_not_constraint",
                        "reason": (
                            "The quoted value is a scenario assumption; it "
                            "does not state a binding constraint or schedule."),
                        "source_span": span,
                    })
                    continue
                if source_authority == "observed":
                    arguments.setdefault("context_rejections", []).append({
                        "context_id": str(item.get("event_id") or
                                          f"context-event-{index}"),
                        "reason_code": "observed_value_not_future_constraint",
                        "reason": (
                            "The quoted value is an observation; it does not "
                            "state a binding future constraint or schedule."),
                        "source_span": span,
                    })
                    continue
                if not item.get("entity_scope"):
                    target = str(arguments.get("target_column") or "").strip()
                    candidates = [name.strip() for name in target.split(",")
                                  if name.strip() and name.strip().lower() != "auto"]
                    matches = [name for name in candidates if re.search(
                        rf"(?<![\w-]){re.escape(name)}(?![\w-])", span,
                        re.IGNORECASE)]
                    if len(candidates) == 1:
                        item["entity_scope"] = candidates
                    elif len(matches) == 1:
                        item["entity_scope"] = matches
                    else:
                        raise GnomonError(
                            "INVALID_ARGUMENTS",
                            "compact literal context requires entity_scope "
                            "unless its quote names exactly one requested target",
                            {"item_index": index,
                             "requested_targets": candidates,
                             "targets_named_in_source_span": matches})
                from .future_context import parse_bound_span, parse_override_span
                parsed_bound, _ = parse_bound_span(span)
                parsed_override, _ = parse_override_span(span)
                explicit_bound = bool(re.search(
                    r"\b(?:caps?|capped|ceiling|floor|maximum|minimum|"
                    r"at\s+most|no\s+more\s+than|not\s+exceed|at\s+least|"
                    r"no\s+less\s+than)\b", span, re.IGNORECASE))
                source_class = (
                    "constraint" if parsed_bound is not None
                    and (parsed_override is None or explicit_bound) else
                    "override" if parsed_override is not None
                    and parsed_bound is None else
                    "override" if claim_kind == "exact" else "constraint"
                )
                requested_class = ("override" if claim_kind == "exact"
                                   else "constraint")
                class_normalization = (None if source_class == requested_class else {
                    "code": "literal_claim_reclassified_from_source",
                    "from": requested_class, "to": source_class,
                    "reason": "the quoted text, not the model label, determines semantics",
                })
                item.update({
                    "event_type": f"{source_class}:literal_{claim_kind}",
                    "attributes": {
                        "source_span": span,
                        **({"compiler_normalizations": [
                            *time_normalizations,
                            *([class_normalization] if class_normalization else []),
                        ]} if time_normalizations or class_normalization else {}),
                    },
                    "source": {"type": "user_supplied", "reference": reference},
                    "created_by": "llm",
                })
            elif not item.get("event_type"):
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "context_events requires claim_kind or event_type",
                    {"item_index": index})
            normalized_inline.append(item)
        events = (events or []) + events_from_list(normalized_inline)
    qualitative = arguments.get("qualitative_context_events")
    if qualitative is not None:
        if not isinstance(qualitative, list):
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "qualitative_context_events must be an array of event objects",
            )
        normalized = []
        for raw_item in qualitative:
            item = dict(raw_item) if isinstance(raw_item, dict) else raw_item
            if not isinstance(item, dict):
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "qualitative_context_events items must be objects",
                )
            time_normalizations = normalize_daily_times(item)
            allowed_fields = {
                "event_id", "entity_scope", "effective_start",
                "effective_end", "known_at", "direction", "effect_family",
                "duration", "source_span", "source_reference",
            }
            unknown_fields = sorted(set(item) - allowed_fields)
            if unknown_fields:
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "qualitative_context_events contains unsupported fields",
                    {"unknown_fields": unknown_fields},
                )
            required_fields = {
                "event_id", "effective_start", "effective_end", "known_at",
                "direction", "effect_family", "duration", "source_span",
            }
            missing_fields = sorted(field for field in required_fields
                                    if item.get(field) in (None, ""))
            if missing_fields:
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "qualitative_context_events is missing required fields",
                    {"missing_fields": missing_fields},
                )
            source_span = str(item.get("source_span") or "").strip()
            if not source_span:
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    "qualitative_context_events.source_span is required",
                )
            effective_day = str(item.get("effective_start") or "")[:10]
            if len(effective_day) != 10 or effective_day not in source_span:
                arguments.setdefault("context_rejections", []).append({
                    "context_id": str(item.get("event_id")),
                    "reason_code": "ambiguous_timing",
                    "reason": (
                        "The source does not state the exact effective date "
                        "proposed by the qualitative event."),
                    "source_span": source_span,
                })
                continue
            normalized.append({
                "event_id": item.get("event_id"),
                # A non-reserved namespace makes numeric future-context
                # admission structurally unreachable for this lane.
                "event_type": "qualitative:" + str(item.get("event_id")),
                "entity_scope": item.get("entity_scope") or ["*"],
                "effective_start": item.get("effective_start"),
                "effective_end": item.get("effective_end"),
                "known_at": item.get("known_at"),
                "status": "confirmed",
                "confidence": 1.0,
                "attributes": {
                    "source_span": source_span,
                    **({"compiler_normalizations": time_normalizations}
                       if time_normalizations else {}),
                    "soft_context": {
                        "effect_family": item.get("effect_family"),
                        "direction": item.get("direction"),
                        "duration": item.get("duration"),
                        "entity_kind": "unknown",
                    },
                },
                "source": {
                    "type": "user_supplied",
                    "reference": str(item.get("source_reference") or "inline"),
                },
                "created_by": "llm",
            })
        events = (events or []) + events_from_list(normalized)
    return events


def _materialized_or_public_events(arguments: dict[str, Any]):
    """Consume trusted internal events or validate the public channels."""
    events = arguments.pop("_materialized_context_events", None)
    return events if events is not None else _context_events_from(arguments)


def _align_single_target_context_scope(events: list[Any] | None,
                                       arguments: dict[str, Any]) -> list[Any] | None:
    """Map the public target name onto the engine's singleton identity.

    A single-column run is represented internally as ``__default__`` while
    callers only know the requested target column. Treating ``[target]`` as a
    non-match silently discards correctly scoped context. Other names remain
    untouched, so this does not broaden an unrelated event to all series.
    """
    if not events or arguments.get("series_column"):
        return events
    target = str(arguments.get("target_column") or "").strip()
    if not target or "," in target or target.lower() == "auto":
        return events
    from dataclasses import replace

    aligned = []
    for event in events:
        scope = tuple("*" if name == target else name
                      for name in event.entity_scope)
        aligned.append(replace(event, entity_scope=scope)
                       if scope != event.entity_scope else event)
    return aligned


def _materialise_context(
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve or register immutable typed context before numeric execution."""
    if "_materialized_context_events" in arguments:
        # Private transport used only after this function has validated the
        # public channels. A direct runner caller must not be able to inject
        # pre-trusted objects through it.
        raise GnomonError(
            "INVALID_ARGUMENTS", "reserved internal context field supplied")
    carries_context = any(arguments.get(key) is not None for key in (
        "context_ref", "context_events", "context_events_file",
        "qualitative_context_events"))
    if not carries_context:
        return arguments, None
    from .context import event_to_dict
    from .context_store import ContextReceiptStore
    from .soft_context import make_context_receipt

    supplied = [key for key in ("context_ref", "context_events",
                                "context_events_file",
                                "qualitative_context_events")
                if arguments.get(key) is not None]
    if "context_ref" in supplied and len(supplied) > 1:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "context_ref replaces all inline/file context event channels; "
            "do not supply both.",
            {"conflicts": supplied},
        )
    store = ContextReceiptStore.default()
    if arguments.get("context_ref"):
        reference = str(arguments["context_ref"])
        try:
            receipt = store.get(reference)
        except KeyError as error:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "context_ref is unknown in this project namespace.",
                {"context_ref": reference},
                repair_options=[{
                    "action": "resupply_context",
                    "description": (
                        "Send context_events, qualitative_context_events, or "
                        "context_events_file again to receive a fresh context_ref."),
                }],
            ) from error
        events = []
        for raw in receipt.get("events") or []:
            item = dict(raw)
            item["attributes"] = {
                **dict(item.get("attributes") or {}),
                "context_receipt_id": receipt["receipt_id"],
            }
            events.append(item)
        # The immutable store preserves whether the original channel was a
        # trusted file or an unverified inline claim. Rehydrate that recorded
        # creator exactly; do not demote a file event or promote an inline one.
        from .context import events_from_list
        materialized = events_from_list(
            events, trust_declared_creator=True)
        return ({**{key: value for key, value in arguments.items()
                    if key not in {"context_ref", "context_events_file",
                                   "context_events",
                                   "qualitative_context_events"}},
                 "_materialized_context_events": materialized,
                 "_context_was_supplied": True}, {
            "status": "hit", "context_ref": reference,
            "receipt_id": receipt["receipt_id"], "compiler_reused": True,
            "store_schema_version": "0.1",
        })

    parsed = _context_events_from(arguments) or []
    event_ids = {str(event.event_id) for event in parsed}
    rejection_ids = {
        str(item.get("context_id") or item.get("event_id") or "").strip()
        for item in arguments.get("context_rejections") or []
        if isinstance(item, dict)
    }
    disposition_conflicts = sorted(event_ids & (rejection_ids - {""}))
    if disposition_conflicts:
        raise GnomonError(
            "CONTEXT_DISPOSITION_CONFLICT",
            "A context claim cannot be both executable and rejected in the "
            "same request.",
            {"context_ids": disposition_conflicts},
            repair_options=[{
                "action": "choose_context_disposition",
                "description": (
                    "For each listed context_id, keep either its executable "
                    "event or its typed rejection, never both."),
            }],
        )
    raw_events = [event_to_dict(event) for event in parsed]
    receipt = make_context_receipt(
        documents=[], events=raw_events, hypotheses=[], rejected=[],
        rejected_hypotheses=[],
        proposer={"kind": "validated_typed_context", "version": "0.1"},
    )
    reference = store.put(receipt)
    bound_events = []
    for raw in raw_events:
        item = dict(raw)
        item["attributes"] = {
            **dict(item.get("attributes") or {}),
            "context_receipt_id": receipt["receipt_id"],
        }
        bound_events.append(item)
    # Carry already-validated typed objects to the numeric runner. Serialising
    # and reparsing them through the public inline channel would erase the
    # operator-controlled file boundary and incorrectly make them
    # scenario-only.
    from dataclasses import replace

    materialized = []
    for event, raw in zip(parsed, bound_events):
        materialized.append(replace(
            event, attributes=dict(raw.get("attributes") or {})))
    return ({**{key: value for key, value in arguments.items()
                if key not in {"context_events_file", "context_events",
                               "qualitative_context_events"}},
             "_materialized_context_events": materialized,
             "_context_was_supplied": True}, {
        "status": "stored", "context_ref": reference,
        "receipt_id": receipt["receipt_id"], "compiler_reused": False,
        "store_schema_version": "0.1",
    })


def _run_preflight_context(arguments: dict[str, Any]) -> dict[str, Any]:
    from .contracts import GnomonError
    from .preflight import preflight_context_events

    events = _materialized_or_public_events(arguments)
    if not events:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "Supply events to preflight: context_events (inline array) or "
            "context_events_file.",
        )
    return preflight_context_events(
        str(arguments["input"]),
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        horizon=int(arguments["horizon"]),
        context_events=events,
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        repair=arguments.get("repair", "safe"),
    )
