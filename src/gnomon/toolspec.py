"""Canonical machine-facing tool specifications.

One definition of Gnomon's agent-facing tools — names, JSON Schemas, and
in-process runners over the runtime — consumed by the MCP server. This is
the single source of truth for the public agent contract.
"""

from __future__ import annotations

from typing import Any, Callable

from .context import load_events_file
from .contracts import ForecastArtifact, GnomonError, REPAIR_OPTIONS
from .runtime import capabilities, forecast, inspect_dataset

#: The inline event channel, shared by every tool that accepts context
#: events. Complete enough that a model holding only this schema can
#: construct a valid event — that is the point of it existing.
_CONTEXT_EVENTS_PROPERTY: dict[str, Any] = {
    "context_events": {
        "type": "array",
        "description": (
            "Inline context events; combined with context_events_file. "
            "Admission remains strict; preflight deterministic claims first."
        ),
        "items": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "Unique id for this event."},
                "event_type": {"type": "string", "description": "constraint:<label> (stated bound), override:<label> (stated level/state), structural:<label> (stated cessation; needs attributes.effect), or a plain type for the fold-ablation gate."},
                "entity_scope": {"type": "array", "items": {"type": "string"}, "description": "Series names the event applies to; [\"*\"] for all."},
                "effective_start": {"type": "string", "description": "ISO-8601 with an explicit timezone offset."},
                "effective_end": {"type": "string", "description": "ISO-8601 with an explicit offset; not before effective_start."},
                "known_at": {"type": "string", "description": "When the information became knowable; ISO-8601 with an explicit offset."},
                "attributes": {"type": "object", "description": "Per-class payload: source_span (the claim sentence, quoted VERBATIM — the only admissible source of numbers), effect (trend_ceases | level_matches_seasonal_high | level_matches_seasonal_low), or claim ({kind: min|max, value: number})."},
                "source": {"type": "object", "properties": {"type": {"type": "string"}, "reference": {"type": "string"}}, "description": "Where the information came from."},
                "created_by": {"type": "string", "description": "user | llm | pipeline."},
            },
            "required": ["event_id", "event_type", "entity_scope",
                         "effective_start", "effective_end", "known_at"],
        },
    },
    "context_ref": {
        "type": "string",
        "description": (
            "Prior context receipt; replaces both event inputs. Timing and "
            "admission are rechecked."
        ),
    },
}

#: The inline covariate channel, shared by every tool that accepts
#: point-in-time covariates. Mutually exclusive with covariates_file;
#: covariate_mapping stays required either way.
_COVARIATES_PROPERTY: dict[str, Any] = {
    "covariates": {
        "type": "array",
        "description": (
            "Point-in-time covariate vintages supplied inline: row "
            "objects with the covariate time column (default "
            "`timestamp`), the known-at column (default `known_at`), and "
            "one key per covariate named in covariate_mapping. Same "
            "validation as covariates_file; mutually exclusive with it."
        ),
        "items": {"type": "object"},
    },
}

#: The covariate declaration, shared by every tool that takes one.
#: parse_mapping treats all three spellings identically; the packed
#: string stays first in the description because it is what existing
#: callers hold, but an agent building the mapping fresh should not have
#: to pack facts into a delimiter grammar to state them.
_COVARIATE_MAPPING_PROPERTY: dict[str, Any] = {
    "covariate_mapping": {
        "type": ["string", "array"],
        "items": {"type": ["string", "object"]},
        "description": (
            "Covariate declarations: comma-separated "
            "name:type:future_known entries, an array of those strings, "
            "or objects with name, type (continuous, binary, or "
            "cyclic_<positive-period>), and "
            "availability (future_known) keys."
        ),
    },
}

#: The inline data channel, shared by every tool that reads observations.
#: The same reason context_events and covariates have one: an MCP client
#: holds no filesystem, and a path-only `input` makes every data-reading
#: tool unreachable for it.
_OBSERVATIONS_PROPERTY: dict[str, Any] = {
    "observations": {
        "type": "array",
        "maxItems": 500,
        "description": (
            "Inline row objects; exclusive with input, maximum 500. Reuse "
            "the returned data_ref."
        ),
        "items": {"type": "object"},
    },
}

_DATA_REF_PROPERTY: dict[str, Any] = {
    "data_ref": {
        "type": "string",
        "description": (
            "Prior call reference; replaces input/observations and binds "
            "schema and cutoff."
        ),
    },
}

_INPUT_PROPERTIES: dict[str, Any] = {
    "input": {"type": "string", "description": "Path to a local CSV, TSV, JSON, JSONL, Parquet, or Excel file of time-series observations, or `store:<dataset>` from the bitemporal store (see gnomon_list_datasets). Callers without a filesystem pass `observations` inline instead."},
    **_OBSERVATIONS_PROPERTY,
    **_DATA_REF_PROPERTY,
    "time_column": {"type": "string", "description": (
        "Timestamp column. Omit to infer when exactly one column "
        "qualifies (disclosed as an assumption); ambiguity fails loudly. "
        "Required for store:<dataset> inputs."
    )},
    "target_column": {"type": "string", "description": (
        "Numeric column to operate on. Omit to infer when exactly one "
        "non-time column qualifies (disclosed as an assumption); "
        "ambiguity fails loudly. Required for store:<dataset> inputs."
    )},
    "series_column": {"type": "string", "description": "Optional column identifying independent series."},
    "frequency": {
        "type": "string",
        "pattern": "^([1-9][0-9]*)?(s|min|h)$|^(D|W|MS)$",
        "description": (
            "Observation frequency: s (seconds), min (minutes), h (hourly), "
            "D (daily), W (weekly), MS (month start), or any whole-second "
            "sub-daily step as <N>s, <N>min, or <N>h (e.g. 5min, 90s, 2h). "
            "Omit to infer; ambiguity fails loudly."
        ),
    },
    "regrid": {
        "type": "string", "enum": ["business_daily", "month_start"],
        "description": (
            "Declared calendar regrid, applied before validation: "
            "business_daily forward-fills weekends and holidays so Mon-Fri "
            "market data lands on the continuous daily grid (implies "
            "frequency=D); month_start restamps monthly data (typically "
            "month-end stamps) to the first of each month (implies "
            "frequency=MS). A statement about the data's calendar, not a "
            "repair of messiness: fills and restamps are disclosed as "
            "warnings but not charged against the repair ceiling."
        ),
    },
}

#: Replay controls, shared by every verb that reads data. `gnomon_forecast`
#: and `gnomon_inspect` were the two that lacked them, which made the
#: bitemporal store and `--as-of` replay CLI-only — invisible to the agents
#: the MCP server exists to serve.
_REPLAY_PROPERTIES: dict[str, Any] = {
    "as_of": {
        "type": "string",
        "description": (
            "Replay instant (ISO-8601): only data known at or before "
            "this is visible. Meaningful for `store:<dataset>` inputs; a "
            "plain file carries one vintage."
        ),
    },
    "store_path": {
        "type": "string",
        "description": (
            "Override the temporal-store path for `store:<dataset>` inputs "
            "(default ~/.local/share/gnomon)."
        ),
    },
}

_TEMPORAL_QUESTIONS_PROPERTY: dict[str, Any] = {
    # Kept intentionally tiny: detailed validation is returned by the
    # compiler, while every schema byte is repaid on every agent turn.
    "questions": {"type": "array", "items": {"type": "object"}},
}

FORECAST_PREVIEW_ROWS = 12

#: Hard ceiling on a tool response's serialised size. Tuned to hold a
#: single-series brief forecast (~5KB with its full support assessment)
#: with headroom; anything past it is bulk, and bulk lives in the
#: artifact. The trim never touches the epistemic contract — see
#: _PROTECTED_KEYS.
RESPONSE_BUDGET_BYTES = 8192
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
    "context_outcome", "question", "answer", "executable", "calibration",
    "direction_probabilities", "primary_forecast_unchanged",
})

_TRIM_HEAD = 3
_TRIM_TAIL = 2


def _holds_protected(node: Any) -> bool:
    """True when anything in this subtree is a :data:`_PROTECTED_KEYS`
    entry — trimming the list that holds it would take a disclosure with
    it. A six-channel batched forecast's ``results`` list is the case
    that matters: six entries is past the head/tail window, and cutting
    it to five would silently drop a channel's support state."""
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
        # A long list whose entries carry the contract (per-channel
        # results) is descended, never cut: the bulk inside each entry is
        # still trimmable, the entries themselves are not.
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


def enforce_response_budget(payload: Any, budget_bytes: int = RESPONSE_BUDGET_BYTES) -> Any:
    """Trim a runner's response to RESPONSE_BUDGET_BYTES.

    Within budget, the payload passes untouched. Over it, long arrays
    outside the protected subtrees are cut to their first/last entries
    (numeric ones with min/max/mean alongside), `truncated: true` is set,
    and a note points at the artifact for the complete data. Error
    envelopes are never trimmed — repair options are the recovery path."""
    if not isinstance(payload, dict) or payload.get("status") == "error" \
            or "error" in payload:
        return payload
    import json as json_module
    try:
        raw = json_module.dumps(payload, default=str)
    except (TypeError, ValueError):
        return payload
    if len(raw) <= budget_bytes:
        return payload
    trimmed: list[dict[str, Any]] = []
    result = _trim_bulk(payload, "", trimmed)
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


_TIER_ORDER = {
    "invalid": 0, "unsupported": 1, "inconclusive": 2,
    "best_effort": 3, "conditionally_supported": 4,
    "context_trusted": 4, "degraded": 4, "weakly_supported": 4,
    "supported": 5, "supported_ensemble": 5,
}


def apply_response_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Add the compact agent-facing envelope without rewriting artifacts.

    Existing verb payloads remain authoritative and byte-compatible on disk;
    this projection adds stable routing fields to MCP responses. Repeated
    warning text is grouped with a count instead of asking an agent to infer
    prevalence from prose. The complete per-series warnings remain in the
    artifact and in each result.
    """
    if payload.get("status") == "error" or "error" in payload:
        return payload
    result = dict(payload)

    if "artifact_id" not in result:
        for key in ("forecast_id", "investigation_id", "anomaly_id",
                    "decision_id", "monitor_id", "route_id"):
            if result.get(key):
                result["artifact_id"] = result[key]
                break

    entries = [item for item in result.get("results", [])
               if isinstance(item, dict)]
    tiers: list[str] = []
    warning_series: dict[str, set[str]] = {}
    warning_examples: dict[str, list[str]] = {}
    recoveries: list[dict[str, Any]] = []
    for entry in entries:
        assessment = entry.get("support_assessment") or {}
        tier = assessment.get("status") or entry.get("support")
        if tier:
            tiers.append(str(tier))
        series = str(entry.get("series") or "__default__")
        for warning in entry.get("warnings") or []:
            text = str(warning)
            warning_series.setdefault(text, set()).add(series)
            examples = warning_examples.setdefault(text, [])
            if series not in examples and len(examples) < 3:
                examples.append(series)
        for action in assessment.get("recovery_actions") or []:
            if isinstance(action, dict) and action not in recoveries:
                recoveries.append(action)

    if tiers and "tier_floor" not in result:
        result["tier_floor"] = min(tiers, key=lambda item: _TIER_ORDER.get(item, 0))
    if warning_series and "limitation_groups" not in result:
        result["limitation_groups"] = [
            {
                "code": "RUNTIME_WARNING",
                "message": message,
                "affected_series_count": len(warning_series[message]),
                "examples": warning_examples[message],
            }
            for message in sorted(warning_series)
        ]
    if recoveries and "recovery_actions" not in result:
        result["recovery_actions"] = recoveries
    return result


def apply_temporal_grounding(payload: dict[str, Any]) -> dict[str, Any]:
    """Echo the data boundary and wall clock on every data-bearing response."""
    from datetime import datetime, timezone

    result = dict(payload)
    now = datetime.now(timezone.utc)
    result["wall_clock_now"] = now.isoformat()
    series_end = result.get("series_end")
    frequency = result.get("frequency")
    if series_end is None:
        series = result.get("series") or []
        ends = [row.get("end") for row in series if isinstance(row, dict) and row.get("end")]
        if ends:
            series_end = max(ends)
        frequency = (result.get("schema") or {}).get("frequency")
    if series_end is None:
        rows = result.get("results") or []
        first = next((row.get("forecast", [])[0].get("timestamp")
                      for row in rows if isinstance(row, dict) and row.get("forecast")), None)
        frequency = frequency or ((result.get("task") or {}).get("schema") or {}).get("frequency")
        if first and frequency:
            from .temporal import frequency_step
            step = frequency_step(str(frequency))
            if step is not None:
                parsed = datetime.fromisoformat(str(first).replace("Z", "+00:00"))
                series_end = (parsed - step).isoformat()
    if series_end is not None:
        result["series_end"] = str(series_end)
        try:
            parsed_end = datetime.fromisoformat(str(series_end).replace("Z", "+00:00"))
            if parsed_end.tzinfo is None:
                parsed_end = parsed_end.replace(tzinfo=timezone.utc)
            gap = now - parsed_end.astimezone(timezone.utc)
            from .temporal import frequency_step
            step = frequency_step(str(frequency)) if frequency else None
            if step is not None and gap > step:
                result["staleness"] = (
                    f"The latest observation is {gap.days} days behind the wall clock "
                    f"({series_end} versus {result['wall_clock_now']})."
                )
        except (TypeError, ValueError):
            pass
    return result


def triage_wide_response(payload: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    """Bound a wide response while leaving the immutable artifact complete."""
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) <= top_k:
        return payload
    ranked = sorted(rows, key=lambda row: (
        -float(row.get("notability", 0.0)), str(row.get("series", ""))))
    remainder = ranked[top_k:]
    tier_counts: dict[str, int] = {}
    for row in remainder:
        assessment = row.get("support_assessment") or {}
        tier = str(assessment.get("status") or row.get("support") or "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    return {
        **payload,
        "results": ranked[:top_k],
        "series_triage": {
            "total": len(rows), "returned": top_k,
            "ranking": "threshold crossing, then relative forecast movement",
            "remainder_count": len(remainder),
            "remainder_tiers": dict(sorted(tier_counts.items())),
            "full_results": ("Use gnomon_get_artifact with series/fields/where/"
                             "order_by/limit selectors on artifact_path."),
        },
    }


def compact_support_details(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep claim-bearing support fields inline; sensitivity lives in artifact."""
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) <= 3:
        return payload
    compacted = []
    # The per-result support contract remains frozen for existing consumers.
    # Sensitivity is diagnostic bulk; the complete block remains in artifact.
    keep = {"status", "reasons", "recovery_actions", "assumptions",
            "disclosures", "legacy_support", "measured_coverage"}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("support_assessment"), dict):
            compacted.append(row)
            continue
        assessment = row["support_assessment"]
        projected = {key: value for key, value in assessment.items()
                     if key in keep and value not in (None, [], {})}
        compacted.append({**row, "support_assessment": projected})
    return {**payload, "results": compacted}

#: Tools whose missing time_column/target_column are inferred from the file
#: when it leaves no choice — the same set of verbs the CLI infers for.
#: gnomon_ingest is deliberately absent: a write to the store under guessed
#: columns would persist the guess.
_SCHEMA_INFERENCE_TOOLS: frozenset[str] = frozenset({
    "gnomon_inspect", "gnomon_describe", "gnomon_forecast",
    "gnomon_validate_covariates",
    "gnomon_preflight_context", "gnomon_route",
    "gnomon_investigate_change", "gnomon_detect_anomalies",
    "gnomon_decide", "gnomon_monitor", "gnomon_run",
})


def disclose_assumptions(payload: Any, assumptions: list[str]) -> Any:
    """Attach caller-level inferences to every result's support assessment.

    An inference the caller is not told about is a guess. These ride in the
    same `assumptions` list as `known_time_assumed`, so an agent reading the
    envelope finds them where it already looks. Payloads without a results
    list carry them at the top level instead.
    """
    if not assumptions or not isinstance(payload, dict):
        return payload
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return {**payload, "assumptions": assumptions}
    decorated = []
    for result in results:
        if not isinstance(result, dict):
            decorated.append(result)
            continue
        assessment = dict(result.get("support_assessment") or {})
        assessment["assumptions"] = list(assessment.get("assumptions", [])) + assumptions
        decorated.append({**result, "support_assessment": assessment})
    return {**payload, "results": decorated}


def _resolve_schema_arguments(
    arguments: dict[str, Any], tool_name: str,
) -> tuple[dict[str, Any], list[str]]:
    """Fill ``time_column``/``target_column`` from the file, or say why not.

    The CLI learned this in v0.4 because a bare missing-argument failure on
    an obvious two-column file was the most common first-run failure there
    is; the tool surface had kept the old behaviour. Same rules as the CLI:
    strict inference (exactly one column qualifies), every inference
    disclosed as an assumption, ambiguity refused with the candidates named
    — in tool-parameter vocabulary, not CLI flags."""
    if arguments.get("time_column") and arguments.get("target_column"):
        return arguments, []
    from .contracts import GnomonError

    source = str(arguments.get("input") or "")
    missing = [name for name in ("time_column", "target_column")
               if not arguments.get(name)]
    if source.startswith("store:"):
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "time_column and target_column are required for store:<dataset> "
            "inputs; a stored dataset has no file header to infer from.",
            {"input": source, "missing_parameters": missing},
            repair_options=[{
                "action": "supply_arguments",
                "description": "Pass time_column and target_column "
                               "explicitly; gnomon_list_datasets shows each "
                               "dataset's variables.",
                "arguments": missing,
            }],
        )
    from .data import infer_schema_columns

    inferred = infer_schema_columns(source)
    resolved = dict(arguments)
    assumptions: list[str] = []
    for parameter, key, candidates_key in (
        ("time_column", "time", "time_candidates"),
        ("target_column", "target", "target_candidates"),
    ):
        if resolved.get(parameter):
            continue
        chosen = inferred[key]
        if chosen is None:
            candidates = list(inferred[candidates_key])
            if parameter == "target_column" and candidates \
                    and tool_name in {"gnomon_inspect", "gnomon_describe"}:
                # Inspection is read-only and cheap, and a caller who
                # names no target on a wide file is asking about the
                # file: inspect every qualifying column rather than
                # refuse. Not a guess — nothing is chosen over anything
                # else, and the expansion is disclosed. Forecasting
                # still refuses loudly: acting on one guessed column
                # (or paying for all of them) is a real choice.
                resolved[parameter] = ",".join(candidates)
                assumptions.append(
                    f"target_column was not supplied and "
                    f"{len(candidates)} columns qualify "
                    f"({', '.join(candidates)}); all of them were "
                    f"inspected. Pass target_column to narrow."
                )
                continue
            repairs: list[dict[str, Any]] = [{
                "action": "supply_arguments",
                "description": f"Pass {parameter} explicitly."
                               + (f" Candidates: {', '.join(candidates)}."
                                  if candidates else ""),
                "arguments": [parameter],
            }]
            if candidates:
                repairs = [
                    {
                        "action": "supply_arguments",
                        "description": f"Retry with {parameter}={candidate!r}.",
                        "tool_call": {
                            "name": tool_name,
                            "arguments": {**resolved, parameter: candidate},
                        },
                    }
                    for candidate in candidates
                ]
            if parameter == "target_column" and tool_name == "gnomon_forecast":
                repairs.append({
                    "action": "forecast_all_candidates",
                    "description": "Or batch every numeric column in one "
                                   "run: pass target_column \"auto\".",
                    "tool_call": {
                        "name": tool_name,
                        "arguments": {**resolved, parameter: "auto"},
                    },
                })
            raise GnomonError(
                "AMBIGUOUS_SCHEMA",
                f"{parameter} was not supplied and cannot be inferred: "
                + (f"{len(candidates)} columns qualify "
                   f"({', '.join(candidates)})."
                   if candidates else "no column qualifies.")
                + f" Pass {parameter} explicitly; every other parameter is "
                  f"inferred from the file.",
                {"parameter": parameter, "candidates": candidates,
                 "columns_examined": list(inferred["time_candidates"])
                 + list(inferred["target_candidates"])},
                repair_options=repairs,
            )
        resolved[parameter] = chosen
        assumptions.append(
            f"{parameter} was not supplied; inferred as {chosen!r}, the only "
            f"column in the input that qualifies."
        )
    return resolved, assumptions


def _default_forecast_horizon(arguments: dict[str, Any]) -> int:
    """One seasonal period of the input's own grid — the CLI's default."""
    from .pipeline import load_stage
    from .temporal import detect_season

    target_spec = str(arguments["target_column"])
    if "," in target_spec or target_spec.strip().lower() == "auto":
        # Borrow one concrete column so the loader sees a real name; the
        # grid and season are shared across a wide file's channels.
        from .data import resolve_target_spec

        target_spec = resolve_target_spec(
            str(arguments["input"]), target_spec,
            time_column=arguments.get("time_column"),
            series_column=arguments.get("series_column"),
        )[0]
    loaded = load_stage(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=target_spec,
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        store_path=arguments.get("store_path"),
    )
    longest = max(loaded.groups.values(), key=len, default=[])
    season, _, _ = detect_season([item.value for item in longest], loaded.frequency)
    return max(1, int(season))


def forecast_summary(artifact: ForecastArtifact, path: Any) -> dict[str, Any]:
    """The compact forecast payload shared by the CLI and every adapter.

    The first forecast rows are inlined so an agent can quote numbers without
    a second read; the full series always lives in forecast.csv."""
    from .support import artifact_headline, forecast_notability
    from .temporal_profile import compact_temporal_profile

    def response_facts(item: Any) -> dict[str, Any] | None:
        if not item.temporal_facts:
            return None
        facts = dict(item.temporal_facts)
        if facts.get("temporal_profile") and item.support not in {
                "best_effort", "unsupported", "invalid", "inconclusive"}:
            facts["temporal_profile"] = compact_temporal_profile(
                facts["temporal_profile"])
        else:
            facts.pop("temporal_profile", None)
        return facts
    payload = {
        "schema_version": "0.1",
        "status": "complete",
        "forecast_id": artifact.forecast_id,
        "artifact_path": str(path),
        # One deterministic sentence, template-generated from the
        # assessment, naming the weakest tier present — the sentence an
        # agent may relay verbatim. Required in every format.
        "headline": artifact_headline(artifact.results),
        **_forecast_temporal_boundary(artifact),
        "results": [
            {
                "series": item.series, "support": item.support,
                "support_assessment": item.support_assessment,
                "selected_model": item.selected_model,
                "interval_coverage": item.interval_coverage,
                "warnings": item.warnings,
                # Epistemic disclosures ride in every format: brief already
                # carried notes; full dropping them meant the verbose mode
                # disclosed less than the compact one.
                "notes": item.notes,
                "forecast_preview": item.forecast[:FORECAST_PREVIEW_ROWS],
                "forecast_rows": len(item.forecast),
                **({
                    "primary_forecast_preview":
                        item.primary_forecast[:FORECAST_PREVIEW_ROWS],
                    "primary_forecast_rows": len(item.primary_forecast),
                    "forecast_role": "context_conditioned_projection",
                    "primary_forecast_location":
                        "artifact.results[].primary_forecast",
                } if item.primary_forecast else {
                    "forecast_role": "primary_forecast",
                }),
                "threshold": item.threshold,
                "context": item.context,
                "context_outcome": item.context_outcome,
                **({"sensitivity_scenarios": [{
                    "events": scenario.get("events", []),
                    "support": scenario.get("support"),
                    "primary_forecast_changed": False,
                    "assumed_effect": scenario.get("assumed_effect"),
                    "assumed_effect_unit": scenario.get("assumed_effect_unit"),
                    "assumptions": scenario.get("assumptions", []),
                    "forecast_rows": len(scenario.get("forecast", [])),
                    "location": "artifact.results[].sensitivity_scenarios",
                } for scenario in item.sensitivity_scenarios]}
                   if item.sensitivity_scenarios else {}),
                "covariates": item.covariates,
                "notability": forecast_notability(item),
                "execution_identity": _execution_identity(artifact, item),
                **({"temporal_facts": response_facts(item)}
                   if item.temporal_facts else {}),
            }
            for item in artifact.results
        ],
    }
    return _attach_multiseries_triage(payload)


def brief_summary(artifact: ForecastArtifact, path: Any) -> dict[str, Any]:
    """The compact forecast payload: q50 path, one q10–q90 interval, the
    selection, and every disclosure — roughly summary.md as JSON.

    What it drops is bulk only: the extra quantile levels, the raw
    `point` path beside its bias correction, and the context/covariate
    gate detail (all still in the artifact on disk, which is written
    unchanged). What it may never drop is epistemics: the support state,
    every warning, every abstention reason, every recovery action, and
    every disclosure ride along verbatim — an abstention serialises the
    same structured support assessment full mode carries. Hiding
    disclosures is the one thing this codebase exists to not do.
    """
    from .support import forecast_notability
    from .temporal_profile import compact_temporal_profile

    def response_facts(item: Any) -> dict[str, Any] | None:
        if not item.temporal_facts:
            return None
        facts = dict(item.temporal_facts)
        if facts.get("temporal_profile") and item.support not in {
                "best_effort", "unsupported", "invalid", "inconclusive"}:
            facts["temporal_profile"] = compact_temporal_profile(
                facts["temporal_profile"])
        else:
            facts.pop("temporal_profile", None)
        return facts
    results = []
    for item in artifact.results:
        results.append({
            "series": item.series,
            "support": item.support,
            "selected_model": item.selected_model,
            "interval_coverage": item.interval_coverage,
            # Verbatim, never summarised: the same objects full mode carries.
            "warnings": item.warnings,
            "support_assessment": item.support_assessment,
            "notes": item.notes,
            "forecast": [
                {"timestamp": row["timestamp"], "q50": row["q50"],
                 "q10": row["q10"], "q90": row["q90"],
                 # The unstrippable label rides on every row in every
                 # format; brief may drop quantile levels, never the tier.
                 **({"tier": row["tier"]} if "tier" in row else {})}
                for row in item.forecast
            ],
            # The row count survives even when the budget trims the rows.
            "forecast_rows": len(item.forecast),
            **({
                "primary_forecast_preview": [
                    {"timestamp": row["timestamp"], "q50": row["q50"],
                     "q10": row["q10"], "q90": row["q90"],
                     **({"tier": row["tier"]} if "tier" in row else {})}
                    for row in item.primary_forecast[:FORECAST_PREVIEW_ROWS]
                ],
                "primary_forecast_rows": len(item.primary_forecast),
                "forecast_role": "context_conditioned_projection",
                "primary_forecast_location":
                    "artifact.results[].primary_forecast",
            } if item.primary_forecast else {
                "forecast_role": "primary_forecast",
            }),
            "notability": forecast_notability(item),
            "execution_identity": _execution_identity(artifact, item),
            **({"temporal_facts": response_facts(item)}
               if item.temporal_facts else {}),
            **({"threshold": item.threshold} if item.threshold else {}),
            **({"context_outcome": item.context_outcome}
               if item.context_outcome else {}),
            **({"sensitivity_scenarios": [{
                "events": scenario.get("events", []),
                "support": scenario.get("support"),
                "primary_forecast_changed": False,
                "assumed_effect": scenario.get("assumed_effect"),
                "assumed_effect_unit": scenario.get("assumed_effect_unit"),
                "assumptions": scenario.get("assumptions", []),
                "forecast_rows": len(scenario.get("forecast", [])),
                "location": "artifact.results[].sensitivity_scenarios",
            } for scenario in item.sensitivity_scenarios]}
               if item.sensitivity_scenarios else {}),
        })
    from .support import artifact_headline
    payload = {
        "schema_version": "0.1",
        "status": "complete",
        "format": "brief",
        "forecast_id": artifact.forecast_id,
        "artifact_path": str(path),
        "headline": artifact_headline(artifact.results),
        **_forecast_temporal_boundary(artifact),
        "note": (
            "Brief output: q50 with the q10-q90 interval per step. The full "
            "artifact (all quantile levels, evidence, lineage) is on disk at "
            "artifact_path, unchanged."
        ),
        "results": results,
    }
    return _attach_multiseries_triage(payload)


def _execution_identity(artifact: ForecastArtifact, item: Any) -> dict[str, Any]:
    """Expose the persisted evaluation/publication seam in the first response."""
    candidate = next((evidence.payload for evidence in artifact.evidence
                      if evidence.kind == "final_candidate"
                      and evidence.series == item.series), None)
    evaluated = dict(candidate) if candidate else {
        "kind": "builtin", "name": item.selected_model,
        "data_fingerprint": artifact.source_fingerprint,
    }
    published = {"name": item.selected_model,
                 "data_fingerprint": artifact.source_fingerprint}
    return {
        "evaluated": evaluated,
        "published": published,
        "publish_matches_evaluated": (
            evaluated.get("name") == published["name"]),
        "runtime_version": artifact.runtime_version or None,
    }


def _attach_multiseries_triage(payload: dict[str, Any]) -> dict[str, Any]:
    """Bounded, typed summary for wide forecasts; the artifact keeps all rows."""
    results = [item for item in payload.get("results", [])
               if isinstance(item, dict)]
    if len(results) <= 3:
        return payload
    ranked = sorted(results, key=lambda item: (
        -float(item.get("notability") or 0.0), str(item.get("series") or "")))
    notable = [{"series": item.get("series"),
                "notability": item.get("notability"),
                "support": item.get("support")}
               for item in ranked[:3]]
    return {**payload, "triage": {
        "series_count": len(results),
        "ranking_rule": "threshold crossing, then relative path movement",
        "notable": notable,
        "remainder_count": len(results) - len(notable),
        # A bounded response is not data loss: every omitted series remains
        # addressable in the immutable artifact named below.
        "remainder_preserved": True,
        "full_results": {
            "tool_call": {"name": "gnomon_get_artifact", "arguments": {
                "artifact_path": payload.get("artifact_path"),
                "order_by": "notability", "limit": len(results),
            }},
        },
    }}


def _forecast_temporal_boundary(artifact: ForecastArtifact) -> dict[str, Any]:
    """Derive the last observed instant from the first forecast grid point."""
    from datetime import datetime
    from .temporal import frequency_step

    first = next((item.forecast[0].get("timestamp") for item in artifact.results
                  if item.forecast), None)
    step = frequency_step(artifact.task.schema.frequency)
    if first is None or step is None:
        return {"frequency": artifact.task.schema.frequency}
    parsed = datetime.fromisoformat(str(first).replace("Z", "+00:00"))
    return {"series_end": (parsed - step).isoformat(),
            "frequency": artifact.task.schema.frequency}


#: Prose longer than this is elided from the brief capabilities view.
#: Machine-readable facts (names, flags, enums, parameter spellings) are
#: all shorter; what crosses the line is explanatory prose, which the
#: full view and the per-section view carry verbatim.
_CAPABILITIES_PROSE_LIMIT = 100


def _brief_capabilities(full: dict[str, Any]) -> dict[str, Any]:
    """The default capabilities view, sized to the response budget.

    Every top-level section of the full payload is present and every
    capability *name* (tools, models, operators, macros, features,
    frequencies, flags) survives verbatim — a brief view that hid a
    capability would make the command lie about the build. What is
    elided, and said to be elided, is prose: long explanatory strings
    and the per-operator / per-TSFM metadata blocks, all of which the
    caller gets verbatim with ``format: "full"`` or ``sections``.
    """
    elided: list[str] = []

    def compact(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                where = f"{path}.{key}" if path else str(key)
                if isinstance(value, str) and len(value) > _CAPABILITIES_PROSE_LIMIT:
                    elided.append(where)
                    continue
                out[key] = compact(value, where)
            return out
        if isinstance(node, list):
            return [compact(item, path) for item in node]
        return node

    brief: dict[str, Any] = {}
    for key, value in full.items():
        if key in ("operators", "macros") and isinstance(value, dict):
            # The names are the capability; the per-entry contracts
            # (summary, minimum data, abstention policy) are detail.
            brief[key] = {"available": sorted(value)}
            elided.append(f"{key}.<details>")
        elif key == "models" and isinstance(value, dict):
            models = dict(value)
            matrix = models.get("tsfm_capabilities")
            if isinstance(matrix, dict):
                models["tsfm_capabilities"] = {"models": sorted(matrix)}
                elided.append("models.tsfm_capabilities.<details>")
            brief[key] = compact(models, "models")
        else:
            brief[key] = compact(value, key)
    brief["view"] = {
        "format": "brief",
        "sections_available": sorted(full),
        "elided": sorted({path.split(".", 1)[0] for path in elided}),
        "note": (
            "Brief view: every section and every capability name is "
            "present; sections in `elided` omit prose/detail. Use format "
            "'full' or request named sections for verbatim detail."
        ),
    }
    return brief


def _run_capabilities(arguments: dict[str, Any]) -> dict[str, Any]:
    full = capabilities()
    requested = arguments.get("sections")
    if requested:
        from .contracts import GnomonError

        if not isinstance(requested, list) or not all(
                isinstance(name, str) for name in requested):
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "sections must be an array of section names; "
                f"available: {', '.join(sorted(full))}.",
            )
        unknown = [name for name in requested if name not in full]
        if unknown:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                f"Unknown capabilities section(s) "
                f"{', '.join(sorted(unknown))}; available: "
                f"{', '.join(sorted(full))}.",
                {"unknown_sections": sorted(unknown),
                 "sections_available": sorted(full)},
                repair_options=[{
                    "action": "supply_arguments",
                    "description": "Pass section names from "
                                   "sections_available.",
                    "arguments": ["sections"],
                }],
            )
        return {
            "schema_version": full["schema_version"],
            "runtime_version": full["runtime_version"],
            **{name: full[name] for name in requested},
            "view": {"format": "sections",
                     "sections_available": sorted(full)},
        }
    if arguments.get("format") == "full":
        return full
    return _brief_capabilities(full)


def _run_inspect(arguments: dict[str, Any]) -> dict[str, Any]:
    target_spec = str(arguments["target_column"])
    if "," in target_spec or target_spec.strip().lower() == "auto":
        return _run_inspect_multi(arguments, target_spec)
    return inspect_dataset(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        store_path=arguments.get("store_path"),
        regrid=arguments.get("regrid"),
    )


def _run_inspect_multi(arguments: dict[str, Any], target_spec: str) -> dict[str, Any]:
    """The multi-target branch of gnomon_inspect: a comma list or `auto`
    in target_column inspects several columns of a wide file in one call
    — the same expansion gnomon_forecast batches with, so the natural
    first call on a multi-channel file is one inspect, not one per
    column (or, before this branch existed, an AMBIGUOUS_SCHEMA round).

    One report per target, keyed by column name (a mapping, not a list,
    so the response budget's array trim can never drop a channel's
    report). A column that fails to load carries its error envelope in
    place — its failure is that column's diagnosis, and it must not
    block the others any more than an abstaining channel blocks a
    batched forecast."""
    from .contracts import GnomonError
    from .data import resolve_target_spec

    targets = resolve_target_spec(
        str(arguments["input"]), target_spec,
        time_column=arguments.get("time_column"),
        series_column=arguments.get("series_column"),
    )
    if len(targets) == 1:
        return _run_inspect({**arguments, "target_column": targets[0]})
    shared: dict[str, Any] = {}
    reports: dict[str, dict[str, Any]] = {}
    errors: dict[str, GnomonError] = {}
    for name in targets:
        try:
            report = _run_inspect({**arguments, "target_column": name})
        except GnomonError as error:
            errors[name] = error
            reports[name] = {"status": "error", "error": {
                "code": error.code, "message": error.message,
                **({"details": error.details} if error.details else {}),
                **({"repair_options": error.repair_options}
                   if error.repair_options else {}),
            }}
            continue
        if not shared:
            shared = {key: report[key] for key in
                      ("input_path", "source_fingerprint", "columns",
                       "schema")}
            shared["schema"] = {**report["schema"],
                                "target_column": ",".join(targets)}
        series = report["series"]
        if len(series) == 1 and series[0].get("name") == "__default__":
            # The loader names a single unscoped group "__default__"; in
            # the combined view the column is the honest label.
            series = [{**series[0], "name": name}]
        reports[name] = {"status": report["status"], "series": series,
                         "data_quality": report["data_quality"]}
    if not reports or len(errors) == len(targets):
        # Every column failed: the first failure is the file's diagnosis.
        raise next(iter(errors.values()))
    valid = [name for name in targets if name not in errors]
    ranked = sorted(valid, key=lambda name: (
        -float((reports[name].get("series") or [{}])[0]
               .get("change", {}).get("absolute_final_step", 0.0)), name))
    return {
        "schema_version": "0.1",
        "status": "valid" if not errors else "partial",
        **shared,
        "targets": reports,
        "triage": {
            "ranking_rule": "largest absolute final-step change",
            "most_notable": ranked[0] if ranked else None,
            "notable": ranked[:3],
            "remainder_count": max(0, len(ranked) - 3),
            "remainder_preserved": True,
        },
        "suggested_next": (
            f"gnomon_forecast with target_column "
            f"\"{','.join(valid)}\" batches every inspected channel "
            f"into one run and one artifact."
        ),
    }


def _run_describe(arguments: dict[str, Any]) -> dict[str, Any]:
    """Fast descriptive evidence: no model selection and no backtest toll."""
    from statistics import mean, median

    from .data import resolve_target_spec
    from .operators import anomaly_score, changepoint_detection, seasonality_analysis
    from .pipeline import load_stage
    from .temporal_profile import compact_temporal_profile, temporal_profile

    target_spec = str(arguments["target_column"])
    targets = (resolve_target_spec(
        str(arguments["input"]), target_spec,
        time_column=arguments.get("time_column"),
        series_column=arguments.get("series_column"),
    ) if "," in target_spec or target_spec.lower() == "auto" else [target_spec])
    reports: dict[str, Any] = {}
    execution_inputs: dict[str, tuple[list[float], int]] = {}
    for target in targets:
        loaded = load_stage(
            arguments["input"], time_column=arguments["time_column"],
            target_column=target, series_column=arguments.get("series_column"),
            frequency=arguments.get("frequency"),
            as_of=_parse_as_of(arguments.get("as_of")),
            store_path=arguments.get("store_path"), regrid=arguments.get("regrid"),
            repair=arguments.get("repair", "safe"),
        )
        for group_name, observations in sorted(loaded.groups.items()):
            values = [item.value for item in observations]
            timestamps = [item.timestamp for item in observations]
            x_mean = (len(values) - 1) / 2
            denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
            slope = (sum((index - x_mean) * (value - mean(values))
                         for index, value in enumerate(values)) / denominator
                     if denominator else 0.0)
            seasonality = seasonality_analysis(values, loaded.frequency)
            changes = changepoint_detection(timestamps, values)
            anomalies = anomaly_score(
                timestamps, values, season=int(seasonality.get("period") or 1))
            notable_anomalies = sorted(
                anomalies.get("anomalies", []),
                key=lambda row: -abs(float(row["score"])),
            )[:5]
            # A long-form panel has one value column and a separate public
            # series identity.  The loader's ``target:group`` name remains
            # useful inside forecast artifacts, but exposing ``value:cpu``
            # to question authors makes an implementation detail part of the
            # reasoning API.  Describe and typed questions therefore use the
            # group identity at this boundary.
            name = (group_name if arguments.get("series_column")
                    and group_name != "__default__"
                    else target if group_name == "__default__"
                    else f"{target}:{group_name}")
            profile = temporal_profile(
                values, season=int(seasonality.get("period") or 1))
            execution_inputs[name] = (
                [float(value) for value in values],
                int(seasonality.get("period") or 1),
            )
            reports[name] = {
                "observations": len(values), "series_start": timestamps[0].isoformat(),
                "series_end": timestamps[-1].isoformat(), "frequency": loaded.frequency,
                "level": {"latest": values[-1], "mean": mean(values),
                          "median": median(values), "minimum": min(values),
                          "maximum": max(values)},
                "trend": {"slope_per_step": slope,
                          "direction": "up" if slope > 0 else "down" if slope < 0 else "flat"},
                "change": {
                    "final_step": values[-1] - values[-2] if len(values) > 1 else 0.0,
                    "absolute_final_step": abs(values[-1] - values[-2])
                    if len(values) > 1 else 0.0,
                },
                "seasonality": seasonality,
                "changepoints": changes,
                "anomalies": {"count": len(anomalies.get("anomalies", [])),
                              "most_extreme": notable_anomalies,
                              "support": anomalies.get("support")},
                "temporal_profile": compact_temporal_profile(profile),
            }
    ranked = sorted(reports, key=lambda name: (
        -float(reports[name]["change"]["absolute_final_step"]), name))
    temporal_answers: list[dict[str, Any]] = []
    if arguments.get("questions") is not None:
        from .temporal_question import compile_temporal_questions
        from .temporal_reasoning import answer_scoped_question

        questions = compile_temporal_questions(
            arguments["questions"], available_targets=reports,
            default_verb="describe")
        for question in questions:
            temporal_answers.append(answer_scoped_question(
                question, reports=reports, execution_inputs=execution_inputs,
                forecast_values=arguments.get("_forecast_values")))
    return _json_temporal_values({
        "schema_version": "0.1", "status": "valid",
        "headline": f"Described {len(reports)} series through "
                    f"{max(report['series_end'] for report in reports.values())}.",
        "reports": reports,
        **({"answers": temporal_answers} if temporal_answers else {}),
        "triage": {
            "ranking_rule": "largest absolute final-step change",
            "most_notable": ranked[0] if ranked else None,
            "notable": ranked[:3],
            "remainder_count": max(0, len(ranked) - 3),
            "remainder_preserved": True,
        },
        "series_end": max(report["series_end"] for report in reports.values()),
        # Description is a complete answer, not a compulsory funnel into an
        # expensive backtest. Hosts may offer forecasting separately when the
        # user actually asks what happens next.
        "suggested_next": [],
    })


def _json_temporal_values(value: Any) -> Any:
    """Normalize operator timestamps before MCP's strict JSON boundary."""
    from datetime import date, datetime

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_temporal_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_temporal_values(item) for item in value]
    return value


def _run_ingest(arguments: dict[str, Any]) -> dict[str, Any]:
    """Append a file's observations to the bitemporal store as vintages."""
    from .ids import SYSTEM_CLOCK
    from .temporal_store import TemporalStore

    store = TemporalStore(arguments.get("store_path"))
    report = store.ingest_csv(
        str(arguments["input"]),
        dataset=str(arguments["dataset"]),
        time_column=str(arguments["time_column"]),
        target_column=str(arguments["target_column"]),
        series_column=arguments.get("series_column"),
        known_at_column=arguments.get("known_at_column"),
        variable=arguments.get("variable"),
        clock=SYSTEM_CLOCK,
    )
    return report.to_dict()


def _run_list_datasets(arguments: dict[str, Any]) -> dict[str, Any]:
    from .temporal_store import TemporalStore

    store = TemporalStore(arguments.get("store_path"))
    datasets = store.list_datasets()
    return {
        "schema_version": "0.1",
        "status": "ok",
        "datasets": [
            {**item,
             "input_ref": f"store:{item['dataset']}",
             "known_time_provenance": store.known_time_provenance(str(item["dataset"]))}
            for item in datasets
        ],
    }


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


def _run_validate_covariates(arguments: dict[str, Any]) -> dict[str, Any]:
    from .contracts import GnomonError
    from .covariates import validate_covariate_file

    inline = arguments.get("covariates")
    if not arguments.get("covariates_file") and inline is None:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "Supply covariates to validate: covariates (inline rows) or "
            "covariates_file.",
        )
    if arguments.get("covariates_file") and inline is not None:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "Provide covariates_file or inline covariates, not both.",
        )
    return validate_covariate_file(
        arguments["input"], arguments.get("covariates_file"),
        arguments["covariate_mapping"],
        time_column=arguments["time_column"], target_column=arguments["target_column"],
        horizon=int(arguments["horizon"]), series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        covariate_time_column=arguments.get("covariate_time_column", "timestamp"),
        covariate_known_at_column=arguments.get("covariate_known_at_column", "known_at"),
        covariate_series_column=arguments.get("covariate_series_column"),
        covariate_rows=inline,
    )


def _context_events_from(arguments: dict[str, Any]):
    """Events from the file channel, the inline channel, or both.

    The inline channel exists because an MCP client holds no
    filesystem: with a file-only parameter the admission lanes were
    unreachable from the published tool surface — a model could be
    told about future_events and still have no way to supply an event.
    Both channels validate loudly through the same contract check.
    """
    from .context import events_from_list
    from .contracts import GnomonError

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
        events = (events or []) + events_from_list(inline)
    return events


def _materialise_context(
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve or register immutable typed context before numeric execution."""
    carries_context = any(arguments.get(key) is not None for key in (
        "context_ref", "context_events", "context_events_file"))
    if not carries_context:
        return arguments, None
    from .context import event_to_dict
    from .context_store import ContextReceiptStore
    from .soft_context import make_context_receipt

    supplied = [key for key in ("context_ref", "context_events",
                                "context_events_file")
                if arguments.get(key) is not None]
    if "context_ref" in supplied and len(supplied) > 1:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "context_ref replaces context_events and context_events_file; "
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
                    "description": ("Send context_events or context_events_file "
                                    "again to receive a fresh context_ref."),
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
        return ({**{key: value for key, value in arguments.items()
                    if key not in {"context_ref", "context_events_file"}},
                 "context_events": events}, {
            "status": "hit", "context_ref": reference,
            "receipt_id": receipt["receipt_id"], "compiler_reused": True,
            "store_schema_version": "0.1",
        })

    parsed = _context_events_from(arguments) or []
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
    return ({**{key: value for key, value in arguments.items()
                if key != "context_events_file"},
             "context_events": bound_events}, {
        "status": "stored", "context_ref": reference,
        "receipt_id": receipt["receipt_id"], "compiler_reused": False,
        "store_schema_version": "0.1",
    })


def _attach_temporal_answers(payload: dict[str, Any], artifact: ForecastArtifact,
                             path: Any, arguments: dict[str, Any]) -> None:
    """Attach opt-in diagnostics without touching the primary artifact."""
    if arguments.get("questions") is None:
        return
    from .context_store import ContextReceiptStore, temporal_answer_cache_key
    from .temporal_question import compile_temporal_questions
    from .temporal_reasoning import TEMPORAL_ANSWER_CONTRACT_VERSION

    target_spec = str(arguments.get("target_column") or "")
    panel_prefix = f"{target_spec}:" if arguments.get("series_column") else ""

    def public_name(name: Any) -> str:
        value = str(name)
        return (value[len(panel_prefix):]
                if panel_prefix and value.startswith(panel_prefix) else value)

    result_names = [public_name(result.series) for result in artifact.results
                    if str(result.series) != "__default__"]
    explicit_names = [item.strip() for item in target_spec.split(",")
                      if item.strip() and item.strip().lower() != "auto"]
    available_targets = (result_names if arguments.get("series_column")
                         else explicit_names or result_names)
    if len(artifact.results) == 1 and explicit_names \
            and not arguments.get("series_column"):
        available_targets = explicit_names[:1]
    questions = compile_temporal_questions(
        arguments["questions"], available_targets=available_targets,
        default_verb="describe")
    store = ContextReceiptStore.default()
    answer_keys = [temporal_answer_cache_key(
        artifact_id=artifact.forecast_id, question=question.to_dict(),
        as_of=arguments.get("as_of"),
        answer_contract_version=TEMPORAL_ANSWER_CONTRACT_VERSION,
    ) for question in questions]
    cached = [store.get_temporal_answer(key) for key in answer_keys]
    cache_hits = sum(answer is not None for answer in cached)

    forecast_values = {
        public_name(result.series): [float(row["point"]) for row in result.forecast]
        for result in artifact.results
    }
    if len(artifact.results) == 1:
        forecast_values[str(arguments["target_column"])] = next(iter(
            forecast_values.values()))
    if cache_hits == len(answer_keys):
        full_answers = [dict(answer) for answer in cached if answer is not None]
        for question, answer in zip(questions, full_answers):
            if answer.get("artifact_id") != artifact.forecast_id \
                    or answer.get("question") != question.to_dict():
                raise GnomonError(
                    "TEMPORAL_ANSWER_CACHE_INTEGRITY",
                    "Cached temporal answer does not match its primary or question.")
        cache_status = "hit"
    else:
        describe_arguments = {
            key: arguments.get(key) for key in (
                "input", "time_column", "target_column", "series_column",
                "frequency", "as_of", "store_path", "regrid", "repair",
                "questions")
        }
        describe_arguments["_forecast_values"] = forecast_values
        described = _run_describe(describe_arguments)
        full_answers = [
            {**answer, "artifact_id": artifact.forecast_id}
            for answer in described.get("answers", [])
        ]
        if len(full_answers) != len(answer_keys):
            raise GnomonError(
                "TEMPORAL_ANSWER_CACHE_INTEGRITY",
                "Temporal execution did not return one answer per question.")
        for key, answer in zip(answer_keys, full_answers):
            store.put_temporal_answer(key, answer)
        cache_status = "stored" if cache_hits == 0 else "refreshed"
    # Keep the inline contract quotable. Constituent executions remain in the
    # immutable receipt; repeating every child through each agent turn harms
    # both token economics and preservation of the scalar decision.
    payload["answers"] = []
    for answer in full_answers:
        compact = {key: value for key, value in answer.items()
                   if key not in {"per_series", "calibration"}}
        children = answer.get("per_series") or []
        if children:
            compact["constituent_summary"] = {
                "count": len(children),
                "support": {
                    label: sum((child.get("answer") or {}).get("support") == label
                               for child in children)
                    for label in ("supported", "weak", "abstained")
                },
                "details_in_answer_receipt": True,
            }
        payload["answers"].append(compact)
    import json as _json
    answer_receipt = path / "temporal_answers.json"
    answer_receipt.write_text(_json.dumps({
        "schema_version": "0.2", "artifact_id": artifact.forecast_id,
        "primary_forecast_unchanged": True, "answers": full_answers,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["answer_receipt"] = str(answer_receipt)
    payload["answer_cache"] = {
        "status": cache_status, "hits": cache_hits,
        "questions": len(answer_keys),
        "answer_contract_version": TEMPORAL_ANSWER_CONTRACT_VERSION,
        "primary_forecast_unchanged": True,
    }


def _run_forecast(arguments: dict[str, Any]) -> dict[str, Any]:
    target_spec = str(arguments["target_column"])
    if "," in target_spec or target_spec.strip().lower() == "auto":
        return _run_forecast_multi(arguments, target_spec)
    events = _context_events_from(arguments)
    config = None
    if arguments.get("future_events") or arguments.get("structural_events"):
        # MCP tool calls do not read ambient project config — deliberately,
        # so the admission lanes must be reachable as explicit
        # per-call parameters or they are unreachable from the agent
        # surface entirely (the same gap best_effort had).
        from .config import GnomonConfig

        config = GnomonConfig()
        config.context.future_events = bool(arguments.get("future_events"))
        config.context.structural_events = bool(
            arguments.get("structural_events"))
    covariates = _covariates_from(arguments)
    artifact, path = forecast(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        horizon=int(arguments["horizon"]),
        as_of=_parse_as_of(arguments.get("as_of")),
        store_path=arguments.get("store_path"),
        output=arguments.get("output_dir") or "gnomon-output",
        minimum_baseline_improvement=float(arguments.get("minimum_baseline_improvement", 0.02)),
        context_events=events,
        covariates=covariates,
        threshold=float(arguments["threshold"]) if arguments.get("threshold") is not None else None,
        repair=arguments.get("repair", "safe"),
        regrid=arguments.get("regrid"),
        candidates=arguments.get("candidates"),
        best_effort=bool(arguments.get("best_effort", False)),
        minimum_support=str(arguments.get("minimum_support")
                            or "best_effort"),
        config=config,
        input_provenance=arguments.get("input_provenance"),
    )
    # Brief is the default: the full multi-quantile payload is opt-in
    # (format="full"), and the artifact on disk is identical either way.
    payload = (forecast_summary(artifact, path)
               if arguments.get("format") == "full"
               else brief_summary(artifact, path))
    _attach_temporal_answers(payload, artifact, path, arguments)
    if arguments.get("project"):
        from .tracking import register_artifact
        payload["tracking_ids"] = register_artifact(
            artifact, str(arguments["project"]), str(path),
            context_events=events,
        )
        payload["project"] = str(arguments["project"])
    return payload


def _run_forecast_multi(arguments: dict[str, Any], target_spec: str) -> dict[str, Any]:
    """The multi-target branch of gnomon_forecast: a comma list or `auto`
    in target_column batches several columns into one run and one
    combined artifact — same numbers per channel as separate calls."""
    from .contracts import GnomonError
    from .data import resolve_target_spec
    from .runtime import forecast_multi

    targets = resolve_target_spec(
        str(arguments["input"]), target_spec,
        time_column=arguments.get("time_column"),
        series_column=arguments.get("series_column"),
    )
    if len(targets) == 1:
        return _run_forecast({**arguments, "target_column": targets[0]})
    unsupported = [
        name for name in (
            "series_column", "project",
        ) if arguments.get(name)
    ]
    if unsupported:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            f"{', '.join(unsupported)} cannot be combined with a "
            f"multi-target target_column yet; run those channels one "
            f"target at a time.",
            {"unsupported_with_multi_target": unsupported, "targets": targets},
        )
    events = _context_events_from(arguments)
    covariates = _covariates_from(arguments)
    config = None
    if arguments.get("future_events") or arguments.get("structural_events"):
        from .config import GnomonConfig

        config = GnomonConfig()
        config.context.future_events = bool(arguments.get("future_events"))
        config.context.structural_events = bool(arguments.get("structural_events"))
    artifact, path = forecast_multi(
        str(arguments["input"]),
        time_column=arguments["time_column"],
        target_columns=targets,
        frequency=arguments.get("frequency"),
        horizon=int(arguments["horizon"]),
        as_of=_parse_as_of(arguments.get("as_of")),
        output=arguments.get("output_dir") or "gnomon-output",
        minimum_baseline_improvement=float(arguments.get("minimum_baseline_improvement", 0.02)),
        context_events=events,
        covariates=covariates,
        threshold=float(arguments["threshold"]) if arguments.get("threshold") is not None else None,
        repair=arguments.get("repair", "safe"),
        regrid=arguments.get("regrid"),
        candidates=arguments.get("candidates"),
        best_effort=bool(arguments.get("best_effort", False)),
        minimum_support=str(arguments.get("minimum_support")
                            or "best_effort"),
        input_provenance=arguments.get("input_provenance"),
        config=config,
    )
    payload = (forecast_summary(artifact, path)
               if arguments.get("format") == "full"
               else brief_summary(artifact, path))
    _attach_temporal_answers(payload, artifact, path, arguments)
    return payload


def _run_preflight_context(arguments: dict[str, Any]) -> dict[str, Any]:
    from .contracts import GnomonError
    from .preflight import preflight_context_events

    events = _context_events_from(arguments)
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


def _actual_tuples(raw_rows: Any) -> list[tuple]:
    """Inline actuals rows -> the (series?, timestamp, value) tuples the
    tracking store scores. Loud on malformed rows: a silently dropped
    actual would surface as 'nothing was due', which is the exact
    ambiguity the store's diagnosis machinery exists to prevent."""
    import math

    from .contracts import GnomonError

    if not isinstance(raw_rows, list) or not raw_rows:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "actuals must be a non-empty array of {timestamp, value, series?} objects.",
        )
    tuples: list[tuple] = []
    for index, row in enumerate(raw_rows, 1):
        if not isinstance(row, dict) or not row.get("timestamp"):
            raise GnomonError(
                "INVALID_ARGUMENTS",
                f"actuals[{index}] must be an object with a timestamp.",
            )
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                f"actuals[{index}].value must be a finite number.",
            ) from exc
        if not math.isfinite(value):
            raise GnomonError(
                "INVALID_ARGUMENTS",
                f"actuals[{index}].value must be a finite number.",
            )
        timestamp = str(row["timestamp"])
        known_at = str(row["known_at"]) if row.get("known_at") else None
        series = str(row["series"]) if row.get("series") else None
        if known_at is not None:
            # Knowledge-time backfill: 4-tuple form, series may be None.
            tuples.append((series, timestamp, value, known_at))
        elif series is not None:
            tuples.append((series, timestamp, value))
        else:
            tuples.append((timestamp, value))
    return tuples


def _run_submit_actuals(arguments: dict[str, Any]) -> dict[str, Any]:
    import csv as csv_module

    from .contracts import GnomonError
    from .tracking import TrackingStore
    store = TrackingStore()
    project = str(arguments["project"])
    raw_occurrences = list(arguments.get("effect_occurrences") or [])

    def record_occurrences() -> list[dict[str, Any]]:
        recorded = []
        for index, item in enumerate(raw_occurrences, 1):
            if not isinstance(item, dict):
                raise GnomonError(
                    "INVALID_ARGUMENTS",
                    f"effect_occurrences[{index}] must be an object.",
                )
            recorded.append(store.record_effect_occurrence(
                str(item.get("effect_id", "")), str(item.get("status", "")),
                known_at=str(item.get("known_at", "")), note=item.get("note"),
            ))
        return recorded
    inline = arguments.get("actuals")
    if inline is not None and arguments.get("actuals_file"):
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "Provide actuals_file or inline actuals, not both.",
        )
    if inline is not None:
        tuples = _actual_tuples(inline)
        results = store.submit_actuals(project, tuples)
        if not results:
            return {
                "schema_version": "0.1", "status": "ok", "project": project,
                "effect_occurrences": record_occurrences(),
                **store.explain_unscored(
                    project, [item[-2] for item in tuples]),
            }
        return {"schema_version": "0.1", "status": "ok",
                "scored": len(results),
                "results": [item.__dict__ for item in results],
                "effect_occurrences": record_occurrences()}
    if not arguments.get("actuals_file"):
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "Supply actuals to score: actuals (inline array) or actuals_file.",
        )
    path = str(arguments["actuals_file"])
    time_column = arguments.get("time_column")
    target_column = arguments.get("target_column")
    series_column = arguments.get("series_column")
    results = store.submit_actuals_csv(
        project, path, time_column=time_column,
        target_column=target_column, series_column=series_column,
    )
    if not results:
        # A bare `scored: 0` reads as "nothing was due" whether or not
        # anything was due. Return the diagnosis instead.
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv_module.DictReader(handle)
            columns = reader.fieldnames or []
            rows = list(reader)
        resolved_time, _, _ = store._resolve_actuals_columns(
            columns, time_column, target_column, series_column,
        )
        return {
            "schema_version": "0.1", "status": "ok", "project": project,
            "effect_occurrences": record_occurrences(),
            **store.explain_unscored(project, [row[resolved_time] for row in rows]),
        }
    return {"schema_version": "0.1", "status": "ok", "scored": len(results),
            "results": [item.__dict__ for item in results],
            "effect_occurrences": record_occurrences()}


def _run_open_forecasts(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    rows = TrackingStore().due_forecasts(arguments.get("project"))
    return {"status": "ok", "forecasts": rows}


def _run_model_performance(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    store = TrackingStore()
    if arguments.get("model"):
        rows: Any = store.model_performance(
            str(arguments["project"]), str(arguments["model"]),
        )
    else:
        rows = [item.__dict__ for item in store.leaderboard(str(arguments["project"]))]
    return {"status": "ok", "performance": rows,
            "warning": "Historical telemetry is observational, not causal."}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "gnomon_capabilities",
        "description": (
            "Report what the installed Gnomon runtime supports. Use only for "
            "explicit feature discovery, never as a prerequisite to forecast, "
            "describe, or inspect. The "
            "default view is brief (every section and capability name, long "
            "prose elided); pass format 'full' or sections for the verbatim "
            "detail."
        ),
        "inputSchema": {"type": "object", "properties": {
            "format": {"type": "string", "enum": ["brief", "full"],
                       "description": (
                "brief (default): every section with long prose elided, "
                "within the response budget; full: the complete payload "
                "verbatim."
            )},
            "sections": {"type": "array", "items": {"type": "string"},
                         "description": (
                "Return only these top-level sections, verbatim. Any "
                "response's view.sections_available lists the names."
            )},
        }, "required": []},
        "runner": _run_capabilities,
    },
    {
        "name": "gnomon_inspect",
        "description": (
            "Validate a temporal dataset before forecasting: schema mapping, "
            "frequency, duplicates, missing periods. Prefer this before "
            "gnomon_forecast when mappings or data quality are uncertain. "
            "target_column takes a comma list or \"auto\" to inspect every "
            "channel of a wide file in one call (the default when several "
            "columns qualify)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                "target_column": {"type": "string", "description": (
                    "Numeric column to inspect, a comma list "
                    "(`\"cpu,mem\"`), or `\"auto\"` (every numeric "
                    "non-time column) — one report per channel. Omit to "
                    "infer: a lone qualifying column is chosen, several "
                    "are all inspected; either is disclosed as an "
                    "assumption. Required for store:<dataset> inputs."
                )},
                **_REPLAY_PROPERTIES,
            },
            "required": [],
        },
        "runner": _run_inspect,
    },
    {
        "name": "gnomon_describe",
        "description": (
            "Answer descriptive temporal questions without forecasting or "
            "backtesting: level, trend, seasonality, changepoints, anomalies, "
            "and extremes. Use for what-happened questions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                "target_column": {"type": "string", "description": (
                    "Numeric column, comma list, or auto for every numeric channel."
                )},
                **_REPLAY_PROPERTIES,
                **_TEMPORAL_QUESTIONS_PROPERTY,
            },
            "required": [],
        },
        "runner": _run_describe,
    },
    {
        "name": "gnomon_forecast",
        "description": (
            "Forecast columns (`target_column`: `\"cpu,mem,requests\"` or `\"auto\"`). "
            "Call directly: it infers an unambiguous schema, "
            "validates, backtests, and returns a quotable preview; do not call "
            "capabilities, inspect, or get_artifact first. Each series gets a "
            "selected model or disclosed abstention. Context and covariates "
            "earn influence only through measured fold lift."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                **_REPLAY_PROPERTIES,
                "target_column": {"type": "string", "description": (
                    "Column to forecast, a comma list "
                    "(`\"cpu,mem,requests\"`), or `\"auto\"` (every "
                    "numeric non-time column): one load pass, one "
                    "combined artifact, one result per channel — an "
                    "abstaining channel never blocks the others. Omit to "
                    "infer when exactly one column qualifies (disclosed "
                    "as an assumption)."
                )},
                "horizon": {"type": "integer", "description": (
                    "Future periods, in units of the data frequency. "
                    "Default: one seasonal period, disclosed as an "
                    "assumption."
                )},
                "format": {"type": "string", "enum": ["full", "brief"], "description": (
                    "brief (default): q50 with one q10-q90 interval per "
                    "step plus every disclosure verbatim; full adds all "
                    "quantile levels. The on-disk artifact is identical "
                    "either way."
                )},
                "candidates": {"type": "array", "items": {"type": "string"}, "description": "Restrict the model pool to these names (e.g. gnomon_route's recommendation). The mandatory baselines always compete regardless."},
                "output_dir": {"type": "string", "description": (
                    "Directory for the immutable artifact. Omit to use "
                    "workspace.default_output_dir from gnomon_capabilities "
                    "— do not guess a path; a jailed host refuses paths "
                    "outside its workspace."
                )},
                "minimum_baseline_improvement": {"type": "number", "minimum": 0, "description": "Minimum relative improvement over the strongest baseline to select a candidate (default 0.02; must be >= 0)."},
                "context_events_file": {"type": "string", "description": "Optional validated context-events JSON file (the output of `gnomon context validate`)."},
                **_CONTEXT_EVENTS_PROPERTY,
                "threshold": {"type": "number", "description": "Optional decision threshold: the result reports when and how likely the forecast crosses this value."},
                "project": {"type": "string", "description": "Optional tracking project. When set, register the forecast for realised scoring."},
                "covariates_file": {"type": "string", "description": (
                    "Local CSV of point-in-time covariate vintages: one "
                    "row per (timestamp, known_at); a backtest fold only "
                    "uses rows known at or before its cutoff. Validate "
                    "first with gnomon_validate_covariates."
                )},
                **_COVARIATES_PROPERTY,
                **_COVARIATE_MAPPING_PROPERTY,
                "covariate_time_column": {"type": "string", "description": "Valid-at column (default timestamp)."},
                "covariate_known_at_column": {"type": "string", "description": "Availability timestamp column (default known_at)."},
                **_TEMPORAL_QUESTIONS_PROPERTY,
                "covariate_series_column": {"type": "string", "description": "Optional series column in the covariate CSV."},
                "repair": {"type": "string", "enum": ["off", "safe", "aggressive"], "description": "Messy-data handling (default safe): off rejects anything non-strict; safe normalises cell text; aggressive also fills gaps and snaps timestamps — every fix disclosed in warnings and evidence."},
                "minimum_support": {"type": "string",
                                    "enum": ["supported",
                                             "conditionally_supported",
                                             "best_effort"],
                                    "description": (
                    "Publication floor (default best_effort: always "
                    "answers, a naive fallback is labeled and carries a "
                    "NO RELIABLE FORECAST warning; `supported` restores "
                    "the strict refusal with typed recovery). No tier "
                    "gets easier to earn."
                )},
                "best_effort": {"type": "boolean", "description": (
                    "DEPRECATED: `true` maps to minimum_support "
                    "'best_effort' (now the default); an explicit "
                    "minimum_support wins."
                )},
                "future_events": {"type": "boolean", "description": (
                    "Admit future-dated constraint:/override: context "
                    "events by textual verifiability (default false); an "
                    "influenced forecast reports support "
                    "`context_trusted` with a history-only "
                    "counterfactual. Dry-run with "
                    "gnomon_preflight_context."
                )},
                "structural_events": {"type": "boolean", "description": (
                    "Additionally admit LLM-classified structural: events "
                    "from a closed effect menu (default false); every "
                    "applied quantity derives from the forecast's own "
                    "path, never model-supplied numbers. Experimental."
                )},
            },
            "required": [],
        },
        "runner": _run_forecast,
    },
    {
        "name": "gnomon_validate_covariates",
        "description": (
            "Validate covariate vintages for format, coverage, and "
            "availability at every selection cutoff. Format: one row per "
            "(timestamp, known_at); a fold only uses rows known at or "
            "before its cutoff. Mapping grammar: name:type:future_known "
            "entries. Failures name the empty cutoffs; pass the same "
            "arguments to gnomon_forecast for the leakage-safe ablation."
        ),
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {"type": "integer"},
            "covariates_file": {"type": "string"},
            **_COVARIATES_PROPERTY,
            **_COVARIATE_MAPPING_PROPERTY,
            "covariate_time_column": {"type": "string"},
            "covariate_known_at_column": {"type": "string"},
            "covariate_series_column": {"type": "string"},
        }, "required": ["horizon", "covariate_mapping"]},
        "runner": _run_validate_covariates,
    },
    {
        "name": "gnomon_submit_actuals",
        "description": "Score all due forecasts in a project from complete realised actuals. Panel actuals must include series,timestamp,value. A forecast scores only when every period in its horizon has an actual; when nothing scores, the result explains which window was missing rather than returning a bare zero.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"},
            "actuals_file": {"type": "string", "description": "CSV of realised values. Callers without a filesystem pass `actuals` inline instead."},
            "actuals": {"type": "array", "items": {"type": "object"}, "description": (
                "Realised values supplied inline: objects of "
                "{timestamp, value, series?, known_at?}. `known_at` (ISO) "
                "backfills when the outcome became knowable; rows without "
                "it became knowable at this submission. Mutually exclusive "
                "with actuals_file (which accepts a known_at column)."
            )},
            "time_column": {"type": "string", "description": "Timestamp column in the actuals file. Inferred from a conventional name or a two-column layout when omitted."},
            "target_column": {"type": "string", "description": "Realised value column. Inferred when unambiguous."},
            "series_column": {"type": "string", "description": "Series column, required for multi-series projects."},
            "effect_occurrences": {"type": "array", "description": (
                "Optional confirmations for tracked context scenarios. Actual "
                "values do not prove an event happened; each item supplies "
                "effect_id, status (confirmed/cancelled/revised), known_at, "
                "and optional note."
            ), "items": {"type": "object", "properties": {
                "effect_id": {"type": "string"},
                "status": {"type": "string", "enum": [
                    "confirmed", "cancelled", "revised"]},
                "known_at": {"type": "string"},
                "note": {"type": "string"},
            }, "required": ["effect_id", "status", "known_at"]}},
        }, "required": ["project"]},
        "runner": _run_submit_actuals,
    },
    {
        "name": "gnomon_ingest",
        "description": (
            "Append a file's observations to the bitemporal store as vintages. "
            "Supply known_at_column when the source records when each value "
            "became knowable — that is what makes `as_of` replay meaningful. "
            "Without it Gnomon records known_time = valid_time and says so, "
            "which asserts every value was knowable the moment it applied. "
            "Re-ingesting a corrected file appends revisions; it never "
            "overwrites, so the vintage history accumulates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Path to the CSV to ingest. Callers without a filesystem pass `observations` inline instead."},
                **_OBSERVATIONS_PROPERTY,
                "dataset": {"type": "string", "description": "Dataset name; read it back as `store:<dataset>`."},
                "time_column": {"type": "string", "description": "Valid-time column: when the value applies."},
                "target_column": {"type": "string", "description": "Numeric value column."},
                "known_at_column": {"type": "string", "description": "Known-time column: when the value became knowable. Omit only if the source genuinely has no publication lag."},
                "series_column": {"type": "string", "description": "Optional column identifying independent series."},
                "variable": {"type": "string", "description": "Name to store the measure under (defaults to target_column)."},
                "store_path": {"type": "string", "description": "Override the temporal-store path."},
            },
            "required": ["dataset", "time_column", "target_column"],
        },
        "runner": _run_ingest,
    },
    {
        "name": "gnomon_list_datasets",
        "description": (
            "List datasets in the bitemporal store with their observation and "
            "revision counts, their valid- and known-time ranges, and whether "
            "their known times were recorded or assumed. Each carries the "
            "`store:<dataset>` reference to pass as an input."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "store_path": {"type": "string", "description": "Override the temporal-store path."},
            },
            "required": [],
        },
        "runner": _run_list_datasets,
    },
    {
        "name": "gnomon_preflight_context",
        "description": (
            "Dry-run the admission checks for proposed context events "
            "against the actual data, before spending a forecast. Returns "
            "one verdict per event — would_influence, rejected (with the "
            "typed reason), or ablation_gated (fold admission is measured, "
            "not predictable) — plus the span grammar the parser accepts, "
            "so a rejected proposal can be repaired and resubmitted in one "
            "step. Deterministic verdicts here are the verdicts the "
            "forecast will reach on the same data; nothing is written."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                "horizon": {"type": "integer", "description": "Future periods the events would apply to, in units of the data frequency."},
                "context_events_file": {"type": "string", "description": "Context-events JSON file to preflight (the output of `gnomon context validate`, or the same shape)."},
                **_CONTEXT_EVENTS_PROPERTY,
                "repair": {"type": "string", "enum": ["off", "safe", "aggressive"], "description": "Messy-data handling, matched to what the forecast will use (default safe)."},
            },
            "required": ["horizon"],
        },
        "runner": _run_preflight_context,
    },
]


def _parse_as_of(raw: Any):
    if not raw:
        return None
    from .data import _parse_timestamp
    return _parse_timestamp(str(raw), 0)


def _run_investigate_change(arguments: dict[str, Any]) -> dict[str, Any]:
    from .macros import investigate_change
    events = _context_events_from(arguments)
    payload, path = investigate_change(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        context_events=events,
        suspected_cause=arguments.get("suspected_cause"),
        output=arguments.get("output_dir") or "gnomon-output",
        input_provenance=arguments.get("input_provenance"),
        regrid=arguments.get("regrid"),
    )
    return {**payload, "artifact_path": str(path)}


def _run_route(arguments: dict[str, Any]) -> dict[str, Any]:
    from .pipeline import load_stage
    from .router import route
    from .tracking import TrackingStore
    loaded = load_stage(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
    )
    project = arguments.get("project")
    store = TrackingStore() if project else None
    decisions = [
        route(arguments.get("task") or "forecast",
              [item.value for item in items], loaded.frequency,
              horizon=int(arguments.get("horizon") or 1),
              series=name, project=project, store=store)
        for name, items in sorted(loaded.groups.items())
    ]
    return {"schema_version": "0.1", "decisions": decisions}


def _run_detect_anomalies(arguments: dict[str, Any]) -> dict[str, Any]:
    from .macros import detect_anomalies
    payload, path = detect_anomalies(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        threshold=(float(arguments["threshold"])
                   if arguments.get("threshold") is not None else None),
        labels=arguments.get("labels"),
        output=arguments.get("output_dir") or "gnomon-output",
        input_provenance=arguments.get("input_provenance"),
        regrid=arguments.get("regrid"),
    )
    return {**payload, "artifact_path": str(path)}


def _run_decide(arguments: dict[str, Any]) -> dict[str, Any]:
    from .macros import decide
    payload, path = decide(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        horizon=int(arguments["horizon"]),
        threshold=float(arguments["threshold"]),
        actions=list(arguments["actions"]),
        utilities=arguments.get("utilities"),
        max_acceptable_risk=(
            float(arguments["max_acceptable_risk"])
            if arguments.get("max_acceptable_risk") is not None else None
        ),
        series_column=arguments.get("series_column"),
        series_name=arguments.get("series_name"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        project=arguments.get("project"),
        output=arguments.get("output_dir") or "gnomon-output",
        input_provenance=arguments.get("input_provenance"),
        regrid=arguments.get("regrid"),
        questions=arguments.get("questions"),
    )
    return {**payload, "artifact_path": str(path)}


def _run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore

    section = str(arguments.get("section") or "all")
    if section == "open_forecasts":
        # Preserve the established open-forecast projection.
        return _run_open_forecasts(arguments)
    if section == "performance":
        # Preserve the established performance projection and project
        # requirement.
        if not arguments.get("project"):
            from .contracts import GnomonError
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "section='performance' needs a project: realised "
                "performance is recorded per tracking project.",
            )
        return _run_model_performance(arguments)
    if section == "effects":
        if not arguments.get("project"):
            from .contracts import GnomonError
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "section='effects' needs a project: effect memory is scoped "
                "to a tracking project.",
            )
        effects = TrackingStore().event_effects(
            str(arguments["project"]),
            event_type=arguments.get("event_type"),
            series=arguments.get("series"),
            include_unresolved=not bool(arguments.get("resolved_only")),
        )
        return {"schema_version": "0.1", "project": arguments["project"],
                "effects": effects}
    if section == "effect_prior":
        from .contracts import GnomonError
        from .effect_registry import prior_from_dict
        from .effect_resolution import resolve_effect_evidence

        required = ("project", "event_type", "series", "as_of")
        missing = [name for name in required if not arguments.get(name)]
        if missing:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "section='effect_prior' needs project, event_type, series, and as_of.",
                {"missing": missing},
            )
        try:
            priors = [prior_from_dict(raw)
                      for raw in (arguments.get("external_priors") or [])]
            resolved = resolve_effect_evidence(
                TrackingStore(), project=str(arguments["project"]),
                event_type=str(arguments["event_type"]),
                series=str(arguments["series"]), as_of=str(arguments["as_of"]),
                external_priors=priors,
                target=str(arguments.get("target") or "*"),
                domain=str(arguments.get("domain") or "*"),
                population=str(arguments.get("population") or "*"),
                unit=str(arguments.get("unit") or "*"),
                human_assumption=arguments.get("human_assumption"),
            )
        except (TypeError, ValueError) as exc:
            raise GnomonError("INVALID_ARGUMENTS", str(exc)) from exc
        return {"schema_version": "0.1", "project": arguments["project"],
                "event_type": arguments["event_type"],
                "series": arguments["series"], "as_of": arguments["as_of"],
                "resolution": resolved}
    status = TrackingStore().status(arguments.get("project"))
    if section == "decisions":
        return {
            "schema_version": status["schema_version"],
            "project": status["project"],
            "unresolved_decisions": status["unresolved_decisions"],
            "decision_summary": status["decision_summary"],
        }
    return status


def _run_resolve_outcome(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    artifact = TrackingStore().resolve_decision_outcome(
        str(arguments["decision_id"]),
        realised_scenario=arguments.get("realised_scenario"),
        realised_utilities=arguments.get("realised_utilities"),
        constraint_violations=arguments.get("constraint_violations"),
        note=arguments.get("note"),
    )
    return {"status": "ok", "decision": artifact.to_dict()}


def _run_monitor(arguments: dict[str, Any]) -> dict[str, Any]:
    from .macros import monitor
    payload, path = monitor(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        horizon=int(arguments["horizon"]),
        threshold=float(arguments["threshold"]),
        alert_cost=float(arguments["alert_cost"]) if arguments.get("alert_cost") is not None else None,
        miss_cost=float(arguments["miss_cost"]) if arguments.get("miss_cost") is not None else None,
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        project=arguments.get("project"),
        output=arguments.get("output_dir") or "gnomon-output",
        input_provenance=arguments.get("input_provenance"),
        regrid=arguments.get("regrid"),
        questions=arguments.get("questions"),
    )
    return {**payload, "artifact_path": str(path)}


def _run_get_artifact(arguments: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path
    from .artifacts import read_artifact
    from .versioning import RUNTIME_VERSION
    directory = Path(arguments["artifact_path"])
    artifact = read_artifact(directory)
    rows = artifact.get("results")
    selection: dict[str, Any] | None = None
    if isinstance(rows, list) and any(arguments.get(key) is not None for key in
                                      ("series", "fields", "where", "order_by", "limit")):
        selected = list(rows)
        names = arguments.get("series")
        if isinstance(names, str):
            names = [names]
        if names:
            wanted = {str(name) for name in names}
            selected = [row for row in selected if str(row.get("series")) in wanted]
        where = arguments.get("where") or {}
        if where:
            selected = [row for row in selected if all(row.get(key) == value
                                                        for key, value in where.items())]
        order_by = arguments.get("order_by")
        if order_by == "notability":
            from .support import forecast_notability
            for row in selected:
                row.setdefault("notability", forecast_notability(row))
            selected.sort(key=lambda row: (-float(row.get("notability", 0.0)),
                                           str(row.get("series", ""))))
        elif order_by == "series":
            selected.sort(key=lambda row: str(row.get("series", "")))
        total = len(selected)
        if arguments.get("limit") is not None:
            selected = selected[:int(arguments["limit"])]
        fields = arguments.get("fields")
        if fields:
            keep = {str(field) for field in fields} | {"series"}
            selected = [{key: value for key, value in row.items() if key in keep}
                        for row in selected]
        artifact = {**artifact, "results": selected}
        selection = {"matched": total, "returned": len(selected),
                     "order_by": order_by, "fields": fields}
    payload: dict[str, Any] = {
        "schema_version": "0.1",
        "artifact": artifact,
        **({"selection": selection} if selection else {}),
    }
    stored = artifact.get("runtime_version")
    if stored != RUNTIME_VERSION:
        # The agent is told to quote artifacts verbatim, so an artifact
        # computed by another build must say so where the quoting happens.
        payload["runtime_note"] = (
            f"This artifact was produced by runtime "
            f"{stored or 'pre-0.5.0 (unstamped)'}; the running build is "
            f"{RUNTIME_VERSION}. Ids cover the runtime version, so "
            f"re-running the task will produce a fresh artifact under a "
            f"new id rather than updating this one."
        )
    lineage_path = directory / "lineage.json"
    if arguments.get("include_lineage") and lineage_path.is_file():
        import json as _json
        payload["lineage"] = _json.loads(lineage_path.read_text(encoding="utf-8"))
    return payload


def _run_unified(arguments: dict[str, Any]) -> dict[str, Any]:
    """Experimental single execution verb; complexity stays measurable."""
    from .contracts import GnomonError

    question = arguments.get("question") or {}
    # Models commonly send the discriminant directly (``"forecast"``)
    # despite the advertised object schema. Accept that unambiguous natural
    # form instead of throwing AttributeError inside the tool server.
    if isinstance(question, str):
        kind, question_fields = question, {}
    elif isinstance(question, dict):
        kind = question.get("kind")
        question_fields = {key: value for key, value in question.items()
                           if key != "kind"}
    else:
        raise GnomonError("INVALID_ARGUMENTS",
                          "question must be an object or a kind string.",
                          {"allowed": ["describe", "forecast", "investigate",
                                       "detect", "decide", "monitor"]})
    merged = {**arguments, **question_fields}
    merged.pop("question", None)
    if kind == "robust_decision":
        from datetime import datetime, timezone
        from .decision_model import robust_scenario_decision
        from .tracking import TrackingStore

        required = ("decision_id", "project", "forecast_id", "actions",
                    "utilities", "scenario_ids")
        missing = [name for name in required if not merged.get(name)]
        if missing:
            raise GnomonError(
                "INVALID_ARGUMENTS",
                "robust_decision needs a complete stated utility matrix.",
                {"missing": missing},
            )
        try:
            artifact = robust_scenario_decision(
                decision_id=str(merged["decision_id"]),
                project=str(merged["project"]),
                forecast_id=str(merged["forecast_id"]),
                actions=list(merged["actions"]),
                utilities=dict(merged["utilities"]),
                scenario_ids=list(merged["scenario_ids"]),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        except (TypeError, ValueError) as exc:
            raise GnomonError("INVALID_ARGUMENTS", str(exc)) from exc
        TrackingStore().save_decision_artifact(artifact)
        return {"schema_version": "0.1", "decision": artifact.to_dict()}
    runners = {
        "describe": _run_describe,
        "forecast": _run_forecast,
        "investigate": _run_investigate_change,
        "detect": _run_detect_anomalies,
        "decide": _run_decide,
        "monitor": _run_monitor,
    }
    runner = runners.get(str(kind))
    if runner is None:
        raise GnomonError("INVALID_ARGUMENTS", "question.kind is required.",
                          {"allowed": sorted(runners)})
    if kind == "forecast" and merged.get("horizon") is None:
        merged["horizon"] = _default_forecast_horizon(merged)
    return runner(merged)


def _run_track(arguments: dict[str, Any]) -> dict[str, Any]:
    from .contracts import GnomonError

    action = arguments.get("action")
    if action == "status":
        return _run_status(arguments)
    if action == "submit_actuals":
        return _run_submit_actuals(arguments)
    if action == "resolve_outcome":
        return _run_resolve_outcome(arguments)
    if action == "record_adapter_shadow":
        from .tracking import TrackingStore
        return TrackingStore().record_adapter_shadow_outcome(
            project=str(arguments["project"]),
            outcome_id=str(arguments["outcome_id"]),
            candidate=str(arguments["candidate"]),
            revision=arguments.get("revision"),
            baseline=str(arguments["baseline"]),
            candidate_error=float(arguments["candidate_error"]),
            baseline_error=float(arguments["baseline_error"]),
            known_at=str(arguments["known_at"]),
        )
    if action == "assess_adapter_shadow":
        from .tracking import TrackingStore
        return TrackingStore().assess_adapter_shadow(
            project=str(arguments["project"]),
            candidate=str(arguments["candidate"]),
            revision=arguments.get("revision"),
            baseline=str(arguments["baseline"]),
            as_of=arguments.get("as_of"),
            min_outcomes=int(arguments.get("min_outcomes", 30)),
            min_improvement=float(arguments.get("min_improvement", .05)),
            min_win_rate=float(arguments.get("min_win_rate", .60)),
        )
    raise GnomonError("INVALID_ARGUMENTS", "action is required.",
                      {"allowed": ["status", "submit_actuals", "resolve_outcome",
                                   "record_adapter_shadow",
                                   "assess_adapter_shadow"]})


def _run_explain_run(arguments: dict[str, Any]) -> dict[str, Any]:
    """Compact explanation of a stored run: claims, support, warnings.
    Statements come verbatim from the verified lineage — nothing is composed."""
    import json as _json
    from pathlib import Path
    from .artifacts import read_artifact
    directory = Path(arguments["artifact_path"])
    artifact = read_artifact(directory)
    explanation: dict[str, Any] = {
        "schema_version": "0.1",
        "artifact_id": (
            artifact.get("investigation_id") or artifact.get("decision_id")
            or artifact.get("monitor_id") or artifact.get("forecast_id")
        ),
        "created_at": artifact.get("created_at"),
        "support_assessments": {},
        "warnings": {},
        "claims": [],
    }
    for result in artifact.get("results", []):
        name = result.get("series", "__default__")
        if result.get("support_assessment") is not None:
            explanation["support_assessments"][name] = result["support_assessment"]
        if result.get("warnings"):
            explanation["warnings"][name] = result["warnings"]
    if artifact.get("support_assessment") is not None:
        explanation["support_assessments"]["__task__"] = artifact["support_assessment"]
    for trigger in artifact.get("triggers", []):
        explanation["support_assessments"][trigger.get("series", "__default__")] = (
            trigger.get("support_assessment")
        )
    lineage_path = directory / "lineage.json"
    if lineage_path.is_file():
        lineage = _json.loads(lineage_path.read_text(encoding="utf-8"))
        explanation["claims"] = [
            {"claim_id": claim["claim_id"], "claim_class": claim["claim_class"],
             "statement": claim["statement"], "evidence_ids": claim["evidence_ids"]}
            for claim in lineage.get("claims", [])
        ]
    summary = directory / "summary.md"
    if summary.is_file():
        explanation["summary_md"] = summary.read_text(encoding="utf-8")
    return explanation


def _run_install_tsfm(arguments: dict[str, Any]) -> dict[str, Any]:
    from .contracts import GnomonError
    from .tsfm import TSFMUnavailable, available_tsfms
    from .tsfm_sandbox import TSFM_PIP_SPECS, install_status, start_install

    name = str(arguments["name"])
    if name not in TSFM_PIP_SPECS:
        raise GnomonError(
            "UNKNOWN_TSFM",
            f"Unknown TSFM: {name!r}. Installable names are listed in "
            f"details.available.",
            {"available": sorted(TSFM_PIP_SPECS),
             "eligible_adapters": available_tsfms()},
        )
    try:
        status = (install_status(name) if arguments.get("status_only")
                  else start_install(name))
    except TSFMUnavailable as exc:
        raise GnomonError("SANDBOX_UNAVAILABLE", str(exc), {"tsfm": name})
    notes = {
        "installing": (
            "Installation runs as a detached process and can take minutes "
            "on first install (torch dominates). Poll with "
            "status_only=true; state=ready means the sandbox is usable."
        ),
        "ready": (
            "Sandbox ready. Pass the name in gnomon_forecast's "
            "`candidates` to enter it in the evaluated competition — "
            "TSFMs compete against the baselines on identical folds, "
            "never win by default."
        ),
        "failed": (
            "The last install attempt died; log_tail holds the evidence. "
            "Calling again without status_only retries from scratch."
        ),
        "absent": (
            "No sandbox and no install running. Call without status_only "
            "to start one."
        ),
    }
    return {"schema_version": "0.1", "tsfm": name,
            "pip_specs": TSFM_PIP_SPECS[name], **status,
            "note": notes[status["state"]]}


def _registry_tools() -> list[dict[str, Any]]:
    """Agent tools generated from the macro registry — one source of truth
    for schemas across CLI, Python API, and MCP."""
    from .registry import MACROS
    runners = {
        "gnomon_investigate_change": _run_investigate_change,
        "gnomon_detect_anomalies": _run_detect_anomalies,
        "gnomon_decide": _run_decide,
        "gnomon_monitor": _run_monitor,
    }
    tools = []
    for spec in MACROS.values():
        if spec.tool_name not in runners:
            continue  # gnomon_forecast keeps its frozen v0.2 definition above
        tools.append({
            "name": spec.tool_name,
            "description": spec.summary,
            "inputSchema": spec.input_schema,
            "runner": runners[spec.tool_name],
        })
    return tools


TOOLS.extend(_registry_tools())
TOOLS.extend([
    {
        "name": "gnomon_run",
        "description": (
            "Experimental unified temporal execution verb. Set question.kind "
            "to describe, forecast, investigate, detect, decide, monitor, or "
            "robust_decision. Robust decisions use caller-supplied utilities "
            "without inventing scenario probabilities."
        ),
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES, **_REPLAY_PROPERTIES,
            **_CONTEXT_EVENTS_PROPERTY,
            "context_events_file": {"type": "string", "description": (
                "Validated context-events JSON produced by the host context compiler.")},
            "future_events": {"type": "boolean", "description": (
                "Admit future constraint/override events only after verbatim-source validation.")},
            "structural_events": {"type": "boolean", "description": (
                "Enable the separately gated structural-event lane.")},
            "question": {"type": "object", "properties": {
                "kind": {"type": "string", "enum": [
                    "describe", "forecast", "investigate", "detect", "decide", "monitor",
                    "robust_decision"]},
                "suspected_cause": {"type": "string"},
            }, "required": ["kind"]},
            "horizon": {"type": "integer", "minimum": 1},
            "threshold": {"type": "number"},
            "actions": {"type": "array", "items": {"oneOf": [
                {"type": "string"},
                {"type": "object", "properties": {
                    "name": {"type": "string"},
                    "feasible": {"type": "boolean"},
                    "constraint_results": {"type": "object"},
                }, "required": ["name"]},
            ]}},
            "utilities": {"type": "object"},
            "decision_id": {"type": "string"},
            "forecast_id": {"type": "string"},
            "scenario_ids": {"type": "array", "items": {"type": "string"}},
            "alert_cost": {"type": "number"},
            "miss_cost": {"type": "number"},
            "output_dir": {"type": "string"},
            "project": {"type": "string"},
            "minimum_support": {"type": "string", "enum": [
                "supported", "conditionally_supported", "best_effort"]},
            "format": {"type": "string", "enum": ["brief", "full"]},
        }, "required": ["question"]},
        "runner": _run_unified,
    },
    {
        "name": "gnomon_track",
        "description": (
            "Experimental tracking verb. action selects status, "
            "submit_actuals, or resolve_outcome."
        ),
        "inputSchema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": [
                "status", "submit_actuals", "resolve_outcome",
                "record_adapter_shadow", "assess_adapter_shadow"]},
            "project": {"type": "string"},
            "section": {"type": "string", "enum": [
                "open_forecasts", "performance", "decisions", "all"]},
            "actuals": {"type": "array", "items": {"type": "object"}},
            "actuals_file": {"type": "string"},
            "actuals_time": {"type": "string"},
            "actuals_target": {"type": "string"},
            "actuals_series": {"type": "string"},
            "decision_id": {"type": "string"},
            "realised_scenario": {"type": "string"},
            "realised_utilities": {"type": "object"},
            "constraint_violations": {"type": "array", "items": {"type": "string"}},
            "note": {"type": "string"},
            "outcome_id": {"type": "string"},
            "candidate": {"type": "string"},
            "revision": {"type": "string"},
            "baseline": {"type": "string"},
            "candidate_error": {"type": "number", "minimum": 0},
            "baseline_error": {"type": "number", "minimum": 0},
            "known_at": {"type": "string"},
            "as_of": {"type": "string"},
            "min_outcomes": {"type": "integer", "minimum": 1},
            "min_improvement": {"type": "number"},
            "min_win_rate": {"type": "number", "minimum": 0, "maximum": 1},
        }, "required": ["action"]},
        "runner": _run_track,
    },
    {
        "name": "gnomon_get_artifact",
        "description": (
            "Read a stored artifact directory: full artifact.json and, "
            "optionally, the typed lineage. All numbers live here; quote them "
            "verbatim."
        ),
        "inputSchema": {"type": "object", "properties": {
            "artifact_path": {"type": "string", "description": "Artifact directory returned by a macro."},
            "include_lineage": {"type": "boolean", "description": "Include lineage.json (artifacts/evidence/claims)."},
            "series": {"type": ["string", "array"], "description": "Series name or names to return from artifact results."},
            "fields": {"type": "array", "items": {"type": "string"}, "description": "Result fields to return; series is always retained."},
            "where": {"type": "object", "description": "Exact-match filters over result fields."},
            "order_by": {"type": "string", "enum": ["notability", "series"], "description": "Deterministic result ordering."},
            "limit": {"type": "integer", "minimum": 1, "description": "Maximum selected results to return."},
        }, "required": ["artifact_path"]},
        "runner": _run_get_artifact,
    },
    {
        "name": "gnomon_status",
        "description": (
            "The one tracking read: open forecasts, due horizons, "
            "unresolved decisions, and realised-performance summaries. "
            "Narrow with `section` (open_forecasts / performance / "
            "decisions / effects / effect_prior) to get exactly the tracking evidence "
            "returned. Descriptive evidence an agent can cite — never "
            "causal; do not treat observational rankings as causal "
            "evidence."
        ),
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string", "description": (
                "Optional project filter; required for "
                "section='performance'."
            )},
            "section": {"type": "string",
                        "enum": ["open_forecasts", "performance",
                                 "decisions", "effects", "effect_prior", "all"],
                        "description": (
                            "Slice to return (default all). open_forecasts: "
                            "unscored forecasts with due horizons; "
                            "performance: realised per-model performance "
                            "for a project; decisions: unresolved decisions "
                            "and the resolution summary; effects: frozen "
                            "context scenarios and realised effect estimates; "
                            "effect_prior: resolve the governed evidence ladder."
                        )},
            "model": {"type": "string", "description": (
                "With section='performance': narrow to one model's "
                "realised runs."
            )},
            "event_type": {"type": "string", "description": (
                "With section='effects': restrict to one event type."
            )},
            "series": {"type": "string", "description": (
                "With section='effects': restrict to one series."
            )},
            "resolved_only": {"type": "boolean", "description": (
                "With section='effects': omit scenarios awaiting outcomes."
            )},
            "as_of": {"type": "string", "description": (
                "With section='effect_prior': timezone-aware knowledge cutoff."
            )},
            "target": {"type": "string"},
            "domain": {"type": "string"},
            "population": {"type": "string"},
            "unit": {"type": "string"},
            "external_priors": {"type": "array", "items": {"type": "object"},
                                "description": (
                                    "Versioned external effect priors; each needs known_at."
                                )},
            "human_assumption": {"type": "object", "description": (
                "Explicit sensitivity assumption with location and known_at; "
                "never treated as learned probabilistic evidence."
            )},
        }, "required": []},
        "runner": _run_status,
    },
    {
        "name": "gnomon_resolve_outcome",
        "description": (
            "Resolve DecisionArtifacts produced by `gnomon_decide` with what "
            "actually happened: "
            "realised scenario and/or per-action realised utilities. Returns "
            "realised utility, regret vs the best feasible action in "
            "hindsight, ex-ante optimality, and risk calibration — bare "
            "'correct' is retired."
        ),
        "inputSchema": {"type": "object", "properties": {
            "decision_id": {"type": "string"},
            "realised_scenario": {"type": "string", "description": "e.g. exceed / no_exceed."},
            "realised_utilities": {"type": "object", "description": "Optional per-action realised payoff."},
            "constraint_violations": {"type": "array", "items": {"type": "string"}},
            "note": {"type": "string"},
        }, "required": ["decision_id"]},
        "runner": _run_resolve_outcome,
    },
    {
        "name": "gnomon_route",
        "description": (
            "Which method for this task on this data? A disclosed, advisory "
            "routing decision: verified capability filter, then a realised-"
            "performance prior from the tracking store when enough scored "
            "history exists (never claimed cold), with the series fingerprint "
            "and every exclusion reason in the output. Feed `candidates` (or "
            "`recommendation`) to `gnomon_forecast`'s `candidates` parameter to "
            "act on the answer. Evaluated runs still backtest whatever pool "
            "they are given against the mandatory baselines, so routing "
            "narrows the contest but never decides it."
        ),
        "inputSchema": {"type": "object", "properties": {
            "input": {"type": "string", "description": "Path to a CSV/Parquet file or store:<dataset>. Callers without a filesystem pass `observations` inline instead."},
            **_OBSERVATIONS_PROPERTY,
            "time_column": {"type": "string"},
            "target_column": {"type": "string"},
            "series_column": {"type": "string"},
            "frequency": {"type": "string"},
            "task": {"type": "string", "enum": ["forecast", "detect_anomalies"],
                     "description": "Task to route (default forecast)."},
            "horizon": {"type": "integer", "description": "Forecast horizon (default 1)."},
            "project": {"type": "string", "description": (
                "Tracking project: consults the realised-performance prior and "
                "records the routing decision for replay."
            )},
        }, "required": []},
        "runner": _run_route,
    },
    {
        "name": "gnomon_explain_run",
        "description": (
            "Compact explanation of a stored run: verified claim statements, "
            "per-series support assessments, and warnings. Statements come "
            "from the verified lineage; never paraphrase abstentions away."
        ),
        "inputSchema": {"type": "object", "properties": {
            "artifact_path": {"type": "string", "description": "Artifact directory returned by a macro."},
        }, "required": ["artifact_path"]},
        "runner": _run_explain_run,
    },
    {
        "name": "gnomon_install_tsfm",
        "description": (
            "Install a time-series foundation model into its isolated "
            "sandbox venv, without blocking: the install runs as a "
            "detached process and each call reports the current state "
            "(absent / installing / ready / failed). Eligible names are "
            "in gnomon_capabilities under models.tsfm_available; when "
            "state is ready, pass the name in gnomon_forecast's "
            "`candidates`. Packages come from the pinned per-model spec "
            "via uv — expect minutes on first install. TSFMs remain "
            "candidates: they compete against the baselines on identical "
            "folds and never win by default."
        ),
        "inputSchema": {"type": "object", "properties": {
            "name": {"type": "string", "description": (
                "TSFM adapter name (e.g. chronos_bolt_mini); the "
                "installable set is in gnomon_capabilities under "
                "models.tsfm_available."
            )},
            "status_only": {"type": "boolean", "description": (
                "Report the sandbox state without starting an install."
            )},
        }, "required": ["name"]},
        "runner": _run_install_tsfm,
    },
])


#: Task profiles: named subsets of the canonical surface for hosts that know
#: what kind of session they are running. Selected by `gnomon mcp serve
#: --profile` or the GNOMON_MCP_PROFILE env var.
_CORE_PROFILE = frozenset({
    "gnomon_capabilities", "gnomon_inspect", "gnomon_forecast",
    "gnomon_investigate_change", "gnomon_detect_anomalies",
    "gnomon_explain_run",
})
PROFILES: dict[str, frozenset[str]] = {
    "core": _CORE_PROFILE,
    "describe": _CORE_PROFILE | {"gnomon_describe"},
    "evidence": frozenset({"gnomon_describe", "gnomon_forecast"}),
    "mega": frozenset({"gnomon_inspect", "gnomon_run", "gnomon_track"}),
    "decision": _CORE_PROFILE | {
        "gnomon_decide", "gnomon_monitor", "gnomon_route",
        "gnomon_status", "gnomon_resolve_outcome",
    },
    "data": _CORE_PROFILE | {
        "gnomon_ingest", "gnomon_list_datasets", "gnomon_submit_actuals",
    },
}
_SURFACE_EXPERIMENT_TOOLS = frozenset({
    "gnomon_describe", "gnomon_run", "gnomon_track",
})


def active_profile() -> str:
    import os
    name = os.environ.get("GNOMON_MCP_PROFILE", "evidence")
    if name != "full" and name not in PROFILES:
        raise ValueError(
            f"Unknown GNOMON_MCP_PROFILE {name!r}; expected one of "
            f"{sorted(PROFILES)} or 'full'."
        )
    return name


def visible_tools() -> list[dict[str, Any]]:
    """The canonical tool surface filtered by the active profile."""
    tools = TOOLS
    profile = active_profile()
    if profile == "full":
        return [tool for tool in tools
                if tool["name"] not in _SURFACE_EXPERIMENT_TOOLS]
    allowed = PROFILES[profile]
    return [tool for tool in tools if tool["name"] in allowed]


_SESSION_DATA_REFS: dict[str, dict[str, Any]] = {}
_MAX_SESSION_DATA_REFS = 128
_DATA_BINDING_KEYS = frozenset({
    "input", "input_provenance", "time_column", "target_column",
    "series_column", "frequency", "regrid", "as_of", "store_path", "repair",
})


def _resolve_data_ref(arguments: dict[str, Any]) -> dict[str, Any]:
    token = arguments.get("data_ref")
    if not token:
        return arguments
    from .contracts import GnomonError
    bound = _SESSION_DATA_REFS.get(str(token))
    if bound is None:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "data_ref is unknown or expired in this MCP session.",
            {"data_ref": str(token)},
            repair_options=[{
                "action": "resupply_data",
                "description": "Send input or observations again to receive a fresh data_ref.",
            }],
        )
    conflicts = sorted(
        key for key in _DATA_BINDING_KEYS
        if key in arguments and key in bound and arguments[key] != bound[key]
    )
    if conflicts:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "data_ref already binds the data and schema; do not override "
            + ", ".join(conflicts) + ".",
            {"data_ref": str(token), "conflicts": conflicts},
        )
    return {**bound, **{key: value for key, value in arguments.items()
                       if key != "data_ref"}, "_data_ref": str(token)}


def _register_data_ref(arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
    import secrets
    token = "data_" + secrets.token_urlsafe(18)
    bound = {key: arguments[key] for key in _DATA_BINDING_KEYS
             if key in arguments and arguments[key] is not None}
    _SESSION_DATA_REFS[token] = bound
    while len(_SESSION_DATA_REFS) > _MAX_SESSION_DATA_REFS:
        # Insertion-ordered dict: discard the oldest session binding. The
        # caller gets the same typed expired-reference recovery as a restart.
        _SESSION_DATA_REFS.pop(next(iter(_SESSION_DATA_REFS)))
    return {**arguments, "_data_ref": token}, token


def _materialise_observations(arguments: dict[str, Any]) -> dict[str, Any]:
    """Turn the inline ``observations`` array into a temp-file ``input``.

    The rows become a CSV in a fresh temp directory and the call
    proceeds exactly as a file-based one — same loaders, same repair
    ladder, same fingerprinting — so the inline channel can never
    develop separate semantics from the file channel.
    """
    arguments = _resolve_data_ref(arguments)
    rows = arguments.get("observations")
    if rows is None:
        return arguments
    from .contracts import GnomonError

    if arguments.get("input"):
        raise GnomonError(
            "INVALID_ARGUMENTS", "Provide input or observations, not both.",
        )
    if (not isinstance(rows, list) or not rows
            or not all(isinstance(row, dict) and row for row in rows)):
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "observations must be a non-empty array of row objects keyed "
            "by column name.",
        )
    if len(rows) > 500:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "observations accepts at most 500 inline rows; use a file, "
            "store:<dataset>, or a data_ref returned by an earlier call.",
            {"observations": len(rows), "maximum": 500},
        )
    import csv
    import tempfile
    from pathlib import Path

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    path = Path(tempfile.mkdtemp(prefix="gnomon-inline-")) / "observations.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    passed = {key: value for key, value in arguments.items()
              if key != "observations"}
    # The temp file erases the channel; this is the last place that knows
    # the rows were typed by the caller rather than read from their disk,
    # and the artifact's task block wants the fact (`provenance: inline`).
    return {**passed, "input": str(path), "input_provenance": "inline"}


def runner_for(name: str) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    for tool in visible_tools():
        if tool["name"] == name:
            runner = tool["runner"]
            takes_data = "observations" in (
                tool.get("inputSchema", {}).get("properties") or {})

            def wrapped(arguments: dict[str, Any], _runner=runner,
                        _takes_data=takes_data, _name=name) -> dict[str, Any]:
                if _takes_data and not arguments.get("input") \
                        and arguments.get("observations") is None \
                        and not arguments.get("data_ref"):
                    raise GnomonError(
                        "INVALID_ARGUMENTS",
                        "Supply the data: input (a file path or "
                        "store:<dataset>) or observations (inline rows).",
                    )
                # Which channels carried caller-typed *measurements*, noted
                # before materialisation erases the distinction. Inline data
                # is a first-class channel — validated, fingerprinted,
                # repaired exactly like a file — but a file at least existed
                # outside this conversation; rows the model typed did not,
                # and a reader weighing the numbers is owed that fact.
                # Context events are deliberately absent: they are claims,
                # not measurements — always caller-authored, whatever the
                # channel — and their trust story is the source field and
                # the admission gate, not the file/inline distinction.
                inline_channels = [
                    label for key, label in (
                        ("observations", "observations"),
                        ("covariates", "covariate vintages"),
                        ("actuals", "actuals"),
                    ) if isinstance(arguments.get(key), list)
                ]
                arguments, context_cache = _materialise_context(arguments)
                arguments = _materialise_observations(arguments)
                assumptions: list[str] = []
                if inline_channels:
                    assumptions.append(
                        f"{' and '.join(inline_channels)} were supplied "
                        f"inline by the caller; Gnomon validated their shape "
                        f"and fingerprinted their content, but cannot attest "
                        f"their origin."
                    )
                if _name in _SCHEMA_INFERENCE_TOOLS:
                    arguments, inferred = _resolve_schema_arguments(
                        arguments, _name)
                    assumptions.extend(inferred)
                data_ref = arguments.pop("_data_ref", None)
                if _takes_data and data_ref is None:
                    arguments, data_ref = _register_data_ref(arguments)
                    arguments.pop("_data_ref", None)
                if _name == "gnomon_forecast" \
                        and arguments.get("horizon") is None:
                    # One season ahead is the smallest horizon that can show
                    # a seasonal pattern, and it is derivable from the data.
                    horizon = _default_forecast_horizon(arguments)
                    arguments = {**arguments, "horizon": horizon}
                    assumptions.append(
                        f"horizon was not supplied; defaulted to {horizon}, "
                        f"one seasonal period of the inferred grid."
                    )
                try:
                    computed = _runner(arguments)
                except GnomonError as error:
                    if error.code == "IRREGULAR_TIME_GRID":
                        retry_arguments = {**arguments, "repair": "aggressive"}
                        if data_ref:
                            retry_arguments = {
                                key: value for key, value in retry_arguments.items()
                                if key not in _DATA_BINDING_KEYS
                            }
                            retry_arguments.update({
                                "data_ref": data_ref, "repair": "aggressive"})
                        error.repair_options = [{
                            "action": "retry_with_aggressive_repair",
                            "description": (
                                "Retry once with capped interpolation and "
                                "timestamp snapping; every repair is disclosed."),
                            "tool_call": {"name": _name,
                                          "arguments": retry_arguments},
                        }, *(error.repair_options
                             if error.repair_options is not None else
                             REPAIR_OPTIONS.get(error.code, []))]
                    raise
                payload = disclose_assumptions(computed, assumptions)
                if context_cache and isinstance(payload, dict):
                    payload = {
                        **payload,
                        "context_ref": context_cache["context_ref"],
                        "context_cache": context_cache,
                    }
                if data_ref and isinstance(payload, dict):
                    payload = {**payload, "data_ref": data_ref}
                    # Runners cannot know the token until the shared wrapper
                    # registers it. Resolve ready-to-issue follow-ups here.
                    for action in payload.get("suggested_next") or []:
                        call = action.get("tool_call") if isinstance(action, dict) else None
                        call_args = call.get("arguments") if isinstance(call, dict) else None
                        if isinstance(call_args, dict) and call_args.get("data_ref") == "<data_ref>":
                            call_args["data_ref"] = data_ref
                if _name == "gnomon_capabilities":
                    # The budget trimmer cuts long arrays, and in a
                    # capabilities payload every array is a capability
                    # list — cutting one would misreport the build. The
                    # runner's own brief default is its budget mechanism;
                    # format 'full' and sections are the caller's explicit
                    # ask for the verbatim payload.
                    return payload
                if _takes_data and isinstance(payload, dict):
                    payload = apply_temporal_grounding(payload)
                # Preserve the established bulk budget decision, then add the
                # small protected routing envelope. Otherwise the envelope
                # itself can push a previously in-budget forecast over the
                # trim threshold and unexpectedly remove rows.
                budget = (DESCRIBE_RESPONSE_BUDGET_BYTES
                          if _name == "gnomon_describe"
                          else RESPONSE_BUDGET_BYTES)
                payload = triage_wide_response(payload)
                payload = compact_support_details(payload)
                return apply_response_contract(
                    enforce_response_budget(payload, budget))

            return wrapped
    return None
