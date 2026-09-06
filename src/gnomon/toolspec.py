"""Legacy MCP dispatch and compatibility exports.

Schemas, input preparation, operations and response projections have independent
owners. Default execution uses GnomonSession, never this compatibility module.
"""

from __future__ import annotations

import json

from typing import Any, Callable

from .contracts import GnomonError, REPAIR_OPTIONS
from .product_contract import resolve_mcp_profile
from .response_budget import (
    CAPABILITIES_RESPONSE_BUDGET_BYTES as CAPABILITIES_RESPONSE_BUDGET_BYTES,
    DESCRIBE_RESPONSE_BUDGET_BYTES as DESCRIBE_RESPONSE_BUDGET_BYTES,
    RESPONSE_BUDGET_BYTES as RESPONSE_BUDGET_BYTES,
    enforce_response_budget as enforce_response_budget,
)
from .tool_schema import (
    CONTEXT_EVENTS_PROPERTY as _CONTEXT_EVENTS_PROPERTY,  # noqa: F401 - compatibility export
    COVARIATE_MAPPING_PROPERTY as _COVARIATE_MAPPING_PROPERTY,  # noqa: F401
    COVARIATES_PROPERTY as _COVARIATES_PROPERTY,  # noqa: F401
    INPUT_PROPERTIES as _INPUT_PROPERTIES,  # noqa: F401
    OBSERVATIONS_PROPERTY as _OBSERVATIONS_PROPERTY,  # noqa: F401
    REPLAY_PROPERTIES as _REPLAY_PROPERTIES,  # noqa: F401
    TEMPORAL_QUESTIONS_PROPERTY as _TEMPORAL_QUESTIONS_PROPERTY,  # noqa: F401
)
from .tool_profiles import PROFILES as PROFILES
from .tool_catalog import TOOL_SCHEMAS
from . import tool_operations
from .tool_response import (
    FORECAST_PREVIEW_ROWS as FORECAST_PREVIEW_ROWS,
    FORECAST_PREVIEW_SMALL_HORIZON as FORECAST_PREVIEW_SMALL_HORIZON,
    _attach_multiseries_triage as _attach_multiseries_triage,
    _attach_tsfm_on_ramp as _attach_tsfm_on_ramp,
    _bounded_forecast_preview as _bounded_forecast_preview,
    _brief_capabilities as _brief_capabilities,
    _compact_sensitivity_projection as _compact_sensitivity_projection,
    _execution_identity as _execution_identity,
    _forecast_temporal_boundary as _forecast_temporal_boundary,
    _model_assisted_summary as _model_assisted_summary,
    _result_authority_projection as _result_authority_projection,
    _triage_artifact_identity as _triage_artifact_identity,
    apply_response_contract as apply_response_contract,
    apply_temporal_grounding as apply_temporal_grounding,
    brief_summary as brief_summary,
    compact_publication_for_wire as compact_publication_for_wire,
    compact_support_details as compact_support_details,
    disclose_assumptions as disclose_assumptions,
    forecast_summary as forecast_summary,
    triage_wide_response as triage_wide_response,
)
from .tool_input import (
    _default_forecast_horizon as _default_forecast_horizon,
    _parse_as_of as _parse_as_of,
    _resolve_schema_arguments as _resolve_schema_arguments,
)
from .tool_context import (
    _align_single_target_context_scope as _align_single_target_context_scope,
    _context_events_from as _context_events_from,
    _covariates_from as _covariates_from,
    _materialise_context as _materialise_context,
    _materialized_or_public_events as _materialized_or_public_events,
    _run_preflight_context as _run_preflight_context,
)
from .tool_publication import (
    _attach_publication as _attach_publication,
)
from .tool_operations import (
    _actual_tuples as _actual_tuples,
    _attach_temporal_answers as _attach_temporal_answers,
    _json_temporal_values as _json_temporal_values,
    _run_capabilities as _run_capabilities,
    _run_decide as _run_decide,
    _run_describe as _run_describe,
    _run_detect_anomalies as _run_detect_anomalies,
    _run_explain_run as _run_explain_run,
    _run_forecast as _run_forecast,
    _run_forecast_multi as _run_forecast_multi,
    _run_get_artifact as _run_get_artifact,
    _run_ingest as _run_ingest,
    _run_inspect as _run_inspect,
    _run_inspect_multi as _run_inspect_multi,
    _run_install_tsfm as _run_install_tsfm,
    _run_investigate_change as _run_investigate_change,
    _run_list_datasets as _run_list_datasets,
    _run_model_performance as _run_model_performance,
    _run_monitor as _run_monitor,
    _run_open_forecasts as _run_open_forecasts,
    _run_resolve_outcome as _run_resolve_outcome,
    _run_route as _run_route,
    _run_select_scenario as _run_select_scenario,
    _run_status as _run_status,
    _run_submit_actuals as _run_submit_actuals,
    _run_validate_covariates as _run_validate_covariates,
)

TOOLS = [{**spec, "runner": getattr(tool_operations, spec["runner"])}
         for spec in TOOL_SCHEMAS]

_SCHEMA_INFERENCE_TOOLS: frozenset[str] = frozenset({
    "gnomon_inspect", "gnomon_describe", "gnomon_forecast",
    "gnomon_validate_covariates",
    "gnomon_preflight_context", "gnomon_route",
    "gnomon_investigate_change", "gnomon_detect_anomalies",
    "gnomon_decide", "gnomon_monitor",
})


def active_profile() -> str:
    return resolve_mcp_profile()


def visible_tools() -> list[dict[str, Any]]:
    """The canonical tool surface filtered by the active profile."""
    tools = TOOLS
    profile = active_profile()
    if profile == "execution":
        from .session import GnomonSession
        with GnomonSession.from_config() as session:
            return session.tools()
    if profile == "full":
        return list(tools)
    allowed = PROFILES[profile]
    return [tool for tool in tools if tool["name"] in allowed]


def profiles_for_tool(name: str) -> list[str]:
    """Named profiles that can expose ``name``, including virtual ``full``."""
    profiles = sorted(
        profile for profile, names in PROFILES.items() if name in names
    )
    known = any(tool["name"] == name for tool in TOOLS)
    if known:
        profiles.append("full")
    return profiles


def enforce_profile_tool_calls(value: Any) -> Any:
    """Never hand an agent a ready call that this server will refuse."""
    visible = {tool["name"] for tool in visible_tools()}

    def visit(item: Any) -> Any:
        if isinstance(item, list):
            return [visit(entry) for entry in item]
        if not isinstance(item, dict):
            return item
        result: dict[str, Any] = {}
        for key, nested in item.items():
            if key == "tool_call" and isinstance(nested, dict):
                name = str(nested.get("name") or "")
                if name and name not in visible:
                    result["tool_unavailable_in_profile"] = {
                        "tool": name,
                        "active_profile": active_profile(),
                        "profiles": profiles_for_tool(name),
                        "note": (
                            "The referenced data remains available at the "
                            "response's artifact path; this server profile "
                            "does not expose the suggested follow-up tool."
                        ),
                    }
                    continue
            result[key] = visit(nested)
        return result

    return visit(value)


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
    if active_profile() == "execution":
        # Stateful execution belongs to an explicitly owned GnomonSession.
        return None
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
                source_kind = None
                source = arguments.get("input")
                if isinstance(source, str) and source.startswith(
                        ("prom://", "prom+http://", "prom+https://")):
                    from .sources import materialize_agent_source
                    resolved, source_kind = materialize_agent_source(source)
                    arguments = {
                        **arguments,
                        "input": resolved,
                        "input_provenance": source_kind,
                        "time_column": arguments.get("time_column") or "timestamp",
                        "target_column": arguments.get("target_column") or "value",
                        "series_column": arguments.get("series_column") or "series",
                    }
                arguments, context_cache = _materialise_context(arguments)
                arguments = _materialise_observations(arguments)
                assumptions: list[str] = []
                if source_kind == "prometheus":
                    assumptions.append(
                        "Input was retrieved through the governed read-only "
                        "Prometheus connector. The host was allowlisted, the "
                        "response was bounded and fingerprinted, and sample "
                        "timestamps are treated as their availability times."
                    )
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
                                "Retry once with capped interpolation; bounded "
                                "timestamp jitter is already handled by safe "
                                "repair, and every repair is disclosed."),
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
                if arguments.get("format") != "full":
                    payload = compact_publication_for_wire(payload)
                payload = triage_wide_response(payload)
                payload = compact_support_details(payload)
                if isinstance(payload, dict) and "verb" not in payload:
                    payload = {**payload,
                               "verb": _name.removeprefix("gnomon_")}
                contracted = apply_response_contract(
                    enforce_response_budget(payload, budget))
                profiled = enforce_profile_tool_calls(contracted)
                # The reasoning/profile envelopes are intentionally added
                # after the bulk pass so they can never be discarded.  A
                # response that was in budget before those envelopes may no
                # longer be in budget afterward, though, so move diagnostic
                # sensitivity to the complete artifact when that is enough.
                # Do not run the array trimmer a second time: brief horizons
                # of at most FORECAST_PREVIEW_SMALL_HORIZON deliberately keep
                # every row so an agent can see support-tier transitions.
                if isinstance(profiled, dict) and not profiled.get("truncated"):
                    try:
                        final_size = len(json.dumps(profiled, default=str))
                    except (TypeError, ValueError):
                        final_size = 0
                    if final_size > budget:
                        compacted = compact_support_details(
                            profiled, force=True)
                        try:
                            compacted_size = len(json.dumps(
                                compacted, default=str))
                        except (TypeError, ValueError):
                            compacted_size = final_size
                        if compacted_size <= budget:
                            profiled = compacted
                return profiled

            return wrapped
    return None
