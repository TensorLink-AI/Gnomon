"""Canonical machine-facing tool specifications.

One definition of Gnomon's agent-facing tools — names, JSON Schemas, and
in-process runners over the runtime — consumed by the MCP server and any
future adapter. The Hermes plugin carries its own copy of the schemas
because it must remain standalone-installable; this module is the source
those copies are checked against.
"""

from __future__ import annotations

from typing import Any, Callable

from .context import load_events_file
from .contracts import ForecastArtifact
from .runtime import capabilities, forecast, inspect_dataset

#: The inline event channel, shared by every tool that accepts context
#: events. Complete enough that a model holding only this schema can
#: construct a valid event — that is the point of it existing.
_CONTEXT_EVENTS_PROPERTY: dict[str, Any] = {
    "context_events": {
        "type": "array",
        "description": (
            "Context events supplied inline — the same objects a "
            "context-events file carries, for callers without a "
            "filesystem. Concatenated with context_events_file when both "
            "are given; both channels validate against the same "
            "contract, loudly. Admission is separate and strict "
            "(constraint:/override:/structural: events need the "
            "future_events/structural_events flags and a verbatim "
            "source_span) — dry-run with gnomon_preflight_context first."
        ),
        "items": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "Unique id for this event."},
                "event_type": {"type": "string", "description": "Namespaced type: constraint:<label> (stated bound), override:<label> (stated level/state), structural:<label> (stated cessation, with attributes.effect), or a plain type for the fold-ablation gate."},
                "entity_scope": {"type": "array", "items": {"type": "string"}, "description": "Series names the event applies to; [\"*\"] for all."},
                "effective_start": {"type": "string", "description": "ISO-8601 with an explicit timezone offset."},
                "effective_end": {"type": "string", "description": "ISO-8601 with an explicit offset; must not precede effective_start."},
                "known_at": {"type": "string", "description": "When the information became knowable; ISO-8601 with an explicit offset."},
                "attributes": {"type": "object", "description": "Per-class payload: source_span (the sentence, quoted VERBATIM, that states the claim — the only admissible source of numbers), effect (structural menu: trend_ceases, level_matches_seasonal_high, level_matches_seasonal_low), or claim ({kind: min|max, value: number})."},
                "source": {"type": "object", "properties": {"type": {"type": "string"}, "reference": {"type": "string"}}, "description": "Where the information came from."},
                "created_by": {"type": "string", "description": "user | llm | pipeline."},
            },
            "required": ["event_id", "event_type", "entity_scope",
                         "effective_start", "effective_end", "known_at"],
        },
    },
}

#: The inline covariate channel, shared by every tool that accepts
#: point-in-time covariates. Mutually exclusive with covariates_file;
#: covariate_mapping stays required either way.
_COVARIATES_PROPERTY: dict[str, Any] = {
    "covariates": {
        "type": "array",
        "description": (
            "Point-in-time covariate vintages supplied inline, for "
            "callers without a filesystem: row objects with the "
            "covariate time column (default `timestamp`, when the value "
            "applies), the known-at column (default `known_at`, when it "
            "became knowable), and one key per covariate named in "
            "covariate_mapping. Same validation and admission gate as "
            "covariates_file; mutually exclusive with it."
        ),
        "items": {"type": "object"},
    },
}

#: The inline data channel, shared by every tool that reads observations.
#: The same reason context_events and covariates have one: an MCP client
#: holds no filesystem, and a path-only `input` makes every data-reading
#: tool unreachable for it.
_OBSERVATIONS_PROPERTY: dict[str, Any] = {
    "observations": {
        "type": "array",
        "description": (
            "The observations supplied inline, for callers without a "
            "filesystem: an array of row objects keyed by your column "
            "names (time_column, target_column, and optionally "
            "series_column), e.g. "
            '[{"timestamp": "2024-01-01T00:00:00+00:00", "value": 12.5}]. '
            "Mutually exclusive with `input`."
        ),
        "items": {"type": "object"},
    },
}

_INPUT_PROPERTIES: dict[str, Any] = {
    "input": {"type": "string", "description": "Path to a local CSV, TSV, JSON, JSONL, Parquet, or Excel file of time-series observations, or `store:<dataset>` to read a dataset from the bitemporal store (see gnomon_list_datasets). Callers without a filesystem pass `observations` inline instead."},
    **_OBSERVATIONS_PROPERTY,
    "time_column": {"type": "string", "description": (
        "Name of the timestamp column. Omit to infer: when exactly one "
        "column parses as timestamps it is chosen and disclosed as an "
        "assumption; ambiguity fails loudly, naming the candidates. "
        "Required for store:<dataset> inputs."
    )},
    "target_column": {"type": "string", "description": (
        "Name of the numeric column to forecast. Omit to infer: when "
        "exactly one non-time column parses as numbers it is chosen and "
        "disclosed as an assumption; ambiguity fails loudly, naming the "
        "candidates. Required for store:<dataset> inputs."
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
}

#: Replay controls, shared by every verb that reads data. `gnomon_forecast`
#: and `gnomon_inspect` were the two that lacked them, which made the
#: bitemporal store and `--as-of` replay CLI-only — invisible to the agents
#: the MCP server exists to serve.
_REPLAY_PROPERTIES: dict[str, Any] = {
    "as_of": {
        "type": "string",
        "description": (
            "Replay instant (ISO-8601). Only data whose known_time is at or "
            "before this is visible; the artifact's snapshot_access evidence "
            "proves what was served. Requires a `store:<dataset>` input to "
            "mean anything, since a plain file carries one vintage."
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

FORECAST_PREVIEW_ROWS = 12

#: Tools whose missing time_column/target_column are inferred from the file
#: when it leaves no choice — the same set of verbs the CLI infers for.
#: gnomon_ingest is deliberately absent: a write to the store under guessed
#: columns would persist the guess.
_SCHEMA_INFERENCE_TOOLS: frozenset[str] = frozenset({
    "gnomon_inspect", "gnomon_forecast", "gnomon_covariate_guide",
    "gnomon_validate_covariates", "gnomon_propose_covariates",
    "gnomon_preflight_context", "gnomon_route",
    "gnomon_investigate_change", "gnomon_detect_anomalies",
    "gnomon_decide", "gnomon_monitor",
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
            repairs: list[dict[str, Any]] = [{
                "action": "supply_arguments",
                "description": f"Pass {parameter} explicitly."
                               + (f" Candidates: {', '.join(candidates)}."
                                  if candidates else ""),
                "arguments": [parameter],
            }]
            if parameter == "target_column" and tool_name == "gnomon_forecast":
                repairs.append({
                    "action": "forecast_all_candidates",
                    "description": "Or batch every numeric column in one "
                                   "run: pass target_column \"auto\".",
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
    return {
        "schema_version": "0.1",
        "status": "complete",
        "forecast_id": artifact.forecast_id,
        "artifact_path": str(path),
        "results": [
            {
                "series": item.series, "support": item.support,
                "support_assessment": item.support_assessment,
                "selected_model": item.selected_model,
                "interval_coverage": item.interval_coverage,
                "warnings": item.warnings,
                "forecast_preview": item.forecast[:FORECAST_PREVIEW_ROWS],
                "forecast_rows": len(item.forecast),
                "threshold": item.threshold,
                "context": item.context,
                "covariates": item.covariates,
            }
            for item in artifact.results
        ],
    }


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
                 "q10": row["q10"], "q90": row["q90"]}
                for row in item.forecast
            ],
            **({"threshold": item.threshold} if item.threshold else {}),
        })
    return {
        "schema_version": "0.1",
        "status": "complete",
        "format": "brief",
        "forecast_id": artifact.forecast_id,
        "artifact_path": str(path),
        "note": (
            "Brief output: q50 with the q10-q90 interval per step. The full "
            "artifact (all quantile levels, evidence, lineage) is on disk at "
            "artifact_path, unchanged."
        ),
        "results": results,
    }


def _run_capabilities(arguments: dict[str, Any]) -> dict[str, Any]:
    return capabilities()


def _run_inspect(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_dataset(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        store_path=arguments.get("store_path"),
    )


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


def _run_covariate_guide(arguments: dict[str, Any]) -> dict[str, Any]:
    from .covariates import covariate_guide
    return covariate_guide(
        arguments["input"], time_column=arguments["time_column"],
        target_column=arguments["target_column"], horizon=int(arguments["horizon"]),
        series_column=arguments.get("series_column"), frequency=arguments.get("frequency"),
    )


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


def _run_forecast(arguments: dict[str, Any]) -> dict[str, Any]:
    target_spec = str(arguments["target_column"])
    if "," in target_spec or target_spec.strip().lower() == "auto":
        return _run_forecast_multi(arguments, target_spec)
    events = _context_events_from(arguments)
    config = None
    if arguments.get("future_events") or arguments.get("structural_events"):
        # MCP tool calls do not read gnomon.yaml — deliberately, for every
        # setting — so the admission lanes must be reachable as explicit
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
        candidates=arguments.get("candidates"),
        best_effort=bool(arguments.get("best_effort", False)),
        config=config,
        input_provenance=arguments.get("input_provenance"),
    )
    payload = (brief_summary(artifact, path)
               if arguments.get("format") == "brief"
               else forecast_summary(artifact, path))
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
            "series_column", "context_events_file", "context_events",
            "covariates_file", "covariates", "project",
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
    artifact, path = forecast_multi(
        str(arguments["input"]),
        time_column=arguments["time_column"],
        target_columns=targets,
        frequency=arguments.get("frequency"),
        horizon=int(arguments["horizon"]),
        as_of=_parse_as_of(arguments.get("as_of")),
        output=arguments.get("output_dir") or "gnomon-output",
        minimum_baseline_improvement=float(arguments.get("minimum_baseline_improvement", 0.02)),
        threshold=float(arguments["threshold"]) if arguments.get("threshold") is not None else None,
        repair=arguments.get("repair", "safe"),
        candidates=arguments.get("candidates"),
        best_effort=bool(arguments.get("best_effort", False)),
        input_provenance=arguments.get("input_provenance"),
    )
    if arguments.get("format") == "brief":
        return brief_summary(artifact, path)
    return forecast_summary(artifact, path)


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
        if row.get("series"):
            tuples.append((str(row["series"]), timestamp, value))
        else:
            tuples.append((timestamp, value))
    return tuples


def _run_submit_actuals(arguments: dict[str, Any]) -> dict[str, Any]:
    import csv as csv_module

    from .contracts import GnomonError
    from .tracking import TrackingStore
    store = TrackingStore()
    project = str(arguments["project"])
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
                **store.explain_unscored(
                    project, [item[-2] for item in tuples]),
            }
        return {"schema_version": "0.1", "status": "ok",
                "scored": len(results),
                "results": [item.__dict__ for item in results]}
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
            **store.explain_unscored(project, [row[resolved_time] for row in rows]),
        }
    return {"schema_version": "0.1", "status": "ok", "scored": len(results),
            "results": [item.__dict__ for item in results]}


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


def _run_proposer_skill(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    rows = TrackingStore().proposer_skill(
        str(arguments["project"]),
        proposer_id=(str(arguments["proposer_id"])
                     if arguments.get("proposer_id") else None),
        event_type=(str(arguments["event_type"])
                    if arguments.get("event_type") else None),
    )
    return {"status": "ok", "proposers": rows,
            "warning": ("Observational, set-level attribution; shrunk "
                        "toward no-skill priors. No proposal currently "
                        "earns forecast influence from these numbers.")}


def _run_record_decision(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    item = TrackingStore().record_decision(
        str(arguments["decision_id"]), str(arguments["project"]),
        str(arguments["forecast_id"]), str(arguments["action"]),
        str(arguments["expected_outcome"]),
    )
    return {"status": "ok", "decision": item.__dict__}


def _run_resolve_decision(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    item = TrackingStore().resolve_decision(
        str(arguments["decision_id"]), str(arguments["actual_outcome"]),
        bool(arguments["correct"]),
    )
    return {"status": "ok", "decision": item.__dict__}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "gnomon_capabilities",
        "description": (
            "Report what the installed Gnomon runtime actually supports. Use for "
            "feature detection instead of assuming a capability exists."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "runner": _run_capabilities,
    },
    {
        "name": "gnomon_inspect",
        "description": (
            "Validate a temporal dataset before forecasting: schema mapping, "
            "frequency, duplicates, missing periods. Prefer this before "
            "gnomon_forecast when mappings or data quality are uncertain."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {**_INPUT_PROPERTIES, **_REPLAY_PROPERTIES},
            "required": [],
        },
        "runner": _run_inspect,
    },
    {
        "name": "gnomon_forecast",
        "description": (
            "Run Gnomon's evaluated forecast: baselines and candidates are "
            "backtested on rolling folds; each series gets a selected model or "
            "an unsupported abstention. Context events (from `gnomon context "
            "validate`) and covariates are admitted only when they demonstrate "
            "stable lift on identical folds; when both are supplied, an "
            "adjudication ladder picks the best of base, context, covariates, "
            "or their combination and records the comparison as evidence. "
            "Read forecast.csv / summary.md in the returned "
            "artifact directory for the numbers and quote them verbatim; never "
            "invent values for an unsupported series."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_INPUT_PROPERTIES,
                **_REPLAY_PROPERTIES,
                "target_column": {"type": "string", "description": (
                    "Name of the numeric column to forecast. Also accepts a "
                    "comma list (`hr,spo2,resp`) or `auto` (every numeric "
                    "non-time column) to batch several columns of a wide "
                    "file into one run — one shared load pass, channels "
                    "evaluated concurrently, one combined artifact with a "
                    "result per column. Each channel's numbers are identical "
                    "to a single-target run; a channel that abstains is "
                    "disclosed in its own result and never blocks the others. "
                    "Omit to infer when exactly one non-time column parses "
                    "as numbers; the inference is disclosed as an assumption."
                )},
                "horizon": {"type": "integer", "description": (
                    "Future periods to forecast, in units of the data "
                    "frequency. Omit to default to one seasonal period of "
                    "the inferred grid, disclosed as an assumption."
                )},
                "format": {"type": "string", "enum": ["full", "brief"], "description": (
                    "Response verbosity (default full). `brief` returns per "
                    "target the q50 path with one q10-q90 interval, the "
                    "selected model, and — verbatim, never summarised — the "
                    "support state, every warning, abstention reason, "
                    "recovery action, and disclosure. The full artifact is "
                    "written to disk unchanged either way; only the response "
                    "payload shrinks."
                )},
                "candidates": {"type": "array", "items": {"type": "string"}, "description": "Restrict the model pool to these names — pass `gnomon_route`'s `candidates` or its `recommendation` to act on a routing decision. The mandatory baselines always compete regardless, so a named candidate still has to beat them."},
                "output_dir": {"type": "string", "description": "Directory for the immutable artifact. Defaults to ./gnomon-output relative to the *server's* working directory, which is often inside the user's repository — pass an explicit path when that matters."},
                "minimum_baseline_improvement": {"type": "number", "minimum": 0, "description": "Minimum relative improvement over the strongest baseline to select a candidate (default 0.02). Must be >= 0; a negative value would let a model that lost the backtest be selected."},
                "context_events_file": {"type": "string", "description": "Optional validated context-events JSON file (the output of `gnomon context validate`)."},
                **_CONTEXT_EVENTS_PROPERTY,
                "threshold": {"type": "number", "description": "Optional decision threshold: the result reports when and how likely the forecast crosses this value."},
                "project": {"type": "string", "description": "Optional tracking project. When set, register the forecast for realised scoring."},
                "covariates_file": {"type": "string", "description": "Local CSV containing point-in-time covariate vintages."},
                **_COVARIATES_PROPERTY,
                "covariate_mapping": {"type": "string", "description": "Comma-separated name:type:future_known entries."},
                "covariate_time_column": {"type": "string", "description": "Valid-at column (default timestamp)."},
                "covariate_known_at_column": {"type": "string", "description": "Availability timestamp column (default known_at)."},
                "covariate_series_column": {"type": "string", "description": "Optional series column in the covariate CSV."},
                "repair": {"type": "string", "enum": ["off", "safe", "aggressive"], "description": "Messy-data handling (default safe): off rejects anything non-strict; safe normalises cell text with disclosure; aggressive additionally fills gaps, snaps timestamps, and resolves conflicts — capped, and every fix is reported in evidence and warnings."},
                "best_effort": {"type": "boolean", "description": (
                    "When the evaluation cannot support the requested "
                    "horizon (default false: an honest abstention with "
                    "recovery actions), publish a naive fallback anyway — "
                    "disclosed as support `best_effort` with a NO RELIABLE "
                    "FORECAST warning and a descriptive, never predictive, "
                    "lineage claim. Never changes a supported forecast."
                )},
                "future_events": {"type": "boolean", "description": (
                    "Admit future-dated constraint:/override: context "
                    "events from context_events_file by textual "
                    "verifiability (default false). Spans are re-parsed "
                    "deterministically; an influenced forecast reports "
                    "support `context_trusted` and carries a history-only "
                    "counterfactual in evidence. Dry-run admission first "
                    "with gnomon_preflight_context."
                )},
                "structural_events": {"type": "boolean", "description": (
                    "Additionally admit LLM-classified structural: events "
                    "(closed effect menu: trend_ceases and the "
                    "seasonal-regime levels) from "
                    "context_events_file (default false). The proposer "
                    "classifies; every applied quantity is derived from "
                    "the forecast's own emitted path — no model-supplied "
                    "numbers. Experimental: "
                    "results/structural-effects/HYPOTHESIS.md."
                )},
            },
            "required": [],
        },
        "runner": _run_forecast,
    },
    {
        "name": "gnomon_covariate_guide",
        "description": "Return point-in-time format, forecast dates, and fold cutoffs. Gnomon does not suggest what data to fetch.",
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {"type": "integer"},
        }, "required": ["horizon"]},
        "runner": _run_covariate_guide,
    },
    {
        "name": "gnomon_validate_covariates",
        "description": "Validate local covariate vintages for format, final-horizon coverage, and availability at every selection cutoff.",
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {"type": "integer"},
            "covariates_file": {"type": "string"},
            **_COVARIATES_PROPERTY,
            "covariate_mapping": {"type": "string"},
            "covariate_time_column": {"type": "string"},
            "covariate_known_at_column": {"type": "string"},
            "covariate_series_column": {"type": "string"},
        }, "required": ["horizon", "covariate_mapping"]},
        "runner": _run_validate_covariates,
    },
    {
        "name": "gnomon_propose_covariates",
        "description": "Evaluate a local point-in-time covariate proposal through leakage-safe ablation and produce a forecast only when it earns admission.",
        "inputSchema": {"type": "object", "properties": {
            **_INPUT_PROPERTIES,
            "horizon": {"type": "integer"},
            "output_dir": {"type": "string"},
            "minimum_baseline_improvement": {"type": "number", "minimum": 0},
            "covariates_file": {"type": "string"},
            **_COVARIATES_PROPERTY,
            "covariate_mapping": {"type": "string"},
            "covariate_time_column": {"type": "string"},
            "covariate_known_at_column": {"type": "string"},
            "covariate_series_column": {"type": "string"},
        }, "required": ["horizon", "covariate_mapping"]},
        "runner": _run_forecast,
    },
    {
        "name": "gnomon_submit_actuals",
        "description": "Score all due forecasts in a project from complete realised actuals. Panel actuals must include series,timestamp,value. A forecast scores only when every period in its horizon has an actual; when nothing scores, the result explains which window was missing rather than returning a bare zero.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"},
            "actuals_file": {"type": "string", "description": "CSV of realised values. Callers without a filesystem pass `actuals` inline instead."},
            "actuals": {"type": "array", "items": {"type": "object"}, "description": (
                "Realised values supplied inline: objects of "
                "{timestamp, value, series?}. Mutually exclusive with "
                "actuals_file."
            )},
            "time_column": {"type": "string", "description": "Timestamp column in the actuals file. Inferred from a conventional name or a two-column layout when omitted."},
            "target_column": {"type": "string", "description": "Realised value column. Inferred when unambiguous."},
            "series_column": {"type": "string", "description": "Series column, required for multi-series projects."},
        }, "required": ["project"]},
        "runner": _run_submit_actuals,
    },
    {
        "name": "gnomon_list_open_forecasts",
        "description": "List unscored forecasts and distinguish horizons that are due from those still awaiting observations.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"},
        }, "required": []},
        "runner": _run_open_forecasts,
    },
    {
        "name": "gnomon_model_performance",
        "description": "Read descriptive realised model performance for a project. Do not treat observational rankings as causal evidence.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"}, "model": {"type": "string"},
        }, "required": ["project"]},
        "runner": _run_model_performance,
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
        output=arguments.get("output_dir") or "gnomon-output",
        input_provenance=arguments.get("input_provenance"),
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
    )
    return {**payload, "artifact_path": str(path)}


def _run_status(arguments: dict[str, Any]) -> dict[str, Any]:
    from .tracking import TrackingStore
    return TrackingStore().status(arguments.get("project"))


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
    )
    return {**payload, "artifact_path": str(path)}


def _run_get_artifact(arguments: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path
    from .artifacts import read_artifact
    from .versioning import RUNTIME_VERSION
    directory = Path(arguments["artifact_path"])
    artifact = read_artifact(directory)
    payload: dict[str, Any] = {
        "schema_version": "0.1",
        "artifact": artifact,
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
        "name": "gnomon_get_artifact",
        "description": (
            "Read a stored artifact directory: full artifact.json and, "
            "optionally, the typed lineage. All numbers live here; quote them "
            "verbatim."
        ),
        "inputSchema": {"type": "object", "properties": {
            "artifact_path": {"type": "string", "description": "Artifact directory returned by a macro."},
            "include_lineage": {"type": "boolean", "description": "Include lineage.json (artifacts/evidence/claims)."},
        }, "required": ["artifact_path"]},
        "runner": _run_get_artifact,
    },
    {
        "name": "gnomon_status",
        "description": (
            "Pollable status: open forecasts, due horizons, unresolved "
            "decisions, and realised-performance summaries. Descriptive "
            "evidence an agent can cite — never causal."
        ),
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string", "description": "Optional project filter."},
        }, "required": []},
        "runner": _run_status,
    },
    {
        "name": "gnomon_resolve_outcome",
        "description": (
            "The current decision resolver, for DecisionArtifacts produced by "
            "`gnomon_decide`. (The v0.2 `gnomon_record_decision` / "
            "`gnomon_resolve_decision` pair is deprecated and resolves only its "
            "own records.) "
            "Resolve a recorded DecisionArtifact with what actually happened: "
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
        "name": "gnomon_proposer_skill",
        "description": ("Shrunk per-proposer, per-event-type skill from "
                        "resolved context-event proposals (realised lift vs "
                        "the history-only counterfactual, in WAPE). "
                        "Observational; attribution is set-level because the "
                        "admission gate decides on event sets."),
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string"},
            "proposer_id": {"type": "string"},
            "event_type": {"type": "string"},
        }, "required": ["project"]},
        "runner": _run_proposer_skill,
    },
])


# --- Experimental planner surface (advanced; macros remain the default) ---

def planner_enabled() -> bool:
    import os
    return os.environ.get("GNOMON_EXPERIMENTAL_PLANNER") == "1"


def _run_compile_task(arguments: dict[str, Any]) -> dict[str, Any]:
    from dataclasses import asdict
    from .plan import compile_task
    plan = compile_task(str(arguments["task_type"]), dict(arguments.get("params") or {}))
    return {"schema_version": "0.1", "plan_id": plan.plan_id(),
            "plan": {"task": plan.task, "steps": [asdict(step) for step in plan.steps]}}


def _run_validate_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    from .plan import plan_from_dict, validate_plan
    plan = plan_from_dict(dict(arguments["plan"]))
    violations = validate_plan(plan)
    return {"schema_version": "0.1", "plan_id": plan.plan_id(),
            "valid": not violations, "violations": violations}


def _run_execute_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    from .execution import execute_plan
    from .plan import plan_from_dict
    payload, path = execute_plan(
        plan_from_dict(dict(arguments["plan"])),
        output=arguments.get("output_dir") or "gnomon-output",
        as_of=_parse_as_of(arguments.get("as_of")),
        store_path=arguments.get("store_path"),
    )
    return {**payload, "artifact_path": str(path)}


def _run_get_run(arguments: dict[str, Any]) -> dict[str, Any]:
    from .artifacts import read_artifact
    return {"schema_version": "0.1", "run": read_artifact(arguments["run_path"])}


_PLAN_PROPERTY = {"type": "object", "description": "A TemporalPlan: {task, steps: [{step_id, operator, inputs, produces, failure_policy}]}."}

PLANNER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "gnomon_compile_task",
        "description": "Compile one of the four canonical task types into a validated TemporalPlan (experimental).",
        "inputSchema": {"type": "object", "properties": {
            "task_type": {"type": "string", "enum": ["forecast", "investigate_change", "decide", "monitor"]},
            "params": {"type": "object", "description": "The macro's parameters."},
        }, "required": ["task_type", "params"]},
        "runner": _run_compile_task,
    },
    {
        "name": "gnomon_validate_plan",
        "description": "Deterministically validate a TemporalPlan: operators, references, leakage, claim-class feasibility, budget, duplicates (experimental).",
        "inputSchema": {"type": "object", "properties": {"plan": _PLAN_PROPERTY},
                        "required": ["plan"]},
        "runner": _run_validate_plan,
    },
    {
        "name": "gnomon_execute_plan",
        "description": "Execute a validated TemporalPlan with step checkpointing, content-addressed caching, and deterministic replay (experimental).",
        "inputSchema": {"type": "object", "properties": {
            "plan": _PLAN_PROPERTY,
            "output_dir": {"type": "string"},
            "as_of": {"type": "string"},
            "store_path": {"type": "string"},
        }, "required": ["plan"]},
        "runner": _run_execute_plan,
    },
    {
        "name": "gnomon_get_run",
        "description": "Read a stored plan-run artifact: step provenance and outputs (experimental).",
        "inputSchema": {"type": "object", "properties": {
            "run_path": {"type": "string", "description": "Run artifact directory."},
        }, "required": ["run_path"]},
        "runner": _run_get_run,
    },
]


def v02_compat_enabled() -> bool:
    import os
    return os.environ.get("GNOMON_V02_COMPAT") == "1"


#: The deprecated v0.2 lifecycle pair, off by default. Two tool slots and
#: two long deprecation arguments were a per-session tax on every agent;
#: GNOMON_V02_COMPAT=1 restores them, schemas and behaviour unchanged, for
#: clients that still write v0.2 decision records.
V02_COMPAT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "gnomon_record_decision",
        "description": ("Link a free-text agent decision and expected "
                        "outcome to a tracked forecast (v0.2 lifecycle; "
                        "superseded by gnomon_decide)."),
        "inputSchema": {"type": "object", "properties": {
            "decision_id": {"type": "string"}, "project": {"type": "string"},
            "forecast_id": {"type": "string"}, "action": {"type": "string"},
            "expected_outcome": {"type": "string"},
        }, "required": ["decision_id", "project", "forecast_id", "action", "expected_outcome"]},
        "runner": _run_record_decision,
    },
    {
        "name": "gnomon_resolve_decision",
        "description": ("Record the realised outcome and a yes/no verdict "
                        "for a v0.2 decision record (superseded by "
                        "gnomon_resolve_outcome)."),
        "inputSchema": {"type": "object", "properties": {
            "decision_id": {"type": "string"}, "actual_outcome": {"type": "string"},
            "correct": {"type": "boolean"},
        }, "required": ["decision_id", "actual_outcome", "correct"]},
        "runner": _run_resolve_decision,
    },
]


def visible_tools() -> list[dict[str, Any]]:
    """The tool surface as gated for this process: macros always; the v0.2
    compat pair only behind GNOMON_V02_COMPAT=1; the raw planner only
    behind GNOMON_EXPERIMENTAL_PLANNER=1."""
    return (TOOLS
            + (V02_COMPAT_TOOLS if v02_compat_enabled() else [])
            + (PLANNER_TOOLS if planner_enabled() else []))


def _materialise_observations(arguments: dict[str, Any]) -> dict[str, Any]:
    """Turn the inline ``observations`` array into a temp-file ``input``.

    The rows become a CSV in a fresh temp directory and the call
    proceeds exactly as a file-based one — same loaders, same repair
    ladder, same fingerprinting — so the inline channel can never
    develop separate semantics from the file channel.
    """
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
                        and arguments.get("observations") is None:
                    from .contracts import GnomonError

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
                return disclose_assumptions(_runner(arguments), assumptions)

            return wrapped
    return None
