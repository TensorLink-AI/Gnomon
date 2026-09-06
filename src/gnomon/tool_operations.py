"""Legacy tool operations. Registration and wire projection are owned separately."""

from __future__ import annotations


from typing import Any

from .contracts import ForecastArtifact, GnomonError
from .runtime import capabilities, forecast, inspect_dataset
from .tool_response import (
    _brief_capabilities,
    brief_summary,
    forecast_summary,
)
from .tool_context import (
    _align_single_target_context_scope,
    _covariates_from,
    _materialized_or_public_events,
    _run_preflight_context as _run_preflight_context,
)
from .tool_publication import (
    _attach_publication,
)
from .tool_input import (
    _parse_as_of,
)


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
            "series_count": len(ranked),
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
    from .operators import anomaly_score, regime_detection, seasonality_analysis
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
    # A typed question attached to a forecast must execute against the same
    # seasonal contract as that immutable primary.  Descriptive inspection
    # still detects seasonality independently, but recomputing the period here
    # can disagree with the period already used by the forecast runtime (most
    # visibly when a frequency default is admitted before two cycles exist).
    execution_seasons = arguments.get("_execution_seasons") or {}
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
            changes = regime_detection(timestamps, values)
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
            execution_season = int(
                execution_seasons.get(name)
                or seasonality.get("period")
                or 1)
            profile = temporal_profile(
                values, season=int(seasonality.get("period") or 1))
            execution_inputs[name] = (
                [float(value) for value in values],
                execution_season,
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
    from .temporal_contracts import classify_dataset_contract
    dataset_contract = classify_dataset_contract(
        list(reports),
        observations={name: int(report["observations"])
                      for name, report in reports.items()},
        frequency=next((str(report.get("frequency")) for report in reports.values()
                        if report.get("frequency")), None),
        time_column=arguments.get("time_column"),
        series_column=arguments.get("series_column"),
        label_column=arguments.get("label_column"),
        truncated=bool(arguments.get("_input_truncated", False)),
    )
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
                forecast_values=arguments.get("_forecast_values"),
                conditional_effects=arguments.get("_conditional_effects")))
    payload = _json_temporal_values({
        "schema_version": "0.1", "status": "valid",
        "headline": f"Described {len(reports)} series through "
                    f"{max(report['series_end'] for report in reports.values())}.",
        "dataset_contract": dataset_contract.to_dict(),
        "reports": reports,
        **({"answers": temporal_answers} if temporal_answers else {}),
        "triage": {
            "series_count": len(ranked),
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
    if arguments.get("format") == "brief" and temporal_answers:
        from .temporal_planner import compact_evidence_plan

        compact_answers = []
        for item in temporal_answers:
            answer = dict(item.get("answer") or {})
            reasoning = answer.get("reasoning")
            compact_reasoning = (compact_evidence_plan(reasoning)
                                 if isinstance(reasoning, dict) else None)
            if compact_reasoning:
                adjudication = dict(
                    compact_reasoning.get("adjudication") or {})
                adjudication.pop("ranked_hypotheses", None)
                adjudication.pop("weight_meaning", None)
                compact_reasoning["adjudication"] = adjudication
            question = dict(item.get("question") or {})
            compact_answers.append({
                key: value for key, value in {
                    "question": {key: question.get(key) for key in (
                        "id", "verb", "target", "property", "horizon")
                        if question.get(key) is not None},
                    "headline": item.get("headline"),
                    "best_estimate": item.get("best_estimate"),
                    "reasoning": compact_reasoning,
                    "limitations": item.get("limitations"),
                }.items() if value is not None
            })
        payload["answers"] = compact_answers
        payload.pop("reports", None)
        payload["view"] = {
            "format": "brief",
            "full_available": True,
            "note": ("Compact typed answers and per-series diagnostics; "
                     "use format='full' for complete reasoning receipts."),
        }
    return payload


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
    execution_seasons = {
        public_name(result.series): int(
            ((result.temporal_facts or {}).get("seasonal_period_steps") or 1))
        for result in artifact.results
    }
    if len(artifact.results) == 1:
        forecast_values[str(arguments["target_column"])] = next(iter(
            forecast_values.values()))
        execution_seasons[str(arguments["target_column"])] = next(iter(
            execution_seasons.values()))
    conditional_effects: dict[str, dict[str, Any]] = {}
    for result in artifact.results:
        public = public_name(result.series)
        effect = (result.context or {}).get("effect") if result.context else None
        if effect or result.context_outcome or result.conditional_forecasts \
                or result.sensitivity_scenarios:
            conditional_effects[public] = {
                **({"measured_effect": effect} if effect else {}),
                **({"outcome": result.context_outcome}
                   if result.context_outcome else {}),
                "conditional_forecast_count": len(result.conditional_forecasts),
                "sensitivity_scenario_count": len(result.sensitivity_scenarios),
                "provenance": "artifact_context_contract",
            }
    if len(artifact.results) == 1 and conditional_effects:
        conditional_effects[str(arguments["target_column"])] = next(iter(
            conditional_effects.values()))
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
        describe_arguments["_execution_seasons"] = execution_seasons
        describe_arguments["_conditional_effects"] = conditional_effects
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
        calibration = answer.get("calibration")
        if isinstance(calibration, dict):
            folds = int(calibration.get("folds") or
                        calibration.get("calibration_ratios") or 0)
            compact["calibration_status"] = {
                "available": folds > 0,
                "applicable": folds > 0,
                "folds": folds,
                **({"requested_horizon": calibration["requested_horizon"]}
                   if calibration.get("requested_horizon") is not None else {}),
                **({"reason": "no_applicable_calibration"}
                   if folds == 0 else {}),
            }
        else:
            compact["calibration_status"] = {
                "available": False, "applicable": False,
                "reason": "not_reported",
            }
        reasoning = ((answer.get("answer") or {}).get("reasoning"))
        if isinstance(reasoning, dict):
            from .temporal_planner import compact_evidence_plan
            compact_answer = dict(compact.get("answer") or {})
            compact_answer["reasoning"] = compact_evidence_plan(reasoning)
            compact["answer"] = compact_answer
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


def _forecast_config(arguments: dict[str, Any], events: Any, *,
                     infer_structural: bool):
    """One admission policy builder, preserving the legacy batch activation rule."""
    from .config import GnomonConfig

    future = bool(arguments.get("future_events") or any(
        event.event_type.startswith(("constraint:literal_", "override:literal_"))
        for event in events or []))
    structural = bool(arguments.get("structural_events") or (infer_structural and any(
        event.event_type.startswith("structural:") for event in events or [])))
    weighted = arguments.get("model_admission") == "evidence_weighted"
    if not (future or structural or weighted):
        return None
    config = GnomonConfig()
    config.context.future_events = future
    config.context.structural_events = structural
    if weighted:
        registry = arguments.get("model_evidence_registry")
        if not registry:
            raise GnomonError(
                "MISSING_MODEL_EVIDENCE_REGISTRY",
                "model_admission=evidence_weighted requires "
                "model_evidence_registry; a model name is not evidence.",
                {"required": ["model_evidence_registry"]},
            )
        config.models.admission_policy = "evidence_weighted"
        config.models.evidence_registry_path = str(registry)
    return config


def _run_forecast(arguments: dict[str, Any]) -> dict[str, Any]:
    target_spec = str(arguments["target_column"])
    if "," in target_spec or target_spec.strip().lower() == "auto":
        return _run_forecast_multi(arguments, target_spec)
    events = _align_single_target_context_scope(
        _materialized_or_public_events(arguments), arguments)
    config = _forecast_config(arguments, events, infer_structural=True)
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
    _attach_publication(payload, artifact, path, arguments)
    _attach_temporal_answers(payload, artifact, path, arguments)
    from .agent_response import build_agent_response_contract
    response_contract = build_agent_response_contract(payload)
    if response_contract is not None:
        payload["agent_response_contract"] = response_contract
    if arguments.get("project"):
        from .tracking import register_artifact
        payload["tracking_ids"] = register_artifact(
            artifact, str(arguments["project"]), str(path),
            context_events=events,
        )
        payload["project"] = str(arguments["project"])
        if payload.get("publication"):
            from .publication import record_publication
            from .tracking import TrackingStore
            payload["publication_synthesis_id"] = record_publication(
                TrackingStore(), project=str(arguments["project"]),
                forecast_id=artifact.forecast_id,
                series=artifact.results[0].series,
                payload=payload["publication"])
    return payload


def _run_forecast_multi(arguments: dict[str, Any], target_spec: str) -> dict[str, Any]:
    """The multi-target branch of gnomon_forecast: a comma list or `auto`
    in target_column batches several columns into one run and one
    combined artifact — same numbers per channel as separate calls."""
    from .contracts import GnomonError
    if (arguments.get("temporal_dossiers")
            or arguments.get("context_submission")
            or arguments.get("scenario_selection")):
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "Cross-series dossier or scenario ranking requires one target; "
            "plain strict, best_effort, and scenario publication modes are "
            "supported in one batched call.")
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
    events = _materialized_or_public_events(arguments)
    covariates = _covariates_from(arguments)
    config = _forecast_config(arguments, events, infer_structural=False)
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
    if (arguments.get("publication_mode") is not None
            or arguments.get("automation_policy")
            or arguments.get("_context_was_supplied")):
        publications = []
        for index, result in enumerate(artifact.results):
            child: dict[str, Any] = {}
            _attach_publication(
                child, artifact, path,
                {**arguments, "target_column": result.series},
                result_index=index)
            publication = child.get("publication")
            if not publication:
                continue
            publications.append({
                "series": result.series,
                "mode": publication.get("mode"),
                "recommended_scenario_id": publication.get(
                    "recommended_scenario_id"),
                "recommended_support": publication.get("recommended_support"),
                "primary_forecast_unchanged": publication.get(
                    "primary_forecast_unchanged"),
                "scenario_count": publication.get("scenario_count"),
                "context_summary": publication.get("context_summary"),
                "context_dispositions": publication.get(
                    "context_dispositions") or [],
                "automation": publication.get("automation"),
                "publication_seal_sha256": publication.get(
                    "publication_seal_sha256"),
                "publication_path": child.get("publication_path"),
            })
        payload["publications"] = publications
        payload["publication_summary"] = {
            "mode": str(arguments.get("publication_mode") or "strict"),
            "series_count": len(publications),
            "primary_forecast_unchanged": all(
                item.get("primary_forecast_unchanged") is True
                for item in publications),
            "automation_eligible": bool(publications) and all(
                (item.get("automation") or {}).get("eligible") is True
                for item in publications),
            "scenario_count": sum(int(item.get("scenario_count") or 0)
                                  for item in publications),
        }
    _attach_temporal_answers(payload, artifact, path, arguments)
    from .agent_response import build_agent_response_contract
    response_contract = build_agent_response_contract(payload)
    if response_contract is not None:
        payload["agent_response_contract"] = response_contract
    return payload


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


def _run_investigate_change(arguments: dict[str, Any]) -> dict[str, Any]:
    from .macros import investigate_change
    events = _materialized_or_public_events(arguments)
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
    from .operators import validate_action_utilities
    actions = arguments.get("actions")
    problems: list[str] = []
    if not isinstance(actions, list):
        problems.append(f"actions is {type(actions).__name__}, not a list")
    else:
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                problems.append(
                    f"item {index} is {type(action).__name__}, not an object "
                    "with a 'name'")
            elif not isinstance(action.get("name"), str) or not action["name"].strip():
                problems.append(f"item {index} has no non-empty 'name'")
            elif "feasible" in action and not isinstance(action["feasible"], bool):
                problems.append(f"item {index}: 'feasible' must be true or false")
            elif "residual_risk" in action:
                try:
                    float(action["residual_risk"])
                except (TypeError, ValueError):
                    problems.append(
                        f"item {index}: 'residual_risk' must be a number")
    if problems:
        raise GnomonError(
            "INVALID_ACTIONS",
            "actions does not match the expected shape: "
            + "; ".join(problems) + ".",
            {"example": [
                {"name": "scale_up", "feasible": True,
                 "residual_risk": 0.1},
                {"name": "do_nothing"},
            ], "problems": problems},
        )
    feasible_names = set()
    for action in actions:
        feasible = bool(action.get("feasible", True))
        if (feasible and arguments.get("max_acceptable_risk") is not None
                and "residual_risk" in action):
            feasible = (float(action["residual_risk"])
                        <= float(arguments["max_acceptable_risk"]))
        if feasible:
            feasible_names.add(str(action["name"]))
    utilities = validate_action_utilities(
        list(actions), ("exceed", "no_exceed"), arguments.get("utilities"),
        feasible_names=feasible_names,
    )
    payload, path = decide(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        horizon=int(arguments["horizon"]),
        threshold=float(arguments["threshold"]),
        actions=list(actions),
        utilities=utilities,
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
    from .monitoring import default_state_path, record_monitor_evaluation
    output = arguments.get("output_dir") or "gnomon-output"
    payload, path = monitor(
        arguments["input"],
        time_column=arguments["time_column"],
        target_column=arguments["target_column"],
        horizon=int(arguments["horizon"]),
        threshold=float(arguments["threshold"]),
        alert_cost=float(arguments["alert_cost"]) if arguments.get("alert_cost") is not None else None,
        miss_cost=float(arguments["miss_cost"]) if arguments.get("miss_cost") is not None else None,
        action_cost=float(arguments["action_cost"]) if arguments.get("action_cost") is not None else None,
        mitigation_effectiveness=float(arguments.get("mitigation_effectiveness", 1.0)),
        series_column=arguments.get("series_column"),
        frequency=arguments.get("frequency"),
        as_of=_parse_as_of(arguments.get("as_of")),
        project=arguments.get("project"),
        output=output,
        input_provenance=arguments.get("input_provenance"),
        regrid=arguments.get("regrid"),
        questions=arguments.get("questions"),
    )
    payload["firing_rate"] = record_monitor_evaluation(
        payload, state_path=default_state_path(output),
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


def _run_select_scenario(arguments: dict[str, Any]) -> dict[str, Any]:
    """Rerank sealed paths without rerunning or rewriting a forecast."""
    import json as _json
    from pathlib import Path

    from .contracts import GnomonError
    from .publication import (select_publication,
                              write_selected_publication)

    source = Path(str(arguments["publication_path"]))
    if not source.is_file():
        raise GnomonError(
            "INVALID_ARGUMENTS", "publication_path must name an existing file")
    try:
        publication = _json.loads(source.read_text(encoding="utf-8"))
        selected = select_publication(
            publication, dict(arguments["scenario_selection"]))
        selected_path = write_selected_publication(source, selected)
    except (OSError, ValueError, TypeError, _json.JSONDecodeError) as exc:
        raise GnomonError("INVALID_ARGUMENTS", str(exc)) from exc
    scenario_id = selected["recommended_scenario_id"]
    return {
        "schema_version": "0.1", "status": "ok", "verb": "select_scenario",
        "headline": (
            f"Selected {scenario_id} as the human-facing recommendation. "
            "The governed primary forecast is unchanged and this selection "
            "does not authorize automation."
        ),
        "artifact_id": selected.get("artifact_id"),
        "publication_path": str(selected_path),
        "supersedes_publication_seal_sha256": selected[
            "supersedes_publication_seal_sha256"],
        "publication_seal_sha256": selected["publication_seal_sha256"],
        "recommended_scenario_id": scenario_id,
        "recommended_forecast": selected["recommended_forecast"],
        "recommended_support": selected["recommended_support"],
        "support": selected["recommended_support"],
        "primary_forecast_unchanged": True,
        "scenario_selection": selected["scenario_selection"],
        "recommendation_authority": selected["recommendation_authority"],
        "automation": selected["automation"],
    }


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
