#!/usr/bin/env python3
"""Evaluate candidate learning in chronological, same-series deployments.

This benchmark is deliberately model-free. It tests the product's learning
contract—not whether an LLM can guess a held-out path—using recurring series,
outcomes that become available at explicit knowledge times, shadow-candidate
scoring, and the real publication/tracking machinery.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from typing import Any

from gnomon.llm_dossier import validate_temporal_dossier
from gnomon.publication import publish_result, record_publication
from gnomon.tracking import TrackingStore


def _rows(start: datetime, level: float, horizon: int = 2) -> list[dict[str, Any]]:
    return [{
        "timestamp": (start + timedelta(days=step)).isoformat(),
        "point": level, "q10": level - 1.0,
        "q50": level, "q90": level + 1.0,
    } for step in range(1, horizon + 1)]


def _dossier(
        cutoff: datetime, candidate_level: float,
        compiler_model: str) -> dict[str, Any]:
    future = [(cutoff + timedelta(days=step)).isoformat()
              for step in (1, 2)]
    text = "A recurring operational event may change the next two readings."
    raw = {
        "claims": [{
            "source_span": text, "relation": "supports_increase",
            "effective_start": future[0], "effective_end": future[-1],
            "confidence": .7,
        }],
        "forecast_candidate": {
            "quantiles": _rows(cutoff, candidate_level),
            "rationale": "A sealed external conditional prior.",
        },
    }
    return validate_temporal_dossier(
        raw, context_text=text, cutoff=cutoff.isoformat(),
        future_timestamps=future, history=list(range(95, 105)),
        compiler_model=compiler_model)[0]


def _resolve_publication(
        store: TrackingStore, *, project: str, forecast_id: str,
        series: str, outcome: list[float], resolved_at: datetime) -> None:
    for receipt in store.temporal_synthesis_receipts(
            project, resolved=False, series=series):
        if receipt["forecast_id"] != forecast_id:
            continue
        store.resolve_temporal_synthesis(
            project=project, forecast_id=forecast_id, series=series,
            question_id=receipt["question_id"],
            synthesis_id=receipt["synthesis_id"],
            outcome={"points": outcome},
            resolved_at=resolved_at.isoformat())


def run_stream(
    store: TrackingStore, *, project: str, series: str,
    candidate_truth: list[tuple[float, float]],
    start: datetime, outcome_delay_days: int = 0,
    compiler_model: str = "outcomelearningbench-fixed-candidate",
) -> dict[str, Any]:
    """Run chronological origins of (candidate level, realised level)."""
    cases = []
    for index, (candidate_level, realised_level) in enumerate(candidate_truth):
        cutoff = start + timedelta(days=index * 3)
        primary = _rows(cutoff, 100.0)
        dossier = _dossier(cutoff, candidate_level, compiler_model)
        evidence = store.candidate_outcome_summary(
            project, series=series, resolved_before=cutoff.isoformat())
        publication = publish_result(
            {"support": "supported", "forecast": primary},
            mode="best_effort", dossiers=[dossier],
            candidate_outcome_evidence=evidence,
            artifact_id=f"{series}-{index}")
        selected = next(item for item in publication["candidate_portfolio"]
                        if item["scenario_id"] ==
                        publication["recommended_scenario_id"])
        actual = [realised_level, realised_level]

        def wape(rows: list[dict[str, Any]]) -> float:
            denominator = sum(abs(value) for value in actual) or 1.0
            return sum(abs(float(row["q50"]) - value)
                       for row, value in zip(rows, actual, strict=True)) / denominator

        forecast_id = f"{series}-{index}"
        record_publication(
            store, project=project, forecast_id=forecast_id,
            series=series, payload=publication)
        known_at = cutoff + timedelta(days=2 + outcome_delay_days)
        _resolve_publication(
            store, project=project, forecast_id=forecast_id, series=series,
            outcome=actual, resolved_at=known_at)
        candidate = next(item for item in publication["candidate_portfolio"]
                         if item["role"] == "model_authored")
        cases.append({
            "origin": index + 1, "cutoff": cutoff.isoformat(),
            "outcome_known_at": known_at.isoformat(),
            "selected_role": selected["role"],
            "selection_method": publication["recommendation_authority"][
                "selection_method"],
            "primary_wape": wape(primary),
            "candidate_wape": wape(candidate["forecast"]),
            "selected_wape": wape(selected["forecast"]),
            "outcome_evidence_resolved": sum(
                int(item.get("resolved", 0)) for item in evidence),
            "primary_forecast_unchanged": publication[
                "primary_forecast_unchanged"],
            "automation_eligible": publication["automation"]["eligible"],
        })
    return {
        "series": series, "candidate_proposer": compiler_model,
        "cases": cases,
        "mean_primary_wape": sum(item["primary_wape"] for item in cases) / len(cases),
        "mean_candidate_wape": sum(item["candidate_wape"] for item in cases) / len(cases),
        "mean_selected_wape": sum(item["selected_wape"] for item in cases) / len(cases),
        "outcome_informed_selections": sum(
            item["selection_method"] == "resolved_outcome_human_prior_policy"
            for item in cases),
        "immutability_failures": sum(
            item["primary_forecast_unchanged"] is not True for item in cases),
        "automation_violations": sum(
            item["automation_eligible"] is True for item in cases),
    }


def run_suite(database: Path) -> dict[str, Any]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stable = run_stream(
        TrackingStore(database.with_name("stable.db")), project="stable",
        series="demand", candidate_truth=[(110.0, 110.0)] * 14,
        start=start)
    harmful = run_stream(
        TrackingStore(database.with_name("harmful.db")), project="harmful",
        series="demand", candidate_truth=[(110.0, 90.0)] * 14,
        start=start)
    reversal = run_stream(
        TrackingStore(database.with_name("reversal.db")), project="reversal",
        series="demand",
        candidate_truth=[(110.0, 110.0)] * 8 + [(110.0, 90.0)] * 8,
        start=start)
    post_reversal = reversal["cases"][8:]
    first_selected_after_reversal = next((
        item["origin"] for item in post_reversal
        if item["selection_method"] == "resolved_outcome_human_prior_policy"),
        None)
    first_demoted_after_selection = next((
        item["origin"] for item in post_reversal
        if first_selected_after_reversal is not None
        and item["origin"] > first_selected_after_reversal
        and item["selection_method"] != "resolved_outcome_human_prior_policy"),
        None)
    reversal["known_regime_change_origin"] = 9
    reversal["first_selected_after_regime_change"] = first_selected_after_reversal
    reversal["first_demoted_after_regime_change"] = first_demoted_after_selection
    reversal["bad_recommendations_before_demotion"] = sum(
        item["selection_method"] == "resolved_outcome_human_prior_policy"
        for item in post_reversal[:(
            first_demoted_after_selection - 9
            if first_demoted_after_selection is not None else len(post_reversal))])
    delayed = run_stream(
        TrackingStore(database.with_name("delayed.db")), project="delayed",
        series="demand", candidate_truth=[(110.0, 110.0)] * 10,
        start=start, outcome_delay_days=60)

    contamination_store = TrackingStore(database.with_name("contamination.db"))
    run_stream(
        contamination_store, project="shared", series="other",
        candidate_truth=[(110.0, 110.0)] * 10, start=start)
    contamination = run_stream(
        contamination_store, project="shared", series="target",
        candidate_truth=[(110.0, 90.0)] * 4,
        start=start + timedelta(days=40))

    proposer_store = TrackingStore(database.with_name("proposer-change.db"))
    run_stream(
        proposer_store, project="replacement", series="demand",
        candidate_truth=[(110.0, 110.0)] * 10, start=start,
        compiler_model="model-a")
    proposer_change = run_stream(
        proposer_store, project="replacement", series="demand",
        candidate_truth=[(110.0, 90.0)] * 4,
        start=start + timedelta(days=40), compiler_model="model-b")

    result = {
        "benchmark": "outcome-learning-prequential",
        "schema_version": "0.1",
        "families": {
            "stable_beneficial": stable,
            "stable_harmful": harmful,
            "regime_reversal": reversal,
            "delayed_outcomes": delayed,
            "unrelated_series_contamination": contamination,
            "proposer_identity_change": proposer_change,
        },
    }
    result["gates"] = {
        "stable_prior_eventually_used": stable["outcome_informed_selections"] > 0,
        "harmful_prior_never_used": harmful["outcome_informed_selections"] == 0,
        "delayed_future_outcomes_not_used": delayed["outcome_informed_selections"] == 0,
        "unrelated_series_not_used": contamination[
            "outcome_informed_selections"] == 0,
        "different_proposer_history_not_used": proposer_change[
            "outcome_informed_selections"] == 0,
        "reversal_demoted_within_two_resolved_losses": (
            reversal["first_demoted_after_regime_change"] is not None
            and reversal["first_demoted_after_regime_change"] <= 11
            and reversal["bad_recommendations_before_demotion"] <= 2),
        "no_immutability_failures": all(
            family["immutability_failures"] == 0
            for family in (stable, harmful, reversal, delayed, contamination,
                           proposer_change)),
        "no_automation_violations": all(
            family["automation_violations"] == 0
            for family in (stable, harmful, reversal, delayed, contamination,
                           proposer_change)),
    }
    result["passed"] = all(result["gates"].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-outcome-learning-") as temp:
        result = run_suite(Path(temp) / "registry.db")
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
