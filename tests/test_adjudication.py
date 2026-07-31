"""The enrichment championship ladder: combined runs are adjudicated, not
rejected, and the artifact proves the choice."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

from aion.adjudication import COMBINED_MODEL_NAME, Candidate, rank_candidates
from aion.context import ContextEvent, ContextSource
from aion.covariates import load_covariates
from aion.runtime import capabilities, forecast

START = datetime(2025, 1, 1, tzinfo=timezone.utc)
COUNT = 130
HORIZON = 7
PROMO_EFFECT = 30.0
COVARIATE_EFFECT = 20.0


def _covariate_signal(index: int) -> float:
    # Deterministic but deliberately not periodic at the daily season of 7.
    return 1.0 if ((index * 17 + 3) % 11) < 5 else 0.0


def _promo(index: int) -> bool:
    return index % 10 in (5, 6)


def _write_observations(tmp_path, *, promo_effect: float, covariate_effect: float):
    observations = tmp_path / "observations.csv"
    with observations.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for index in range(COUNT):
            value = (
                50 + index * 0.2
                + covariate_effect * _covariate_signal(index)
                + (promo_effect if _promo(index) else 0.0)
            )
            writer.writerow([(START + timedelta(days=index)).isoformat(), value])
    return observations


def _write_covariates(tmp_path, column=_covariate_signal):
    covariates = tmp_path / "covariates.csv"
    with covariates.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "known_at", "campaign"])
        for index in range(COUNT + HORIZON):
            writer.writerow([
                (START + timedelta(days=index)).isoformat(),
                (START - timedelta(days=30)).isoformat(),
                column(index),
            ])
    return load_covariates(str(covariates), "campaign:binary:future_known")


def _events(*, sourced: bool = True) -> list[ContextEvent]:
    events = []
    for day in range(COUNT + HORIZON):
        if day % 10 == 5:
            start = START + timedelta(days=day)
            events.append(ContextEvent(
                event_id=f"promo_{day:03d}",
                event_type="promotion",
                entity_scope=("*",),
                effective_start=start.isoformat(),
                effective_end=(start + timedelta(days=1, hours=23)).isoformat(),
                known_at=(START - timedelta(days=1)).isoformat(),
                source=ContextSource("calendar", "promo-calendar.ics") if sourced else None,
            ))
    return events


def _run(tmp_path, observations, *, events, dataset):
    # The weekly season is pinned: the promo (10-day) and covariate (11-day)
    # cycles otherwise skew autodetection, which is not what these tests probe.
    return forecast(
        str(observations), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        context_events=events, covariates=dataset, seasonal_period=7,
    )


def _adjudication_record(artifact):
    records = [item for item in artifact.evidence if item.kind == "enrichment_adjudication"]
    assert len(records) == 1
    return records[0]


def test_combined_run_admits_the_joint_challenger(tmp_path) -> None:
    """Both enrichments carry independent signal: the previously-rejected
    combination is evaluated on identical folds and wins the ladder."""
    observations = _write_observations(
        tmp_path, promo_effect=PROMO_EFFECT, covariate_effect=COVARIATE_EFFECT,
    )
    artifact, path = _run(tmp_path, observations, events=_events(), dataset=_write_covariates(tmp_path))
    assert path.is_dir()
    result = artifact.results[0]
    assert result.context["admitted"] is True
    assert result.covariates["admitted"] is True
    assert result.selected_model == COMBINED_MODEL_NAME
    assert len(result.forecast) == HORIZON

    record = _adjudication_record(artifact)
    payload = record.payload
    assert payload["winner"] == "combined"
    assert payload["selected_model"] == COMBINED_MODEL_NAME
    assert payload["covariates_fingerprint"].startswith("sha256:")
    names = [candidate["name"] for candidate in payload["candidates"]]
    assert names == ["base", "context", "covariates", "combined"]
    # Fold-identical comparison: every scored candidate has one score per
    # recorded selection cutoff.
    assert payload["fold_cutoffs"]
    for candidate in payload["candidates"]:
        assert candidate["excluded_reason"] is None
        assert len(candidate["fold_scores"]) == len(payload["fold_cutoffs"])
    means = {candidate["name"]: candidate["mean_score"] for candidate in payload["candidates"]}
    assert means["combined"] <= min(means.values())
    # The record survives into the typed lineage without extra wiring.
    assert "enrichment_adjudication" in (path / "lineage.json").read_text()


def test_base_is_retained_when_no_challenger_clears_its_gate(tmp_path) -> None:
    observations = _write_observations(
        tmp_path, promo_effect=PROMO_EFFECT, covariate_effect=COVARIATE_EFFECT,
    )
    artifact, _ = _run(
        tmp_path, observations,
        events=_events(sourced=False),
        dataset=_write_covariates(tmp_path, column=lambda index: 0.0),
    )
    result = artifact.results[0]
    assert result.context["admitted"] is False
    assert result.covariates["admitted"] is False
    payload = _adjudication_record(artifact).payload
    assert payload["winner"] == "base"
    assert "gate" in payload["reason"]
    assert result.selected_model == payload["selected_model"]
    assert result.selected_model not in (COMBINED_MODEL_NAME, "event_adjusted", "covariate_linear")


def test_single_admitted_challenger_matches_the_single_enrichment_run(tmp_path) -> None:
    """With no promo effect in the data, only covariates are admitted; the
    ladder must reproduce the covariate-only run exactly (apples-to-apples)."""
    observations = _write_observations(
        tmp_path, promo_effect=0.0, covariate_effect=COVARIATE_EFFECT,
    )
    combined_artifact, _ = _run(
        tmp_path, observations, events=_events(), dataset=_write_covariates(tmp_path),
    )
    single_artifact, _ = forecast(
        str(observations), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "single"),
        covariates=_write_covariates(tmp_path), seasonal_period=7,
    )
    combined_result = combined_artifact.results[0]
    payload = _adjudication_record(combined_artifact).payload
    assert combined_result.context["admitted"] is False
    assert payload["winner"] == "covariates"
    assert combined_result.selected_model == "covariate_linear"
    assert combined_result.forecast == single_artifact.results[0].forecast
    assert combined_result.support == single_artifact.results[0].support


def test_redundant_enrichments_are_not_double_counted(tmp_path) -> None:
    """When the covariate column and the events encode the same promo signal,
    stacking both would double-apply the effect; the ladder must prefer a
    single-enrichment candidate."""
    observations = _write_observations(
        tmp_path, promo_effect=PROMO_EFFECT, covariate_effect=0.0,
    )
    artifact, _ = _run(
        tmp_path, observations, events=_events(),
        dataset=_write_covariates(tmp_path, column=lambda index: 1.0 if _promo(index) else 0.0),
    )
    payload = _adjudication_record(artifact).payload
    assert payload["winner"] != "combined"
    assert artifact.results[0].selected_model != COMBINED_MODEL_NAME


def test_single_enrichment_runs_do_not_adjudicate(tmp_path) -> None:
    observations = _write_observations(
        tmp_path, promo_effect=0.0, covariate_effect=COVARIATE_EFFECT,
    )
    artifact, _ = forecast(
        str(observations), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "out"),
        covariates=_write_covariates(tmp_path), seasonal_period=7,
    )
    assert not any(item.kind == "enrichment_adjudication" for item in artifact.evidence)
    assert artifact.results[0].selected_model == "covariate_linear"


def test_ranking_tie_breaks_are_deterministic() -> None:
    base = Candidate("base", "drift", [], mean_score=0.5)
    context = Candidate("context", "event_adjusted", ["context_events"], mean_score=0.5)
    covariates = Candidate("covariates", "covariate_linear", ["covariates"], mean_score=0.5)
    combined = Candidate("combined", COMBINED_MODEL_NAME, ["context_events", "covariates"], mean_score=0.5)
    unscored = Candidate("context", "event_adjusted", ["context_events"], excluded_reason="failed")
    # Equal scores: fewest enrichments wins, then fixed assembly order.
    ranked = rank_candidates([base, context, covariates, combined, unscored])
    assert [candidate.name for candidate in ranked] == ["base", "context", "covariates", "combined"]
    # A better score beats fewer enrichments.
    combined_better = Candidate("combined", COMBINED_MODEL_NAME, ["context_events", "covariates"], mean_score=0.4)
    assert rank_candidates([base, context, combined_better])[0] is combined_better


def test_capabilities_flag_enrichment_adjudication() -> None:
    assert capabilities()["features"]["enrichment_adjudication"] is True


def test_combined_forecast_carries_both_effects(tmp_path) -> None:
    """The winning combined forecast reflects the promo bump on the promo day
    inside the horizon, on top of the covariate-driven level."""
    observations = _write_observations(
        tmp_path, promo_effect=PROMO_EFFECT, covariate_effect=COVARIATE_EFFECT,
    )
    artifact, _ = _run(tmp_path, observations, events=_events(), dataset=_write_covariates(tmp_path))
    result = artifact.results[0]
    assert result.selected_model == COMBINED_MODEL_NAME
    # Day 135 (index 5 of the horizon) is a promo day with covariate off;
    # day 134 is a non-promo day with covariate on. The forecast gap must
    # reflect both learned effects: roughly +PROMO_EFFECT - COVARIATE_EFFECT.
    promo_row = result.forecast[5]
    plain_row = result.forecast[4]
    assert promo_row["point"] - plain_row["point"] > (PROMO_EFFECT - COVARIATE_EFFECT) / 2
