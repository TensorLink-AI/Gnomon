from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

from aion.contracts import AionError
from aion.covariates import (
    assess_covariates,
    covariate_guide,
    load_covariates,
    parse_mapping,
    validate_covariate_file,
)
from aion.evaluation import evaluate
from aion.runtime import capabilities, forecast
from aion.tsfm import capability_matrix, eligible_tsfms


START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _signal(index: int) -> float:
    # Deterministic but deliberately not periodic at the daily season of 7.
    return 1.0 if ((index * 17 + 3) % 11) < 5 else 0.0


def _write_files(tmp_path, count: int = 100, horizon: int = 7):
    observations = tmp_path / "observations.csv"
    with observations.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for index in range(count):
            timestamp = START + timedelta(days=index)
            writer.writerow([timestamp.isoformat(), 50 + index * 0.2 + 20 * _signal(index)])

    covariates = tmp_path / "covariates.csv"
    with covariates.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "known_at", "campaign"])
        for index in range(count + horizon):
            timestamp = START + timedelta(days=index)
            writer.writerow([
                timestamp.isoformat(),
                (START - timedelta(days=30)).isoformat(),
                _signal(index),
            ])
    return observations, covariates


def test_mapping_requires_explicit_future_known() -> None:
    with pytest.raises(AionError) as exc:
        parse_mapping("temperature:continuous")
    assert exc.value.code == "INVALID_COVARIATE_MAPPING"
    with pytest.raises(AionError) as exc:
        parse_mapping("temperature:continuous:observed")
    assert exc.value.code == "UNSUPPORTED_COVARIATE_AVAILABILITY"


def test_loader_selects_latest_vintage_known_at_cutoff(tmp_path) -> None:
    path = tmp_path / "vintages.csv"
    valid = START + timedelta(days=10)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "known_at", "weather"])
        writer.writerow([valid.isoformat(), START.isoformat(), 10])
        writer.writerow([valid.isoformat(), (START + timedelta(days=5)).isoformat(), 20])
    dataset = load_covariates(str(path), "weather:continuous:future_known")
    assert dataset.value_at("weather", "__default__", valid, START + timedelta(days=2)) == 10
    assert dataset.value_at("weather", "__default__", valid, START + timedelta(days=6)) == 20


def test_guide_and_validation_report_point_in_time_constraints(tmp_path) -> None:
    observations, covariates = _write_files(tmp_path)
    guide = covariate_guide(
        str(observations), time_column="timestamp", target_column="value",
        horizon=7, frequency="D",
    )
    assert guide["series"][0]["selection_fold_cutoffs"]
    assert "known_at" in guide["format"]["critical_rule"]
    result = validate_covariate_file(
        str(observations), str(covariates), "campaign:binary:future_known",
        time_column="timestamp", target_column="value", horizon=7, frequency="D",
    )
    assert result["valid"] is True


def test_validation_rejects_reconstructed_future_values(tmp_path) -> None:
    observations, covariates = _write_files(tmp_path)
    # Rewrite every vintage as if it only became known after the dataset ended.
    rows = list(csv.DictReader(covariates.open()))
    with covariates.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "known_at", "campaign"])
        writer.writeheader()
        for row in rows:
            row["known_at"] = (START + timedelta(days=200)).isoformat()
            writer.writerow(row)
    result = validate_covariate_file(
        str(observations), str(covariates), "campaign:binary:future_known",
        time_column="timestamp", target_column="value", horizon=7, frequency="D",
    )
    assert result["valid"] is False
    codes = {issue["code"] for issue in result["validation"][0]["issues"]}
    assert "MISSING_HISTORICAL_VINTAGES" in codes
    assert "MISSING_FORECAST_VALUES" in codes


def test_covariate_ablation_admits_stable_signal(tmp_path) -> None:
    observations, covariates_path = _write_files(tmp_path)
    dataset = load_covariates(str(covariates_path), "campaign:binary:future_known")
    values = [50 + index * 0.2 + 20 * _signal(index) for index in range(100)]
    timestamps = [START + timedelta(days=index) for index in range(100)]
    future = [START + timedelta(days=index) for index in range(100, 107)]
    base = evaluate(values, 7, 7, 0.02, frequency="D", tsfm_names=[])
    result = assess_covariates(
        values, timestamps, future, dataset, "__default__", 7, 7, 0.02, base,
    )
    assert result.admitted is True
    assert result.retained == ["campaign"]
    assert len(result.points) == 7


def test_forecast_records_covariate_evidence(tmp_path) -> None:
    observations, covariates_path = _write_files(tmp_path)
    dataset = load_covariates(str(covariates_path), "campaign:binary:future_known")
    artifact, path = forecast(
        str(observations), time_column="timestamp", target_column="value",
        horizon=7, frequency="D", output=str(tmp_path / "out"), covariates=dataset,
    )
    assert path.is_dir()
    assert artifact.results[0].covariates is not None
    assert artifact.results[0].selected_model == "covariate_linear"
    evidence = [item for item in artifact.evidence if item.kind == "covariate_ablation"]
    assert evidence and evidence[0].payload["source_fingerprint"].startswith("sha256:")


def test_capabilities_are_adapter_level_and_machine_actionable() -> None:
    matrix = capability_matrix()
    assert matrix["toto2_22m"]["multivariate_targets"] is True
    assert matrix["toto2_22m"]["future_known_covariates"] is False
    eligible, excluded = eligible_tsfms(
        history_length=100, horizon=7, frequency="D", require_future_covariates=True,
    )
    assert eligible == []
    assert all("future-known" in reasons[0] for reasons in excluded.values())
    assert capabilities()["features"]["point_in_time_covariates"] is True
