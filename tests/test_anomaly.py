"""Evaluated anomaly detection: detectors, grader, macro, determinism."""

from __future__ import annotations

import json
import math

import pytest

from gnomon.anomaly import (
    DEFAULT_THRESHOLD,
    DETECTORS,
    MIN_DETECTION_HISTORY,
    detect_anomalies,
    forecast_interval_scores,
    grade_detectors,
    local_slope_scores,
    robust_zscore_scores,
    rolling_median_scores,
)


def _seasonal_series(n: int = 96, season: int = 12) -> list[float]:
    return [10.0 + 3.0 * math.sin(2 * math.pi * (i % season) / season)
            + 0.15 * ((i * 37) % 11 - 5) for i in range(n)]


def _trending_series(n: int = 80) -> list[float]:
    return [5.0 + 0.4 * i + 0.2 * ((i * 29) % 7 - 3) for i in range(n)]


def _daily_csv(values: list[float]) -> str:
    from datetime import date, timedelta
    start = date(2025, 1, 1)
    lines = ["timestamp,value"]
    for i, value in enumerate(values):
        lines.append(f"{(start + timedelta(days=i)).isoformat()},{value}")
    return "\n".join(lines)


class TestDetectors:
    def test_robust_zscore_flags_planted_spike(self):
        values = _seasonal_series()
        values[50] += 25.0
        scores = robust_zscore_scores(values, 12)
        assert abs(scores[50]) >= DEFAULT_THRESHOLD

    def test_rolling_median_handles_trend(self):
        values = _trending_series()
        values[40] += 15.0
        scores = rolling_median_scores(values, 1)
        assert abs(scores[40]) >= DEFAULT_THRESHOLD
        flagged = [i for i, s in enumerate(scores) if abs(s) >= DEFAULT_THRESHOLD]
        assert flagged == [40]

    def test_forecast_interval_scores_alignment_and_warmup(self):
        values = _seasonal_series()
        scores = forecast_interval_scores(values, 12)
        assert len(scores) == len(values)
        assert all(score == 0.0 for score in scores[:8])

    def test_detectors_are_deterministic(self):
        values = _seasonal_series()
        for score_function in DETECTORS.values():
            assert score_function(values, 12) == score_function(values, 12)


class TestGrader:
    def test_grades_every_candidate(self):
        grading = grade_detectors(_seasonal_series(), 12)
        assert set(grading["grades"]) == set(DETECTORS)
        for grade in grading["grades"].values():
            assert 0.0 <= grade["f1"] <= 1.0
            assert grade["trials"] > 0

    def test_grading_is_deterministic(self):
        values = _seasonal_series()
        assert grade_detectors(values, 12) == grade_detectors(values, 12)

    def test_seed_derives_from_content(self):
        a = grade_detectors(_seasonal_series(), 12)
        b = grade_detectors(_trending_series(), 1)
        assert a["injection"]["seed"] != b["injection"]["seed"]

    def test_clean_series_flags_are_excluded(self):
        values = _seasonal_series()
        values[30] += 40.0  # pre-existing anomaly must not count as FP or hit
        grading = grade_detectors(values, 12)
        best = max(grade["f1"] for grade in grading["grades"].values())
        assert best > 0.0


class TestDetectAnomalies:
    def test_abstains_below_minimum_history(self):
        timestamps = [f"t{i}" for i in range(MIN_DETECTION_HISTORY - 1)]
        result = detect_anomalies(timestamps, [1.0] * (MIN_DETECTION_HISTORY - 1))
        assert result["detector"] is None
        assert result["support"]["status"] == "inconclusive"

    def test_finds_planted_spike_with_disclosure(self):
        values = _seasonal_series()
        values[60] += 25.0
        timestamps = [f"t{i}" for i in range(len(values))]
        result = detect_anomalies(timestamps, values, season=12)
        assert result["detector"] in DETECTORS
        assert result["selection_basis"] == "synthetic_injection_macro_f1"
        assert any(item["timestamp"] == "t60" for item in result["anomalies"])
        assert set(result["detector_grades"]) == set(DETECTORS)
        assert result["support"]["status"] in ("supported", "conditionally_supported")

    def test_no_anomalies_is_a_supported_conclusion(self):
        values = _seasonal_series()
        timestamps = [f"t{i}" for i in range(len(values))]
        result = detect_anomalies(timestamps, values, season=12)
        if not result["anomalies"] and result["support"]["status"] == "supported":
            reasons = [r["code"] for r in result["support"]["reasons"]]
            assert "no_anomalies_detected" in reasons

    def test_label_selection_basis(self):
        values = _seasonal_series()
        values[45] += 25.0
        timestamps = [f"t{i}" for i in range(len(values))]
        result = detect_anomalies(timestamps, values, season=12, label_indices=[45])
        assert result["selection_basis"] == "label_f1"
        assert "label_grades" in result
        assert result["label_grades"][result["detector"]]["recall"] == 1.0

    def test_extra_detector_can_win(self):
        values = _seasonal_series()
        values[70] += 25.0
        timestamps = [f"t{i}" for i in range(len(values))]
        oracle = lambda vals, season: [  # noqa: E731
            10.0 if i == 70 else 0.0 for i in range(len(vals))
        ]
        result = detect_anomalies(
            timestamps, values, season=12,
            extra_detectors={"oracle": oracle},
        )
        assert "oracle" in result["detector_grades"]

    def test_output_is_deterministic(self):
        values = _seasonal_series()
        values[33] -= 22.0
        timestamps = [f"t{i}" for i in range(len(values))]
        first = detect_anomalies(timestamps, values, season=12)
        second = detect_anomalies(timestamps, values, season=12)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


class TestRegistryAndMacro:
    def test_operator_is_registered(self):
        from gnomon.registry import OPERATORS
        spec = OPERATORS["detect_anomalies"]
        assert spec.seeded is True
        assert spec.deterministic is True
        assert spec.claim_classes == ("descriptive",)
        assert spec.runner is detect_anomalies

    def test_macro_is_registered_with_schema(self):
        from gnomon.registry import MACROS
        spec = MACROS["detect_anomalies"]
        assert spec.tool_name == "gnomon_detect_anomalies"
        assert "threshold" in spec.input_schema["properties"]
        assert "labels" in spec.input_schema["properties"]

    def test_capabilities_flag(self):
        from gnomon.runtime import capabilities
        payload = capabilities()
        assert payload["features"]["anomaly_detection"] is True
        assert payload["features"]["graded_detector_selection"] is True
        assert "detect_anomalies" in payload["operators"]

    def test_macro_end_to_end(self, tmp_path):
        values = _seasonal_series(60, 12)
        values[40] += 25.0
        csv_path = tmp_path / "series.csv"
        csv_path.write_text(_daily_csv(values), encoding="utf-8")
        from gnomon.macros import detect_anomalies as macro
        payload, artifact_path = macro(
            str(csv_path), time_column="timestamp", target_column="value",
            output=str(tmp_path / "out"),
        )
        assert payload["status"] == "complete"
        assert payload["results"][0]["detector"] is not None
        assert artifact_path.exists()
        lineage = json.loads((artifact_path / "lineage.json").read_text())
        assert any(claim["claim_class"] == "descriptive" for claim in lineage["claims"])

    def test_cli_detect(self, tmp_path, capsys):
        values = _seasonal_series(60, 12)
        csv_path = tmp_path / "series.csv"
        csv_path.write_text(_daily_csv(values), encoding="utf-8")
        from gnomon.cli import main
        code = main([
            "detect", str(csv_path), "--time", "timestamp",
            "--target", "value", "--output", str(tmp_path / "out"),
        ])
        assert code == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["threshold"] == pytest.approx(DEFAULT_THRESHOLD)


def test_local_slope_detector_finds_a_trend_anomaly():
    # A flat series that drifts at the wrong rate for a stretch: every
    # value stays in range, so only slope reveals it.
    values = [10.0 + (i % 3) * 0.1 for i in range(60)]
    for offset, index in enumerate(range(25, 40)):
        values[index] += 0.6 * (offset + 1)
    for index in range(40, 60):
        values[index] += 0.6 * 15
    scores = local_slope_scores(values, 1)
    flagged = [i for i, s in enumerate(scores) if abs(s) >= DEFAULT_THRESHOLD]
    assert flagged, "slope change went unflagged"
    assert all(24 <= i <= 41 for i in flagged), flagged


def test_grader_plants_and_discloses_trend_shifts():
    values = [10.0 + (i % 5) * 0.2 for i in range(80)]
    grading = grade_detectors(values, 1)
    assert "trend_shift" in grading["injection"]["families"]
    assert "trend_shift" in grading["injection"]["families_planted"]
    assert "local_slope" in grading["grades"]


def test_support_names_the_families_the_grade_covers():
    values = [10.0 + (i % 7) * 0.3 for i in range(80)]
    result = detect_anomalies([str(i) for i in range(80)], values)
    support = result["support"]
    families = support["sensitivity"]["graded_families"]
    assert "trend_shift" in families
    assert any("not vouched for" in assumption
               for assumption in support["assumptions"]), support["assumptions"]


def _anomllm_trend_series(n: int = 1000) -> list[float]:
    """Deterministic AnomLLM-shaped trend series: a period-50 sine on a
    piecewise trend (slope 3/n normally, 10/n in the anomaly window),
    normalised to [-1, 1] like the benchmark's generator."""
    values, level = [], 0.0
    for step in range(n):
        slope = 10.0 if 600 <= step < 700 else 3.0
        level += slope / n
        values.append(math.sin(2 * math.pi * 0.02 * step) + level)
    low, high = min(values), max(values)
    return [2 * (value - low) / (high - low) - 1 for value in values]


def test_seasonal_structure_is_not_reported_as_anomalies():
    """With the season misdetected (24 against a true period of 50),
    rolling_median_residual grades best on injections while flagging
    ~half the observed series — its clean-series flags are excluded from
    injection scoring, so density was invisible to selection. A dense
    candidate must be disqualified, not crowned."""
    values = _anomllm_trend_series()
    result = detect_anomalies([str(i) for i in range(len(values))], values,
                              season=24)
    grades = result["detector_grades"]
    assert grades["rolling_median_residual"]["dense_on_observed"] is True
    assert grades["rolling_median_residual"]["clean_flag_fraction"] > 0.4
    assert result["detector"] != "rolling_median_residual"
    assert len(result["anomalies"]) < 100


def test_grades_record_clean_series_flag_density():
    values = _seasonal_series()
    grading = grade_detectors(values, 12)
    for grade in grading["grades"].values():
        assert 0.0 <= grade["clean_flag_fraction"] <= 1.0
        assert grade["dense_on_observed"] == (
            grade["clean_flag_fraction"] > 0.2
        )
    assert any(not grade["dense_on_observed"]
               for grade in grading["grades"].values())


def test_all_candidates_dense_abstains_instead_of_reporting_noise(monkeypatch):
    import gnomon.anomaly as anomaly_module

    def everything(values, season):
        return [10.0] * len(values)

    monkeypatch.setattr(anomaly_module, "DETECTORS",
                        {"everything": everything})
    values = _seasonal_series()
    result = detect_anomalies([str(i) for i in range(len(values))], values,
                              season=12)
    assert result["detector"] is None
    assert result["anomalies"] == []
    assert result["support"]["status"] == "inconclusive"
    assert result["support"]["reasons"][0]["code"] == "no_plausible_detector"
    assert result["detector_grades"]["everything"]["dense_on_observed"] is True


def test_labelled_selection_keeps_dense_candidates_eligible():
    """User labels measure real-series precision flag by flag, so density
    needs no separate guard there — and a dense detector simply loses on
    label F1."""
    values = _seasonal_series()
    result = detect_anomalies(
        [str(i) for i in range(len(values))], values, season=12,
        label_indices=[40],
        extra_detectors={"everything": lambda v, s: [10.0] * len(v)},
    )
    assert result["selection_basis"] == "label_f1"
    assert result["detector"] != "everything"


class TestGradingWindow:
    """Long series are graded on a trailing window; short series in full."""

    def test_short_series_grades_in_full(self):
        grading = grade_detectors(_seasonal_series(96, 12), 12)
        assert "grading_window" not in grading["injection"]

    def test_long_series_grades_on_trailing_window(self, monkeypatch):
        import gnomon.anomaly as anomaly
        monkeypatch.setattr(anomaly, "MAX_GRADING_HISTORY", 64)
        values = _seasonal_series(160, 12)
        grading = grade_detectors(values, 12)
        assert grading["injection"]["grading_window"] == {
            "observations": 64, "of": 160, "placement": "trailing",
        }
        # Windowed grading is exactly grading of the trailing slice: same
        # seed, same trials, same grades.
        trailing = grade_detectors(values[-64:], 12)
        assert "grading_window" not in trailing["injection"]
        assert trailing["grades"] == grading["grades"]
        assert trailing["injection"]["seed"] == grading["injection"]["seed"]

    def test_window_stretches_to_cover_long_seasons(self, monkeypatch):
        import gnomon.anomaly as anomaly
        monkeypatch.setattr(anomaly, "MAX_GRADING_HISTORY", 64)
        # 4 seasons of 30 > 64: the window must span whole cycles.
        values = _seasonal_series(300, 30)
        grading = grade_detectors(values, 30)
        assert grading["injection"]["grading_window"]["observations"] == 120

    def test_selected_detector_scores_full_series_with_disclosure(self, monkeypatch):
        import gnomon.anomaly as anomaly
        monkeypatch.setattr(anomaly, "MAX_GRADING_HISTORY", 64)
        values = _seasonal_series(200, 12)
        values[30] += 40.0  # planted well before the grading window
        result = detect_anomalies(
            [str(i) for i in range(len(values))], values, season=12,
        )
        assert len(result["scores"]) == len(values)
        assert any(item["timestamp"] == "30" for item in result["anomalies"])
        assert any("trailing 64 of 200" in note
                   for note in result["support"]["assumptions"])

    def test_no_disclosure_without_window(self):
        values = _seasonal_series(96, 12)
        values[40] += 25.0
        result = detect_anomalies(
            [str(i) for i in range(len(values))], values, season=12,
        )
        assert not any("trailing" in note
                       for note in result["support"]["assumptions"])
