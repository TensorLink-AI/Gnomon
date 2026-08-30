"""The cross-series (VAR) candidate must be admitted like everything else.

Before this, `multivariate_stage` overwrote the univariate forecast outright:
no fold comparison against the models it displaced, no baseline, no evidence
record, and its own validation ran on a trailing window overlapping the
report-only test fold. It was the one path that could publish a number the
admission philosophy never examined. These tests pin the replacement.
"""

import csv
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, "src")

import pytest  # noqa: E402

from gnomon.forecast_adapter import (  # noqa: E402
    ForecastAdapterError, ForecastRequest, conformance_report,
)
from gnomon.multivariate import (  # noqa: E402
    MINIMUM_CORRELATION, VarForecastAdapter, VarFrame, _predict,
)
from gnomon.runtime import forecast  # noqa: E402


class _Item:
    __slots__ = ("timestamp", "value")

    def __init__(self, timestamp, value):
        self.timestamp = timestamp
        self.value = value


def _groups(correlated: bool = True, count: int = 200):
    start = datetime(2024, 1, 1)
    stamps = [start + timedelta(hours=index) for index in range(count)]
    lead = [100 + 10 * math.sin(i / 6) + 0.3 * i for i in range(count)]
    if correlated:
        follow = [0.8 * value + 5 * math.cos(i / 6) + 20
                  for i, value in enumerate(lead)]
    else:
        follow = [50 + ((i * 37) % 11) for i in range(count)]
    return {
        "a": [_Item(t, v) for t, v in zip(stamps, lead)],
        "b": [_Item(t, v) for t, v in zip(stamps, follow)],
    }


def _write_csv(path: Path, groups) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "series", "value"])
        for name, items in sorted(groups.items()):
            for item in items:
                writer.writerow([item.timestamp.isoformat(), name, item.value])


def _gate_records(artifact):
    return {evidence.series: evidence.payload
            for evidence in artifact.evidence
            if evidence.kind == "multivariate_gate"}


class TestVarFrame:
    def test_unaligned_series_are_ineligible(self):
        groups = _groups()
        groups["b"] = groups["b"][:-1]
        frame, reason = VarFrame.build(groups)
        assert frame is None
        assert "identical timestamps" in reason

    def test_uncorrelated_series_are_ineligible(self):
        frame, reason = VarFrame.build(_groups(correlated=False))
        assert frame is None
        assert str(MINIMUM_CORRELATION) in reason

    def test_predictor_refits_at_each_origin(self):
        """A fold forecast must not depend on data after that fold's origin."""
        frame, reason = VarFrame.build(_groups())
        assert frame is not None, reason
        predict_at = frame.predictor("a")
        early = predict_at(100, 12)
        assert len(early) == 12

        # Corrupting history *after* the origin must not change the forecast
        # made at that origin.
        truncated = {name: items[:100] for name, items in _groups().items()}
        for item in truncated["a"]:
            pass
        padded = _groups()
        for item in padded["a"][100:]:
            item.value = 1e6
        padded_frame, _ = VarFrame.build(padded)
        assert padded_frame.predictor("a")(100, 12) == pytest.approx(early)

    def test_adapter_proves_related_series_contract(self):
        adapter = VarForecastAdapter()
        history = list(range(20))
        related = [list(range(1, 21))]
        report = conformance_report(
            adapter, history=history, related_series=related)
        assert report["conformant"] is True
        assert report["capabilities"]["panel"] is True
        request = ForecastRequest(
            tuple(history), 4,
            related_series=(tuple(related[0]),),
        )
        result = adapter.forecast(request)
        assert len(result.point) == 4
        assert result.metadata["features_used"]["related_series"] == 1
        with pytest.raises(ForecastAdapterError, match="related series"):
            adapter.forecast(ForecastRequest(tuple(range(20)), 4))
        with pytest.raises(ForecastAdapterError, match="history-aligned"):
            ForecastRequest(
                tuple(range(20)), 4,
                related_series=(tuple(range(19)),),
            )

    def test_adapter_refactor_preserves_original_var_numbers(self):
        groups = _groups()
        columns = [[item.value for item in groups[name]] for name in ("a", "b")]
        expected = _predict([column[:100] for column in columns], 12)
        frame, reason = VarFrame.build(groups)
        assert frame is not None, reason
        assert frame.predictor("a")(100, 12) == pytest.approx(
            expected[0], abs=1e-9)
        assert frame.predictor("b")(100, 12) == pytest.approx(
            expected[1], abs=1e-9)


class TestMultivariateAdmission:
    def test_var_competes_on_the_same_folds_and_is_recorded(self, tmp_path):
        path = tmp_path / "series.csv"
        _write_csv(path, _groups())
        artifact, _ = forecast(
            str(path), time_column="timestamp", target_column="value",
            series_column="series", horizon=12, multivariate=True,
            output=str(tmp_path / "out"),
        )
        records = _gate_records(artifact)
        assert set(records) == {"a", "b"}
        for result in artifact.results:
            record = records[result.series]
            # The gate's verdict and the selection must agree.
            assert record["admitted"] == (result.selected_model == "var")
            # VAR is scored on the same folds as every other candidate.
            assert result.selection_scores.get("var") is not None
            codes = [check["code"] for check in record["checks"]]
            assert "var_completed_every_fold" in codes
            assert "var_is_the_best_candidate" in codes
            assert record["target_series"] == result.series
            assert record["retained_related_series"] == [
                name for name in ("a", "b") if name != result.series
            ]
            assert record["supplied_related_series"] == [
                name for name in ("a", "b") if name != result.series
            ]
            assert record["adapter"] == "var"
            assert record["adapter_kind"] == "statistical"
            assert record["adapter_protocol_version"] == "0.1"

    def test_var_is_decided_per_series_not_imposed_on_all(self, tmp_path):
        """A VAR that helps one series must not be forced onto the other."""
        path = tmp_path / "series.csv"
        _write_csv(path, _groups(count=160))
        artifact, _ = forecast(
            str(path), time_column="timestamp", target_column="value",
            series_column="series", horizon=12, multivariate=True,
            output=str(tmp_path / "out"),
        )
        selected = {result.series: result.selected_model
                    for result in artifact.results}
        assert selected["a"] == "var", selected
        assert selected["b"] != "var", (
            "series b's univariate candidate scored better; VAR must not be "
            "imposed on it just because it won elsewhere"
        )

    def test_admitted_var_carries_its_own_intervals(self, tmp_path):
        path = tmp_path / "series.csv"
        _write_csv(path, _groups(count=160))
        artifact, _ = forecast(
            str(path), time_column="timestamp", target_column="value",
            series_column="series", horizon=12, multivariate=True,
            output=str(tmp_path / "out"),
        )
        result = next(r for r in artifact.results if r.selected_model == "var")
        assert result.forecast
        assert all(row["q10"] <= row["q50"] <= row["q90"]
                   for row in result.forecast)
        confirmation = result.support_assessment[
            "sensitivity"]["selection_stability"]["confirmation"]
        assert confirmation["candidate"] == "var"
        assert confirmation["passed"] is True
        assert result.test_scores["var"] is not None

    def test_var_confirmation_veto_is_named_in_gate_evidence(self, tmp_path):
        path = tmp_path / "series.csv"
        _write_csv(path, _groups())
        artifact, _ = forecast(
            str(path), time_column="timestamp", target_column="value",
            series_column="series", horizon=12, multivariate=True,
            output=str(tmp_path / "out"),
        )
        result = next(r for r in artifact.results if r.series == "a")
        confirmation = result.support_assessment[
            "sensitivity"]["selection_stability"]["confirmation"]
        record = _gate_records(artifact)["a"]
        checks = {check["code"]: check for check in record["checks"]}

        assert confirmation["candidate"] == "var"
        assert confirmation["passed"] is False
        assert confirmation["fallback_applied"] is True
        assert result.selected_model == confirmation["baseline"]
        assert checks["var_is_the_best_candidate"]["passed"] is True
        assert checks["var_passed_independent_confirmation"]["passed"] is False
        assert record["admitted"] is False
        assert record["decided_by"] == "var_passed_independent_confirmation"

    def test_ineligible_frame_is_disclosed_rather_than_silent(self, tmp_path):
        path = tmp_path / "series.csv"
        _write_csv(path, _groups(correlated=False))
        artifact, _ = forecast(
            str(path), time_column="timestamp", target_column="value",
            series_column="series", horizon=12, multivariate=True,
            output=str(tmp_path / "out"),
        )
        records = _gate_records(artifact)
        assert records, "asking for --multivariate must leave a record either way"
        for payload in records.values():
            assert payload["admitted"] is False
            assert payload["decided_by"] == "series_eligible_for_var"
            assert len(payload["supplied_related_series"]) == 1
            assert payload["retained_related_series"] == []

    def test_no_gate_record_without_the_flag(self, tmp_path):
        path = tmp_path / "series.csv"
        _write_csv(path, _groups())
        artifact, _ = forecast(
            str(path), time_column="timestamp", target_column="value",
            series_column="series", horizon=12,
            output=str(tmp_path / "out"),
        )
        assert _gate_records(artifact) == {}
        assert all(result.selected_model != "var" for result in artifact.results)
