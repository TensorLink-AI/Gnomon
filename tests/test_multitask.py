"""Phase 3 multi-task seams: tasks capability, verbs, reconstruction detector."""

from __future__ import annotations

import math

import pytest

from gnomon.tsfm import TSFMCapabilities, eligible_tsfms, tsfm_capabilities
from gnomon.tsfm_sandbox import SubprocessAdapter
from gnomon.tsfm import TSFMError


def _series(n: int = 64) -> list[float]:
    return [10.0 + 2.0 * math.sin(2 * math.pi * i / 8) for i in range(n)]


class TestTasksCapability:
    def test_default_is_forecast_only(self):
        assert TSFMCapabilities().tasks == ("forecast",)
        assert tsfm_capabilities("chronos_bolt_mini").tasks == ("forecast",)

    def test_moment_declares_multitask(self):
        tasks = tsfm_capabilities("moment_small").tasks
        assert set(tasks) == {"forecast", "detect_anomalies", "impute", "embed"}

    def test_tasks_in_capability_matrix(self):
        from gnomon.tsfm import capability_matrix
        matrix = capability_matrix()
        assert matrix["moment_small"]["tasks"] == ["forecast", "detect_anomalies", "impute", "embed"] or \
            tuple(matrix["moment_small"]["tasks"]) == ("forecast", "detect_anomalies", "impute", "embed")

    def test_eligible_tsfms_filters_by_task(self):
        eligible, excluded = eligible_tsfms(
            history_length=100, horizon=5, frequency="D", task="detect_anomalies",
        )
        assert eligible == ["moment_small"]
        assert any("does not implement task" in reason
                   for reasons in excluded.values() for reason in reasons)

    def test_eligible_tsfms_default_task_unchanged(self):
        eligible, _ = eligible_tsfms(history_length=100, horizon=5, frequency="D")
        assert "chronos_bolt_mini" in eligible


class TestSubprocessVerbs:
    def test_reconstruct_gated_by_verified_task(self):
        adapter = SubprocessAdapter("chronos_bolt_mini")
        with pytest.raises(TSFMError, match="does not implement task"):
            adapter.reconstruct(_series())

    def test_embed_gated_by_verified_task(self):
        adapter = SubprocessAdapter("ttm")
        with pytest.raises(TSFMError, match="does not implement task"):
            adapter.embed(_series())

    def test_worker_script_carries_new_verbs(self):
        from gnomon.tsfm_sandbox import WORKER_SCRIPT
        assert "def run_reconstruct" in WORKER_SCRIPT
        assert "def run_embed" in WORKER_SCRIPT
        assert 'request.get("mode", "predict")' in WORKER_SCRIPT

    def test_stale_worker_script_is_refreshed(self, tmp_path):
        from gnomon.tsfm_sandbox import WORKER_SCRIPT, _ensure_worker_script
        stale = tmp_path / "worker.py"
        stale.write_text("# old worker without verbs")
        path = _ensure_worker_script(tmp_path)
        assert path.read_text() == WORKER_SCRIPT


class FakeReconstructionAdapter:
    """Reconstructs a clean sine — residuals expose the planted spike."""

    def reconstruct(self, history, mask=None):
        return [10.0 + 2.0 * math.sin(2 * math.pi * i / 8)
                for i in range(len(history))]


class TestReconstructionDetector:
    def test_scores_flag_the_planted_spike(self):
        from gnomon.anomaly import DEFAULT_THRESHOLD, _reconstruction_score_function
        values = _series()
        values[40] += 20.0
        scores = _reconstruction_score_function(FakeReconstructionAdapter())(values, 8)
        assert len(scores) == len(values)
        assert abs(scores[40]) >= DEFAULT_THRESHOLD

    def test_detector_can_join_and_win(self):
        from gnomon.anomaly import detect_anomalies
        values = _series()
        values[40] += 20.0
        timestamps = [f"t{i}" for i in range(len(values))]
        extra = {"fake_reconstruction": _score_via_fake()}
        result = detect_anomalies(timestamps, values, season=8, extra_detectors=extra)
        assert "fake_reconstruction" in result["detector_grades"]

    def test_failing_detector_scores_zero_with_disclosure(self):
        from gnomon.anomaly import detect_anomalies

        def broken(values, season):
            raise RuntimeError("sandbox unavailable")

        values = _series()
        values[40] += 20.0
        timestamps = [f"t{i}" for i in range(len(values))]
        result = detect_anomalies(timestamps, values, season=8,
                                  extra_detectors={"broken": broken})
        assert result["detector_grades"]["broken"]["f1"] == 0.0
        assert "error" in result["detector_grades"]["broken"]
        assert result["detector"] != "broken"

    def test_no_sandboxes_means_no_tsfm_candidates(self, monkeypatch, tmp_path):
        import gnomon.tsfm_sandbox as sandbox_module
        monkeypatch.setattr(sandbox_module, "SANDBOX_ROOT", tmp_path)
        from gnomon.anomaly import tsfm_reconstruction_detectors
        assert tsfm_reconstruction_detectors() == {}


def _score_via_fake():
    from gnomon.anomaly import _reconstruction_score_function
    return _reconstruction_score_function(FakeReconstructionAdapter())
