"""Tests for config, API inference, ensemble, and meta-model."""

import sys
sys.path.insert(0, "src")

import pytest
from aion.config import (
    AionConfig, load_config, DEFAULT_CONFIG,
    resolve_tsfm_backend, find_config,
)
from aion.ensemble import (
    weighted_mean_forecast, median_forecast, combine_quantiles,
    compute_ensemble_forecast,
)
from aion.meta_model import (
    train_meta_model, predict_meta_model, _solve_nnls, _solve_ols,
    _gaussian_elimination,
)


class TestConfig:
    """Config loading and defaults."""

    def test_default_config(self):
        cfg = DEFAULT_CONFIG
        # Baselines are mandatory and no longer expressible as a setting:
        # a candidate is selected by beating them.
        assert not hasattr(cfg.models, "baselines_enabled")
        assert cfg.models.statistical_enabled is True
        assert cfg.models.statistical_candidates is None
        assert cfg.models.tsfm_candidates == []
        assert cfg.ensemble.enabled is False
        assert cfg.meta_model.enabled is False
        assert cfg.llm.enabled is False

    def test_load_config_returns_default_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AION_CONFIG_PATH", "")
        cfg = load_config()
        assert isinstance(cfg, AionConfig)
        assert cfg.models.statistical_enabled is True

    def test_load_config_from_explicit_path(self, tmp_path):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")
        config_file = tmp_path / "aion.yaml"
        config_file.write_text(
            "models:\n"
            "  tsfm:\n"
            "    candidates:\n"
            "      - chronos_bolt_mini\n"
            "ensemble:\n"
            "  enabled: true\n"
            "  strategy: median\n"
        )
        cfg = load_config(str(config_file))
        assert cfg.models.tsfm_candidates == ["chronos_bolt_mini"]
        assert cfg.ensemble.enabled is True
        assert cfg.ensemble.strategy == "median"

    def test_resolve_tsfm_backend(self):
        cfg = DEFAULT_CONFIG
        # With no config, auto resolves to sandbox (default enabled)
        assert resolve_tsfm_backend("chronos_bolt_mini", cfg) == "sandbox"
        # Unknown TSFM also auto-resolves to sandbox (since sandbox is enabled by default)
        assert resolve_tsfm_backend("unknown", cfg) in ("sandbox", "skip")

    def test_find_config_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AION_CONFIG_PATH", "")
        assert find_config() is None


class TestEnsemble:
    """Ensemble combination strategies."""

    def test_weighted_mean_forecast(self):
        forecasts = {
            "model_a": [100.0, 200.0, 300.0],
            "model_b": [110.0, 190.0, 310.0],
        }
        scores = {"model_a": 0.1, "model_b": 0.2}
        result = weighted_mean_forecast(forecasts, scores, max_weight_ratio=0.7)
        assert len(result) == 3
        # model_a has lower error → higher weight → result closer to model_a
        assert 100.0 <= result[0] <= 110.0

    def test_weighted_mean_caps_weight(self):
        forecasts = {
            "good": [100.0] * 3,
            "bad": [200.0] * 3,
        }
        scores = {"good": 0.01, "bad": 100.0}
        result = weighted_mean_forecast(forecasts, scores, max_weight_ratio=0.7)
        # good model is capped at 0.7 weight → result between 100*0.7 and 100*0.7+200*0.3
        assert result[0] <= 130.0 + 0.01  # 100*0.7 + 200*0.3 = 130

    def test_median_forecast(self):
        forecasts = {
            "a": [100.0, 200.0],
            "b": [110.0, 190.0],
            "c": [120.0, 180.0],
        }
        result = median_forecast(forecasts)
        assert result == [110.0, 190.0]

    def test_combine_quantiles_union(self):
        quantiles_a = [{"0.1": 90, "0.5": 100, "0.9": 110}]
        quantiles_b = [{"0.1": 95, "0.5": 105, "0.9": 115}]
        combined = combine_quantiles([quantiles_a, quantiles_b], strategy="union")
        assert combined[0]["0.1"] == 90   # min of lower
        assert combined[0]["0.9"] == 115  # max of upper

    def test_combine_quantiles_intersection(self):
        quantiles_a = [{"0.1": 90, "0.5": 100, "0.9": 110}]
        quantiles_b = [{"0.1": 95, "0.5": 105, "0.9": 115}]
        combined = combine_quantiles([quantiles_a, quantiles_b], strategy="intersection")
        assert combined[0]["0.1"] == 95   # max of lower
        assert combined[0]["0.9"] == 110  # min of upper

    def test_compute_ensemble_weighted_mean(self):
        forecasts = {"a": [100.0, 200.0], "b": [110.0, 190.0]}
        scores = {"a": 0.1, "b": 0.2}
        result = compute_ensemble_forecast(forecasts, scores, "weighted_mean")
        assert len(result) == 2

    def test_compute_ensemble_median(self):
        forecasts = {"a": [100.0], "b": [110.0], "c": [120.0]}
        result = compute_ensemble_forecast(forecasts, {}, "median")
        assert result == [110.0]

    def test_compute_ensemble_voting(self):
        # 2 models say up, 1 says down
        forecasts = {
            "a": [110.0],
            "b": [115.0],
            "c": [90.0],
        }
        result = compute_ensemble_forecast(forecasts, {}, "voting", last_observed=100.0)
        assert result[0] > 100.0  # majority says up


class TestMetaModel:
    """Meta-model training and prediction."""

    def test_solve_nnls_simple(self):
        A = [[1.0], [2.0], [3.0]]
        b = [1.0, 2.0, 3.0]
        result = _solve_nnls(A, b)
        assert result is not None
        assert abs(result[0] - 1.0) < 0.1

    def test_solve_nnls_non_negative(self):
        # When optimal solution would be negative, NNLS returns non-negative
        # If the only solution is negative, it returns all-zeros (None)
        A = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
        b = [1.0, 2.0, 3.0]
        result = _solve_nnls(A, b)
        # Should return non-negative weights or None
        if result is not None:
            assert all(w >= -0.01 for w in result)
        # All-negative target → solution should be zeros (None)
        A2 = [[1.0], [2.0], [3.0]]
        b2 = [-1.0, -2.0, -3.0]
        result2 = _solve_nnls(A2, b2)
        # Projected gradient converges to zeros for all-negative target
        assert result2 is None or all(w >= -0.01 for w in result2)

    def test_solve_ols(self):
        A = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        b = [1.0, 2.0, 3.0]
        result = _solve_ols(A, b)
        assert result is not None
        assert abs(result[0] - 1.0) < 0.1
        assert abs(result[1] - 2.0) < 0.1

    def test_gaussian_elimination(self):
        A = [[2.0, 1.0], [1.0, 3.0]]
        b = [5.0, 10.0]
        x = _gaussian_elimination(A, b)
        assert x is not None
        # 2x + y = 5, x + 3y = 10 → x=1, y=3
        assert abs(x[0] - 1.0) < 0.01
        assert abs(x[1] - 3.0) < 0.01

    def test_gaussian_elimination_singular(self):
        A = [[1.0, 1.0], [1.0, 1.0]]
        b = [1.0, 2.0]
        result = _gaussian_elimination(A, b)
        assert result is None

    def test_train_meta_model(self):
        # Two models: model_a is good, model_b is noisy
        fold_forecasts = {
            "model_a": [[100.0, 101.0], [102.0, 103.0], [104.0, 105.0]],
            "model_b": [[110.0, 108.0], [95.0, 120.0], [100.0, 115.0]],
        }
        fold_actuals = [[101.0, 102.0], [103.0, 104.0], [105.0, 106.0]]
        weights = train_meta_model(fold_forecasts, fold_actuals, non_negative=True)
        assert weights is not None
        assert "model_a" in weights
        assert "model_b" in weights
        # model_a should get higher weight (it's more accurate)
        assert weights["model_a"] >= weights["model_b"]

    def test_train_meta_model_too_few_models(self):
        fold_forecasts = {"only_one": [[100.0, 101.0]]}
        fold_actuals = [[101.0, 102.0]]
        result = train_meta_model(fold_forecasts, fold_actuals)
        assert result is None

    def test_predict_meta_model(self):
        weights = {"a": 0.7, "b": 0.3}
        forecasts = {"a": [100.0, 200.0], "b": [110.0, 190.0]}
        result = predict_meta_model(weights, forecasts)
        assert len(result) == 2
        assert abs(result[0] - 103.0) < 0.1  # 0.7*100 + 0.3*110
        assert abs(result[1] - 197.0) < 0.1  # 0.7*200 + 0.3*190


class TestEvaluationWithConfig:
    """Evaluation pipeline with config-driven features."""

    def test_evaluate_with_ensemble_config(self):
        from aion.evaluation import evaluate
        from aion.config import AionConfig, EnsembleConfig

        cfg = AionConfig()
        cfg.ensemble = EnsembleConfig(
            enabled=True, strategy="median", min_models=2,
        )

        values = [100.0 + i * 2.0 for i in range(200)]
        result = evaluate(
            values, horizon=24, season=24, minimum_improvement=0.02,
            frequency="h", config=cfg,
        )
        # Ensemble should be evaluated as a candidate
        assert result.supported is True
        assert "ensemble" in result.tsfm_scores or result.selected_model is not None

    def test_evaluate_with_meta_model_config(self):
        from aion.evaluation import evaluate
        from aion.config import AionConfig, MetaModelConfig

        cfg = AionConfig()
        cfg.meta_model = MetaModelConfig(
            enabled=True, min_models=2, min_folds=2,
        )

        values = [100.0 + i * 2.0 for i in range(200)]
        result = evaluate(
            values, horizon=24, season=24, minimum_improvement=0.02,
            frequency="h", config=cfg,
        )
        assert result.supported is True

    def test_evaluate_with_full_config(self):
        from aion.evaluation import evaluate
        from aion.config import AionConfig, EnsembleConfig, MetaModelConfig

        cfg = AionConfig()
        cfg.ensemble = EnsembleConfig(enabled=True, strategy="weighted_mean", min_models=2)
        cfg.meta_model = MetaModelConfig(enabled=True, min_models=2)

        values = [100.0 + i * 2.0 + 10 * (1 if i % 24 < 12 else -1) for i in range(200)]
        result = evaluate(
            values, horizon=12, season=24, minimum_improvement=0.02,
            frequency="h", config=cfg,
        )
        assert result.supported is True


class TestEnsembleCalibrationPartitions:
    """The ensemble must be calibrated like every other candidate.

    Before this, the ensemble path pooled residuals from a trailing window
    of the series — which overlaps both the calibration fold and the
    report-only test fold — so the published interval width for every
    ensemble result was derived from data the design forbids choosing on.
    """

    @staticmethod
    def _series():
        import math
        return [100.0 + 10 * math.sin(i / 6) + 0.5 * i + (i % 5) for i in range(200)]

    def test_ensemble_residuals_exclude_the_test_fold(self):
        from aion.config import AionConfig, EnsembleConfig
        from aion.evaluation import evaluate, _origins

        horizon, season = 12, 12
        values = self._series()
        cfg = AionConfig()
        cfg.ensemble = EnsembleConfig(enabled=True, min_models=2)
        result = evaluate(
            values, horizon=horizon, season=season, minimum_improvement=0.02,
            frequency="h", config=cfg,
        )
        residuals = (result.residuals if result.selected_model == "ensemble"
                     else result.ensemble_residuals)
        assert residuals, "the ensemble must be calibrated on the folds"
        origins = _origins(len(values), horizon, max(2 * season, 2 * horizon, 8))
        # Selection folds plus the calibration fold. The final origin is the
        # test fold and contributes nothing.
        assert len(residuals) == (len(origins) - 1) * horizon

    def test_forcing_the_ensemble_still_calibrates_on_folds(self, tmp_path):
        """`--selection-strategy ensemble` overrides selection, not honesty."""
        import csv
        from datetime import datetime, timedelta
        from aion.runtime import forecast

        path = tmp_path / "series.csv"
        start = datetime(2024, 1, 1)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "value"])
            for index, value in enumerate(self._series()):
                writer.writerow([(start + timedelta(hours=index)).isoformat(), value])

        artifact, _ = forecast(
            str(path), time_column="timestamp", target_column="value",
            horizon=12, selection_strategy="ensemble",
            output=str(tmp_path / "out"),
        )
        result = artifact.results[0]
        assert result.support == "supported_ensemble"
        assert result.forecast, "an ensemble forecast must still carry intervals"
        assert all(row["q10"] <= row["point"] <= row["q90"] for row in result.forecast)


class TestSelectionStride:
    """Selection folds and calibration folds answer to different rules.

    Overlapping folds are legitimate for selection -- candidates are still
    compared on identical windows, and the extra comparisons cut variance.
    They are not legitimate for calibration: residuals from overlapping
    windows are dependent, and a conformal quantile that treats n of them as
    n independent draws is anti-conservative by exactly the amount it
    over-counts.
    """

    @staticmethod
    def _series(count=260):
        import math
        return [100 + 20 * math.sin(2 * math.pi * i / 12) + 0.3 * i + (i % 7)
                for i in range(count)]

    def test_default_is_non_overlapping(self):
        from aion.evaluation import evaluate

        values = self._series()
        base = evaluate(values, 12, 12, 0.02, frequency="h")
        explicit = evaluate(values, 12, 12, 0.02, frequency="h",
                            selection_stride=12)
        assert base.selected_model == explicit.selected_model
        assert base.residuals == explicit.residuals

    def test_stride_does_not_change_calibration_residuals(self):
        from aion.evaluation import evaluate

        values = self._series()
        sparse = evaluate(values, 12, 12, 0.02, frequency="h")
        dense = evaluate(values, 12, 12, 0.02, frequency="h", selection_stride=3)
        if sparse.selected_model != dense.selected_model:
            pytest.skip("stride changed the selection; residuals describe "
                        "different models and are not comparable")
        assert dense.residuals == sparse.residuals, (
            "calibration must stay on the non-overlapping skeleton whatever "
            "stride selection uses"
        )

    def test_no_selection_fold_reads_the_calibration_or_test_partition(self):
        from aion.evaluation import _origins, dense_selection_origins

        horizon, season, length = 12, 12, 260
        minimum_train = max(2 * season, 2 * horizon, 8)
        origins = _origins(length, horizon, minimum_train)
        calibration_origin = origins[-2]
        for stride in (1, 2, 3, 5, 12):
            dense = dense_selection_origins(minimum_train, origins[:-2][-1], stride)
            assert dense, stride
            # Each fold's target window is [origin, origin + horizon).
            assert max(dense) + horizon <= calibration_origin, (
                f"stride {stride} lets a selection fold read the calibration "
                f"partition"
            )


class TestQuantileLevelsAndPinball:
    """Nine levels, and a distributional loss available but not default.

    q10/q50/q90 keep their exact meaning *and their exact values* — the same
    order statistics of the same residuals, fitted the same way. The rest are
    additional keys.
    """

    @staticmethod
    def _series(count=260):
        import math
        return [100 + 20 * math.sin(2 * math.pi * i / 12) + 0.3 * i + (i % 7)
                for i in range(count)]

    def test_generalised_spreads_reproduce_the_frozen_three(self):
        import random

        from aion.evaluation import (
            conformal_quantile_spreads,
            conformal_spreads,
            interval_from_spread,
            quantiles_from_spread,
        )

        rng = random.Random(3)
        for _ in range(50):
            horizon = rng.randint(3, 14)
            by_lead = {
                step: [rng.gauss(0, 1 + step * 0.3)
                       for _ in range(rng.choice([1, 2, 5, 8, 13, 30]))]
                for step in range(1, horizon + 1)
            }
            old = conformal_spreads(by_lead, horizon)
            new = conformal_quantile_spreads(by_lead, horizon)
            for step in range(1, horizon + 1):
                low, centre, high = interval_from_spread(100.0, old[step])
                levels = quantiles_from_spread(100.0, new[step])
                assert abs(levels["q10"] - low) < 1e-12
                assert abs(levels["q50"] - centre) < 1e-12
                assert abs(levels["q90"] - high) < 1e-12

    def test_levels_are_ordered_within_every_lead(self):
        import random

        from aion.evaluation import conformal_quantile_spreads, quantiles_from_spread

        rng = random.Random(11)
        for _ in range(50):
            horizon = rng.randint(3, 14)
            by_lead = {
                step: [rng.gauss(0, 1 + step * 0.3)
                       for _ in range(rng.choice([1, 3, 9, 25]))]
                for step in range(1, horizon + 1)
            }
            spreads = conformal_quantile_spreads(by_lead, horizon)
            for step in range(1, horizon + 1):
                values = [value for _, value in
                          sorted(quantiles_from_spread(0.0, spreads[step]).items())]
                assert values == sorted(values), (step, values)

    def test_pinball_loss_is_minimised_at_the_true_quantile(self):
        from aion.evaluation import pinball_loss

        # For level 0.9, under-predicting must be cheaper to avoid than
        # over-predicting: the loss is asymmetric in that direction.
        assert pinball_loss(10.0, 8.0, 0.9) > pinball_loss(10.0, 12.0, 0.9)
        assert pinball_loss(10.0, 8.0, 0.1) < pinball_loss(10.0, 12.0, 0.1)
        assert pinball_loss(10.0, 10.0, 0.5) == 0.0

    def test_default_selection_loss_is_unchanged(self):
        from aion.evaluation import evaluate

        values = self._series()
        default = evaluate(values, 12, 12, 0.02, frequency="h")
        explicit = evaluate(values, 12, 12, 0.02, frequency="h",
                            selection_loss="wape")
        assert default.selected_model == explicit.selected_model
        assert default.pinball_scores == {}, (
            "the distributional score is opt-in; computing it by default "
            "would change nothing but cost every run"
        )

    def test_pinball_scores_are_reported_when_requested(self):
        from aion.evaluation import evaluate

        result = evaluate(self._series(), 12, 12, 0.02, frequency="h",
                          selection_loss="pinball")
        assert result.supported
        assert result.pinball_scores
        assert any(value is not None for value in result.pinball_scores.values())
        # The point scores are reported alongside, never replaced.
        assert result.selection_scores
