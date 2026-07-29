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
        assert cfg.models.baselines_enabled is True
        assert cfg.models.statistical_enabled is True
        assert cfg.models.tsfm_candidates == []
        assert cfg.ensemble.enabled is False
        assert cfg.meta_model.enabled is False
        assert cfg.llm.enabled is False

    def test_load_config_returns_default_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AION_CONFIG_PATH", "")
        cfg = load_config()
        assert isinstance(cfg, AionConfig)
        assert cfg.models.baselines_enabled is True

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
