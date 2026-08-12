"""Fold-indexed evidence stays fold-aligned when a member fails mid-run.

A built-in model that failed ``predict`` (or hit an unscoreable fold)
used to have that fold compacted out of its per-fold lists, while the
ensemble and meta-model read those lists *by fold index* — so every
later fold's forecast shifted one slot left, and fold k's actuals were
scored against fold k+1's forecast for any model that failed a strict
subset of folds. The TSFM lists always kept placeholder slots; the
built-in lists now follow the same rule.
"""

from gnomon.config import GnomonConfig

NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]


def _values(periods: int = 160) -> list[float]:
    return [
        100 + index * 0.4 + NOISE[index % 10] * 6 for index in range(periods)
    ]


def test_mid_run_member_failure_does_not_shift_later_folds(monkeypatch):
    """Fail one model on the second selection fold only; every other
    fold's ensemble map must carry the forecast that model produced at
    that fold's own origin, and the failed fold must simply lack it."""
    import gnomon.ensemble as ensemble_module
    import gnomon.evaluation as evaluation_module
    from gnomon.evaluation import evaluate

    real_predict = evaluation_module.predict
    seen_lengths: list[int] = []
    produced: dict[int, list[float]] = {}

    def flaky_predict(name, train, horizon, season):
        length = len(train)
        if length not in seen_lengths:
            seen_lengths.append(length)
        if (name == "drift" and len(seen_lengths) >= 2
                and length == seen_lengths[1]):
            raise ValueError("synthetic mid-run failure")
        forecast = real_predict(name, train, horizon, season)
        if name == "drift" and length not in produced:
            produced[length] = list(forecast)
        return forecast

    captured: list[dict[str, list[float]]] = []
    real_compute = ensemble_module.compute_ensemble_forecast

    def recording_compute(forecasts, scores, strategy="weighted_mean",
                          last_observed=0.0, config=None):
        captured.append({name: list(f) for name, f in forecasts.items()})
        return real_compute(forecasts, scores, strategy=strategy,
                            last_observed=last_observed, config=config)

    monkeypatch.setattr(evaluation_module, "predict", flaky_predict)
    monkeypatch.setattr(ensemble_module, "compute_ensemble_forecast",
                        recording_compute)

    config = GnomonConfig()
    config.ensemble.enabled = True
    result = evaluate(_values(), 7, 7, 0.02, frequency="D", tsfm_names=[],
                      config=config)

    # The per-fold ensemble maps are the first compute calls, one per
    # selection fold, in fold order; later calls (final refit, residual
    # passes) run on longer histories and are not fold-indexed.
    fold_maps = captured[: len(seen_lengths)]
    assert len(fold_maps) >= 3, "expected several ensemble folds"
    assert "drift" not in fold_maps[1], (
        "the failed fold must lack the member, not borrow a later fold"
    )
    for fold_idx, fold_map in enumerate(fold_maps):
        if fold_idx == 1 or "drift" not in fold_map:
            continue
        assert fold_map["drift"] == produced[seen_lengths[fold_idx]], (
            f"fold {fold_idx} carries a forecast drift produced at a "
            "different fold's origin"
        )

    # A model that failed a fold holds no aggregate selection score:
    # every fold must have a real score, the same rule the TSFM
    # aggregation always applied.
    assert result.selection_scores.get("drift") is None
    assert result.selection_scores.get("last_value") is not None


def test_all_folds_failing_keeps_model_out_of_meta_pool(monkeypatch):
    """Placeholder-only lists (every fold failed) must not count toward
    the meta-model's member minimum."""
    import gnomon.evaluation as evaluation_module
    import gnomon.meta_model as meta_module
    from gnomon.evaluation import evaluate

    real_predict = evaluation_module.predict

    def dead_model_predict(name, train, horizon, season):
        if name == "drift":
            raise ValueError("synthetic total failure")
        return real_predict(name, train, horizon, season)

    trained_pools: list[list[str]] = []
    real_train = meta_module.train_meta_model

    def recording_train(fold_forecasts, fold_actuals, **kwargs):
        trained_pools.append(sorted(fold_forecasts))
        return real_train(fold_forecasts, fold_actuals, **kwargs)

    monkeypatch.setattr(evaluation_module, "predict", dead_model_predict)
    monkeypatch.setattr(meta_module, "train_meta_model", recording_train)

    config = GnomonConfig()
    config.meta_model.enabled = True
    evaluate(_values(), 7, 7, 0.02, frequency="D", tsfm_names=[],
             config=config)

    assert trained_pools, "the meta-model never trained"
    for pool in trained_pools:
        assert "drift" not in pool
