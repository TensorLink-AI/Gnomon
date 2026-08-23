from gnomon.evaluation import BASELINES, evaluate, scaled_error_score


def test_fold_local_scaled_error_rejects_uninformative_denominator():
    train = [100.0 + index * 1e-9 for index in range(50)]
    assert scaled_error_score(train, [100.0], [100.0], season=1) is None


def test_near_constant_channel_keeps_robust_baseline_even_for_wape_oracle():
    values = [100.0 + ((index % 3) - 1) * 1e-8 for index in range(120)]
    assessment = evaluate(
        values, horizon=5, season=1, minimum_improvement=.02,
        extra_candidates={
            "oracle_for_test": lambda origin, horizon: values[origin:origin + horizon],
        },
    )
    assert assessment.selected_model in BASELINES
    assert assessment.selection_stability["scaled_error_passed"] is False


def test_scaled_error_is_fold_local_and_finite_on_real_movement():
    train = [float(index) for index in range(30)]
    score = scaled_error_score(train, [30.0, 31.0], [30.0, 31.0], season=1)
    assert score == 0.0
