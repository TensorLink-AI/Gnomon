from gnomon.models import croston_sba, ets, theta


def test_croston_is_intermittent_only_and_non_negative() -> None:
    history = [0, 0, 4, 0, 0, 0, 6, 0, 0, 5, 0, 0, 0, 0, 8, 0]
    points = croston_sba(history, 4, 1)
    assert len(points) == 4
    assert len(set(points)) == 1
    assert points[0] > 0


def test_theta_and_ets_transform_selection_stays_finite() -> None:
    multiplicative = [10.0 * 1.08 ** index for index in range(30)]
    for model in (theta, ets):
        points = model(multiplicative, 5, 1)
        assert len(points) == 5
        assert all(point > 0 for point in points)


def test_classical_family_choice_does_not_require_positive_data() -> None:
    history = [float(index - 10) for index in range(24)]
    assert len(theta(history, 3, 1)) == 3
    assert len(ets(history, 3, 1)) == 3
