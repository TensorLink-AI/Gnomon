from benchmarks.power_analysis import exact_power, required_pairs


def test_power_increases_with_sample_size_and_effect() -> None:
    assert exact_power(480, .55, .025) > exact_power(80, .55, .025)
    assert exact_power(480, .60, .025) > exact_power(480, .55, .025)


def test_required_pairs_meets_declared_power() -> None:
    required = required_pairs(.55, .8, .025)
    assert exact_power(required, .55, .025) >= .8
    assert exact_power(required - 1, .55, .025) < .8
