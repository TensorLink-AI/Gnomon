from benchmarks.capabilitybench import run


def test_capabilitybench_smoke_is_deterministic_and_graduates():
    first = run(seed=77, cases_per_family=10)
    second = run(seed=77, cases_per_family=10)
    assert first == second
    assert first["graduated"] is True
