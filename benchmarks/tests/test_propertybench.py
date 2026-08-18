from benchmarks.propertybench.run_propertybench import run


def test_propertybench_is_deterministic_complete_and_immutable() -> None:
    first = run(seed=1234, replicates=2)
    second = run(seed=1234, replicates=2)
    assert first == second
    assert first["cases"] == 52
    assert first["gates"]["complete"] is True
    assert first["gates"]["primary_immutable"] is True
