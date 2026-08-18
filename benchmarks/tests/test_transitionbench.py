from benchmarks.transitionbench.run_transitionbench import run


def test_transitionbench_smoke_graduates() -> None:
    result = run(seed=8811, replicates=4)
    assert result["graduated"] is True
    assert result["gates"]["all_cases_present"] is True
