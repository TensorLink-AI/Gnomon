from benchmarks.adjudicationbench.run_adjudicationbench import run


def test_adjudicationbench_graduates_on_fresh_seed() -> None:
    result = run(seed=125893, replicates=3)
    assert result["cases"] == 105
    assert result["graduated"] is True
    assert all(result["gates"].values())
