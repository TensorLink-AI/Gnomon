from benchmarks.adjudicationbench.run_adjudicationbench import run


def test_adjudicationbench_graduates_on_fresh_seed() -> None:
    result = run(seed=125893, replicates=3)
    assert result["cases"] == 105
    assert result["graduated"] is True
    assert all(result["gates"].values())
    # The immutability gate is behavioral: it reads the observed per-call
    # input comparison, not the engine's attested constant.
    assert "inputs_unmutated_100pct" in result["gates"]
    assert all(row["inputs_unmutated"] is True for row in result["rows"])
