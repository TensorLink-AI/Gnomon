from benchmarks.adapterbench.run_adapterbench import run


def test_adapterbench_graduates() -> None:
    result = run()
    assert result["graduated"] is True
