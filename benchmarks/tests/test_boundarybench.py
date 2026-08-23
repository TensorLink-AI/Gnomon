from benchmarks.boundarybench.run_boundarybench import run


def test_boundarybench_is_seeded_and_graduates():
    first = run(17, 25)
    second = run(17, 25)
    assert first == second
    assert first["graduated"] is True
    assert sum(row["redundant_calls"] for row in first["rows"]) == 7
