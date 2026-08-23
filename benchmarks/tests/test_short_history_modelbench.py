from benchmarks.modelbench.run_short_history import run


def test_short_history_benchmark_is_deterministic_and_retains_records():
    first = run(seed=91, cases_per_family=8)
    second = run(seed=91, cases_per_family=8)
    assert first == second
    assert len(first["raw_records"]) == 72
    assert all(first["gates"].values())
    assert first["pooling"]["mixed_direction"]["admitted"] == 0
