from benchmarks.contextcachebench.run_contextcachebench import run


def test_contextcachebench_graduates_on_product_path(tmp_path):
    summary = run(cases=2, output_dir=tmp_path, replays_per_case=10)
    assert summary["graduated"] is True
    assert summary["forecast_equivalent"] == 20
