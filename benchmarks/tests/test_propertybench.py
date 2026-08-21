from benchmarks.common.immutability import call_preserving_inputs
from benchmarks.propertybench.run_propertybench import run


def test_propertybench_is_deterministic_complete_and_immutable() -> None:
    first = run(seed=1234, replicates=2)
    second = run(seed=1234, replicates=2)
    assert first == second
    assert first["cases"] == 52
    assert first["gates"]["complete"] is True
    assert first["gates"]["inputs_unmutated"] is True
    assert first["engine_calls"] > 0
    assert first["engine_calls_mutated_inputs"] == 0
    stress = first["seasonality_alignment_stress"]
    assert stress["cases"] == 48
    assert set(stress["by_family"]) == {
        "harmonic", "asymmetric", "amplitude_drift", "low_snr"}
    assert first["gates"][
        "seasonality_alignment_stress_balanced_accuracy_at_least_65pct"] is True


def test_immutability_check_is_behavioral_not_attested() -> None:
    """The gate must be able to fail: a function that edits its inputs in
    place is caught by the before/after comparison, a clean one is not."""
    def mutating(values: list[float]) -> float:
        values[0] += 1.0
        return values[0]

    def clean(values: list[float]) -> float:
        return sum(values)

    _, unmutated = call_preserving_inputs(mutating, [1.0, 2.0])
    assert unmutated is False
    _, unmutated = call_preserving_inputs(clean, [1.0, 2.0])
    assert unmutated is True


def test_expected_class_variants_draw_distinct_randomness() -> None:
    """Panel and forecast-path case seeds include the class index: without
    it the three expected classes replayed the same underlying draws and
    the lanes' effective sample was a third of their case count."""
    result = run(seed=1234, replicates=2)
    panel = result["panel_volatility"]["rows"]
    per_class = len(panel) // 3
    assert per_class * 3 == len(panel)
    path = result["forecast_path_volatility"]["rows"]
    seeds_by_class = [
        {row["seed"] for row in path[start:start + len(path) // 3]}
        for start in range(0, len(path), len(path) // 3)
    ]
    assert seeds_by_class[0].isdisjoint(seeds_by_class[1])
    assert seeds_by_class[1].isdisjoint(seeds_by_class[2])
