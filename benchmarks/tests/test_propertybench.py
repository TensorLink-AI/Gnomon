from benchmarks.propertybench.run_propertybench import run


def test_propertybench_is_deterministic_complete_and_immutable() -> None:
    first = run(seed=1234, replicates=2)
    second = run(seed=1234, replicates=2)
    assert first == second
    assert first["cases"] == 52
    assert first["gates"]["complete"] is True
    assert first["gates"]["inputs_immutable"] is True
    stress = first["seasonality_alignment_stress"]
    assert stress["cases"] == 48
    assert set(stress["by_family"]) == {
        "harmonic", "asymmetric", "amplitude_drift", "low_snr"}
    assert first["gates"][
        "seasonality_alignment_stress_balanced_accuracy_at_least_65pct"] is True
