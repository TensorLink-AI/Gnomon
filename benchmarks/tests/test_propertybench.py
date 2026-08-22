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
    process_volatility = first["future_process_volatility"]
    assert process_volatility["cases"] == 56
    assert set(process_volatility["class_recall"]) == {
        "increased", "decreased", "stable"}
    assert all(row["family"] in {
        "stationary", "gradual_up", "gradual_down", "late_shift_up",
        "late_shift_down", "level_trend", "heavy_tailed",
    } for row in process_volatility["rows"])
    regime_transitions = first["observed_regime_transitions"]
    assert regime_transitions["cases"] == 32
    assert set(regime_transitions["class_recall"]) == {
        "no_change", "regime_shift", "transient_anomaly"}
    assert first["property_gates"]["regime"][
        "unannounced_breaks_not_claimed"] is True
    dependence_stress = first["dependence_stress"]
    assert dependence_stress["cases"] == 32
    assert set(dependence_stress["class_recall"]) == {
        "negative", "weak", "positive"}
