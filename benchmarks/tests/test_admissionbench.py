from benchmarks.admissionbench.run_admissionbench import run


def test_admissionbench_is_seeded_and_policy_beats_fixed_arms():
    first = run(seed=7, cases=500)
    second = run(seed=7, cases=500)
    assert first == second
    assert all(first["gates"].values()), first
