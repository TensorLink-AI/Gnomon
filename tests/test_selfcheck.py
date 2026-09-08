from gnomon.selfcheck import leakage_self_check


def test_installed_leakage_mechanism_check_proves_access_boundary():
    result = leakage_self_check(cases=2, seed=11)
    assert result["checks_passed"] is True
    assert result["evidence"] == "finite_synthetic_checks"
    assert result["general_leakage_safety"] == "not_established"
    assert result["passed"] == 2 and result["failed"] == 0
