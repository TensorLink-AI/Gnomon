from gnomon.selfcheck import leakage_self_check


def test_installed_leakage_mechanism_check_proves_access_boundary():
    result = leakage_self_check(cases=2, seed=11)
    assert result["structural_claim_proven"] is True
    assert result["passed"] == 2 and result["failed"] == 0
