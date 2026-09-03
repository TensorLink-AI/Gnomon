from benchmarks.authoritybench.run import evaluate_cases


def test_authority_matrix_crosses_helpfulness_without_escalation() -> None:
    rows = evaluate_cases()

    assert len(rows) == 8
    assert {row["expected_authority"] for row in rows} == {
        "observed", "forecast", "assumed", "binding"}
    assert {row["helpful_to_target"] for row in rows} == {True, False}
    assert all(row["classification_correct"] for row in rows)
    assert not any(row["authority_escalated"] for row in rows)
