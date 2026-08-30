from benchmarks.multivariatebench.run import SEEDS, generate_case, summarize


def _row(seed: int, *, complete: bool = True, model: str = "ets",
         improvement: float = 0.0) -> dict:
    case = generate_case(seed)
    return {
        "case_id": case.case_id,
        "family": case.family,
        "seed": seed,
        "complete": complete,
        "treatment_model": model,
        "smape_improvement": improvement,
        "relative_improvement": improvement,
    }


def test_frozen_generator_is_deterministic_and_balanced():
    first = [generate_case(seed) for seed in SEEDS]
    second = [generate_case(seed) for seed in SEEDS]
    assert first == second
    assert len({case.case_id for case in first}) == 12
    assert sum(case.family == "lagged_driver" for case in first) == 6
    assert sum(case.family.endswith("control") for case in first) == 6
    assert all(len(case.target) == 204 for case in first)


def test_missing_and_failed_rows_remain_in_gate_denominator():
    rows = [_row(seed) for seed in SEEDS[:-1]]
    report = summarize(rows)
    assert report["denominators"] == {
        "all": 12, "completed": 11, "drivers": 6, "controls": 5,
    }
    assert report["gates"]["completion"] is False
    assert report["decision_ready"] is False


def test_summary_can_represent_a_clean_promotion():
    rows = []
    for seed in SEEDS:
        case = generate_case(seed)
        rows.append(_row(
            seed,
            model="var" if case.family == "lagged_driver" else "ets",
            improvement=.2 if case.family == "lagged_driver" else 0.0,
        ))
    report = summarize(rows)
    assert report["decision_ready"] is True
