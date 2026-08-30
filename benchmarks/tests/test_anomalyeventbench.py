from benchmarks.anomalyeventbench.run import (
    EVENT_INDEX, NEARBY_INDEX, _cases, _event_counts, _generate, _summarise,
)


def test_anomaly_event_cases_are_frozen_and_labels_stay_scorer_side() -> None:
    cases = _cases()
    assert len(cases) == 24
    assert {case["family"] for case in cases} == {
        "level_shift_up", "level_shift_down", "isolated_spike_up",
        "isolated_spike_down", "nearby_opposite_spikes", "stationary_noise",
    }
    shifted = next(case for case in cases if case["kind"] == "shift")
    values = _generate(shifted)
    assert len(values) == 120
    assert shifted["expected_anomaly_indices"] == []
    assert shifted["expected_shift_index"] == EVENT_INDEX


def test_event_match_is_one_to_one_and_remainder_is_false() -> None:
    score = _event_counts(
        [EVENT_INDEX, EVENT_INDEX + 1, NEARBY_INDEX],
        [EVENT_INDEX, NEARBY_INDEX],
    )
    assert score["matched_events"] == 2
    assert score["false_events"] == 1
    assert score["missed_events"] == 0
    assert score["event_precision"] == 2 / 3
    assert score["event_recall"] == 1


def test_summary_keeps_engine_surfaces_and_event_failures_separate() -> None:
    rows = []
    for case in _cases():
        surface = {
            "raw_alerts": 0, "expected_events": len(
                case["expected_anomaly_indices"]),
            "matched_events": 0, "false_events": 0,
            "missed_events": len(case["expected_anomaly_indices"]),
            "event_precision": 1.0, "event_recall": 0.0,
            "rebound_duplicate": False,
            "nearby_events_exact": (
                False if case["kind"] == "nearby" else None),
            "selection_basis": "synthetic_injection_macro_f1",
            "shift_admitted": case["kind"] == "shift",
            "post_admitted_shift_alerts": 0,
        }
        labelled = dict(surface)
        if case["expected_anomaly_indices"]:
            labelled["selection_basis"] = "label_f1"
        rows.append({
            **case, "deterministic_replay": True,
            "product_inputs": {"labels_supplied": bool(
                case["expected_anomaly_indices"])},
            "investigation": dict(surface),
            "unlabelled": dict(surface),
            "labelled": labelled,
        })
    summary = _summarise(rows)
    assert set(summary["surfaces"]) == {
        "investigation", "unlabelled", "labelled"}
    assert summary["gates"]["nearby_events_preserved_when_detected"] is False
    assert summary["passed"] is False
