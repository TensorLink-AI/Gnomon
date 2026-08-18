from gnomon.tracking import TrackingStore


def test_temporal_answer_receipts_are_immutable_and_pollable(tmp_path) -> None:
    store = TrackingStore(tmp_path / "registry.db")
    answer = {
        "question": {"id": "trend", "target": "api", "property": "trend"},
        "answer": {"support": "supported", "direction": "upward"},
        "headline": "Trend is upward.", "limitations": [],
        "artifact_id": "forecast-1",
    }
    store.record_temporal_answers("capacity", "forecast-1", "api", [answer])
    # INSERT OR IGNORE preserves the exact first receipt on replay.
    store.record_temporal_answers(
        "capacity", "forecast-1", "api",
        [{**answer, "headline": "mutated"}],
    )
    rows = store.temporal_answer_receipts("capacity", resolved=False)
    assert len(rows) == 1
    assert rows[0]["payload"]["headline"] == "Trend is upward."
    assert store.status("capacity")["temporal_answers"] == {
        "open": 1, "resolved": 0}
