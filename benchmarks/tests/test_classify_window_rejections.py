"""window_is_future_only rejections: the record self-classifies now.

Engine side: the check records the quantities it compared (window,
last observation, overlap, timezone awareness). Classifier side: those
quantities bucket every rejection into proposer-dating vs boundary
artifact vs the timezone reading — the split the 50-rejection census
could not make.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.classify_window_rejections import classify_rejection

DAY = 86400.0


def test_buckets():
    assert classify_rejection(None, DAY) == "record_predates_data"
    assert classify_rejection({}, DAY) == "record_predates_data"
    assert classify_rejection(
        {"overlap_seconds": 5 * DAY, "window_entirely_historical": True,
         "mixed_timezone_alignment": False}, DAY,
    ) == "entirely_historical"
    assert classify_rejection(
        {"overlap_seconds": 7200.0, "window_entirely_historical": False,
         "mixed_timezone_alignment": True}, DAY,
    ) == "timezone_shaped"
    assert classify_rejection(
        {"overlap_seconds": 0.0, "window_entirely_historical": False,
         "mixed_timezone_alignment": False}, DAY,
    ) == "boundary_overlap"
    assert classify_rejection(
        {"overlap_seconds": 3 * DAY, "window_entirely_historical": False,
         "mixed_timezone_alignment": False}, DAY,
    ) == "misdated_partial"
    # Without a resolvable step, near-boundary cannot be told from
    # misdating — it must not silently become the benign bucket.
    assert classify_rejection(
        {"overlap_seconds": 0.0, "window_entirely_historical": False,
         "mixed_timezone_alignment": False}, None,
    ) == "misdated_partial"


def test_engine_records_the_compared_quantities():
    from gnomon.context import ContextEvent, ContextSource
    from gnomon.future_context import assess_future_events

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(days=day) for day in range(30)]
    future = [start + timedelta(days=30 + day) for day in range(5)]
    event = ContextEvent(
        event_id="w1", event_type="constraint:cap", entity_scope=("*",),
        effective_start=(start + timedelta(days=20)).isoformat(),
        effective_end=(start + timedelta(days=25)).isoformat(),
        known_at=start.isoformat(),
        attributes={"source_span": "values stay below 10"},
        source=ContextSource("dataset", "t#c"), created_by="llm",
    )
    assessment = assess_future_events(
        [event], "s", [1.0] * 30, timestamps, future, 7,
    )
    rejection, = assessment.rejected
    assert rejection["code"] == "window_is_future_only"
    data = rejection["data"]
    assert data["window_entirely_historical"] is True
    assert data["mixed_timezone_alignment"] is False
    assert data["overlap_seconds"] == 9 * 86400.0
    assert classify_rejection(data, 86400.0) == "entirely_historical"
