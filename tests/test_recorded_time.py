from datetime import datetime, timezone
import sqlite3

import pytest

from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock
from gnomon.temporal_store import TemporalObservation, TemporalStore


def at(day):
    return datetime(2025, 1, day, tzinfo=timezone.utc)


def row(value, known_day=2):
    return TemporalObservation("one", "value", at(1), at(known_day), value)


def test_source_available_and_recorded_replay_are_distinct(tmp_path):
    store = TemporalStore(tmp_path / "store.db")
    store.ingest_rows("data", [row(1)], source_fingerprint="one", clock=FixedClock(at(3)))
    before = store.snapshot("data", as_of=at(5), recorded_as_of=at(5)).snapshot_id
    store.ingest_rows("data", [row(2, 4)], source_fingerprint="two", clock=FixedClock(at(10)))
    assert store.snapshot("data", as_of=at(5)).series("one", "value")[0].value == 2
    replay = store.snapshot("data", as_of=at(5), recorded_as_of=at(5))
    assert replay.series("one", "value")[0].value == 1
    assert replay.snapshot_id == before
    assert replay.access_summary()["replay_mode"] == "recorded"
    # Resupplying the first vintage must not rewrite first-seen time.
    store.ingest_rows("data", [row(1)], source_fingerprint="one", clock=FixedClock(at(12)))
    assert store.snapshot("data", recorded_as_of=at(5)).series("one", "value")[0].recorded_at == at(3)


def test_pre_release_store_is_rejected_instead_of_guessed_or_migrated(tmp_path):
    path = tmp_path / "pre-release.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE observations (dataset TEXT, entity TEXT, variable TEXT, valid_time TEXT, "
                     "known_time TEXT, value REAL, revision INTEGER, source_ref TEXT)")
        conn.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?)",
                     ("data", "one", "value", at(1).isoformat(), at(2).isoformat(), 9, 0, "old"))
    before = path.read_bytes()
    with pytest.raises(GnomonError, match="new empty file"):
        TemporalStore(path)
    assert path.read_bytes() == before


def test_dataset_fingerprint_hashes_values_not_row_counts(tmp_path):
    first, second = TemporalStore(tmp_path / "a.db"), TemporalStore(tmp_path / "b.db")
    first.ingest_rows("data", [row(1)], source_fingerprint="one", clock=FixedClock(at(3)))
    second.ingest_rows("data", [row(99)], source_fingerprint="two", clock=FixedClock(at(3)))
    assert first.dataset_fingerprint("data") != second.dataset_fingerprint("data")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_store_refuses_nonfinite_values(tmp_path, value):
    with pytest.raises(GnomonError, match="finite"):
        TemporalStore(tmp_path / "store.db").ingest_rows("data", [row(value)], source_fingerprint="bad")
