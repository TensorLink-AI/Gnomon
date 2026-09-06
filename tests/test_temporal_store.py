from datetime import datetime, timezone
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock
from gnomon.temporal_store import (
    InMemoryTemporalStore,
    Snapshot,
    TemporalObservation,
    TemporalStore,
    join_as_of,
)

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))


def _obs(valid_day: int, known_day: int, value: float, revision: int = 0,
         entity: str = "alpha", variable: str = "sales") -> TemporalObservation:
    return TemporalObservation(
        entity=entity, variable=variable,
        valid_time=datetime(2026, 1, valid_day),
        known_time=datetime(2026, 1, known_day),
        value=value, revision=revision,
    )


def test_snapshot_structurally_excludes_future_known_rows():
    snapshot = Snapshot([_obs(1, 1, 10.0), _obs(2, 5, 20.0)], as_of=datetime(2026, 1, 3))
    series = snapshot.series("alpha", "sales")
    assert [item.value for item in series] == [10.0]
    # Even an explicit later cutoff cannot see past as_of.
    late = snapshot.series("alpha", "sales", cutoff=datetime(2026, 1, 30))
    assert [item.value for item in late] == [10.0]


def test_latest_revision_wins_and_vintages_are_reconstructable():
    rows = [
        _obs(1, 1, 100.0, revision=0),
        _obs(1, 10, 105.0, revision=1),  # corrected later
    ]
    snapshot = Snapshot(rows, as_of=None)
    assert [item.value for item in snapshot.series("alpha", "sales")] == [105.0]
    # As known before the correction, the original value stands.
    early = snapshot.series("alpha", "sales", cutoff=datetime(2026, 1, 5))
    assert [item.value for item in early] == [100.0]


def test_join_as_of_returns_vintages_not_final_values():
    rows = [
        _obs(1, 1, 100.0, revision=0),
        _obs(1, 10, 105.0, revision=1),
        _obs(2, 2, 50.0, revision=0),
    ]
    snapshot = Snapshot(rows, as_of=None)
    grid = [datetime(2026, 1, 1), datetime(2026, 1, 2), datetime(2026, 1, 3)]
    assert join_as_of(snapshot, "alpha", "sales", grid, cutoff=datetime(2026, 1, 4)) == [100.0, 50.0, None]
    assert join_as_of(snapshot, "alpha", "sales", grid) == [105.0, 50.0, None]


def test_access_log_records_reads():
    snapshot = Snapshot([_obs(1, 1, 10.0)], as_of=None)
    assert not hasattr(snapshot, "variables") and not hasattr(snapshot, "access_log")
    snapshot.series("alpha", "sales")
    summary = snapshot.access_summary()
    assert summary["accesses"][0]["entity"] == "alpha"
    assert summary["accesses"][0]["max_known_time"] == "2026-01-01T00:00:00"


def test_plain_observations_reject_duplicates():
    from gnomon.data import Observation
    rows = [
        Observation(datetime(2026, 1, 1), 1.0, "alpha"),
        Observation(datetime(2026, 1, 1), 2.0, "alpha"),
    ]
    with pytest.raises(GnomonError) as caught:
        InMemoryTemporalStore.from_plain_observations(rows, "sales", "sha256:x")
    assert caught.value.code == "DUPLICATE_TIMESTAMPS"


def _write_csv(path: Path, rows: list[str], header: str = "timestamp,value") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_ingest_appends_revisions(tmp_path):
    store = TemporalStore(tmp_path / "store.db")
    first = _write_csv(tmp_path / "v1.csv", ["2026-01-01,100", "2026-01-02,200"])
    report = store.ingest_csv(
        str(first), dataset="sales", time_column="timestamp", target_column="value",
        clock=CLOCK,
    )
    assert report.rows_added == 2
    assert report.revisions_created == 0
    assert any("known_time_assumed" in warning for warning in report.warnings)

    # Re-supplying the identical file changes nothing.
    again = store.ingest_csv(
        str(first), dataset="sales", time_column="timestamp", target_column="value",
        clock=CLOCK,
    )
    assert again.rows_added == 0
    assert again.duplicates_skipped == 2

    # A corrected history becomes new revision rows, not an overwrite.
    # Both rows are new vintages: 100 -> 101 is a correction, and 200
    # restated on Jan 5 is the source confirming 200 at a second instant.
    # "As known on Jan 1 it was 200" and "as known on Jan 5 it was 200" are
    # different facts, and only the second survives a later contradiction.
    corrected = _write_csv(
        tmp_path / "v2.csv",
        ["2026-01-01,101,2026-01-05", "2026-01-02,200,2026-01-05"],
        header="timestamp,value,published",
    )
    revised = store.ingest_csv(
        str(corrected), dataset="sales", time_column="timestamp",
        target_column="value", known_at_column="published", clock=CLOCK,
    )
    assert revised.rows_added == 2
    assert revised.revisions_created == 2
    assert revised.duplicates_skipped == 0
    assert revised.reverts_recorded == 1

    snapshot = store.snapshot("sales")
    assert [item.value for item in snapshot.series("__default__", "value")] == [101.0, 200.0]
    early = snapshot.series("__default__", "value", cutoff=datetime(2026, 1, 3))
    assert [item.value for item in early] == [100.0, 200.0]


def test_a_revert_keeps_its_own_vintage(tmp_path):
    """100 -> 150 -> back to 100 is three vintages, not two.

    Deduplicating on value alone dropped the third, so a replay positioned
    after the revert returned the superseded 150 — the store answering a
    point-in-time question with a value that was wrong at that instant.
    """
    store = TemporalStore(tmp_path / "store.db")

    def _ingest(name: str, value: str, published: str):
        path = _write_csv(
            tmp_path / name, [f"2026-01-01,{value},{published}"],
            header="timestamp,value,published",
        )
        return store.ingest_csv(
            str(path), dataset="revisions", time_column="timestamp",
            target_column="value", known_at_column="published", clock=CLOCK,
        )

    _ingest("r1.csv", "100", "2026-01-05")
    _ingest("r2.csv", "150", "2026-01-08")
    third = _ingest("r3.csv", "100", "2026-01-11")

    assert third.rows_added == 1, "the reverted vintage was dropped"
    assert third.duplicates_skipped == 0
    assert third.reverts_recorded == 1

    snapshot = store.snapshot("revisions")
    at = lambda day: snapshot.value_as_of(  # noqa: E731
        "__default__", "value", datetime(2026, 1, 1),
        cutoff=datetime(2026, 1, day),
    )
    assert at(6) == 100.0
    assert at(9) == 150.0
    assert at(11) == 100.0, "replay after the revert returned the superseded value"


def test_an_exact_repeat_is_still_a_duplicate(tmp_path):
    """Same valid_time, same known_time, same value: a genuine no-op."""
    store = TemporalStore(tmp_path / "store.db")
    path = _write_csv(
        tmp_path / "v.csv", ["2026-01-01,100,2026-01-05"],
        header="timestamp,value,published",
    )
    kwargs = dict(
        dataset="sales", time_column="timestamp", target_column="value",
        known_at_column="published", clock=CLOCK,
    )
    assert store.ingest_csv(str(path), **kwargs).rows_added == 1
    repeat = store.ingest_csv(str(path), **kwargs)
    assert repeat.rows_added == 0
    assert repeat.duplicates_skipped == 1
    assert repeat.reverts_recorded == 0


def test_known_time_provenance_comes_from_the_ingest(tmp_path):
    """Not inferred from the data: same-day publication is not an assumption.

    A dataset whose values genuinely become knowable the day they apply has
    ``valid_time == known_time`` on every row, which the old inference read
    as "assumed". Provenance can tell them apart.
    """
    store = TemporalStore(tmp_path / "store.db")
    same_day = _write_csv(
        tmp_path / "same_day.csv",
        ["2026-01-01,100,2026-01-01", "2026-01-02,200,2026-01-02"],
        header="timestamp,value,published",
    )
    store.ingest_csv(
        str(same_day), dataset="realtime", time_column="timestamp",
        target_column="value", known_at_column="published", clock=CLOCK,
    )
    assert store.known_time_provenance("realtime") == "recorded"
    assert store.snapshot("realtime").assumed_known_time is False


def test_a_mixed_dataset_reports_partially_assumed(tmp_path):
    """One assumed ingest beside one real one is neither, and says so."""
    store = TemporalStore(tmp_path / "store.db")
    plain = _write_csv(tmp_path / "plain.csv", ["2026-01-01,100"])
    store.ingest_csv(
        str(plain), dataset="mixed", time_column="timestamp",
        target_column="value", clock=CLOCK,
    )
    dated = _write_csv(
        tmp_path / "dated.csv", ["2026-01-02,200,2026-01-06"],
        header="timestamp,value,published",
    )
    store.ingest_csv(
        str(dated), dataset="mixed", time_column="timestamp",
        target_column="value", known_at_column="published", clock=CLOCK,
    )
    assert store.known_time_provenance("mixed") == "partially_assumed"
    summary = store.snapshot("mixed").access_summary()
    assert summary["known_time_provenance"] == "partially_assumed"
    assert summary["known_time_assumed"] is True


def test_timezone_mismatch_after_the_first_row_is_structured(tmp_path):
    """The check reads every observation, not just ``observations[0]``."""
    observations = [
        TemporalObservation(
            entity="alpha", variable="sales",
            valid_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            known_time=datetime(2026, 1, 1, tzinfo=timezone.utc), value=1.0,
        ),
        TemporalObservation(
            entity="alpha", variable="sales",
            valid_time=datetime(2026, 1, 2), known_time=datetime(2026, 1, 2),
            value=2.0,
        ),
    ]
    with pytest.raises(GnomonError) as raised:
        Snapshot(observations, datetime(2026, 1, 3, tzinfo=timezone.utc))
    assert raised.value.code == "SNAPSHOT_TIMEZONE_MISMATCH"
