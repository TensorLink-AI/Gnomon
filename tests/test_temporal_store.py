from datetime import datetime, timezone
from pathlib import Path

import json
import pytest

from gnomon.cli import main
from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock
from gnomon.runtime import forecast
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


def test_documented_vintage_quickstart_is_supported(tmp_path):
    """`docs/quickstart-mcp.md` §4, run verbatim on the shipped fixture.

    The revisions fixture used to hold ten dates, so the documented
    horizon-7 replay abstained; the workaround the docs implied — ingesting
    the plain file first — asserted same-day knowledge of revised figures.
    The fixture now carries the whole history and one ingest suffices.
    """
    examples = Path(__file__).resolve().parent.parent / "examples"
    source = examples / "messy_requests_revisions.csv"
    store = TemporalStore(tmp_path / "store.db")
    report = store.ingest_csv(
        str(source), dataset="requests", time_column="timestamp",
        target_column="requests", known_at_column="published", clock=CLOCK,
    )
    assert report.rows_added == report.rows_seen
    assert report.revisions_created > 0
    assert store.known_time_provenance("requests") == "recorded"

    artifact, _ = forecast(
        "store:requests", time_column="timestamp", target_column="requests",
        horizon=7, as_of=datetime(2026, 6, 3),
        store_path=str(tmp_path / "store.db"),
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    result = artifact.results[0]
    assert result.support in {"supported", "weakly_supported"}, result.support_assessment
    assert result.selected_model == result.strongest_baseline
    assert len(result.forecast) == 7

    access = next(
        item for item in artifact.evidence if item.kind == "snapshot_access"
    ).payload
    served = [entry["max_known_time"] for entry in access["accesses"]]
    assert served == ["2026-06-03T00:00:00"], "the replay saw post-cutoff vintages"


def _daily_csv(path: Path, days: int) -> Path:
    from datetime import date, timedelta
    start = date(2026, 3, 1)
    rows = [
        f"{(start + timedelta(days=i)).isoformat()},{100 + i * 2 + [3, 5, 1, -2, -4, 0, 2][i % 7]}"
        for i in range(days)
    ]
    return _write_csv(path, rows)


def test_forecast_as_of_replay_uses_only_prior_data(tmp_path):
    source = _daily_csv(tmp_path / "history.csv", 40)
    as_of = datetime(2026, 3, 25)
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=5, output=str(tmp_path / "out"), clock=CLOCK, as_of=as_of,
    )
    # The forecast starts where knowledge at as_of ended, not at the file's end.
    first_forecast = artifact.results[0].forecast[0]["timestamp"]
    assert first_forecast == "2026-03-26T00:00:00"
    # The snapshot access log proves nothing published after as_of was read.
    snapshot_evidence = [item for item in artifact.evidence if item.kind == "snapshot_access"]
    assert len(snapshot_evidence) == 1
    payload = snapshot_evidence[0].payload
    assert payload["as_of"] == as_of.isoformat()
    for access in payload["accesses"]:
        assert access["max_known_time"] <= as_of.isoformat()
    assert artifact.task.as_of == as_of.isoformat()


def test_aggressive_repair_cannot_use_rows_after_as_of(tmp_path):
    from gnomon.pipeline import load_stage
    from gnomon.repair import RepairLog

    source = tmp_path / "gapped.csv"
    source.write_text(
        "timestamp,value\n"
        "2026-01-01,10\n"
        "2026-01-02,20\n"
        "2026-01-04,1000\n",
        encoding="utf-8",
    )
    log = RepairLog()
    loaded = load_stage(
        str(source), time_column="timestamp", target_column="value",
        series_column=None, frequency="D", as_of=datetime(2026, 1, 3),
        repair="aggressive", repair_log=log,
    )
    rows = loaded.groups["__default__"]
    assert [row.timestamp.day for row in rows] == [1, 2]
    assert [row.value for row in rows] == [10.0, 20.0]
    assert "interpolated" not in {item.code for item in log.actions()}


def test_forecast_as_of_before_all_data_is_structured_error(tmp_path):
    source = _daily_csv(tmp_path / "history.csv", 10)
    with pytest.raises(GnomonError) as caught:
        forecast(
            str(source), time_column="timestamp", target_column="value",
            horizon=3, output=str(tmp_path / "out"), clock=CLOCK,
            as_of=datetime(2020, 1, 1),
        )
    assert caught.value.code == "EMPTY_SNAPSHOT"


def test_forecast_from_persistent_store(tmp_path):
    source = _daily_csv(tmp_path / "history.csv", 40)
    store_path = tmp_path / "store.db"
    store = TemporalStore(store_path)
    store.ingest_csv(
        str(source), dataset="requests", time_column="timestamp",
        target_column="value", clock=CLOCK,
    )
    artifact, _ = forecast(
        "store:requests", time_column="timestamp", target_column="value",
        horizon=5, output=str(tmp_path / "out"), clock=CLOCK,
        store_path=str(store_path),
    )
    assert artifact.results[0].forecast
    assert artifact.task.input_path == "store:requests"


def test_missing_dataset_is_structured_error(tmp_path):
    with pytest.raises(GnomonError) as caught:
        forecast(
            "store:nope", time_column="timestamp", target_column="value",
            horizon=3, output=str(tmp_path / "out"), clock=CLOCK,
            store_path=str(tmp_path / "store.db"),
        )
    assert caught.value.code == "DATASET_NOT_FOUND"


def test_cli_ingest_and_store_list(tmp_path, capsys):
    source = _daily_csv(tmp_path / "history.csv", 5)
    store_path = str(tmp_path / "store.db")
    assert main([
        "ingest", str(source), "--dataset", "smoke",
        "--time", "timestamp", "--target", "value", "--store-path", store_path,
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["rows_added"] == 5
    assert main(["store", "list", "--store-path", store_path]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["datasets"][0]["dataset"] == "smoke"
    assert listing["datasets"][0]["observations"] == 5
