from datetime import datetime, timezone
from pathlib import Path

import json
import pytest

from aion.cli import main
from aion.contracts import AionError
from aion.ids import FixedClock
from aion.runtime import forecast
from aion.temporal_store import (
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
    from aion.data import Observation
    rows = [
        Observation(datetime(2026, 1, 1), 1.0, "alpha"),
        Observation(datetime(2026, 1, 1), 2.0, "alpha"),
    ]
    with pytest.raises(AionError) as caught:
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
    corrected = _write_csv(
        tmp_path / "v2.csv",
        ["2026-01-01,101,2026-01-05", "2026-01-02,200,2026-01-05"],
        header="timestamp,value,published",
    )
    revised = store.ingest_csv(
        str(corrected), dataset="sales", time_column="timestamp",
        target_column="value", known_at_column="published", clock=CLOCK,
    )
    assert revised.rows_added == 1
    assert revised.revisions_created == 1
    assert revised.duplicates_skipped == 1

    snapshot = store.snapshot("sales")
    assert [item.value for item in snapshot.series("__default__", "value")] == [101.0, 200.0]
    early = snapshot.series("__default__", "value", cutoff=datetime(2026, 1, 3))
    assert [item.value for item in early] == [100.0, 200.0]


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


def test_forecast_as_of_before_all_data_is_structured_error(tmp_path):
    source = _daily_csv(tmp_path / "history.csv", 10)
    with pytest.raises(AionError) as caught:
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
    with pytest.raises(AionError) as caught:
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
