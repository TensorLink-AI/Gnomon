from datetime import datetime, timezone
from pathlib import Path

import json
import pytest

from gnomon.artifacts import read_artifact
from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock, content_id
from gnomon.runtime import forecast
from gnomon.versioning import ensure_readable, readable_schema_versions

REPO = Path(__file__).resolve().parent.parent


def test_current_version_is_readable():
    ensure_readable(readable_schema_versions()[0])


def test_unknown_version_is_rejected():
    with pytest.raises(GnomonError) as excinfo:
        ensure_readable("9.9")
    assert excinfo.value.code == "UNSUPPORTED_SCHEMA_VERSION"


def test_read_artifact_roundtrip(tmp_path):
    artifact, artifact_dir = forecast(
        str(REPO / "examples" / "daily_requests.csv"),
        time_column="timestamp", target_column="requests", horizon=3,
        output=str(tmp_path), clock=FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc)),
    )
    data = read_artifact(artifact_dir)
    assert data["forecast_id"] == artifact.forecast_id


def test_read_artifact_rejects_future_schema(tmp_path):
    directory = tmp_path / "fake"
    directory.mkdir()
    (directory / "artifact.json").write_text(json.dumps({"schema_version": "9.9"}))
    with pytest.raises(GnomonError):
        read_artifact(directory)


def test_content_id_stability():
    first = content_id("dataset", {"b": 2, "a": 1})
    second = content_id("dataset", {"a": 1, "b": 2})
    assert first == second
    assert first.startswith("dataset_")
    assert first != content_id("dataset", {"a": 1, "b": 3})


# -- runtime version is part of every artifact id's identity ----------------

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))


def _series_csv(tmp_path: Path) -> Path:
    path = tmp_path / "series.csv"
    rows = "\n".join(
        f"2026-01-{day:02d},{100 + day}" for day in range(1, 29)
    )
    path.write_text("timestamp,value\n" + rows + "\n", encoding="utf-8")
    return path


def test_different_runtime_versions_mint_different_forecast_ids(tmp_path, monkeypatch):
    """A content id names a question and its answerer. The same task run by
    a different build must not share an id: first-write-wins would then
    serve one build's numbers as the other's — observed live when a
    threshold fix left a stale artifact answering `decide` with the
    pre-fix probabilities."""
    import gnomon.runtime as runtime_module

    source = _series_csv(tmp_path)
    current, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=3, output=str(tmp_path / "now"), clock=CLOCK,
    )
    monkeypatch.setattr(runtime_module, "RUNTIME_VERSION", "0.4.9")
    previous, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=3, output=str(tmp_path / "then"), clock=CLOCK,
    )
    assert current.forecast_id != previous.forecast_id
    assert current.runtime_version == "0.5.0"


def test_artifact_json_carries_the_runtime_stamp(tmp_path):
    source = _series_csv(tmp_path)
    _, directory = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=3, output=str(tmp_path / "out"), clock=CLOCK,
    )
    stored = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    from gnomon.versioning import RUNTIME_VERSION
    assert stored["runtime_version"] == RUNTIME_VERSION


def test_get_artifact_names_a_foreign_build(tmp_path):
    """Agents are told to quote artifacts verbatim, so an artifact computed
    by another build must say so at the tool that serves quotes."""
    from gnomon.toolspec import runner_for

    source = _series_csv(tmp_path)
    _, directory = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=3, output=str(tmp_path / "out"), clock=CLOCK,
    )
    current = runner_for("gnomon_get_artifact")({"artifact_path": str(directory)})
    assert "runtime_note" not in current

    artifact_path = directory / "artifact.json"
    doctored = json.loads(artifact_path.read_text(encoding="utf-8"))
    doctored["runtime_version"] = "0.4.0"
    artifact_path.write_text(json.dumps(doctored), encoding="utf-8")
    stale = runner_for("gnomon_get_artifact")({"artifact_path": str(directory)})
    assert "0.4.0" in stale["runtime_note"]

    doctored.pop("runtime_version")  # artifacts from before the stamp existed
    artifact_path.write_text(json.dumps(doctored), encoding="utf-8")
    unstamped = runner_for("gnomon_get_artifact")({"artifact_path": str(directory)})
    assert "unstamped" in unstamped["runtime_note"]
