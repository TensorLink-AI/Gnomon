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
