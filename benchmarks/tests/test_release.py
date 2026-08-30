import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.release import build, validate


def test_release_builder_removes_case_data_and_records_digest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "score": .8, "rows": [{"prompt": "private"}],
        "results": [{"response": "also private"}],
        "raw_records": [{"ground_truth": "private future"}],
        "raw_paired_channel_records": [{"task_id": "also private"}],
        "breakdown": {"easy": .9},
        "shards": ["/root/Gnomon/results/run/shard-0"],
        "external_dataset": "/runner/external/data.parquet",
    }), encoding="utf-8")
    output = tmp_path / "release"
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "release": "test", "output_dir": str(output),
        "benchmarks": [{
            "benchmark": "example", "arm": "core", "scope": "full",
            "source": str(source), "file": "example.json",
            "evaluated_commit": "evaluated123",
            "harness_commit": "harness123",
            "dataset_identity": "fixture-v1",
            "configuration_identity": "config-v1",
        }],
    }), encoding="utf-8")
    monkeypatch.setattr("benchmarks.release._git_sha", lambda: "abc123")
    build(spec)
    payload = json.loads((output / "example.json").read_text())
    assert "rows" not in payload["summary"]
    assert "results" not in payload["summary"]
    assert "raw_records" not in payload["summary"]
    assert "raw_paired_channel_records" not in payload["summary"]
    assert payload["summary"]["breakdown"] == {"easy": .9}
    assert payload["summary"]["shards"] == ["results/run/shard-0"]
    assert payload["summary"]["external_dataset"] == "<absolute-path-redacted>"
    assert payload["release_metadata"]["source_sha256"] == hashlib.sha256(
        source.read_bytes()).hexdigest()
    validate(output)


def test_release_builder_refuses_publishable_unknown_provenance(
        tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"score": 1}\n', encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "release": "test", "output_dir": str(tmp_path / "release"),
        "benchmarks": [{"benchmark": "example", "scope": "full",
                        "source": str(source), "file": "example.json"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown provenance"):
        build(spec)


def test_release_validator_rejects_digest_tampering(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": "1.1",
        "benchmarks": [{
            "benchmark": "x", "arm": None, "file": "summary.json",
            "scope": "full", "status": "complete", "sha256": "wrong",
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate(tmp_path)


def test_release_validator_rejects_sensitive_fields(tmp_path: Path) -> None:
    payload = {
        "release_metadata": {"scope": "full"},
        "summary": {"api_key": "not-allowed"},
    }
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": "1.1",
        "benchmarks": [{
            "benchmark": "x", "arm": None, "file": "summary.json",
            "scope": "full", "status": "complete", "sha256": digest,
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="sensitive fields"):
        validate(tmp_path)
