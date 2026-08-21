import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.release import build, validate


PROVENANCE = {
    "evaluated_commit": "abc123def",
    "harness_commit": "abc123def",
    "dataset": "in-repo synthetic generators",
    "provider_model": "none (deterministic evaluation; no LLM)",
    "configuration": "seed 1, replicates 2",
    "validity_limitations": ["single fixed seed"],
}


def _spec(tmp_path: Path, source: Path, **overrides) -> Path:
    entry = {
        "benchmark": "example", "arm": "core", "scope": "full",
        "source": str(source), "file": "example.json",
        "provenance": dict(PROVENANCE),
    }
    entry.update(overrides)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "release": "test", "output_dir": str(tmp_path / "release"),
        "benchmarks": [entry],
    }), encoding="utf-8")
    return spec


def test_release_builder_removes_case_data_and_records_digest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "score": .8, "rows": [{"prompt": "private"}],
        "breakdown": {"easy": .9},
        "shards": ["/root/Gnomon/results/run/shard-0"],
    }), encoding="utf-8")
    monkeypatch.setattr("benchmarks.release._git_sha", lambda: "abc123")
    output = Path(str(build(_spec(tmp_path, source))))
    payload = json.loads((output / "example.json").read_text())
    assert "rows" not in payload["summary"]
    assert payload["summary"]["breakdown"] == {"easy": .9}
    assert payload["summary"]["shards"] == ["results/run/shard-0"]
    assert payload["release_metadata"]["source_sha256"] == hashlib.sha256(
        source.read_bytes()).hexdigest()
    assert payload["release_metadata"]["provenance"] == PROVENANCE
    validate(output)


def test_release_builder_refuses_missing_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    spec = _spec(tmp_path, source)
    stripped = json.loads(spec.read_text())
    del stripped["benchmarks"][0]["provenance"]
    spec.write_text(json.dumps(stripped), encoding="utf-8")
    with pytest.raises(ValueError, match="missing provenance"):
        build(spec)


def test_release_builder_refuses_empty_validity_limitations(
        tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    provenance = dict(PROVENANCE, validity_limitations=[])
    with pytest.raises(ValueError, match="validity limitation"):
        build(_spec(tmp_path, source, provenance=provenance))


def _manifest_with(tmp_path: Path, payload: dict, **record_overrides) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    record = {
        "benchmark": "x", "arm": None, "file": "summary.json",
        "scope": "full", "status": "complete",
        "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    }
    record.update(record_overrides)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": "1.1",
        "benchmarks": [record],
    }), encoding="utf-8")


def test_release_validator_rejects_digest_tampering(tmp_path: Path) -> None:
    _manifest_with(tmp_path, {}, sha256="wrong")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate(tmp_path)


def test_release_validator_rejects_sensitive_fields(tmp_path: Path) -> None:
    _manifest_with(tmp_path, {
        "release_metadata": {"scope": "full", "provenance": dict(PROVENANCE)},
        "summary": {"api_key": "not-allowed"},
    })
    with pytest.raises(ValueError, match="sensitive fields"):
        validate(tmp_path)


def test_release_validator_rejects_missing_row_provenance(
        tmp_path: Path) -> None:
    _manifest_with(tmp_path, {
        "release_metadata": {"scope": "full"},
        "summary": {},
    })
    with pytest.raises(ValueError, match="missing provenance"):
        validate(tmp_path)


def test_release_validator_rejects_undated_withdrawal(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": "1.1",
        "benchmarks": [],
        "withdrawn": [{"benchmark": "x", "reason": "invalid instrument"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="withdrawn"):
        validate(tmp_path)


def test_committed_release_validates() -> None:
    release = Path(__file__).resolve().parents[2] / \
        "results/benchmark-releases/2026-08-21"
    validate(release)
