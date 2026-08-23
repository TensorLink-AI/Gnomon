import json

import pytest

from benchmarks.temporalbench.merge_shards import merge_shards


def _shard(path, task_id, value):
    path.mkdir()
    (path / "details").mkdir()
    record = {"task_id": task_id, "success": True, "value": value}
    (path / "gnomonbench.jsonl").write_text(json.dumps(record) + "\n")
    (path / "details" / f"{task_id}.json").write_text(
        json.dumps({"answer": value}))


def _usage(path, requests, prompt_tokens):
    (path / "summary.json").write_text(json.dumps({"llm_usage": {
        "requests": requests,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": requests * 2,
    }}))


def _manifest(path, **extra):
    payload = {"benchmark": "temporalbench", "target": "tiers=T2,T4",
               "condition": "gnomon-mcp", "command": str(path), **extra}
    (path / "manifest.json").write_text(json.dumps(payload))


def test_merge_shards_builds_resumable_union(tmp_path):
    left, right, target = tmp_path / "left", tmp_path / "right", tmp_path / "all"
    _shard(left, "b", 2)
    _shard(right, "a", 1)
    result = merge_shards(target, [left, right])
    assert result == {"records": 2, "details": 2}
    rows = [json.loads(line) for line in
            (target / "gnomonbench.jsonl").read_text().splitlines()]
    assert [row["task_id"] for row in rows] == ["a", "b"]


def test_merge_shards_preserves_verified_manifest(tmp_path):
    left, right, target = tmp_path / "left", tmp_path / "right", tmp_path / "all"
    _shard(left, "a", 1)
    _shard(right, "b", 2)
    _manifest(left)
    _manifest(right)

    merge_shards(target, [left, right])

    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["benchmark"] == "temporalbench"
    assert manifest["rows"] == 2
    assert len(manifest["merged_shards"]) == 2
    assert len(manifest["source_commands"]) == 2


def test_merge_shards_accepts_documented_resume_recovery(tmp_path):
    left, right, target = tmp_path / "left", tmp_path / "right", tmp_path / "all"
    _shard(left, "a", 1)
    _shard(right, "b", 2)
    _manifest(left, model="same", resume=True)
    _manifest(right, model="same")

    merge_shards(target, [left, right])

    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["rows"] == 2
    assert "resume" not in manifest


def test_merge_shards_accepts_different_partition_controls(tmp_path):
    left, right, target = tmp_path / "left", tmp_path / "right", tmp_path / "all"
    _shard(left, "a", 1)
    _shard(right, "b", 2)
    _manifest(left, model="same", offset=0, limit=5)
    _manifest(right, model="same", offset=5, limit=2)

    merge_shards(target, [left, right])

    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["rows"] == 2
    assert "offset" not in manifest
    assert "limit" not in manifest


def test_merge_shards_accepts_interrupted_and_complete_execution_states(tmp_path):
    left, right, target = tmp_path / "left", tmp_path / "right", tmp_path / "all"
    _shard(left, "a", 1)
    _shard(right, "b", 2)
    _manifest(left, model="same", run_status="in_progress")
    _manifest(right, model="same", run_status="complete")

    merge_shards(target, [left, right])

    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["rows"] == 2
    assert "run_status" not in manifest


def test_merge_shards_rejects_incompatible_manifests(tmp_path):
    left, right, target = tmp_path / "left", tmp_path / "right", tmp_path / "all"
    _shard(left, "a", 1)
    _shard(right, "b", 2)
    _manifest(left, model="one")
    _manifest(right, model="two")

    with pytest.raises(ValueError, match="incompatible shard manifest.*model"):
        merge_shards(target, [left, right])


def test_merge_shards_accumulates_usage_once_across_repeated_merge(tmp_path):
    left, right, target = tmp_path / "left", tmp_path / "right", tmp_path / "all"
    _shard(left, "a", 1)
    _shard(right, "b", 2)
    _usage(left, 2, 100)
    _usage(right, 3, 250)
    for path, retries, reason in (
        (left, 1, "HTTP 502"), (right, 2, "timeout")):
        payload = json.loads((path / "summary.json").read_text())
        payload["infrastructure_retries"] = retries
        payload["infrastructure_failures_retried"] = {reason: retries}
        (path / "summary.json").write_text(json.dumps(payload))

    merge_shards(target, [left, right])
    merge_shards(target, [left, right])

    summary = json.loads((target / "summary.json").read_text())
    assert summary["llm_usage"]["requests"] == 5
    assert summary["llm_usage"]["prompt_tokens"] == 350
    assert summary["infrastructure_retries"] == 3
    assert summary["infrastructure_failures_retried"] == {
        "HTTP 502": 1, "timeout": 2}
    assert len(summary["merged_usage_sources"]) == 2


def test_merge_shards_rejects_conflicts(tmp_path):
    left, right, target = tmp_path / "left", tmp_path / "right", tmp_path / "all"
    _shard(left, "a", 1)
    _shard(right, "a", 2)
    with pytest.raises(ValueError, match="conflicting"):
        merge_shards(target, [left, right])


def test_merge_shards_recovers_complete_lines_from_interrupted_partial(tmp_path):
    shard, target = tmp_path / "shard", tmp_path / "all"
    _shard(shard, "a", 1)
    (shard / "gnomonbench.jsonl").rename(
        shard / "gnomonbench.partial.jsonl")

    result = merge_shards(target, [shard])

    assert result == {"records": 1, "details": 1}
    record = json.loads((target / "gnomonbench.jsonl").read_text())
    assert record["task_id"] == "a"


def test_merge_shards_recovers_usage_checkpoint_from_interrupted_shard(tmp_path):
    shard, target = tmp_path / "shard", tmp_path / "all"
    _shard(shard, "a", 1)
    (shard / "gnomonbench.jsonl").rename(
        shard / "gnomonbench.partial.jsonl")
    (shard / "usage.checkpoint.json").write_text(json.dumps({
        "llm_usage": {"requests": 4, "prompt_tokens": 200},
        "completed_details": 1,
    }))

    merge_shards(target, [shard])

    summary = json.loads((target / "summary.json").read_text())
    assert summary["llm_usage"]["requests"] == 4
    assert summary["llm_usage"]["prompt_tokens"] == 200
