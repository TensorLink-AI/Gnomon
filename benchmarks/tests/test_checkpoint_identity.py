import json

import pytest

from benchmarks.common.checkpoint import prepare_run_identity


def test_checkpoint_identity_requires_explicit_matching_resume(tmp_path):
    identity = {"schema_version": 1, "benchmark": "example", "seed": 4}
    state = tmp_path / "cases.jsonl"
    prepare_run_identity(
        tmp_path, identity, resume=False, state_paths=[state])
    assert json.loads((tmp_path / "run_identity.json").read_text()) == identity

    with pytest.raises(SystemExit, match="pass --resume"):
        prepare_run_identity(
            tmp_path, identity, resume=False, state_paths=[state])
    prepare_run_identity(tmp_path, identity, resume=True, state_paths=[state])

    with pytest.raises(SystemExit, match="identity mismatch"):
        prepare_run_identity(
            tmp_path, {**identity, "seed": 5}, resume=True,
            state_paths=[state])


def test_checkpoint_identity_refuses_unidentified_legacy_state(tmp_path):
    state = tmp_path / "cases.jsonl"
    state.write_text("{}\n")
    with pytest.raises(SystemExit, match="without run_identity"):
        prepare_run_identity(
            tmp_path, {"schema_version": 1}, resume=True,
            state_paths=[state])
