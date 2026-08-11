"""Tests for the run_all orchestrator's command building (no execution)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.run_all import REGISTRY, build_command, summary_path

CONFIG = {
    "model": "openai/gpt-4o",
    "output_root": "results/batch",
    "defaults": {"limit": 25, "temperature": 0.3},
}


def _cmd(benchmark, name, args, config=CONFIG):
    return build_command(
        {"benchmark": benchmark, "name": name, "args": args}, config
    )


def test_temporalbench_defaults_injected():
    cmd = _cmd("temporalbench", "tb-x",
               {"condition": "control", "data_dir": "~/tb"})
    assert cmd[:3] == [sys.executable, "-m",
                       "benchmarks.temporalbench.run_temporalbench"]
    text = " ".join(cmd)
    assert "--model openai/gpt-4o" in text
    assert "--limit 25" in text
    assert "--temperature 0.3" in text
    assert "--output-dir results/batch/tb-x" in text


def test_explicit_args_beat_defaults():
    cmd = _cmd("timesage_mt", "ts",
               {"condition": "direct", "data_dir": "~/ts",
                "model": "openai/gpt-4o-mini", "limit": 5})
    text = " ".join(cmd)
    assert "--model openai/gpt-4o-mini" in text
    assert "--limit 5" in text and "--limit 25" not in text


def test_cik_gets_no_limit_flag():
    cmd = _cmd("cik", "cik", {"method": "gnomon-pure", "seeds": 3})
    text = " ".join(cmd)
    assert "--method gnomon-pure" in text and "--seeds 3" in text
    assert "--limit" not in text and "--n-samples" not in text
    assert "--output-dir results/batch/cik" in text


def test_mtbench_control_subcommand_scope():
    cmd = _cmd("mtbench", "mtb-c",
               {"subcommand": "control", "mtbench_root": "~/mtb",
                "script": "run.py",
                "positional": ["--", "--dataset", "x"]})
    assert cmd[3] == "control"
    text = " ".join(cmd)
    assert "--model openai/gpt-4o" in text
    assert "--output-dir" not in text and "--limit" not in text
    assert cmd[-3:] == ["--", "--dataset", "x"]


def test_mtbench_gnomon_gets_output_and_limit():
    cmd = _cmd("mtbench", "mtb-a",
               {"subcommand": "gnomon", "dataset_folder": "~/d",
                "mode": "agent"})
    text = " ".join(cmd)
    assert cmd[3] == "gnomon"
    assert "--output-dir results/batch/mtb-a" in text
    assert "--limit 25" in text


def test_boolean_true_becomes_bare_flag():
    cmd = _cmd("cik", "cik", {"method": "control", "no_cache": True})
    assert "--no-cache" in cmd
    assert "True" not in cmd


def test_anomllm_nothing_injected():
    cmd = _cmd("anomllm", "an", {"anomllm_root": "~/a", "data": "trend"})
    text = " ".join(cmd)
    assert "--model" not in text and "--output-dir" not in text \
        and "--limit" not in text


def test_unknown_benchmark_raises():
    with pytest.raises(ValueError, match="unknown benchmark"):
        _cmd("nope", "x", {})


def test_summary_path_prefers_explicit_output_dir():
    run = {"benchmark": "cik", "name": "c",
           "args": {"output_dir": "/tmp/out"}}
    assert summary_path(run, CONFIG) == Path("/tmp/out/summary.json")
    run2 = {"benchmark": "cik", "name": "c", "args": {}}
    assert summary_path(run2, CONFIG) == Path("results/batch/c/summary.json")


def test_registry_covers_all_adapters():
    adapters = {p.name for p in
                (Path(__file__).resolve().parents[1]).iterdir()
                if p.is_dir() and list(p.glob("run_*.py"))}
    assert adapters == set(REGISTRY)


def test_orchestrator_completes_more_than_one_run(tmp_path, monkeypatch):
    """A batch of two must not die after the first.

    `args` held the parsed CLI namespace, and the manifest block inside the
    loop rebound it to the run's own arg dict — so the second iteration
    raised AttributeError on `args.dry_run`. Every dry-run test passed
    (dry runs `continue` before the rebind) and single-run batches passed,
    which is why the documented way to run the suite was broken for months.
    """
    import json
    import sys

    import benchmarks.run_all as run_all

    config = {
        "model": "m",
        "output_root": str(tmp_path / "out"),
        "runs": [
            {"benchmark": "leaktrap", "name": "one", "args": {"subcommand": "gnomon"}},
            {"benchmark": "leaktrap", "name": "two", "args": {"subcommand": "gnomon"}},
        ],
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    executed = []

    class _Result:
        returncode = 0

    def fake_run(command, cwd=None, **kwargs):
        if kwargs.get("capture_output"):
            # manifest.code_revision() shells out to git; leave it alone.
            return real_run(command, cwd=cwd, **kwargs)
        executed.append(command)
        # Adapters write their own summary; emulate just enough for the
        # manifest/summary block to exercise the same path as a real run.
        name = "one" if "one" in " ".join(command) else "two"
        directory = tmp_path / "out" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text('{"ok": true}', encoding="utf-8")
        return _Result()

    real_run = run_all.subprocess.run
    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv",
                        ["run_all", "--config", str(config_path)])
    assert run_all.main() == 0
    assert len(executed) == 2, "the second run must actually execute"


def test_orchestrator_merges_the_adapters_manifest_instead_of_replacing_it(
    tmp_path, monkeypatch,
):
    """The adapter records facts only it knows (base_url, best_effort, its
    real target); overwriting them dropped exactly the fields report.py's
    comparability refusal reads — two orchestrated TemporalBench runs with
    different tier sets compared without refusal because both manifests
    lost `target`."""
    import json
    import sys

    import benchmarks.run_all as run_all
    from benchmarks.common.manifest import read_manifest, write_manifest

    config = {
        "model": "m",
        "output_root": str(tmp_path / "out"),
        "runs": [
            {"benchmark": "temporalbench", "name": "tb",
             "args": {"condition": "gnomon-agent", "data_dir": "~/tb"}},
        ],
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class _Result:
        returncode = 0

    def fake_run(command, cwd=None, **kwargs):
        if kwargs.get("capture_output"):
            return real_run(command, cwd=cwd, **kwargs)
        directory = tmp_path / "out" / "tb"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text('{"ok": true}',
                                                encoding="utf-8")
        # What the real adapter writes at the end of its run.
        write_manifest(
            directory, benchmark="temporalbench", condition="gnomon-agent",
            target="tiers=T2,T4", base_url="http://local:8000/v1",
            best_effort=True, command="child-argv",
        )
        return _Result()

    real_run = run_all.subprocess.run
    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv",
                        ["run_all", "--config", str(config_path)])
    assert run_all.main() == 0
    manifest = read_manifest(tmp_path / "out" / "tb")
    # The adapter's own record survives...
    assert manifest["target"] == "tiers=T2,T4"
    assert manifest["base_url"] == "http://local:8000/v1"
    assert manifest["best_effort"] is True
    assert manifest["command"] == "child-argv"
    # ...and the orchestrator adds what only it knows.
    assert manifest["run_name"] == "tb"
    assert manifest["status"] == "ok"
    assert manifest["config_path"] == str(config_path)


def test_failed_run_does_not_trust_a_stale_adapter_manifest(
    tmp_path, monkeypatch,
):
    """Adapters write their manifest at the end of a successful run; on a
    non-zero exit whatever sits in the directory may be a previous run's
    record, and stale provenance is worse than sparse."""
    import json
    import sys

    import benchmarks.run_all as run_all
    from benchmarks.common.manifest import read_manifest, write_manifest

    directory = tmp_path / "out" / "tb"
    directory.mkdir(parents=True)
    write_manifest(directory, benchmark="temporalbench",
                   target="tiers=T1", base_url="http://stale:1/v1")

    config = {
        "model": "m",
        "output_root": str(tmp_path / "out"),
        "runs": [
            {"benchmark": "temporalbench", "name": "tb",
             "args": {"condition": "control", "data_dir": "~/tb"}},
        ],
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class _Result:
        returncode = 3

    def fake_run(command, cwd=None, **kwargs):
        if kwargs.get("capture_output"):
            return real_run(command, cwd=cwd, **kwargs)
        return _Result()

    real_run = run_all.subprocess.run
    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv",
                        ["run_all", "--config", str(config_path)])
    assert run_all.main() == 1
    manifest = read_manifest(directory)
    assert manifest["status"] == "exit 3"
    assert "base_url" not in manifest, "stale adapter fields must not survive"
    assert "target" not in manifest


def test_manifestless_child_does_not_inherit_the_previous_runs_manifest(
    tmp_path, monkeypatch,
):
    """For a child that writes no manifest of its own (mtbench's control
    arm runs the official script), the only manifest in the directory is
    the one run_all wrote for the PREVIOUS run. Re-reading it through the
    merge stamped run 2 with run 1's provenance — model, condition,
    command — under status ok, silently defeating the comparability
    refusal this file exists to feed."""
    import json
    import sys

    import benchmarks.run_all as run_all
    from benchmarks.common.manifest import read_manifest, write_manifest

    directory = tmp_path / "out" / "tb"
    directory.mkdir(parents=True)
    # What run_all itself wrote for the previous run of this config.
    write_manifest(directory, benchmark="temporalbench", model="m1",
                   condition="old-condition", command="old-cmd",
                   run_name="tb", status="ok")

    config = {
        "model": "m2",
        "output_root": str(tmp_path / "out"),
        "runs": [
            {"benchmark": "temporalbench", "name": "tb",
             "args": {"condition": "control", "data_dir": "~/tb"}},
        ],
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class _Result:
        returncode = 0

    def fake_run(command, cwd=None, **kwargs):
        if kwargs.get("capture_output"):
            return real_run(command, cwd=cwd, **kwargs)
        (directory / "summary.json").write_text('{"ok": true}',
                                                encoding="utf-8")
        return _Result()  # child writes a summary but no manifest

    real_run = run_all.subprocess.run
    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv",
                        ["run_all", "--config", str(config_path)])
    assert run_all.main() == 0
    manifest = read_manifest(directory)
    assert manifest["model"] == "m2", "run 2 must not carry run 1's model"
    assert manifest["condition"] == "control"
    assert manifest["command"] != "old-cmd"


def test_failed_rerun_does_not_ingest_a_stale_summary(tmp_path, monkeypatch):
    """A rerun whose child fails before writing summary.json must not
    record the previous run's summary as its own in combined_summary.json
    — that is the manifest staleness bug with a different file name."""
    import json
    import sys

    import benchmarks.run_all as run_all

    directory = tmp_path / "out" / "tb"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text('{"stale": true}',
                                            encoding="utf-8")

    config = {
        "model": "m",
        "output_root": str(tmp_path / "out"),
        "runs": [
            {"benchmark": "temporalbench", "name": "tb",
             "args": {"condition": "control", "data_dir": "~/tb"}},
        ],
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class _Result:
        returncode = 3

    def fake_run(command, cwd=None, **kwargs):
        if kwargs.get("capture_output"):
            return real_run(command, cwd=cwd, **kwargs)
        return _Result()

    real_run = run_all.subprocess.run
    monkeypatch.setattr(run_all.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv",
                        ["run_all", "--config", str(config_path)])
    assert run_all.main() == 1
    combined = json.loads(
        (tmp_path / "out" / "combined_summary.json").read_text())
    (outcome,) = combined["runs"]
    assert outcome["status"] == "exit 3"
    assert "summary" not in outcome, "a failed run has no summary of its own"
