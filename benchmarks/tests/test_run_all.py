"""Tests for the run_all orchestrator's command building (no execution)."""

import sys
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.catalog import CATALOG
from benchmarks.run_all import (
    REGISTRY, build_command, completed_gate_failure, summary_path,
)

CONFIG = {
    "model": "openai/gpt-4o",
    "output_root": "results/batch",
    "defaults": {"limit": 25, "temperature": 0.3},
}


def test_registry_and_claim_catalog_cover_the_same_benchmarks():
    # Every batch-orchestrated adapter needs a claim boundary. Frozen protocol
    # runners with bespoke output/checkpoint contracts may remain catalogued
    # without being forced through this generic orchestrator.
    assert set(REGISTRY) <= set(CATALOG)
    assert CATALOG["temporalbench"].layer == "reasoning_harness"
    assert CATALOG["propertybench"].layer == "engine"
    assert CATALOG["effectbench"].layer == "safety_contract"


@pytest.mark.parametrize("module", [
    "benchmarks.effectbench.run_effectbench",
    "benchmarks.admissionbench.run_admissionbench",
    "benchmarks.contextbench.run_llm",
    "benchmarks.contextbench.run_surfaces",
    "benchmarks.modelbench.build_tsfm_registry",
])
def test_documented_module_entrypoints_import_cleanly(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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


@pytest.mark.parametrize(("benchmark", "gate"), [
    ("workflow", "release_gate_pass"),
    ("contextbench", "decision_ready"),
])
def test_completed_gate_failure_requires_a_real_negative_summary(
        tmp_path, benchmark, gate):
    config = {"output_root": str(tmp_path)}
    run = {"benchmark": benchmark, "name": "case", "args": {}}
    output = tmp_path / "case"
    output.mkdir()
    (output / "summary.json").write_text(
        '{"' + gate + '": false}', encoding="utf-8")
    assert completed_gate_failure(run, config, 2) is True
    assert completed_gate_failure(run, config, 3) is False
    (output / "summary.json").write_text(
        '{"' + gate + '": true}', encoding="utf-8")
    assert completed_gate_failure(run, config, 2) is False


def test_argparse_exit_two_is_not_scored_evidence(tmp_path):
    run = {"benchmark": "contextbench", "name": "missing", "args": {}}
    assert completed_gate_failure(
        run, {"output_root": str(tmp_path)}, 2) is False


def test_registry_covers_all_adapters():
    root = Path(__file__).resolve().parents[1]
    active = {
        p.name for p in root.iterdir()
        if p.is_dir() and (
            (p / "run.py").is_file() or any(p.glob("run_*.py")))
    }
    active.add("capabilitybench")
    assert active <= set(CATALOG)
    assert set(REGISTRY) <= active


def test_reasoningbench_gets_model_output_and_case_limit():
    cmd = _cmd("reasoningbench", "reason", {})
    text = " ".join(cmd)
    assert "--model openai/gpt-4o" in text
    assert "--cases 25" in text
    assert "--output-dir results/batch/reason" in text


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
