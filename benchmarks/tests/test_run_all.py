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
    cmd = _cmd("cik", "cik", {"method": "aion-pure", "seeds": 3})
    text = " ".join(cmd)
    assert "--method aion-pure" in text and "--seeds 3" in text
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


def test_mtbench_aion_gets_output_and_limit():
    cmd = _cmd("mtbench", "mtb-a",
               {"subcommand": "aion", "dataset_folder": "~/d",
                "mode": "agent"})
    text = " ".join(cmd)
    assert cmd[3] == "aion"
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
