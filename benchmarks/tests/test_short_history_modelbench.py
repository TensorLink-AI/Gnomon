import subprocess
import sys
from pathlib import Path

from benchmarks.modelbench.run_short_history import run
from benchmarks.modelbench.run_production_selector import run as run_production_selector


def test_short_history_benchmark_is_deterministic_and_retains_records():
    first = run(seed=91, cases_per_family=8)
    second = run(seed=91, cases_per_family=8)
    assert first == second
    assert len(first["raw_records"]) == 72
    assert all(first["gates"].values())
    assert first["pooling"]["mixed_direction"]["admitted"] == 0


def test_short_history_runner_works_from_documented_direct_entrypoint(tmp_path):
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "benchmarks/modelbench/run_short_history.py",
         "--seed", "91", "--cases-per-family", "40", "--output-dir",
         str(tmp_path / "result")],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert completed.returncode in {0, 2}, completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr
    assert (tmp_path / "result" / "summary.json").exists()


def test_pooling_admission_is_stable_across_independent_seeds():
    results = [run(seed=seed, cases_per_family=80)
               for seed in (82631, 19087, 55109, 9103, 9104)]
    strong = [result["pooling"]["comparable_strong"] for result in results]
    admitted = sum(item["admitted"] for item in strong)
    uplift = sum(item["outcomes"]["uplift"] for item in strong)
    assert admitted >= 250  # useful yield, not safety by universal fallback
    assert uplift / admitted >= .90
    assert sum(result["pooling"]["comparable_marginal"]["admitted"]
               for result in results) == 0
    assert sum(result["pooling"]["null"]["admitted"]
               for result in results) == 0
    assert sum(result["pooling"]["mixed_direction"]["admitted"]
               for result in results) == 0


def test_production_selector_screen_is_deterministic_and_prefix_only():
    first = run_production_selector(seed=91, cases_per_family=4)
    second = run_production_selector(seed=91, cases_per_family=4)
    assert first == second
    assert len(first["raw_records"]) == 20
    assert 0 < first["overall"]["completion_rate"] <= 1
    assert first["gates"]["future_observations_used_zero"]
    assert first["gates"]["no_silent_fallback"]
    assert first["gates"]["selection_provenance_complete"]
    assert {row["length_lane"] for row in first["raw_records"]} == {
        "short_horizon", "fold_starved_long_horizon",
    }
    fallbacks = [row for row in first["raw_records"]
                 if not row["engine_supported"]]
    assert all(row["fallback_disclosed"] for row in fallbacks)
    assert all(row["published_support"] == "best_effort"
               for row in fallbacks)


def test_production_selector_runner_retains_failed_gate_output(tmp_path):
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable,
         "benchmarks/modelbench/run_production_selector.py",
         "--seed", "91", "--cases-per-family", "4", "--output-dir",
         str(tmp_path / "result")],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert completed.returncode in {0, 2}, completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr
    assert (tmp_path / "result" / "summary.json").exists()
    assert (tmp_path / "result" / "manifest.json").exists()
