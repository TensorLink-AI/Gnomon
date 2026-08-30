import subprocess
import sys
from pathlib import Path

from benchmarks.decisioninputbench.run import CASES, run


def test_decision_input_benchmark_is_deterministic_and_complete():
    first = run()
    second = run()
    assert first == second
    assert first["cases"] == len(CASES) == 11
    assert len(first["raw_records"]) == 11
    assert {row["case_id"] for row in first["raw_records"]} == {
        case["id"] for case in CASES
    }


def test_decision_input_runner_retains_failing_reproduction(tmp_path):
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "benchmarks/decisioninputbench/run.py",
         "--output-dir", str(tmp_path / "result")],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert completed.returncode in {0, 2}, completed.stderr
    assert (tmp_path / "result" / "summary.json").exists()
    assert (tmp_path / "result" / "manifest.json").exists()
