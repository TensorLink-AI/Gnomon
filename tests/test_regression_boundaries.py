"""Release CI preserves numeric regressions without treating experiments as proof."""

from pathlib import Path

import yaml


def test_ci_separates_production_and_historical_adapter_gates():
    root = Path(__file__).resolve().parents[1]
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]
    production = [step.get("run", "") for step in jobs["test"]["steps"]]
    historical = [step.get("run", "") for step in jobs["benchmark-compatibility"]["steps"]]
    assert "pytest -q tests" in production
    assert "pytest -q benchmarks/tests" in historical
    assert jobs["test"]["strategy"]["matrix"]["python-version"] == ["3.11", "3.12", "3.13"]
    assert any("--dry-run" in command for command in historical)
    assert not any("--dry-run" not in command and "benchmarks.run_all" in command for command in historical)
