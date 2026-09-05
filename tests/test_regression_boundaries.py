"""Release CI preserves numeric regressions without treating experiments as proof."""

from pathlib import Path

import yaml


def test_ci_separates_production_and_current_evaluation_harness_gates():
    root = Path(__file__).resolve().parents[1]
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]
    production = [step.get("run", "") for step in jobs["test"]["steps"]]
    historical = [step.get("run", "") for step in jobs["evaluation-harness"]["steps"]]
    assert "pytest -q tests" in production
    assert "pytest -q benchmarks/tests" in historical
    assert jobs["test"]["strategy"]["matrix"]["python-version"] == ["3.11", "3.12", "3.13"]
    assert not any("benchmarks.run_all" in command for command in historical)
    assert not (root / ".github/workflows/benchmarks.yml").exists()


def test_release_requires_verified_build_and_marks_prereleases():
    root = Path(__file__).resolve().parents[1]
    jobs = yaml.safe_load((root / ".github/workflows/release.yml").read_text())["jobs"]
    commands = "\n".join(step.get("run", "") for step in jobs["build"]["steps"])
    assert "pytest -q tests benchmarks/tests" in commands
    assert "twine check dist/*" in commands
    assert "offline_wheel_smoke.py" in commands
    assert jobs["publish"]["needs"] == "build"
    assert jobs["publish"]["environment"]["name"] == "pypi"
    assert jobs["publish"]["permissions"] == {"id-token": "write"}
    release = "\n".join(step.get("run", "") for step in jobs["github-release"]["steps"])
    assert "--prerelease --latest=false" in release
