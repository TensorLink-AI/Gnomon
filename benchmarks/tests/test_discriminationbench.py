from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.discriminationbench.run_discriminationbench import run  # noqa: E402


def test_all_gates_pass_on_the_default_configuration() -> None:
    summary = run(seed=20260824, cases=200)
    assert summary["gates"] == {name: True for name in summary["gates"]}
    assert summary["identifiable_rate"] == 1.0
    assert summary["accuracy"] > .6
    assert summary["clear_separation"]["accuracy"] >= .9


def test_the_benchmark_is_deterministic_for_a_seed() -> None:
    assert run(seed=7, cases=60) == run(seed=7, cases=60)


def test_every_property_is_exercised_and_scored() -> None:
    summary = run(seed=11, cases=200)
    for name in ("trend", "level", "volatility", "disturbance"):
        entry = summary["per_property"][name]
        assert entry["cases"] > 0
        assert entry["accuracy"] is not None
