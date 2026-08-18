import sys

sys.path.insert(0, ".")

from benchmarks.effectbench.generate import generate
from benchmarks.effectbench.run import run


def test_effectbench_has_required_denominators_and_passes_reference_system(tmp_path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "run"
    generate(corpus, seed=8127, cases=80)
    summary = run(corpus, output)
    assert summary["gates"]["complete"]
    assert summary["gates"]["false_influence_controls_present"]
    assert summary["gates"]["heldout_event_types_present"]
    assert summary["gates"]["false_influence_below_1pct"]
    assert summary["gates"]["heldout_event_types_abstain"]
    assert summary["gates"]["effect_coverage_not_demonstrably_below_80pct"]
    assert summary["gates"]["robust_decision_reduces_regret"]
    assert summary["decision_ready"] is True
