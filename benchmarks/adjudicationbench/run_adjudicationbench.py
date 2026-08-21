"""Fresh-seed benchmark for evidence authority and synthesis admission.

This benchmark does not use TemporalBench questions or labels.  It perturbs
receipt order and property vocabularies to test invariants: supported fitted
answers remain binding, weak disagreement stays advisory, independently
supported opposition can earn synthesis, context needs outcome provenance,
and the adjudicator never manufactures calibrated probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.immutability import call_preserving_inputs  # noqa: E402
from gnomon.temporal_adjudication import adjudicate_temporal_evidence  # noqa: E402


LABELS = {
    "level": ("higher", "similar", "lower"),
    "trend": ("upward", "constant", "downward"),
    "seasonality": ("strengthened", "stable", "weakened"),
    "volatility": ("increased", "stable", "decreased"),
    "regime": ("shift", "no_shift", "reversal"),
    "extreme": ("more_frequent", "stable", "less_frequent"),
    "dependence": ("stronger", "stable", "weaker"),
}


def _result(canonical: str, support: str, probabilities: bool) -> dict:
    answer = {}
    if probabilities:
        answer["property_distribution"] = {
            "probabilities": {canonical: .7, "other": .3}, "folds": 12,
            "balanced_accuracy": .75, "brier_skill": .08,
        }
    return {"best_estimate": {"value": canonical, "support": support},
            "answer": answer}


def run(seed: int = 92417, replicates: int = 20) -> dict[str, object]:
    rng = random.Random(seed)
    rows = []
    for prop, labels in LABELS.items():
        for replicate in range(replicates):
            canonical, alternative, third = rng.sample(list(labels), 3)
            scenarios = [
                ("binding_canonical", "supported", [
                    {"kind": "observed_transition", "direction": alternative,
                     "support": "supported"},
                    {"kind": "historical_analogues", "direction": alternative,
                     "support": "supported"}], None, False),
                ("single_opposition", "weak", [
                    {"kind": "observed_transition", "direction": alternative,
                     "support": "supported"}], None, False),
                ("independent_opposition", "weak", [
                    {"kind": "observed_transition", "direction": alternative,
                     "support": "supported"},
                    {"kind": "cross_series_observed_evidence",
                     "direction": alternative, "support": "supported"}],
                 None, True),
                ("conflicting_opposition", "weak", [
                    {"kind": "observed_transition", "direction": alternative,
                     "support": "supported"},
                    {"kind": "cross_series_observed_evidence",
                     "direction": third, "support": "supported"}], None, False),
                ("unvalidated_context", "weak", [], {
                    "distribution": {"location": 2.0, "sample_size": 0},
                    "provenance": {"provenance_class": "human_assumption",
                                   "outcome_validated": False}}, False),
            ]
            for scenario, support, evidence, effect, expected in scenarios:
                rng.shuffle(evidence)
                fitted = replicate % 2 == 0
                # Behavioral immutability: hand the adjudicator the original
                # result/evidence/effect objects and compare them element-wise
                # to a pre-call deep copy — its own primary_forecast_unchanged
                # diagnostic is a constant and cannot gate anything.
                adjudication, inputs_unmutated = call_preserving_inputs(
                    adjudicate_temporal_evidence,
                    _result(canonical, support, fitted), evidence=evidence,
                    conditional_effect=effect)
                eligibility = adjudication["synthesis_eligibility"]["eligible"]
                candidates = adjudication["candidates"]
                invented = (not fitted and any(
                    "fitted_probability" in item for item in candidates))
                preserved = (fitted and any(
                    item.get("value") == canonical
                    and item.get("fitted_probability") == .7
                    for item in candidates))
                rows.append({
                    "property": prop, "scenario": scenario,
                    "expected_eligible": expected, "eligible": eligibility,
                    "correct": eligibility is expected,
                    # Attested by the engine; recorded, never gated on.
                    "primary_unchanged": adjudication[
                        "primary_forecast_unchanged"],
                    "inputs_unmutated": inputs_unmutated,
                    "invented_probability": invented,
                    "fitted_probability_preserved": preserved if fitted else None,
                })
    scenarios = sorted({row["scenario"] for row in rows})
    by_scenario = {
        name: {"cases": len(subset := [row for row in rows
                                      if row["scenario"] == name]),
               "accuracy": statistics.mean(row["correct"] for row in subset)}
        for name in scenarios
    }
    gates = {
        "authority_accuracy_100pct": all(row["correct"] for row in rows),
        # Behavioral, not attested: primary_unchanged is the engine's own
        # constant-True diagnostic, so the gate is the observed before/after
        # comparison of every adjudicate call's inputs instead.
        "inputs_unmutated_100pct": bool(rows) and all(
            row["inputs_unmutated"] for row in rows),
        "invented_probability_0pct": not any(
            row["invented_probability"] for row in rows),
        "fitted_probability_preserved_100pct": all(
            row["fitted_probability_preserved"] is not False for row in rows),
        "every_property_and_scenario_present": len(rows) == (
            len(LABELS) * replicates * 5),
    }
    return {"schema_version": "0.1", "seed": seed, "replicates": replicates,
            "cases": len(rows), "by_scenario": by_scenario, "gates": gates,
            "graduated": all(gates.values()), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=92417)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run(args.seed, args.replicates)
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, sort_keys=True))
    return 0 if result["graduated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
