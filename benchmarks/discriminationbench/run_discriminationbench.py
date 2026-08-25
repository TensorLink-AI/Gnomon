"""Known-truth evaluation of held-out hypothesis discrimination.

Every case is generated with a known true interpretation (the seeded
process that produced the values), so the mechanism is falsifiable without
an LLM: does `gnomon.discrimination` point at the truth, does its
"separation" grade mean what it says, and does it ever exclude the truth
outright?  This measures the discriminating-evidence mechanism itself —
model uplift from consuming it is a separate, matched LLM experiment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.manifest import code_revision, write_manifest  # noqa: E402
from gnomon.discrimination import discriminate  # noqa: E402


def _series(rng: random.Random) -> tuple[str, str, list[float]]:
    """One generated case: (property, true_interpretation, values)."""
    n = rng.choice([40, 60, 90])
    k = max(4, n // 5)
    sigma = 1.0
    base = rng.uniform(20, 200)
    noise = [rng.gauss(0, sigma) for _ in range(n)]
    property = rng.choice(["trend", "level", "volatility", "disturbance"])
    if property == "trend":
        truth = rng.choice(["upward", "downward", "constant"])
        slope = (0.0 if truth == "constant" else
                 rng.uniform(.15, .5) * (1 if truth == "upward" else -1))
        values = [base + slope * index + noise[index] for index in range(n)]
    elif property == "level":
        truth = rng.choice(["higher", "lower", "similar"])
        shift = (0.0 if truth == "similar" else
                 rng.uniform(3, 6) * sigma * (1 if truth == "higher" else -1))
        values = [base + (shift if index >= n - 2 * k else 0.0) + noise[index]
                  for index in range(n)]
    elif property == "volatility":
        truth = rng.choice(["increased", "decreased", "stable"])
        factor = {"increased": 3.0, "decreased": 1 / 3, "stable": 1.0}[truth]
        values = [base + noise[index] * (factor if index >= n - 2 * k else 1.0)
                  for index in range(n)]
    else:
        truth = rng.choice(["sudden_spike", "level_shift", "stable"])
        at = n - k - 2
        jump = rng.uniform(9, 14) * sigma
        values = [base + noise[index] for index in range(n)]
        if truth == "sudden_spike":
            values[at] += jump
        elif truth == "level_shift":
            for index in range(at, n):
                values[index] += jump
    return property, truth, values


def run(seed: int = 20260824, cases: int = 400) -> dict[str, object]:
    rng = random.Random(seed)
    rows = []
    for _ in range(cases):
        property, truth, values = _series(rng)
        payload = discriminate(values, property=property)
        weights = {row["value"]: row["relative_weight"]
                   for row in payload.get("hypotheses", [])}
        rows.append({
            "property": property, "truth": truth,
            "identifiable": bool(payload.get("identifiable")),
            "best": payload.get("best"),
            "separation": payload.get("separation"),
            "correct": payload.get("best") == truth,
            # The truth was excluded when its hypothesis is listed with
            # exactly zero weight — the surrogate set argued it cannot be
            # the explanation while it in fact was.
            "truth_excluded": weights.get(truth) == 0.0,
        })
    identifiable = [row for row in rows if row["identifiable"]]
    clear = [row for row in identifiable if row["separation"] == "clear"]
    null_truths = {"constant", "similar", "stable"}
    transition_cases = [row for row in identifiable
                        if row["truth"] not in null_truths]
    null_cases = [row for row in identifiable if row["truth"] in null_truths]
    false_clear_transitions = [
        row for row in null_cases
        if row["separation"] == "clear" and row["best"] not in null_truths
    ]

    def accuracy(subset: list[dict]) -> float | None:
        return (sum(row["correct"] for row in subset) / len(subset)
                if subset else None)

    per_property = {}
    for name in ("trend", "level", "volatility", "disturbance"):
        subset = [row for row in identifiable if row["property"] == name]
        per_property[name] = {
            "cases": len(subset), "accuracy": accuracy(subset),
            "clear_rate": (sum(row["separation"] == "clear" for row in subset)
                           / len(subset)) if subset else None,
        }
    summary = {
        "schema_version": "0.1", "seed": seed, "cases": cases,
        "identifiable_rate": len(identifiable) / cases,
        "accuracy": accuracy(identifiable),
        "clear_separation": {
            "cases": len(clear), "share": len(clear) / max(len(identifiable), 1),
            "accuracy": accuracy(clear),
        },
        "transition_accuracy": accuracy(transition_cases),
        "null_accuracy": accuracy(null_cases),
        "truth_excluded_rate":
            sum(row["truth_excluded"] for row in identifiable)
            / max(len(identifiable), 1),
        "false_clear_transition_rate":
            len(false_clear_transitions) / max(len(null_cases), 1),
        "per_property": per_property,
    }
    summary["gates"] = {
        # The generator only emits histories long enough for the split.
        "always_identifiable": summary["identifiable_rate"] == 1.0,
        # Above chance overall (three-way vocabularies -> 1/3 chance).
        "accuracy_beats_chance": (summary["accuracy"] or 0) >= .6,
        # "clear" must mean what it says.
        "clear_separation_is_reliable":
            (summary["clear_separation"]["accuracy"] or 0) >= .9,
        # The mechanism may be unsure; it must almost never rule the
        # truth out.
        "truth_rarely_excluded": summary["truth_excluded_rate"] <= .05,
        # Quiet series must not confidently manufacture transitions.
        "no_confident_false_transitions":
            summary["false_clear_transition_rate"] <= .1,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--cases", type=int, default=400)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    run_revision = code_revision()
    summary = run(arguments.seed, arguments.cases)
    summary["evaluated_commit"] = run_revision
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    output = arguments.output
    if arguments.output_dir:
        output = arguments.output_dir / "summary.json"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
        write_manifest(
            output.parent, benchmark="discriminationbench",
            condition="held-out-hypothesis-fit",
            target=f"seed={arguments.seed};cases={arguments.cases}",
            code_revision=run_revision,
        )
    print(encoded)
    return 0 if all(summary["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
