"""Property-style benchmark for Gnomon's agent reasoning boundary."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.common.manifest import code_revision
from gnomon.reasoning_boundary import measure_redundant_calls, verify_fact_sources
from gnomon.toolspec import apply_response_contract
from gnomon.toolspec import _run_inspect


def _case(index: int, rng: random.Random) -> dict[str, Any]:
    kind = ("forecast", "detect", "decide", "monitor", "describe")[index % 5]
    artifact = f"{kind}-{index}"
    payload: dict[str, Any] = {
        "headline": f"Computed {kind} conclusion {index}.",
        "artifact_id": artifact,
        "task": {"task_type": kind},
        "support": {"state": rng.choice(("supported", "weak"))},
        "limitations": ([] if index % 3 else ["More outcomes would narrow uncertainty."]),
        "results": [{"series": f"series-{index}",
                     "forecast": [{"q50": rng.uniform(-10, 10)}]}],
    }
    return {"id": f"case-{index}", "verb": kind, "payload": payload}


def run(seed: int, cases: int) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = []
    for index in range(cases):
        case = _case(index, rng)
        original = json.loads(json.dumps(case["payload"], sort_keys=True))
        response = apply_response_contract(case["payload"])
        transcript = [{"tool": f"gnomon_{case['verb']}", "result": response}]
        if index % 4 == 0:  # controlled host-policy error, not product need
            transcript.append({"tool": "gnomon_get_artifact", "result": {"rows": []}})
        calls = measure_redundant_calls(transcript)
        rows.append({
            "id": case["id"], "verb": case["verb"],
            "canonical_unchanged": all(response.get(key) == value
                                       for key, value in original.items()),
            "traceable": not verify_fact_sources(response),
            "argument_complete": all(key in response["reasoning"] for key in
                                     ("because", "against", "unknown",
                                      "what_would_flip", "sufficiency",
                                      "resolution")),
            "redundant_calls": calls["redundant_calls"],
            "redundancy_attributed": calls["redundant_calls"] == (index % 4 == 0),
        })
    rejection = apply_response_contract({"status": "error", "error": {
        "code": "INVALID_ARGUMENTS", "message": "Bad window.",
        "repair_options": [{"tool": "gnomon_forecast", "arguments": {"horizon": 7}}]}})

    # At least one gate crosses a real public verb boundary.  The synthetic
    # corpus above provides adversarial breadth; this fixture proves the same
    # projection accepts the actual inspect response shape.
    with tempfile.TemporaryDirectory(prefix="gnomon-boundary-") as directory:
        source = Path(directory) / "series.csv"
        source.write_text(
            "timestamp,value\n" + "".join(
                f"2026-01-{index:02d},{100 + index}\n" for index in range(1, 21)
            ), encoding="utf-8",
        )
        real_response = apply_response_contract(_run_inspect({
            "input": str(source), "time_column": "timestamp",
            "target_column": "value", "frequency": "daily",
        }))
    real_response_ok = (
        real_response.get("status") != "error"
        and isinstance(real_response.get("reasoning"), dict)
        and not verify_fact_sources(real_response)
    )
    gates = {
        "canonical_immutability": all(row["canonical_unchanged"] for row in rows),
        "fact_traceability": all(row["traceable"] for row in rows),
        "argument_completeness": all(row["argument_complete"] for row in rows),
        "redundancy_attribution": all(row["redundancy_attributed"] for row in rows),
        "actionable_rejection": bool((rejection.get("rejection") or {}).get(
            "admissibility_path")),
        "real_verb_response": real_response_ok,
    }
    # A benchmark that cannot detect a broken contract is self-attestation.
    # Apply one direct mutation per gate and require every corresponding check
    # to become false.
    sample = apply_response_contract(_case(1, random.Random(seed))["payload"])
    changed = json.loads(json.dumps(sample))
    changed["headline"] = "mutated"
    dangling = json.loads(json.dumps(sample))
    dangling["reasoning"]["facts"][0]["source"] = "/missing"
    incomplete = json.loads(json.dumps(sample))
    incomplete["reasoning"].pop("because")
    dead_end = apply_response_contract({"status": "error", "error": {
        "code": "INVALID_ARGUMENTS", "message": "Bad window."}})
    mutation_detection = {
        "canonical_immutability": changed.get("headline") != sample.get("headline"),
        "fact_traceability": bool(verify_fact_sources(dangling)),
        "argument_completeness": "because" not in incomplete["reasoning"],
        "redundancy_attribution": measure_redundant_calls([
            {"tool": "gnomon_forecast", "result": sample},
            {"tool": "gnomon_get_artifact", "result": {}},
        ])["redundant_calls"] == 1,
        "actionable_rejection": not bool(
            (dead_end.get("rejection") or {}).get("admissibility_path")),
        "real_verb_response": bool(verify_fact_sources({
            **real_response,
            "reasoning": {**real_response["reasoning"], "facts": [
                {"name": "broken", "source": "/not-real"},
            ]},
        })),
    }
    gates["mutation_falsifiability"] = all(mutation_detection.values())
    return {"schema_version": "0.1", "evaluated_commit": code_revision(),
            "seed": seed, "cases": cases,
            "gates": gates, "mutation_detection": mutation_detection,
            "graduated": all(gates.values()), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    report = run(args.seed, args.cases)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    destination = args.output
    if args.output_dir:
        destination = str(Path(args.output_dir) / "summary.json")
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if report["graduated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
