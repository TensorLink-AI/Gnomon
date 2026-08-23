"""Property-style benchmark for Gnomon's agent reasoning boundary."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from gnomon.reasoning_boundary import measure_redundant_calls, verify_fact_sources
from gnomon.toolspec import apply_response_contract


def _case(index: int, rng: random.Random) -> dict[str, Any]:
    kind = ("forecast", "detect", "decide", "monitor", "describe")[index % 5]
    artifact = f"{kind}-{index}"
    payload: dict[str, Any] = {
        "headline": f"Computed {kind} conclusion {index}.",
        "artifact_id": artifact,
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
                                      "what_would_flip", "sufficiency")),
            "redundant_calls": calls["redundant_calls"],
            "redundancy_attributed": calls["redundant_calls"] == (index % 4 == 0),
        })
    rejection = apply_response_contract({"status": "error", "error": {
        "code": "INVALID_ARGUMENTS", "message": "Bad window.",
        "repair_options": [{"tool": "gnomon_forecast", "arguments": {"horizon": 7}}]}})
    gates = {
        "canonical_immutability": all(row["canonical_unchanged"] for row in rows),
        "fact_traceability": all(row["traceable"] for row in rows),
        "argument_completeness": all(row["argument_complete"] for row in rows),
        "redundancy_attribution": all(row["redundancy_attributed"] for row in rows),
        "actionable_rejection": bool((rejection.get("rejection") or {}).get(
            "admissibility_path")),
    }
    return {"schema_version": "0.1", "seed": seed, "cases": cases,
            "gates": gates, "graduated": all(gates.values()), "rows": rows}


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
