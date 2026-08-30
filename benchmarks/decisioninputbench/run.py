"""Frozen reproduction for malformed and non-identifying decision inputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.manifest import code_revision, write_manifest
from gnomon.contracts import GnomonError
from gnomon.operators import evaluate_actions


PROBABILITIES = {"exceed": 0.4, "no_exceed": 0.6}
ACTIONS = [{"name": "act"}, {"name": "wait"}]

CASES: tuple[dict[str, Any], ...] = (
    {"id": "valid_non_tied", "expected": "select_act", "actions": ACTIONS,
     "utilities": {"act": {"exceed": 10.0, "no_exceed": 2.0},
                   "wait": {"exceed": -3.0, "no_exceed": 1.0}}},
    {"id": "missing_action", "expected": "typed_rejection", "actions": ACTIONS,
     "utilities": {"act": {"exceed": 10.0, "no_exceed": 2.0}}},
    {"id": "missing_scenario", "expected": "typed_rejection", "actions": ACTIONS,
     "utilities": {"act": {"exceed": 10.0},
                   "wait": {"exceed": -3.0, "no_exceed": 1.0}}},
    {"id": "unknown_scenarios", "expected": "typed_rejection", "actions": ACTIONS,
     "utilities": {"act": {"up": 10.0}, "wait": {"down": 1.0}}},
    {"id": "mixed_scenarios", "expected": "typed_rejection", "actions": ACTIONS,
     "utilities": {"act": {"exceed": 10.0, "no_exceed": 2.0, "typo": 999.0},
                   "wait": {"exceed": -3.0, "no_exceed": 1.0}}},
    {"id": "flat_value", "expected": "typed_rejection", "actions": ACTIONS,
     "utilities": {"act": 10.0, "wait": 1.0}},
    {"id": "non_numeric", "expected": "typed_rejection", "actions": ACTIONS,
     "utilities": {"act": {"exceed": "high", "no_exceed": 2.0},
                   "wait": {"exceed": -3.0, "no_exceed": 1.0}}},
    {"id": "empty", "expected": "typed_rejection", "actions": ACTIONS,
     "utilities": {}},
    {"id": "infeasible_incomplete", "expected": "select_wait",
     "actions": [{"name": "act", "feasible": False}, {"name": "wait"}],
     "utilities": {"wait": {"exceed": -3.0, "no_exceed": 1.0}}},
    {"id": "exact_tie", "expected": "tie_abstention", "actions": ACTIONS,
     "utilities": {"act": {"exceed": 5.0, "no_exceed": 5.0},
                   "wait": {"exceed": 5.0, "no_exceed": 5.0}}},
    {"id": "no_feasible_action", "expected": "no_feasible",
     "actions": [{"name": "act", "feasible": False},
                 {"name": "wait", "feasible": False}],
     "utilities": {}},
)


def run() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        try:
            result = evaluate_actions(
                list(case["actions"]), dict(PROBABILITIES),
                utilities=case["utilities"],
            )
            rows.append({
                "case_id": case["id"], "expected": case["expected"],
                "termination": "result", "selected": result.get("selected"),
                "support": (result.get("support") or {}).get("status"),
                "reason_codes": [item.get("code") for item in
                                 (result.get("support") or {}).get("reasons", [])],
                "selection_margin": result.get("selection_margin"),
                "evaluations": result.get("evaluations", []),
                "error": None,
            })
        except GnomonError as exc:
            error = exc.to_dict()["error"]
            rows.append({
                "case_id": case["id"], "expected": case["expected"],
                "termination": "typed_rejection", "selected": None,
                "support": "invalid", "reason_codes": [exc.code],
                "selection_margin": None, "evaluations": [],
                "error": error,
            })
        except Exception as exc:  # retained as a product failure
            rows.append({
                "case_id": case["id"], "expected": case["expected"],
                "termination": "internal_error", "selected": None,
                "support": None, "reason_codes": [],
                "selection_margin": None, "evaluations": [],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })

    by_id = {row["case_id"]: row for row in rows}
    malformed = [row for row in rows if row["expected"] == "typed_rejection"]
    no_unsafe_selection = all(
        row["selected"] is None for row in rows
        if row["expected"] in {"typed_rejection", "tie_abstention", "no_feasible"}
    )
    gates = {
        "all_cases_terminate_without_internal_error": all(
            row["termination"] != "internal_error" for row in rows),
        "malformed_cases_are_typed_invalid_utilities": all(
            row["termination"] == "typed_rejection"
            and (row["error"] or {}).get("code") == "INVALID_UTILITIES"
            for row in malformed),
        "malformed_repairs_are_executable": all(
            any(item.get("tool") == "gnomon_decide"
                and isinstance(item.get("arguments"), dict)
                and "utilities" in item["arguments"]
                for item in (row["error"] or {}).get("repair_options", []))
            for row in malformed),
        "no_unsafe_selection": no_unsafe_selection,
        "exact_tie_is_inconclusive_abstention": (
            by_id["exact_tie"]["selected"] is None
            and by_id["exact_tie"]["support"] == "inconclusive"
            and "utility_tie" in by_id["exact_tie"]["reason_codes"]
            and by_id["exact_tie"]["selection_margin"] == 0.0),
        "valid_non_tied_selection_preserved": (
            by_id["valid_non_tied"]["selected"] == "act"
            and by_id["valid_non_tied"]["support"] == "supported"),
        "infeasible_action_utilities_not_required": (
            by_id["infeasible_incomplete"]["selected"] == "wait"),
        "no_feasible_action_still_abstains": (
            by_id["no_feasible_action"]["selected"] is None
            and by_id["no_feasible_action"]["support"] == "unsupported"),
        "numeric_outputs_finite": all(
            math.isfinite(float(item["expected_utility"]))
            for row in rows for item in row["evaluations"]
            if item.get("expected_utility") is not None),
    }
    return {
        "schema_version": "0.1",
        "benchmark": "decision-input-integrity",
        "scenario_probabilities": PROBABILITIES,
        "cases": len(rows),
        "internal_errors": sum(row["termination"] == "internal_error"
                               for row in rows),
        "unsafe_selections": sum(
            row["selected"] is not None
            and row["expected"] in {
                "typed_rejection", "tie_abstention", "no_feasible"}
            for row in rows),
        "gates": gates,
        "raw_records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run()
    result["evaluated_commit"] = code_revision()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_manifest(
            args.output_dir, benchmark="decisioninputbench",
            condition="current-production-boundary",
            target="frozen-s1-cases",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

