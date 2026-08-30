"""Run the frozen v0.8 D1 label-free decision-contract matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gnomon.agent_response import (
    build_agent_response_contract,
    verify_agent_response_contract,
)


PROPERTIES = (
    "level", "trend", "volatility", "seasonality", "regime", "extremes",
)


def _answer(index: int, *, support: str, sufficiency: str,
            opposition: str = "none", interval: bool = True,
            relationship: str | None = None,
            conditional: bool = False) -> dict[str, Any]:
    prop = PROPERTIES[index % len(PROPERTIES)]
    canonical = "upward" if support != "abstained" else None
    interpretations = []
    if canonical is not None:
        interpretations.append({
            "value": canonical, "support": support,
            "compatible": True, "supporting": ["fitted_executable"],
            "conflicting": (["observed_transition"]
                            if opposition != "none" else []),
        })
    if opposition != "none":
        interpretations.append({
            "value": "downward",
            "support": "supported" if opposition == "supported" else "weak",
            "compatible": opposition != "contradicted",
            "supporting": (["observed_transition"]
                           if opposition != "contradicted" else []),
            "conflicting": ["fitted_executable"],
            "conditional_only": opposition == "conditional",
        })
    answer = {
        "question": {
            "id": f"decision-{index:02d}", "verb": "predict",
            "property": prop, "target": "value", "horizon": 12,
        },
        "artifact_id": "forecast:decision-contract-matrix",
        "best_estimate": {
            "value": canonical,
            "display_value": canonical or "Uncertain",
            "support": support,
            "automation_eligible": support == "supported" and index == 0,
        },
        "support": {
            "state": support,
            "automation_eligible": support == "supported" and index == 0,
        },
        "answer": {
            "support": support,
            "automation_eligible": support == "supported" and index == 0,
            "interval": [-0.2, 0.4] if interval else None,
            "reasoning": {
                "packet": {
                    "interpretations": interpretations,
                    "sufficiency": sufficiency,
                    "selector": ("gnomon_canonical"
                                 if support == "supported" else "model"),
                    "selection_must_cite_evidence": support != "supported",
                },
                "primary_forecast_unchanged": True,
            },
        },
        "calibration_status": {
            "available": interval,
            "applicable": interval,
            **({"reason": "no_applicable_calibration"} if not interval else {}),
        },
        "limitations": ([] if interval else
                        ["No applicable interval calibration is available."]),
    }
    if conditional:
        answer["conditional_effect"] = {
            "role": "conditional_evidence_only",
            "primary_forecast_unchanged": True,
        }
    if relationship:
        answer["context_assessment"] = {
            "relationship_to_primary": relationship,
            "canonical_primary_preserved": True,
        }
    return answer


def matrix() -> list[dict[str, Any]]:
    specs = [
        ("supported", "sufficient", "none", True, None, False),
        ("supported", "sufficient", "supported", True, None, False),
        ("weak", "mixed", "supported", True, None, False),
        ("weak", "mixed", "contradicted", True, None, False),
        ("weak", "mixed", "conditional", True, None, True),
        ("weak", "mixed", "none", False, None, False),
        ("weak", "insufficient", "none", False, None, False),
        ("abstained", "insufficient", "none", False, None, False),
        ("abstained", "mixed", "supported", False, None, False),
        ("weak", "mixed", "none", True,
         "no_distinct_numeric_path", False),
        ("weak", "mixed", "supported", True, None, True),
        ("supported", "sufficient", "none", False, None, False),
    ]
    return [
        _answer(index, support=support, sufficiency=sufficiency,
                opposition=opposition, interval=interval,
                relationship=relationship, conditional=conditional)
        for index, (support, sufficiency, opposition, interval,
                    relationship, conditional) in enumerate(specs)
    ]


def run() -> dict[str, Any]:
    answers = matrix()
    payload = {
        "forecast_id": "forecast:decision-contract-matrix",
        "artifact_path": "/sealed/forecast:decision-contract-matrix",
        "results": [{"series": "value", "support": "weak"}],
        "answers": answers,
        "answer_receipt": "/sealed/forecast:decision-contract-matrix/temporal_answers.json",
    }
    contract = build_agent_response_contract(payload)
    decisions = ((contract or {}).get("decisions") or [])
    required = {
        "question_id", "property", "inference_mode", "conclusion",
        "value_status", "support", "authority", "selector",
        "interpretations", "conditions", "uncertainty",
        "decision_eligible", "automation_eligible",
        "primary_forecast_unchanged", "provenance", "required",
    }
    rows = []
    for index, expected in enumerate(answers):
        decision = decisions[index] if index < len(decisions) else {}
        best = expected["best_estimate"]
        context = expected.get("context_assessment") or {}
        exact = bool(decision) and (
            decision.get("question_id") == expected["question"]["id"]
            and decision.get("conclusion") == best.get("value")
            and decision.get("support") == best.get("support")
            and decision.get("automation_eligible")
            == bool(best.get("automation_eligible"))
            and decision.get("primary_forecast_unchanged") is True
            and (not context.get("relationship_to_primary")
                 or decision.get("relationship_to_primary")
                 == context["relationship_to_primary"])
        )
        rows.append({
            "case": expected["question"]["id"],
            "contract_present": bool(decision),
            "complete": required <= set(decision),
            "exact": exact,
        })
    summary = {
        "benchmark": "decisioncontractbench",
        "schema_version": 1,
        "cases": len(rows),
        "contract_emitted": contract is not None,
        "decision_contracts": len(decisions),
        "complete": sum(row["complete"] for row in rows),
        "exact": sum(row["exact"] for row in rows),
        "seal_recomputes": bool(
            contract and verify_agent_response_contract(payload, contract)),
        "rows": rows,
    }
    summary["gates"] = {
        "all_decisions_present": len(decisions) == len(rows),
        "all_required_fields": all(row["complete"] for row in rows),
        "all_exact": all(row["exact"] for row in rows),
        "seal_recomputes": summary["seal_recomputes"],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    summary = run()
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all(summary["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
