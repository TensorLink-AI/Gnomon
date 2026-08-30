"""Run the frozen v0.8 A1 publication-policy matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gnomon.publication import publish_result, verify_publication


def _result() -> dict[str, Any]:
    return {
        "series": "value", "support": "supported", "selected_model": "ets",
        "forecast": [{
            "timestamp": f"2026-01-{day:02d}T00:00:00+00:00",
            "point": 10.0, "q10": 9.0, "q50": 10.0, "q90": 11.0,
        } for day in range(3, 15)],
    }


def _calibration(**updates: Any) -> dict[str, Any]:
    row = {
        "artifact_id": "forecast:a1-matrix", "series": "value",
        "selected_model": "ets", "horizon": 12, "nominal_coverage": 0.8,
        "measured_interval_coverage": 0.83, "coverage_points": 12,
        "residual_fold_count": 1,
        "residuals_pooled_across_selection": False,
        "cutoff_status": "artifact_snapshot",
        "prospective_validation_status": "passed",
    }
    row.update(updates)
    return row


def matrix() -> list[dict[str, Any]]:
    complete = {"authorize": True, "policy_id": "ops-v08",
                "minimum_support": "supported"}
    return [
        {"case": "legacy", "policy": complete, "calibration": None,
         "authority": True, "expected": True, "reason": "authorized"},
        {"case": "advisory", "policy": {**complete, "action_tier": "advisory"},
         "calibration": _calibration(), "authority": True, "expected": False,
         "reason": "advisory_tier"},
        {"case": "reversible_matched", "policy": {
            **complete, "action_tier": "reversible_low_impact"},
         "calibration": _calibration(), "authority": True, "expected": True,
         "reason": "authorized_reversible_low_impact"},
        {"case": "reversible_absent", "policy": {
            **complete, "action_tier": "reversible_low_impact"},
         "calibration": None, "authority": True, "expected": False,
         "reason": "calibration_not_action_eligible"},
        {"case": "wrong_series", "policy": {
            **complete, "action_tier": "reversible_low_impact"},
         "calibration": _calibration(series="other"), "authority": True,
         "expected": False, "reason": "calibration_not_action_eligible"},
        {"case": "wrong_model", "policy": {
            **complete, "action_tier": "reversible_low_impact"},
         "calibration": _calibration(selected_model="theta"), "authority": True,
         "expected": False, "reason": "calibration_not_action_eligible"},
        {"case": "wrong_horizon", "policy": {
            **complete, "action_tier": "reversible_low_impact"},
         "calibration": _calibration(horizon=3), "authority": True,
         "expected": False, "reason": "calibration_not_action_eligible"},
        {"case": "pooled_selection", "policy": {
            **complete, "action_tier": "reversible_low_impact"},
         "calibration": _calibration(residuals_pooled_across_selection=True),
         "authority": True, "expected": False,
         "reason": "calibration_not_action_eligible"},
        {"case": "coverage_out_of_band", "policy": {
            **complete, "action_tier": "reversible_low_impact"},
         "calibration": _calibration(measured_interval_coverage=0.5),
         "authority": True, "expected": False,
         "reason": "calibration_not_action_eligible"},
        {"case": "too_few_points", "policy": {
            **complete, "action_tier": "reversible_low_impact"},
         "calibration": _calibration(coverage_points=4), "authority": True,
         "expected": False, "reason": "calibration_not_action_eligible"},
        {"case": "high_impact", "policy": {
            **complete, "action_tier": "high_impact"},
         "calibration": _calibration(), "authority": True, "expected": False,
         "reason": "high_impact_not_supported"},
        {"case": "untrusted_reversible", "policy": {
            **complete, "action_tier": "reversible_low_impact"},
         "calibration": _calibration(), "authority": False, "expected": False,
         "reason": "untrusted_authorization_channel"},
    ]


def run() -> dict[str, Any]:
    rows = []
    for case in matrix():
        try:
            publication = publish_result(
                _result(), artifact_id="forecast:a1-matrix",
                automation_policy=case["policy"],
                automation_authority=case["authority"],
                calibration_evidence=case["calibration"],
            )
            automation = publication["automation"]
            rows.append({
                "case": case["case"], "status": "complete",
                "eligible": automation["eligible"],
                "reason_code": automation["reason_code"],
                "expected": case["expected"],
                "exact": (automation["eligible"] is case["expected"]
                          and automation["reason_code"] == case["reason"]),
                "seal_valid": verify_publication(publication),
                "primary_unchanged": publication["primary_forecast_unchanged"],
            })
        except (TypeError, ValueError) as exc:
            rows.append({
                "case": case["case"], "status": "error", "exact": False,
                "seal_valid": False, "primary_unchanged": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    summary = {
        "benchmark": "calibrationactionbench", "schema_version": 1,
        "cases": len(rows), "completed": sum(
            row["status"] == "complete" for row in rows),
        "exact": sum(row["exact"] for row in rows),
        "sealed": sum(row["seal_valid"] for row in rows),
        "primary_unchanged": sum(row["primary_unchanged"] for row in rows),
        "rows": rows,
    }
    summary["gates"] = {
        "complete": summary["completed"] == len(rows),
        "exact": summary["exact"] == len(rows),
        "sealed": summary["sealed"] == len(rows),
        "primary_unchanged": summary["primary_unchanged"] == len(rows),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    summary = run()
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if all(summary["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
