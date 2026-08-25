"""Report the governed-covariate funnel in a completed or partial CiK run.

The report keeps four denominators separate: LLM proposal, provenance
validation, host binding, and fold-safe forecast admission. A table that is
well cited but not predictive is not counted as a product win.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _artifact_covariates(path: str | None) -> tuple[bool, bool]:
    if not path:
        return False, False
    try:
        from gnomon.artifacts import read_artifact
        artifact = read_artifact(path)
    except Exception:
        return False, False
    assessments = [item.get("covariates") or {}
                   for item in artifact.get("results") or []]
    return (any(item.get("considered") for item in assessments),
            any(item.get("admitted") for item in assessments))


def collect(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / "runs").glob("*/*/extra_info")):
        try:
            extra = ast.literal_eval(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, SyntaxError):
            continue
        summary = extra.get("context_compilation") or {}
        receipt = {}
        receipt_path = Path(summary.get("receipt_path") or "")
        if receipt_path.exists():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                receipt = {}
        covariates = receipt.get("covariates") or {}
        considered, admitted = _artifact_covariates(extra.get("artifact_path"))
        proposed = int(covariates.get(
            "tables_proposed", summary.get("covariate_tables_proposed", 0)) or 0)
        validated = int(covariates.get(
            "tables_validated", summary.get("covariate_tables", 0)) or 0)
        records.append({
            "task": path.parent.parent.name,
            "seed": path.parent.name,
            "tables_proposed": proposed,
            "rows_proposed": int(covariates.get(
                "rows_proposed", summary.get("covariate_rows_proposed", 0)) or 0),
            "tables_validated": validated,
            "rows_validated": int(covariates.get(
                "rows_validated", summary.get("covariate_rows_validated", 0)) or 0),
            "table_bound": validated == 1,
            "covariates_considered": considered,
            "covariates_admitted": admitted,
            "route": extra.get("route"),
        })
    return records


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    cases = len(records)
    count = lambda predicate: sum(bool(predicate(row)) for row in records)
    proposal_cases = count(lambda row: row["tables_proposed"] > 0)
    validated_cases = count(lambda row: row["tables_validated"] > 0)
    bound_cases = count(lambda row: row["table_bound"])
    considered_cases = count(lambda row: row["covariates_considered"])
    admitted_cases = count(lambda row: row["covariates_admitted"])
    return {
        "cases_with_receipts": cases,
        "proposal_cases": proposal_cases,
        "validated_cases": validated_cases,
        "bound_cases": bound_cases,
        "fold_considered_cases": considered_cases,
        "fold_admitted_cases": admitted_cases,
        "tables_proposed": sum(row["tables_proposed"] for row in records),
        "rows_proposed": sum(row["rows_proposed"] for row in records),
        "tables_validated": sum(row["tables_validated"] for row in records),
        "rows_validated": sum(row["rows_validated"] for row in records),
        "proposal_validation_rate": (
            validated_cases / proposal_cases if proposal_cases else None),
        "validation_to_admission_rate": (
            admitted_cases / validated_cases if validated_cases else None),
        "all_routes_governed": all(row["route"] == "gnomon" for row in records),
        "note": "Validation proves cited provenance; fold admission separately "
                "proves predictive usefulness. Neither implies the other.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    records = collect(args.run_dir)
    summary = summarise(records)
    output = args.run_dir / "covariate_behavior_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
