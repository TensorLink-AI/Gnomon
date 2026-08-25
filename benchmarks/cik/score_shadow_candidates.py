"""Score retained CiK LLM candidates from the same canonical compiler call.

This is diagnostic evidence, not a publication selector. Missing or invalid
candidates are imputed at the official RCRPS cap so candidate availability
cannot improve the aggregate by dropping difficult tasks.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.cik.gnomon_forecaster import samples_from_quantile_rows  # noqa: E402
from benchmarks.cik.run_cik import RCRPS_CAP  # noqa: E402
from gnomon.llm_dossier import verify_temporal_dossier_seal  # noqa: E402


def run(input_dir: Path, n_samples: int) -> dict:
    import numpy as np
    from cik_benchmark import ALL_TASKS

    classes = {task.__name__: task for task in ALL_TASKS}
    canonical: dict[tuple[str, int], float] = {}
    with (input_dir / "scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("rcrps"):
                canonical[(row["task"], int(row["seed"]))] = float(row["rcrps"])

    rows = []
    for (task_name, seed), canonical_score in sorted(canonical.items()):
        extra_path = input_dir / "runs" / task_name / str(seed) / "extra_info"
        extra = ast.literal_eval(extra_path.read_text(encoding="utf-8"))
        lane = extra.get("llm_candidate_shadow") or {}
        receipt_path = Path(
            (extra.get("context_compilation") or {}).get("receipt_path", ""))
        reason = ""
        score = None
        if not lane:
            reason = "no_admissible_candidate"
        elif not receipt_path.exists():
            reason = "missing_retained_receipt"
        else:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            dossier = receipt.get("dossier") or {}
            if not verify_temporal_dossier_seal(dossier):
                reason = "invalid_dossier_seal"
            elif dossier.get("seal_sha256") != lane.get("seal_sha256"):
                reason = "candidate_receipt_identity_mismatch"
            else:
                quantiles = lane["forecast_candidate"]["quantiles"]
                samples = np.asarray(samples_from_quantile_rows(
                    quantiles, n_samples), dtype=float)[:, :, None]
                score = float(classes[task_name](seed=seed).evaluate(
                    samples)["metric"])
        rows.append({
            "task": task_name, "seed": seed,
            "canonical_rcrps": canonical_score,
            "candidate_rcrps": score,
            "candidate_available": score is not None,
            "candidate_wins": score is not None and score < canonical_score,
            "reason": reason,
        })

    output = input_dir / "shadow_candidate_scores.csv"
    fieldnames = [
        "task", "seed", "canonical_rcrps", "candidate_rcrps",
        "candidate_available", "candidate_wins", "reason",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    imputed = [min(float(row["candidate_rcrps"]), RCRPS_CAP)
               if row["candidate_rcrps"] is not None else RCRPS_CAP
               for row in rows]
    summary = {
        "runs": len(rows),
        "candidates_available": sum(row["candidate_available"] for row in rows),
        "candidate_wins_vs_canonical": sum(row["candidate_wins"] for row in rows),
        "mean_candidate_rcrps_capped_imputed": (
            sum(imputed) / len(imputed) if imputed else None),
        "mean_canonical_rcrps": (
            sum(float(row["canonical_rcrps"]) for row in rows) / len(rows)
            if rows else None),
        "diagnostic_only": True,
        "candidate_publication_authorised": False,
    }
    (input_dir / "shadow_candidate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--n-samples", type=int, default=25)
    args = parser.parse_args()
    run(args.input_dir, args.n_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
