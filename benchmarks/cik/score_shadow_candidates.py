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
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.cik.gnomon_forecaster import samples_from_quantile_rows  # noqa: E402
from benchmarks.cik.run_cik import RCRPS_CAP  # noqa: E402
from gnomon.llm_dossier import verify_temporal_dossier_seal  # noqa: E402


def run(input_dir: Path, n_samples: int, *, publication_mode: str | None = None) -> dict:
    import numpy as np
    from cik_benchmark import ALL_TASKS

    classes = {task.__name__: task for task in ALL_TASKS}
    canonical: dict[tuple[str, int], float] = {}
    with (input_dir / "scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("rcrps"):
                canonical[(row["task"], int(row["seed"]))] = float(row["rcrps"])

    stem = ("best_effort_publication" if publication_mode == "best_effort"
            else "shadow_candidate")
    checkpoint_path = input_dir / f"{stem}_checkpoint.json"
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        checkpoint = {}
    rows = []
    ordered = sorted(canonical.items())
    for ordinal, ((task_name, seed), canonical_score) in enumerate(ordered, 1):
        key = f"{task_name}::seed={seed}"
        if key in checkpoint:
            rows.append(checkpoint[key])
            continue
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
                if publication_mode == "best_effort":
                    from gnomon.publication import publish_result, verify_publication
                    artifact_path = Path(extra.get("artifact_path") or "")
                    try:
                        artifact = json.loads(
                            (artifact_path / "artifact.json").read_text(
                                encoding="utf-8"))
                        result = artifact["results"][0]
                    except (OSError, KeyError, IndexError, json.JSONDecodeError):
                        reason = "missing_immutable_primary"
                        rows.append({
                            "task": task_name, "seed": seed,
                            "canonical_rcrps": canonical_score,
                            "candidate_rcrps": None,
                            "candidate_available": False,
                            "candidate_wins": False, "reason": reason,
                        })
                        continue
                    publication = publish_result(
                        result, mode="best_effort", dossiers=[dossier],
                        artifact_id=artifact.get("forecast_id"))
                    if not verify_publication(publication):
                        reason = "invalid_product_publication"
                        rows.append({
                            "task": task_name, "seed": seed,
                            "canonical_rcrps": canonical_score,
                            "candidate_rcrps": None,
                            "candidate_available": False,
                            "candidate_wins": False, "reason": reason,
                        })
                        continue
                    quantiles = publication["recommended_forecast"]
                samples = np.asarray(samples_from_quantile_rows(
                    quantiles, n_samples), dtype=float)[:, :, None]
                score = float(classes[task_name](seed=seed).evaluate(
                    samples)["metric"])
        row = {
            "task": task_name, "seed": seed,
            "canonical_rcrps": canonical_score,
            "candidate_rcrps": score,
            "candidate_available": score is not None,
            "candidate_wins": score is not None and score < canonical_score,
            "recommended_rcrps": (score if score is not None
                                    else canonical_score)
                if publication_mode == "best_effort" else score,
            "reason": reason,
        }
        rows.append(row)
        checkpoint[key] = row
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(checkpoint, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(checkpoint_path)
        if ordinal % 20 == 0 or ordinal == len(ordered):
            print(f"[{ordinal}/{len(ordered)}] scored publication candidates",
                  flush=True)

    if publication_mode == "best_effort":
        for row in rows:
            row["recommended_rcrps"] = (
                row.get("candidate_rcrps")
                if row.get("candidate_rcrps") is not None
                else row["canonical_rcrps"])
    output = input_dir / f"{stem}_scores.csv"
    fieldnames = [
        "task", "seed", "canonical_rcrps", "candidate_rcrps",
        "candidate_available", "candidate_wins", "recommended_rcrps", "reason",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if publication_mode == "best_effort":
        recommended = [float(row.get("recommended_rcrps")
                             if row.get("recommended_rcrps") is not None
                             else row["canonical_rcrps"]) for row in rows]
    else:
        recommended = [float(row["candidate_rcrps"])
                       if row["candidate_rcrps"] is not None else RCRPS_CAP
                       for row in rows]
    imputed = [min(value, RCRPS_CAP) for value in recommended]
    canonical_capped = [min(float(row["canonical_rcrps"]), RCRPS_CAP)
                        for row in rows]
    paired = [base - treatment for base, treatment
              in zip(canonical_capped, imputed)]
    rng = random.Random(260826)
    bootstrap = []
    if paired:
        for _ in range(5000):
            bootstrap.append(sum(rng.choice(paired) for _ in paired) / len(paired))
        bootstrap.sort()
    available = [row for row in rows if row["candidate_available"]]
    summary = {
        "runs": len(rows),
        "candidates_available": sum(row["candidate_available"] for row in rows),
        "candidate_wins_vs_canonical": sum(row["candidate_wins"] for row in rows),
        "candidate_losses_vs_canonical": sum(
            row["candidate_available"] and float(row["candidate_rcrps"])
            > float(row["canonical_rcrps"]) for row in rows),
        "mean_recommended_rcrps_capped": (
            sum(imputed) / len(imputed) if imputed else None),
        "mean_canonical_rcrps_capped": (
            sum(canonical_capped) / len(rows)
            if rows else None),
        "mean_candidate_rcrps_when_available": (
            sum(float(row["candidate_rcrps"]) for row in available)
            / len(available) if available else None),
        "mean_canonical_rcrps_on_available_cases": (
            sum(float(row["canonical_rcrps"]) for row in available)
            / len(available) if available else None),
        "paired_mean_capped_rcrps_improvement": (
            sum(paired) / len(paired) if paired else None),
        "paired_bootstrap_95_ci": ([bootstrap[124], bootstrap[4874]]
                                   if bootstrap else None),
        "diagnostic_only": publication_mode is None,
        "publication_mode": publication_mode,
        "human_recommendation_lane_enabled": publication_mode == "best_effort",
        "automation_eligible": False,
    }
    (input_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--publication-mode", choices=["best_effort"])
    args = parser.parse_args()
    run(args.input_dir, args.n_samples,
        publication_mode=args.publication_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
