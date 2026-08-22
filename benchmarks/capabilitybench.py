"""Independent property benchmark for the temporal capability registry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.manifest import code_revision, write_manifest  # noqa: E402
from gnomon.statistical_executables import (  # noqa: E402
    fit_decomposition_executable,
    fit_regression_executable,
    fit_stationarity_executable,
)
from gnomon.temporal_contracts import (  # noqa: E402
    classify_dataset_contract, plan_execution,
)
from gnomon.temporal_question import compile_temporal_question  # noqa: E402


def run(*, seed: int, cases_per_family: int) -> dict:
    rng = random.Random(seed)
    adf_correct = decomposition_valid = helpful_supported = null_supported = 0
    coefficient_errors: list[float] = []
    for case in range(cases_per_family):
        stationary = case % 2 == 0
        values, level = [], 0.0
        for _ in range(320):
            innovation = rng.gauss(0, 1)
            if stationary:
                level = .55 * level + innovation
            else:
                level += innovation
            values.append(level)
        result = fit_stationarity_executable(
            values, target="value", method="adf").execute()
        adf_correct += (result["direction"] == (
            "stationary" if stationary else "unit_root_not_rejected"))

        period = (7, 12, 24)[case % 3]
        seasonal = [
            .01 * index + 4 * math.sin(2 * math.pi * index / period)
            + rng.gauss(0, .15)
            for index in range(period * 8)
        ]
        decomposition = fit_decomposition_executable(
            seasonal, target="value", period=period).execute()["estimate"]
        reconstructed = [
            (decomposition["trend"][index]
             + decomposition["seasonal"][index]
             + decomposition["residual"][index])
            for index in range(len(seasonal))
            if decomposition["trend"][index] is not None
        ]
        expected = [seasonal[index] for index in range(len(seasonal))
                    if decomposition["trend"][index] is not None]
        decomposition_valid += max(abs(a - b) for a, b in
                                   zip(reconstructed, expected)) < 1e-9

        x = [rng.uniform(-2, 2) for _ in range(180)]
        helpful_y = [3 + 2.5 * value + rng.gauss(0, .2) for value in x]
        helpful = fit_regression_executable(
            helpful_y, {"x": x}, target="y").execute()
        helpful_supported += helpful["support"] == "supported"
        coefficient_errors.append(abs(
            helpful["estimate"]["coefficients"]["x"] - 2.5))
        null_y = [rng.gauss(0, 1) for _ in x]
        null = fit_regression_executable(
            null_y, {"x": x}, target="y").execute()
        null_supported += null["support"] == "supported"

    stl = compile_temporal_question({
        "id": "stl", "verb": "decompose", "property": "decomposition",
        "target": "value", "method": "stl", "period": 12,
    }, available_targets=["value"])
    semantic_refusal = plan_execution(
        stl, classify_dataset_contract(["value"])).status == "unsupported"
    total = cases_per_family
    summary = {
        "benchmark": "capabilitybench",
        "schema_version": "0.1",
        "seed": seed,
        "cases_per_family": total,
        "adf_balanced_accuracy": adf_correct / total,
        "decomposition_reconstruction_rate": decomposition_valid / total,
        "regression_helpful_support_rate": helpful_supported / total,
        "regression_null_false_support_rate": null_supported / total,
        "regression_mean_coefficient_error": sum(coefficient_errors) / total,
        "semantic_substitution_refused": semantic_refusal,
    }
    summary["gates"] = {
        "adf_accuracy": summary["adf_balanced_accuracy"] >= .9,
        "decomposition_identity": summary[
            "decomposition_reconstruction_rate"] == 1.0,
        "regression_helpful": summary[
            "regression_helpful_support_rate"] >= .9,
        "regression_null_control": summary[
            "regression_null_false_support_rate"] <= .2,
        "semantic_refusal": semantic_refusal,
    }
    summary["graduated"] = all(summary["gates"].values())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--cases-per-family", type=int, default=40)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    revision = code_revision()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = run(seed=args.seed, cases_per_family=args.cases_per_family)
    summary["evaluated_commit"] = revision
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        output, benchmark="capabilitybench", condition="property-suite",
        target=f"seed={args.seed};cases_per_family={args.cases_per_family}",
        code_revision=revision)
    print(json.dumps(summary, indent=2))
    return 0 if summary["graduated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
