"""Fresh naturalistic confirmation for the production selector.

The selector receives only each history prefix, horizon, season, and cadence.
Held-out values and all dataset metadata remain in this scoring harness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.manifest import code_revision, write_manifest
from benchmarks.modelbench.run_production_selector import (
    DEFAULT_MINIMUM_IMPROVEMENT,
    _loss,
    _published_points,
)
from gnomon.evaluation import evaluate
from gnomon.models import last_value, seasonal_naive


CUTOFF_FRACTIONS = (0.60, 0.70, 0.80, 0.90)
BOOTSTRAP_SEED = 92742
BOOTSTRAP_DRAWS = 4000

DATASETS: tuple[dict[str, Any], ...] = (
    {"id": "pedestrian", "group": "operational",
     "path": "benchmarks/breachbench/data/pedestrian_counts_daily.csv",
     "sha256": "a53d1d3e98c323be36ac52f35a7a06d91ba3a3a7ca61d99330ca0f670236f7f7",
     "frequency": "D", "season": 7, "horizons": (7, 28)},
    {"id": "retail", "group": "operational",
     "path": "benchmarks/breachbench/data/retail_sales_monthly.csv",
     "sha256": "1d524c08bdf2c4bd6b75c9c629e9949ebc10fa2f268ed373b02709b8d41c4815",
     "frequency": "MS", "season": 12, "horizons": (3, 12)},
    {"id": "sensor_temperature", "group": "operational",
     "path": "benchmarks/breachbench/data/sensor_temps_5min.csv",
     "sha256": "cdeab8b7cdeade5ef6a04b2bbd2b2a4d1393a0dc90d7c4201b8a823a02f87387",
     "frequency": "5min", "season": 288, "horizons": (12, 48)},
    {"id": "web_traffic", "group": "operational",
     "path": "benchmarks/breachbench/data/wiki_traffic_daily_log.csv",
     "sha256": "7ee685b87a4685ed484563d2d09a430aa2d886b6f0a6eeb6a6e86aac474a3dbc",
     "frequency": "D", "season": 7, "horizons": (7, 28)},
    {"id": "co2", "group": "environmental",
     "path": "benchmarks/dossierbench/data/co2_weekly_mauna_loa.csv",
     "sha256": "b7dba885a948f608014256d497dfc13d84f1fd7cee100ced2889e8d2de09e419",
     "frequency": "W", "season": 52, "horizons": (4, 13)},
    {"id": "elnino", "group": "environmental",
     "path": "benchmarks/dossierbench/data/elnino_sst_monthly.csv",
     "sha256": "9fe30321f02a8e9b38b4ea7ded66c84e3e7dd935e09e79d9c12abbbee6cca6b1",
     "frequency": "MS", "season": 12, "horizons": (3, 12)},
    {"id": "nile", "group": "historical",
     "path": "benchmarks/dossierbench/data/nile_annual_flow.csv",
     "sha256": "68b52eb6a3315ea651f6c249597544b357bdf3acc1e609117b95e47022180b51",
     "frequency": "D", "season": 1, "horizons": (3, 5)},
    {"id": "sunspots", "group": "historical",
     "path": "benchmarks/dossierbench/data/sunspots_yearly.csv",
     "sha256": "7bb454ed535fda29e7749a9c936c1f0bda5c14f61a7f36b03a42f13c5fb85b9b",
     "frequency": "D", "season": 11, "horizons": (3, 11)},
    {"id": "us_cpi", "group": "macro",
     "path": "benchmarks/dossierbench/data/us_cpi_quarterly.csv",
     "sha256": "fc1e08aed907d5e7ae059b0590eac4446d88092c731886838c4efffb574bc603",
     "frequency": "MS", "season": 4, "horizons": (4, 8)},
    {"id": "us_m1", "group": "macro",
     "path": "benchmarks/dossierbench/data/us_m1_quarterly.csv",
     "sha256": "323a9b29fe3063d45e8e2ccd4666587c1081e0d194384fe68fb51b13a5890f70",
     "frequency": "MS", "season": 4, "horizons": (4, 8)},
    {"id": "us_realgdp", "group": "macro",
     "path": "benchmarks/dossierbench/data/us_realgdp_quarterly.csv",
     "sha256": "d2a471150da59c726ded6220713eeb5d29ca29b4a8c2a3f0628ca45b9f836d01",
     "frequency": "MS", "season": 4, "horizons": (4, 8)},
    {"id": "us_unemployment", "group": "macro",
     "path": "benchmarks/dossierbench/data/us_unemployment_quarterly.csv",
     "sha256": "fb84becea8da840c7513825b30298713184042a1375f65898994bd8c1c63fb61",
     "frequency": "MS", "season": 4, "horizons": (4, 8)},
)


def _read_values(spec: dict[str, Any]) -> list[float]:
    path = ROOT / spec["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"dataset digest mismatch: {spec['id']}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [float(row["value"]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"dataset contains missing/non-finite values: {spec['id']}")
    return values


def _historical_mean(history: list[float], horizon: int) -> list[float]:
    return [mean(history)] * horizon


def _bounded_gain(candidate: float, reference: float) -> float:
    return (reference - candidate) / max(reference, candidate, 1e-12)


def _cluster_interval(rows: list[dict[str, Any]]) -> list[float] | None:
    departure_rows = [row for row in rows if row["departed_from_last_value"]]
    clusters = sorted({row["dataset_id"] for row in departure_rows})
    if not clusters:
        return None
    by_cluster = {
        cluster: [float(row["relative_gain"])
                  for row in departure_rows if row["dataset_id"] == cluster]
        for cluster in clusters
    }
    rng = random.Random(BOOTSTRAP_SEED)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        selected = rng.choices(clusters, k=len(clusters))
        values = [gain for cluster in selected for gain in by_cluster[cluster]]
        samples.append(median(values))
    samples.sort()
    return [samples[int(0.05 * (BOOTSTRAP_DRAWS - 1))],
            samples[int(0.95 * (BOOTSTRAP_DRAWS - 1))]]


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["completed"]]
    departures = [row for row in completed if row["departed_from_last_value"]]
    gains = [float(row["relative_gain"]) for row in completed]
    departure_gains = [float(row["relative_gain"]) for row in departures]
    wins = sum(value > 1e-12 for value in departure_gains)
    losses = sum(value < -1e-12 for value in departure_gains)
    return {
        "cases": len(rows),
        "completed": len(completed),
        "completion_rate": len(completed) / len(rows) if rows else 0.0,
        "engine_supported": sum(bool(row["engine_supported"]) for row in rows),
        "disclosed_fallbacks": sum(bool(row["fallback_disclosed"]) for row in rows),
        "departures": len(departures),
        "departure_wins": wins,
        "departure_losses": losses,
        "departure_ties": len(departures) - wins - losses,
        "admission_precision": wins / len(departures) if departures else None,
        "median_relative_gain_all_cases": median(gains) if gains else None,
        "median_relative_gain_departures": median(departure_gains)
        if departure_gains else None,
        "mean_invalidated_departures": sum(
            bool(row["mean_invalidated_departure"]) for row in departures),
        "strongest_reference_counts": dict(sorted(Counter(
            str(row["strongest_reference"]) for row in completed).items())),
    }


def run(dataset_ids: set[str] | None = None) -> dict[str, Any]:
    selected_specs = [
        spec for spec in DATASETS
        if dataset_ids is None or spec["id"] in dataset_ids
    ]
    unknown = sorted((dataset_ids or set())
                     - {str(spec["id"]) for spec in DATASETS})
    if unknown:
        raise ValueError(f"unknown frozen dataset ids: {', '.join(unknown)}")
    if not selected_specs:
        raise ValueError("at least one frozen dataset is required")
    full_scope = len(selected_specs) == len(DATASETS)
    rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for spec in selected_specs:
        values = _read_values(spec)
        identities.append({
            "dataset_id": spec["id"], "group": spec["group"],
            "path": spec["path"], "sha256": spec["sha256"],
            "observations": len(values),
        })
        for origin, fraction in enumerate(CUTOFF_FRACTIONS):
            horizon = int(spec["horizons"][origin % 2])
            cutoff = math.floor(len(values) * fraction)
            if cutoff <= int(spec["season"]) or cutoff + horizon > len(values):
                raise ValueError(f"invalid frozen cutoff: {spec['id']} origin {origin}")
            history = values[:cutoff]
            actual = values[cutoff:cutoff + horizon]
            assessment = evaluate(
                history, horizon, int(spec["season"]),
                DEFAULT_MINIMUM_IMPROVEMENT,
                frequency=str(spec["frequency"]), tsfm_names=[],
                strict_abstention=False,
            )
            try:
                points, published_model, support, fallback = _published_points(
                    assessment, history, horizon, int(spec["season"]))
            except (ValueError, ArithmeticError, OverflowError):
                points, published_model, support, fallback = [], "none", "unsupported", False
            references = {
                "last_value": last_value(history, horizon, int(spec["season"])),
                "seasonal_naive": seasonal_naive(
                    history, horizon, int(spec["season"])),
                "historical_mean": _historical_mean(history, horizon),
            }
            reference_losses = {
                name: _loss(actual, forecast) for name, forecast in references.items()
            }
            strongest_reference = min(reference_losses,
                                       key=lambda name: (reference_losses[name], name))
            reference_loss = reference_losses[strongest_reference]
            completed = len(points) == horizon and all(
                math.isfinite(value) for value in points)
            candidate_loss = _loss(actual, points) if completed else float("inf")
            departed = completed and any(
                not math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
                for left, right in zip(points, references["last_value"]))
            gain = _bounded_gain(candidate_loss, reference_loss) if completed else None
            rows.append({
                "case_id": f"{spec['id']}-{origin}",
                "dataset_id": spec["id"], "group": spec["group"],
                "cutoff_fraction": fraction, "cutoff_index": cutoff,
                "history_length": cutoff, "horizon": horizon,
                "season": spec["season"], "frequency": spec["frequency"],
                "future_observations_used_by_selector": 0,
                "completed": completed,
                "engine_supported": assessment.supported,
                "engine_selected_model": assessment.selected_model,
                "engine_strongest_baseline": assessment.strongest_baseline,
                "engine_reported_improvement": assessment.improvement,
                "published_model": published_model,
                "published_support": support,
                "fallback_disclosed": fallback,
                "selection_scores": assessment.selection_scores,
                "test_scores": assessment.test_scores,
                "selection_fold_count": assessment.selection_fold_count,
                "selection_guardrail_applied": (
                    assessment.selection_guardrail_applied),
                "selection_stability": assessment.selection_stability,
                "warnings": assessment.warnings,
                "notes": assessment.notes,
                "departed_from_last_value": departed,
                "reference_losses": reference_losses,
                "strongest_reference": strongest_reference,
                "reference_loss": reference_loss,
                "candidate_loss": candidate_loss,
                "relative_gain": gain,
                "mean_invalidated_departure": (
                    departed and reference_losses["historical_mean"]
                    + 1e-12 < candidate_loss),
            })

    overall = _summarise(rows)
    by_group = {
        group: _summarise([row for row in rows if row["group"] == group])
        for group in sorted({str(row["group"]) for row in rows})
    }
    by_dataset = {
        dataset: _summarise(
            [row for row in rows if row["dataset_id"] == dataset])
        for dataset in sorted({str(row["dataset_id"]) for row in rows})
    }
    interval = _cluster_interval(rows)
    group_medians = [item["median_relative_gain_all_cases"]
                     for item in by_group.values()]
    result = {
        "schema_version": "0.1",
        "benchmark": "naturalistic-production-selector-confirmation",
        "scope": "full" if full_scope else "smoke",
        "protocol": "docs/v0.7-q1-naturalistic-confirmation-protocol.md",
        "minimum_baseline_improvement": DEFAULT_MINIMUM_IMPROVEMENT,
        "cutoff_fractions": list(CUTOFF_FRACTIONS),
        "bootstrap": {"unit": "dataset", "seed": BOOTSTRAP_SEED,
                      "draws": BOOTSTRAP_DRAWS, "interval": interval},
        "dataset_identities": identities,
        "overall": overall,
        "by_group": by_group,
        "by_dataset": by_dataset,
        "raw_records": rows,
    }
    common_gates = {
        "all_product_cases_complete": (
            overall["completed"] == len(selected_specs) * len(CUTOFF_FRACTIONS)),
        "no_silent_fallback": all(
            row["engine_supported"] or row["fallback_disclosed"]
            for row in rows if row["completed"]),
        "future_observations_used_zero": all(
            row["future_observations_used_by_selector"] == 0 for row in rows),
        "three_finite_references_measured": all(
            set(row["reference_losses"]) == {
                "last_value", "seasonal_naive", "historical_mean"}
            and all(math.isfinite(float(value))
                    for value in row["reference_losses"].values())
            for row in rows),
        "selection_provenance_complete": all(
            row["published_model"] != "none"
            and row["published_support"] in {
                "supported", "weakly_supported", "degraded", "best_effort"}
            and isinstance(row["selection_scores"], dict)
            and isinstance(row["warnings"], list)
            for row in rows if row["completed"]),
    }
    promotion_gates = {
        "at_least_10_departures": overall["departures"] >= 10,
        "departure_precision_at_least_70pct": (
            overall["admission_precision"] is not None
            and overall["admission_precision"] >= 0.70),
        "more_departure_wins_than_losses": (
            overall["departure_wins"] > overall["departure_losses"]),
        "departure_median_gain_positive": (
            overall["median_relative_gain_departures"] is not None
            and overall["median_relative_gain_departures"] > 0),
        "cluster_bootstrap_lower_nonnegative": (
            interval is not None and interval[0] >= 0),
        "group_median_regression_within_2pct": (
            len(group_medians) == 4
            and min(float(value) for value in group_medians) >= -0.02),
        "no_departure_invalidated_by_historical_mean": (
            overall["mean_invalidated_departures"] == 0),
    }
    result["gates"] = (
        {**common_gates, **promotion_gates} if full_scope else common_gates)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--datasets",
        help="Comma-separated frozen dataset ids for a smoke shard; omit for all 12.",
    )
    args = parser.parse_args()
    dataset_ids = ({item.strip() for item in args.datasets.split(",") if item.strip()}
                   if args.datasets else None)
    result = run(dataset_ids)
    result["evaluated_commit"] = code_revision()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_manifest(
            args.output_dir, benchmark="naturalistic-selector",
            condition=f"current-production-policy-{result['scope']}",
            target=("12-real-series-48-frozen-origins" if dataset_ids is None
                    else "datasets:" + ",".join(sorted(dataset_ids))),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
