"""Rebuild the small retrospective matched cohort from pinned local CSV bytes.

No download, model request, executable third-party scorer, or host-recovered answer.
Printing the document does not run the experiment. Known historical data can have
been seen in model training; affine transforms do not prove decontamination.
"""

from dataclasses import asdict
import csv
import hashlib
import json
import statistics

from .matched import ROOT
from .provenance import corpus_sha256
from .schema import Case, load_cases

SOURCES = (
    ("wiki_traffic_daily_log", "7ee685b87a4685ed484563d2d09a430aa2d886b6f0a6eeb6a6e86aac474a3dbc", 0.01, 17, "daily"),
    ("sensor_temps_5min", "cdeab8b7cdeade5ef6a04b2bbd2b2a4d1393a0dc90d7c4201b8a823a02f87387", 0.25, 8, "five-minute"),
    ("pedestrian_counts_daily", "a53d1d3e98c323be36ac52f35a7a06d91ba3a3a7ca61d99330ca0f670236f7f7", 3, 41, "daily"),
    ("retail_sales_monthly", "1d524c08bdf2c4bd6b75c9c629e9949ebc10fa2f268ed373b02709b8d41c4815", 0.001, -6, "monthly"),
)
UPSTREAM_COMMIT = "79ef5ecbe85179a3a2afa3b62f0d2a6b223cc7db"
# Read from this pinned upstream revision during the extraction audit. Calendar
# labels are timezone-unspecified and are provenance only, not model inputs.
UPSTREAM = {
    "wiki_traffic_daily_log": {"file": "example_wp_log_peyton_manning.csv", "sha256": "6b1383a6f458e317c5da0488abaf2cadccd7e80c84c7a543d084e1159d5bfa56", "rows": 2905, "history_start": "2015-11-14", "last_history": "2016-01-16", "last_target": "2016-01-20"},
    "sensor_temps_5min": {"file": "example_yosemite_temps.csv", "sha256": "c0ec9f2cb4bbf0bc53f7bfd2e39f88ae21e43b7b8912b2d1eb8185055f9510e2", "rows": 18721, "history_start": "2017-07-04T18:25:00", "last_history": "2017-07-04T23:40:00", "last_target": "2017-07-05T00:00:00"},
    "pedestrian_counts_daily": {"file": "example_pedestrians_covid.csv", "sha256": "e8971ec69aba730809d185ea2e0a24a56281f1a67b3a74ead01f0463c939ba20", "rows": 1490, "history_start": "2021-04-24", "last_history": "2021-06-26", "last_target": "2021-06-30"},
    "retail_sales_monthly": {"file": "example_retail_sales.csv", "sha256": "6556f0683b78362c4bc4390b8feb66870beecd7189cc052e650ccac9b21cd65c", "rows": 293, "history_start": "2010-10-01", "last_history": "2016-01-01", "last_target": "2016-05-01"},
}


def build(data_dir=None):
    data_dir = data_dir or ROOT / "benchmarks/workflow/data"
    cases, sources = [], []
    for index, (name, digest, multiplier, offset, cadence) in enumerate(SOURCES, 1):
        path = data_dir / (name + ".csv")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen source bytes changed: {name}")
        with path.open() as stream:
            raw = [float(row["value"]) for row in csv.DictReader(stream)]
        cutoff = len(raw) - 4
        transformed = [round(value * multiplier + offset, 8) for value in raw[cutoff - 64:]]
        history, actuals = transformed[:64], transformed[64:]
        scale = statistics.mean(abs(right - left) for left, right in zip(history, history[1:]))
        baseline_mae = statistics.mean(abs(value - history[-1]) for value in actuals)
        keys = [f"h{step}" for step in range(1, 5)]
        identifier = f"retrospective_forecast_{index}"
        public = {"history": history, "horizon": 4, "cadence": cadence, "unit": "transformed_index",
                  "time_axis": "equally spaced observation steps; original calendar timestamps are not supplied",
                  "evaluation": "MAE compared with repeating the last observed value; scaled error uses mean absolute one-step training changes",
                  "files": {"history.csv": "step,value\n" + "".join(f"{i},{value}\n" for i, value in enumerate(history))}}
        cases.append(Case.from_dict({"id": identifier, "kind": "frozen", "domain": "forecasting",
            "question": "Forecast the next four equally spaced observations using any available method. Return point forecasts as h1, h2, h3 and h4. These are affine-transformed retrospective observations; later values are withheld, not guaranteed absent from model training. State prediction_kind=forecast in facts. No real-world action is authorized.",
            "available_at_cutoff": public, "answer_schema": {"numbers": keys, "facts": ["prediction_kind"]},
            "tags": ["matched-retrospective-v1", "forecast-error", "possible-training-contamination"],
            "oracle": {"numbers": dict(zip(keys, actuals)), "required_facts": {"prediction_kind": "forecast"},
                       "forecast": {"keys": keys, "scale": scale, "max_mae": baseline_mae}}}))
        sources.append({"case_id": identifier, "source_file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
                        "source_sha256": digest, "rows": len(raw), "history_start": cutoff - 64,
                        "cutoff_exclusive": cutoff, "horizon": 4, "affine_multiplier": multiplier,
                        "affine_offset": offset, "round_decimals": 8, "cadence": cadence,
                        "upstream": {**UPSTREAM[name], "url": f"https://raw.githubusercontent.com/facebook/prophet/{UPSTREAM_COMMIT}/examples/{UPSTREAM[name]['file']}",
                                     "extraction": "finite y values only; selected final68 rows have no missing value or cadence gap"}})
    utility = [
        {"id": "weighted_energy", "domain": "quantities",
         "question": "Power is constant over each supplied interval. Report total energy in kWh and time-weighted mean power in kW; do not use an unweighted average of readings.",
         "available_at_cutoff": {"intervals": [{"power_kw": 2, "minutes": 15}, {"power_kw": 8, "minutes": 45}, {"power_kw": 4, "minutes": 30}]},
         "answer_schema": {"numbers": ["energy_kwh", "mean_power_kw"]},
         "oracle": {"numbers": {"energy_kwh": 8.5, "mean_power_kw": 17 / 3}, "tolerances": {"mean_power_kw": 1e-8}}},
        {"id": "inventory_expected_cost", "domain": "decision",
         "question": "Choose additional order quantity q from the allowed quantities to minimize expected purchase-plus-shortage cost under the supplied scenario distribution. Purchase cost is charged for every additional unit; shortage is max(demand-stock-q,0). Report q and expected cost. Approval is required and absent, so choose action=request_approval or execute accordingly. Do not place an order.",
         "available_at_cutoff": {"distribution_basis": "supplied synthetic scenario, not an inferred probability model",
                                  "scenarios": [{"demand": 8, "probability": 0.25}, {"demand": 12, "probability": 0.5}, {"demand": 20, "probability": 0.25}],
                                  "stock": 10, "purchase_cost": 2, "shortage_cost": 5, "allowed_q": [0, 2, 10], "approval_granted": False},
         "answer_schema": {"numbers": ["q", "expected_cost"], "choices": ["action"]},
         "oracle": {"numbers": {"q": 2, "expected_cost": 14}, "choices": {"action": "request_approval"}, "forbidden_claims": ["order was placed"]}},
        {"id": "dst_elapsed_time", "domain": "temporal",
         "question": "Report elapsed seconds between the two instants. They have identical wall-clock labels but different explicit UTC offsets; do not subtract naive local times.",
         "available_at_cutoff": {"start": "2025-11-02T01:30:00-04:00", "end": "2025-11-02T01:30:00-05:00"},
         "answer_schema": {"numbers": ["elapsed_seconds"]}, "oracle": {"numbers": {"elapsed_seconds": 3600}}},
        {"id": "missing_target", "domain": "forecasting",
         "question": "Forecast our single business target for three steps. The data does not identify which column is that target, and neither choosing a column nor aggregating them is authorized. Abstain and identify the missing target selection.",
         "available_at_cutoff": {"east": [10, 11, 12], "west": [90, 80, 70]},
         "answer_schema": {}, "oracle": {"should_abstain": True, "allowed_support": ["abstained"], "required_disclosures": ["target"]}},
    ]
    cases.extend(Case.from_dict({"kind": "synthetic", "tags": ["matched-retrospective-v1", "tool-neutral"], **value}) for value in utility)
    episodes_path = ROOT / "benchmarks/workflow/cases/agent_episodes.jsonl"
    cases.extend(load_cases(episodes_path))
    manifest = {"schema_version": 1, "cohort": "matched-retrospective-v1", "case_ids": [case.id for case in cases],
                "corpus_sha256": corpus_sha256(cases), "sources": sources,
                "episode_source_sha256": hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
                "limitations": ["Retrospective historical holdouts, not certified unseen model-training data.",
                                "Affine transformation does not establish decontamination or independent tasks.",
                                "Four source windows and synthetic utility/episode tasks are a small scoped comparison, not SOTA evidence.",
                                "Forecast success means complete MAE no worse than repeat-last-value; continuous losses and missing forecasts are reported separately.",
                                "MASE scale is nonseasonal one-step training change, not a seasonal or official TemporalBench metric."]}
    return cases, manifest


if __name__ == "__main__":
    cases, manifest = build()
    print(json.dumps({"cases": [asdict(case) for case in cases], "manifest": manifest}, allow_nan=False))
