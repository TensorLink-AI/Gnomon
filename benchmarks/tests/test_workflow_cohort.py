"""Independent numerical checks; no scripted fixture establishes agent uplift."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
from fractions import Fraction
import csv
import json
import math
import os
import statistics

import pytest

from benchmarks.workflow.cohort import build
from benchmarks.workflow.matched import ARMS, ROOT, forecast_error_summary, normalized_rows
from benchmarks.workflow.provenance import corpus_sha256
from benchmarks.workflow.run_workflow import case_payload
from benchmarks.workflow.schema import Case, Observation, Oracle, load_cases
from benchmarks.workflow.scoring import forecast_metrics, score_run


def forecast_case():
    return Case.from_dict({"id": "forecast", "kind": "frozen", "question": "Forecast two steps.",
        "available_at_cutoff": {"history": [1, 3, 5]}, "answer_schema": {"numbers": ["h1", "h2"]},
        "oracle": {"numbers": {"h1": 4, "h2": 8}, "forecast": {"keys": ["h1", "h2"], "scale": 2, "max_mae": 2}}})


def test_free_forecast_scored_by_complete_mae_not_exact_target_matching():
    case = forecast_case()
    obs = Observation(case_id=case.id, status="answered", support="supported", numbers={"h1": 2, "h2": 10})
    row = score_run([case], [obs])["rows"][0]
    assert row["correctness"] == row["accuracy_components"]["numeric"] == 1
    assert row["forecast_metrics"]["mae"] == 2
    assert row["forecast_metrics"]["rmse"] == pytest.approx(2)
    assert row["forecast_metrics"]["mase"] == 1
    assert normalized_rows({"rows": [row]})[0]["forecast_metrics"] == row["forecast_metrics"]
    worse = score_run([case], [replace(obs, numbers={"h1": 2, "h2": 11})])["rows"][0]
    assert worse["correctness"] == 0 and worse["forecast_metrics"]["mae"] == 2.5


@pytest.mark.parametrize("status,numbers", [("answered", {}), ("answered", {"h1": 4}),
    ("error", {"h1": 4, "h2": 8}), ("abstained", {"h1": 4, "h2": 8}),
    ("answered", {"h1": 1e308, "h2": -1e308})])
def test_missing_or_failed_forecasts_do_not_get_zero_loss(status, numbers):
    oracle = forecast_case().oracle
    if len(numbers) == 2 and status == "answered":
        oracle = replace(oracle, forecast={**oracle.forecast, "scale": 1e-308})
    metrics = forecast_metrics(oracle, status, numbers)
    assert not metrics["complete"] and metrics["mae"] is None and metrics["mase"] is None
    assert metrics["within_mae_limit"] is False


@pytest.mark.parametrize("change", [
    {"keys": []}, {"keys": ["h1", "h1"]}, {"keys": ["missing"]}, {"scale": 0},
    {"scale": True}, {"scale": float("inf")}, {"max_mae": -1}, {"max_mae": float("nan")}, {"hidden": 1},
])
def test_forecast_contract_refuses_ambiguous_or_nonfinite_limits(change):
    raw = asdict(forecast_case())
    raw["oracle"]["forecast"].update(change)
    with pytest.raises(ValueError):
        Case.from_dict(raw)


def test_forecast_contract_does_not_weaken_other_numeric_requirements():
    case = forecast_case()
    oracle = replace(case.oracle, numbers={**case.oracle.numbers, "sum": 12})
    case = replace(case, oracle=oracle)
    obs = Observation(case_id=case.id, status="answered", support="supported", numbers={"h1": 4, "h2": 8, "sum": 99})
    assert score_run([case], [obs])["rows"][0]["correctness"] == 0.5


def test_forecast_coverage_and_common_complete_cohort_are_not_survivor_uplift():
    metric = forecast_metrics(forecast_case().oracle, "answered", {"h1": 4, "h2": 10})
    missing = forecast_metrics(forecast_case().oracle, "error", {})
    rows = {arm: [{"task_id": "one", "forecast_metrics": metric}, {"task_id": "two", "forecast_metrics": metric}]
            for arm in ARMS}
    rows["lean"][1]["forecast_metrics"] = missing
    rows["full"][0]["forecast_metrics"] = missing
    summary = forecast_error_summary(rows)
    assert summary["planned_forecasts"] == 2 and summary["all_arm_complete_forecasts"] == 0
    assert summary["arms"]["lean"]["missing_or_invalid"] == 1
    assert summary["arms"]["lean"]["mean_mase_complete_only"] == 0.5
    assert summary["arms"]["ordinary"]["mean_mase_all_arm_complete_only"] is None
    rows["full"].pop()
    with pytest.raises(ValueError, match="membership"):
        forecast_error_summary(rows)


def test_frozen_cohort_rebuilds_exactly_and_source_cutoffs_are_independent_of_answers():
    cases, manifest = build()
    directory = ROOT / "benchmarks/workflow/cases"
    assert manifest == json.loads((directory / "matched-retrospective.manifest.json").read_text())
    assert corpus_sha256(cases) == manifest["corpus_sha256"]
    assert [asdict(case) for case in load_cases(directory / "matched-retrospective.jsonl")] == [asdict(case) for case in cases]
    assert len(cases) == 11 and len({case.id for case in cases}) == 11
    for case, source in zip(cases[:4], manifest["sources"]):
        with (ROOT / source["source_file"]).open() as stream:
            raw = [float(row["value"]) for row in csv.DictReader(stream)]
        start, cutoff = source["history_start"], source["cutoff_exclusive"]
        def transform(value):
            return round(value * source["affine_multiplier"] + source["affine_offset"], 8)
        history = [transform(value) for value in raw[start:cutoff]]
        actual = [transform(value) for value in raw[cutoff:cutoff + 4]]
        assert len(history) == 64 and list(case.oracle.numbers.values()) == actual
        assert case.available_at_cutoff["history"] == history
        csv_rows = list(csv.DictReader(case.available_at_cutoff["files"]["history.csv"].splitlines()))
        assert [float(row["value"]) for row in csv_rows] == history
        baseline = {key: history[-1] for key in case.oracle.forecast["keys"]}
        measured = forecast_metrics(case.oracle, "answered", baseline)
        expected = sum(Fraction.from_float(abs(value - history[-1])) for value in actual) / 4
        assert measured["mae"] == float(expected) and measured["within_mae_limit"]
        scale = statistics.mean(abs(history[i] - history[i-1]) for i in range(1, 64))
        assert case.oracle.forecast["scale"] == scale
        public = json.dumps(case_payload(case))
        assert '"oracle"' not in public and '"max_mae"' not in public and source["source_file"] not in public
        changed = deepcopy(asdict(case))
        changed["oracle"]["numbers"]["h1"] += 1000
        assert case_payload(Case.from_dict(changed)) == case_payload(case)
    assert not any(case.stages or case.oracle.context_behavior or case.oracle.engine_required_facts
                   or case.oracle.requires_publish_parity or case.oracle.requires_quote_match for case in cases)


def test_changed_source_bytes_are_rejected_before_building_a_new_identity(tmp_path):
    (tmp_path / "wiki_traffic_daily_log.csv").write_text("value\n999\n")
    with pytest.raises(ValueError, match="source bytes changed"):
        build(tmp_path)


def test_utility_oracles_have_independent_physical_probability_and_time_checks():
    cases, _ = build()
    energy, inventory, elapsed, missing = cases[4:8]
    intervals = energy.available_at_cutoff["intervals"]
    kwh = sum(Fraction(row["power_kw"] * row["minutes"], 60) for row in intervals)
    hours = sum(Fraction(row["minutes"], 60) for row in intervals)
    assert float(kwh) == energy.oracle.numbers["energy_kwh"]
    assert math.isclose(float(kwh / hours), energy.oracle.numbers["mean_power_kw"])
    inputs = inventory.available_at_cutoff
    costs = {q: inputs["purchase_cost"] * q + sum(Fraction(str(row["probability"])) * inputs["shortage_cost"]
             * max(0, row["demand"] - inputs["stock"] - q) for row in inputs["scenarios"]) for q in inputs["allowed_q"]}
    chosen = min(costs, key=costs.get)
    assert inventory.oracle.numbers == {"q": chosen, "expected_cost": float(costs[chosen])}
    assert not inputs["approval_granted"] and inventory.oracle.choices["action"] == "request_approval"
    data = elapsed.available_at_cutoff
    assert (datetime.fromisoformat(data["end"]) - datetime.fromisoformat(data["start"])).total_seconds() == elapsed.oracle.numbers["elapsed_seconds"]
    assert missing.oracle.should_abstain


def test_empty_forecast_contract_preserves_prior_corpus_hash():
    import hashlib
    case = replace(forecast_case(), oracle=Oracle(numbers={"h1": 4}))
    old = asdict(case)
    old.pop("episode")
    old["oracle"].pop("forecast")
    encoded = json.dumps([old], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert corpus_sha256([case]) == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("index", range(4))
def test_actual_statsforecast_can_use_each_public_frozen_window_without_adapter(tmp_path, index):
    from benchmarks.workflow.software_backend import SoftwareBackend
    image = os.environ.get("GNOMON_TEST_SOFTWARE_IMAGE")
    if not image:
        pytest.skip("explicit installed GNOMON_TEST_SOFTWARE_IMAGE required")
    case = build()[0][index]
    backend = SoftwareBackend(case=case_payload(case), options={"image": image, "docker_host": "unix:///var/run/docker.sock"},
                              workspace=tmp_path, timeout=45)
    try:
        result = backend.call("python", {"code": '''import json,numpy as np,pandas as pd,importlib.util,site
from pathlib import Path
assert importlib.util.find_spec('prophet') is None
source_names={'example_wp_log_peyton_manning.csv','example_yosemite_temps.csv','example_pedestrians_covid.csv','example_retail_sales.csv','wiki_traffic_daily_log.csv','sensor_temps_5min.csv','pedestrian_counts_daily.csv','retail_sales_monthly.csv'}
assert not any(path.name in source_names for base in site.getsitepackages() for path in Path(base).rglob('*.csv'))
from statsforecast.models import Naive,AutoETS
y=pd.read_csv('/tmp/data/history.csv')['value'].to_numpy(dtype=float)
results={name:model.forecast(y=y,h=4)['mean'].tolist() for name,model in [('naive',Naive()),('ets',AutoETS(season_length=1))]}
print(json.dumps(results,allow_nan=False))
'''}, timeout=30).value
        assert result["returncode"] == 0, result
        values = json.loads(result["stdout"])
        for model, prediction in values.items():
            assert len(prediction) == 4 and all(math.isfinite(value) for value in prediction)
            numbers = dict(zip(case.oracle.forecast["keys"], prediction))
            metrics = forecast_metrics(case.oracle, "answered", numbers)
            assert metrics["complete"]
            if model == "naive":
                assert prediction == [case.available_at_cutoff["history"][-1]] * 4
                assert metrics["within_mae_limit"]
        assert "gnomon-forecast" not in backend.provenance["software"]["distributions"]
    finally:
        backend.close()
