from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_spec = spec_from_file_location("check_production_progress", Path(__file__).resolve().parents[1]
                                / "scripts" / "check_production_progress.py")
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
summarize = _module.summarize


def document():
    return {"iteration": 1, "workstreams": [{"id": "delivery", "weight": 100,
            "checks": [{"id": "implemented", "weight": 90, "status": "verified", "evidence": ["test run"]},
                       {"id": "release", "weight": 10, "status": "pending", "evidence": []}]}]}


def test_only_verified_evidence_earns_points():
    source = document()
    before = deepcopy(source)
    assert summarize(source)["earned"] == 90
    assert summarize(source)["complete"] is False
    assert source == before
    source["workstreams"][0]["checks"][1].update(status="verified", evidence=["release checked"])
    assert summarize(source)["complete"] is True


def test_weight_total_must_be_exactly_one_hundred():
    source = document()
    source["workstreams"][0]["weight"] = 99
    source["workstreams"][0]["checks"][1]["weight"] = 9
    with pytest.raises(ValueError, match="total 100"):
        summarize(source)


@pytest.mark.parametrize("mutation", [
    {"evidence": []}, {"evidence": [""]}, {"status": "almost"}, {"weight": True},
])
def test_invented_status_missing_evidence_and_invalid_weights_fail(mutation):
    source = document()
    source["workstreams"][0]["checks"][0].update(mutation)
    with pytest.raises(ValueError):
        summarize(source)


def test_duplicate_checks_fail():
    source = document()
    source["workstreams"][0]["checks"][1]["id"] = "implemented"
    with pytest.raises(ValueError, match="duplicate check"):
        summarize(source)
