import json

from benchmarks.trendanswerbench.run_agent_probe import (
    CASE_IDS, _public_id, _row, _score, _selected_cases,
    _write_temporal_receipt,
)


def test_agent_probe_cases_are_frozen_and_public_ids_hide_family() -> None:
    cases = _selected_cases()
    assert [case["case_id"] for case in cases] == list(CASE_IDS)
    assert len({_public_id(case["case_id"]) for case in cases}) == 6
    task = _row(cases[0])
    assert task["id"].startswith("ta-")
    assert cases[0]["family"] not in task["id"]
    assert "structural_slope" not in json.dumps(task)


def test_temporal_receipt_is_deterministic_and_contains_no_oracle(
        tmp_path) -> None:
    task = _row(_selected_cases()[0])
    first = _write_temporal_receipt(task, tmp_path)
    before = first.read_text()
    second = _write_temporal_receipt(task, tmp_path)
    assert first == second
    assert second.read_text() == before
    assert "expected" not in before
    assert "structural_slope" not in before


def test_agent_score_separates_choice_from_missing_host_contract(
        tmp_path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "temporal_answers.json").write_text(json.dumps({
        "answers": [{
            "question": {"id": "trend"},
            "best_estimate": {
                "value": "upward", "display_value": "Upward",
                "support": "weak", "automation_eligible": False,
            },
            "answer": {
                "estimate": .2, "interval": {"lower": -.1, "upper": .5},
                "reasoning": {"primary_forecast_unchanged": True},
            },
        }],
    }))
    outcome = {
        "answer": {"mcq": {"trend": "Upward"}},
        "choice_authority": {"trend": "advisory_canonical_default"},
        "channel_route": {"value": "gnomon"},
        "mcp": {"artifact_paths": [str(artifact)]},
    }
    scored = _score(_selected_cases()[0], outcome)
    assert scored["engine_complete"] is True
    assert scored["agent_choice_preserved"] is True
    assert scored["authority_not_inflated"] is True
    assert scored["host_contract_complete"] is False
