from __future__ import annotations

import argparse
import json

import pytest

from benchmarks.breachbench.run_breachbench import run as run_breachbench
from benchmarks.breachbench.run_reconciliation import run as run_reconciliation


class FixedClient:
    def completions(self, messages, *, n=1):
        assert n == 1
        return [json.dumps({
            "breach_expected": False,
            "first_breach_step": None,
            "action": "monitor",
            "automation_action": "withhold",
            "evidence_assessment": "indeterminate",
            "breach_probability": .25,
            "selected_source": "synthesis",
            "counterevidence_source": "immutable_primary",
            "confidence": "low",
            "what_would_change": "More independent replay origins.",
        })]


def _base_args(tmp_path):
    return argparse.Namespace(
        model="fixed-model", base_url="https://example.invalid/v1",
        api_key_env="NONE", cases=4, seed=20260826, data_dir=None,
        concurrency=2, max_tokens=400, reasoning_effort="none",
        output_dir=str(tmp_path / "source"), resume=False,
    )


def _reconcile_args(tmp_path):
    return argparse.Namespace(
        source_run=str(tmp_path / "source"), model="fixed-model",
        base_url="https://example.invalid/v1", api_key_env="NONE",
        cases=4, seed=20260826, concurrency=2, max_tokens=400,
        reasoning_effort="none", output_dir=str(tmp_path / "reconciled"),
        resume=False,
    )


def test_reconciliation_consumes_verified_pre_evidence_rows(tmp_path):
    run_breachbench(_base_args(tmp_path), client=FixedClient())
    summary = run_reconciliation(_reconcile_args(tmp_path), client=FixedClient())
    assert set(summary["metrics"]) == {"control", "gnomon", "reconciled"}
    assert summary["invariants"] == {
        "primary_forecast_unchanged": True,
        "prior_support": "prior_assisted",
        "automation_eligible": False,
        "source_requests_verified": True,
        "held_out_future_absent": True,
    }
    rows = [json.loads(line) for line in
            (tmp_path / "reconciled" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(row["automation_action"] == "withhold" for row in rows)
    assert all(row["selection_valid"] is True for row in rows)
    assert all(row.get("what_would_change") for row in rows)
    assert all(row.get("request_sha256") for row in rows)
    selection = summary["selection"]
    assert selection["action_conflicts"] >= 0
    assert selection["chose_prior_action_on_conflict"] <= \
        selection["action_conflicts"]


def test_reconciliation_rejects_tampered_source_request(tmp_path):
    run_breachbench(_base_args(tmp_path), client=FixedClient())
    path = tmp_path / "source" / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["request_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="request identity failed"):
        run_reconciliation(_reconcile_args(tmp_path), client=FixedClient())
