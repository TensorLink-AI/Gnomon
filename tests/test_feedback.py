from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest

from gnomon import feedback


def test_shareable_projection_excludes_private_material_and_redacts() -> None:
    receipt = feedback.create_receipt(
        "tool_error",
        context={
            "surface": "evidence",
            "verb": "gnomon_forecast",
            "error_code": "TIMEOUT",
            "call_count": 2,
        },
        public_summary=(
            "Failed on /tmp/customer.csv; api_key=super-secret and "
            "owner@example.com saw https://private.example/run/2"
        ),
        private_note="raw internal diagnosis",
        artifact_id="/private/artifacts/forecast-123",
    )

    shared = receipt.shareable()
    encoded = json.dumps(shared)
    assert "raw internal diagnosis" not in encoded
    assert "/private/artifacts/forecast-123" not in encoded
    assert receipt.claim_secret not in encoded
    assert "super-secret" not in encoded
    assert "owner@example.com" not in encoded
    assert "private.example" not in encoded
    assert shared["evidence"]["artifact_digest"].startswith("sha256:")
    assert shared["incentive"]["eligibility"] == "pending_independent_verification"
    assert shared["incentive"]["usage_volume_rewarded"] is False


def test_context_is_allowlisted_and_typed() -> None:
    with pytest.raises(feedback.FeedbackError, match="unknown context fields"):
        feedback.create_receipt("routing", context={"raw_prompt": "secret"})
    with pytest.raises(feedback.FeedbackError, match="call_count"):
        feedback.create_receipt("routing", context={"call_count": "3"})
    with pytest.raises(feedback.FeedbackError, match="non-negative"):
        feedback.create_receipt("routing", context={"call_count": -1})


def test_store_round_trip_uses_private_permissions(tmp_path: Path) -> None:
    store = feedback.FeedbackStore(tmp_path / "feedback")
    receipt = feedback.create_receipt(
        "wrong_answer",
        context={"task_kind": "forecast", "corrected": True},
        private_note="local only",
    )
    path = store.record(receipt)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    loaded = store.get(receipt.receipt_id)
    assert loaded == receipt
    assert loaded.local_dict()["private"]["private_note"] == "local only"


def test_store_rejects_corrupt_evidence_digest(tmp_path: Path) -> None:
    store = feedback.FeedbackStore(tmp_path / "feedback")
    receipt = feedback.create_receipt("tool_error", artifact_id="artifact-1")
    path = store.record(receipt)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence"]["artifact_digest"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(feedback.FeedbackError, match="artifact digest"):
        store.get(receipt.receipt_id)


def test_summary_contains_behavioral_aggregates_not_content() -> None:
    receipts = [
        feedback.create_receipt(
            "unnecessary_calls",
            context={
                "surface": "core",
                "task_kind": "forecast",
                "call_count": 3,
                "minimum_call_count": 1,
            },
            public_summary="shareable words",
            private_note="private words",
        ),
        feedback.create_receipt(
            "positive",
            context={"surface": "evidence", "task_kind": "describe", "call_count": 1},
        ),
    ]

    summary = feedback.summarize_receipts(receipts)
    encoded = json.dumps(summary)
    assert summary["by_category"] == {"positive": 1, "unnecessary_calls": 1}
    assert summary["call_behavior"]["total_calls"] == 4
    assert summary["call_behavior"]["total_redundant_calls"] == 2
    assert "shareable words" not in encoded
    assert "private words" not in encoded
    assert summary["privacy"]["contains_free_text"] is False


def test_export_requires_separate_consent(tmp_path: Path) -> None:
    receipt = feedback.create_receipt("missing_capability")
    destination = tmp_path / "out" / "bundle.json"

    with pytest.raises(feedback.FeedbackError, match="explicit consent"):
        feedback.export_bundle([receipt], destination, consent=False)
    assert not destination.exists()

    feedback.export_bundle([receipt], destination, consent=True)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["receipts"] == [receipt.shareable()]
    assert "private" not in payload["receipts"][0]
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_submission_requires_safe_endpoint_and_consent(monkeypatch) -> None:
    receipt = feedback.create_receipt("tool_error", context={"error_code": "E_TEST"})
    with pytest.raises(feedback.FeedbackError, match="explicit consent"):
        feedback.submit_bundle(
            [receipt], "https://feedback.example/v1", consent=False
        )
    with pytest.raises(feedback.FeedbackError, match="HTTPS"):
        feedback.submit_bundle(
            [receipt], "http://feedback.example/v1", consent=True
        )

    seen = {}

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b'{"server_receipt":"accepted-1"}'

    def fake_urlopen(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(feedback.urlrequest, "urlopen", fake_urlopen)
    result = feedback.submit_bundle(
        [receipt],
        "https://feedback.example/v1",
        consent=True,
        token="token-not-in-body",
        timeout=2.5,
    )
    assert result == {
        "status": "submitted",
        "http_status": 202,
        "response": {"server_receipt": "accepted-1"},
    }
    assert seen["timeout"] == 2.5
    assert seen["request"].get_header("Authorization") == "Bearer token-not-in-body"
    assert b"token-not-in-body" not in seen["request"].data


def test_cli_create_preview_and_consent_gate(tmp_path: Path, capsys) -> None:
    store = tmp_path / "store"
    result = feedback.main([
        "--store",
        str(store),
        "create",
        "--category",
        "abstention",
        "--context",
        "surface=evidence",
        "--context",
        "call_count=1",
        "--private-note",
        "not exported",
    ])
    assert result == 0
    created = json.loads(capsys.readouterr().out)
    receipt_id = created["receipt_id"]
    assert created["shared"] is False

    assert feedback.main([
        "--store", str(store), "preview", receipt_id
    ]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["receipts"][0]["receipt_id"] == receipt_id
    assert "not exported" not in json.dumps(preview)

    output = tmp_path / "bundle.json"
    assert feedback.main([
        "--store", str(store), "export", receipt_id, "--output", str(output)
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "FEEDBACK_ERROR"
    assert not output.exists()

    assert feedback.main([
        "--store",
        str(store),
        "export",
        receipt_id,
        "--output",
        str(output),
        "--consent",
    ]) == 0
    capsys.readouterr()
    assert output.exists()
