from __future__ import annotations

import json

import pytest

from gnomon.context_store import (
    ContextReceiptStore, assessment_cache_key, context_cache_key,
    temporal_answer_cache_key,
)
from gnomon.soft_context import make_context_receipt


def _receipt(*, known_at: str = "2026-01-01T00:00:00+00:00"):
    return make_context_receipt(
        documents=[{"content_fingerprint": "sha256:document"}],
        events=[{
            "event_id": "deploy", "event_type": "deployment",
            "entity_scope": ["api"],
            "effective_start": "2026-01-02T00:00:00+00:00",
            "effective_end": "2026-01-03T00:00:00+00:00",
            "known_at": known_at,
        }], hypotheses=[], rejected=[], rejected_hypotheses=[],
        proposer={"model": "compiler-a"})


def test_receipt_store_round_trips_and_finds_by_interpretation_key(tmp_path):
    store = ContextReceiptStore(tmp_path, namespace="team-a")
    key = context_cache_key(
        document_fingerprints=["sha256:document"],
        available_targets=["api"], horizon=12, frequency="h",
        compiler={"model": "compiler-a"}, prompt_version="p1",
        receipt_schema_version="0.1")
    reference = store.put(_receipt(), cache_key=key)
    assert reference.startswith("context_")
    assert store.get(reference) == _receipt()
    assert store.find(key) == _receipt()


def test_context_materializer_preserves_file_trust_and_inline_distrust(
        tmp_path, monkeypatch):
    from gnomon.context import backtest_admissible
    from gnomon.toolspec import _materialise_context

    store = ContextReceiptStore(tmp_path / "store")
    monkeypatch.setattr(ContextReceiptStore, "default", lambda: store)
    event = {
        "event_id": "deploy", "event_type": "deployment",
        "entity_scope": ["*"],
        "effective_start": "2026-01-02T00:00:00+00:00",
        "effective_end": "2026-01-03T00:00:00+00:00",
        "known_at": "2026-01-01T00:00:00+00:00",
        "source": {"type": "calendar", "reference": "ops.ics"},
        "created_by": "llm",
    }
    event_file = tmp_path / "events.json"
    event_file.write_text(json.dumps({
        "schema_version": "0.1", "events": [event]}))

    file_arguments, _ = _materialise_context({
        "context_events_file": str(event_file)})
    inline_arguments, _ = _materialise_context({"context_events": [event]})
    assert backtest_admissible(
        file_arguments["_materialized_context_events"][0])
    assert not backtest_admissible(
        inline_arguments["_materialized_context_events"][0])


def test_one_receipt_can_satisfy_multiple_interpretation_keys(tmp_path):
    store = ContextReceiptStore(tmp_path)
    receipt = _receipt()
    reference = store.put(receipt)
    assert store.put(receipt, cache_key="context_key:a") == reference
    assert store.put(receipt, cache_key="context_key:b") == reference
    assert store.find("context_key:a") == receipt
    assert store.find("context_key:b") == receipt


def test_namespace_cannot_read_another_projects_reference(tmp_path):
    reference = ContextReceiptStore(
        tmp_path, namespace="team-a").put(_receipt())
    with pytest.raises(KeyError):
        ContextReceiptStore(tmp_path, namespace="team-b").get(reference)


def test_receipt_corruption_fails_closed(tmp_path):
    store = ContextReceiptStore(tmp_path)
    reference = store.put(_receipt())
    path = next((tmp_path / "receipts").glob("*.json"))
    body = json.loads(path.read_text())
    body["events"][0]["known_at"] = "2099-01-01T00:00:00+00:00"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="integrity"):
        store.get(reference)


def test_knowledge_revision_and_execution_vintage_change_keys():
    first = _receipt()
    revised = _receipt(known_at="2026-01-02T00:00:00+00:00")
    assert first["receipt_id"] != revised["receipt_id"]
    common = {
        "receipt_id": first["receipt_id"],
        "visible_data_fingerprint": "sha256:data", "series": "api",
        "fitted_candidate_id": "candidate:1",
        "configuration_fingerprint": "config:1", "code_version": "1",
    }
    assert assessment_cache_key(as_of="2026-01-01", **common) != \
        assessment_cache_key(as_of="2026-01-02", **common)


def test_assessment_cache_is_immutable_and_namespaced(tmp_path):
    key = assessment_cache_key(
        receipt_id="context_receipt:x", visible_data_fingerprint="data:a",
        as_of="2026-01-01", series="api", fitted_candidate_id="fit:a",
        configuration_fingerprint="config:a", code_version="1")
    store = ContextReceiptStore(tmp_path, namespace="a")
    store.put_assessment(key, {"admitted": False})
    assert store.get_assessment(key) == {"admitted": False}
    assert ContextReceiptStore(
        tmp_path, namespace="b").get_assessment(key) is None
    with pytest.raises(ValueError, match="another result"):
        store.put_assessment(key, {"admitted": True})


def test_assessment_corruption_fails_closed(tmp_path):
    key = assessment_cache_key(
        receipt_id="context_receipt:x", visible_data_fingerprint="data:a",
        as_of=None, series="api", fitted_candidate_id="fit:a",
        configuration_fingerprint="config:a", code_version="1")
    store = ContextReceiptStore(tmp_path)
    store.put_assessment(key, {"admitted": False})
    path = next((tmp_path / "assessments").glob("*.json"))
    path.write_text('{"admitted": true}\n')
    with pytest.raises(ValueError, match="integrity"):
        store.get_assessment(key)


def test_temporal_answer_cache_binds_primary_question_vintage_and_namespace(
        tmp_path):
    common = {
        "artifact_id": "forecast:a", "question": {
            "id": "v", "verb": "predict", "target": "cpu",
            "property": "volatility", "horizon": 7,
        }, "as_of": "2026-01-01", "answer_contract_version": "0.3",
    }
    key = temporal_answer_cache_key(**common)
    assert key != temporal_answer_cache_key(**{**common, "as_of": "2026-01-02"})
    store = ContextReceiptStore(tmp_path, namespace="a")
    answer = {"best_estimate": {"value": "stable", "support": "weak"},
              "artifact_id": "forecast:a"}
    store.put_temporal_answer(key, answer)
    assert store.get_temporal_answer(key) == answer
    assert ContextReceiptStore(
        tmp_path, namespace="b").get_temporal_answer(key) is None
    with pytest.raises(ValueError, match="another result"):
        store.put_temporal_answer(key, {**answer, "artifact_id": "forecast:b"})


def test_temporal_answer_cache_corruption_fails_closed(tmp_path):
    key = temporal_answer_cache_key(
        artifact_id="forecast:a", question={"id": "q"}, as_of=None,
        answer_contract_version="0.3")
    store = ContextReceiptStore(tmp_path)
    store.put_temporal_answer(key, {"artifact_id": "forecast:a"})
    path = next((tmp_path / "temporal-answers").glob("*.json"))
    path.write_text('{"artifact_id": "forecast:b"}\n')
    with pytest.raises(ValueError, match="integrity"):
        store.get_temporal_answer(key)


def test_cli_forecast_resolves_persistent_context_ref(tmp_path, capsys):
    from datetime import date, timedelta
    from gnomon.cli import main

    store_path = tmp_path / "store"
    reference = ContextReceiptStore(
        store_path, namespace="project").put(_receipt())
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("\n".join(["timestamp,value"] + [
        f"{start + timedelta(days=index)},{index}" for index in range(40)
    ]) + "\n")
    code = main([
        "forecast", str(source), "--horizon", "2",
        "--context-ref", reference, "--context-store", str(store_path),
        "--context-namespace", "project", "--output", str(tmp_path / "out"),
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_ref"] == reference
    assert payload["context_cache"]["status"] == "hit"
