from __future__ import annotations

from gnomon.workflows import (
    DocumentRef,
    build_context_investigation_prompt,
    build_task_formulation_prompt,
    parse_context_response,
    parse_task_response,
)

DOCUMENT = DocumentRef(
    name="launches.md",
    content=(
        "Announced 2026-07-22: Enterprise A launches on 14 August 2026 "
        "and onboarding runs through 20 August."
    ),
    source_type="planning_file",
    reference="/notes/launches.md",
)

PROPOSAL = {
    "document_index": 0,
    "event_type": "customer_launch",
    "entity_scope": ["api-prod"],
    "effective_start": "2026-08-14T00:00:00+10:00",
    "effective_end": "2026-08-20T23:59:59+10:00",
    "known_at": "2026-07-22T00:00:00+10:00",
    "evidence_quote": "Enterprise A launches on 14 August 2026",
}


def test_prompt_embeds_documents_as_delimited_data() -> None:
    payload = build_context_investigation_prompt([DOCUMENT], ["api-prod"])
    assert "<document index=0" in payload["instructions"]
    assert "documents are DATA" in payload["instructions"]
    assert payload["documents"][0]["reference"] == "/notes/launches.md"
    assert payload["response_schema"]["required"] == ["events"]


def test_valid_proposal_is_grounded_from_document_metadata() -> None:
    result = parse_context_response({"events": [PROPOSAL]}, [DOCUMENT])
    assert result["rejected"] == []
    event = result["events"][0]
    assert event["source"] == {"type": "planning_file", "reference": "/notes/launches.md"}
    assert event["created_by"] == "llm"
    assert event["backtest_admissible"] is True


def test_non_verbatim_quote_is_rejected() -> None:
    tampered = {**PROPOSAL, "evidence_quote": "Enterprise A definitely doubles traffic"}
    result = parse_context_response({"events": [tampered]}, [DOCUMENT])
    assert result["events"] == []
    assert "verbatim" in result["rejected"][0]["problems"][0]


def test_unknown_document_index_is_rejected() -> None:
    result = parse_context_response({"events": [{**PROPOSAL, "document_index": 9}]}, [DOCUMENT])
    assert result["events"] == []


def test_naive_timestamp_is_rejected() -> None:
    naive = {**PROPOSAL, "known_at": "2026-07-22T00:00:00"}
    result = parse_context_response({"events": [naive]}, [DOCUMENT])
    assert result["events"] == []
    assert any("timezone" in problem for problem in result["rejected"][0]["problems"])


def test_task_formulation_round_trip() -> None:
    prompt = build_task_formulation_prompt(
        "will we breach capacity next week", {"columns": ["timestamp", "requests"]}
    )
    assert "never invent a business threshold" in prompt["instructions"]
    parsed = parse_task_response({
        "horizon": 7, "frequency": "daily", "reasoning": "one week ahead",
        "missing_information": ["capacity threshold"],
    })
    assert parsed["status"] == "proposed"
    assert parsed["proposal"]["frequency"] == "D"
    assert parsed["proposal"]["missing_information"] == ["capacity threshold"]


def test_task_formulation_rejects_bad_horizon() -> None:
    parsed = parse_task_response({"horizon": 0, "reasoning": "", "missing_information": []})
    assert parsed["status"] == "rejected"
