from gnomon.llm_covariates import (
    inline_covariate_arguments,
    validate_llm_covariate_tables,
)


DOC = "On 2026-08-27 the published temperature forecast is 31.5 degrees."


def _table(value=31.5, *, name="temperature", quote=DOC,
           timestamp="2026-08-27T00:00:00+00:00"):
    return [{
        "name": name,
        "type": "continuous",
        "rows": [{
            "document_index": 0,
            "timestamp": timestamp,
            "source_time_span": "2026-08-27",
            "value": value,
            "evidence_quote": quote,
            "known_at": "1999-01-01T00:00:00+00:00",
        }],
    }]


def test_cited_row_is_host_timestamped_and_loader_ready():
    receipt, rejected = validate_llm_covariate_tables(
        _table(), documents=[DOC],
        known_at="2026-08-25T00:00:00+00:00",
        as_of="2026-08-25T00:00:00+00:00",
    )
    assert rejected == []
    assert receipt["tables_proposed"] == 1
    assert receipt["rows_proposed"] == 1
    assert receipt["tables_validated"] == 1
    assert receipt["rows_validated"] == 1
    row = receipt["tables"][0]["rows"][0]
    assert row["known_at"] == "2026-08-25T00:00:00+00:00"
    assert row["provenance"]["class"] == "llm_extracted_host_verified"
    arguments = inline_covariate_arguments(receipt)
    assert arguments["covariates"][0]["temperature"] == 31.5
    assert arguments["covariate_mapping"][0]["availability"] == "future_known"


def test_model_cannot_invent_value_or_backdate_known_at():
    receipt, rejected = validate_llm_covariate_tables(
        _table(99), documents=[DOC],
        known_at="2026-08-25T00:00:00+00:00",
        as_of="2026-08-25T00:00:00+00:00",
    )
    assert receipt["tables"] == []
    assert any("value" in reason for reason in rejected)


def test_timestamp_must_be_supported_by_exact_source_time():
    receipt, rejected = validate_llm_covariate_tables(
        _table(timestamp="2026-08-28T00:00:00+00:00"), documents=[DOC],
        known_at="2026-08-25T00:00:00+00:00",
        as_of="2026-08-25T00:00:00+00:00",
    )
    assert receipt["tables"] == []
    assert any("timestamp" in reason for reason in rejected)


def test_date_component_cannot_be_reused_as_covariate_value():
    receipt, rejected = validate_llm_covariate_tables(
        _table(27), documents=[DOC],
        known_at="2026-08-25T00:00:00+00:00",
        as_of="2026-08-25T00:00:00+00:00",
    )
    assert receipt["tables"] == []
    assert any("value" in reason for reason in rejected)


def test_multiple_tables_are_retained_but_not_silently_merged():
    receipt, rejected = validate_llm_covariate_tables(
        _table() + _table(name="weather"), documents=[DOC],
        known_at="2026-08-25T00:00:00+00:00",
        as_of="2026-08-25T00:00:00+00:00",
    )
    assert rejected == []
    assert len(receipt["tables"]) == 2
    assert inline_covariate_arguments(receipt) == {}


def test_each_document_uses_host_attested_knowledge_time():
    second = "On 2026-08-28 the published temperature forecast is 32.0 degrees."
    proposal = _table()
    proposal[0]["rows"].append({
        "document_index": 1,
        "timestamp": "2026-08-28T00:00:00+00:00",
        "source_time_span": "2026-08-28",
        "value": 32.0,
        "evidence_quote": second,
    })
    receipt, rejected = validate_llm_covariate_tables(
        proposal, documents=[DOC, second], known_at=None,
        document_known_at=[
            "2026-08-24T00:00:00+00:00",
            "2026-08-25T00:00:00+00:00",
        ],
        as_of="2026-08-25T00:00:00+00:00",
    )
    assert rejected == []
    rows = receipt["tables"][0]["rows"]
    assert [row["known_at"] for row in rows] == [
        "2026-08-24T00:00:00+00:00",
        "2026-08-25T00:00:00+00:00",
    ]
