"""Architectural guards for the canonical tool surface."""

from gnomon import response_budget, tool_schema, toolspec


def test_toolspec_keeps_response_budget_compatibility_exports() -> None:
    assert toolspec.enforce_response_budget is response_budget.enforce_response_budget
    assert toolspec.RESPONSE_BUDGET_BYTES == response_budget.RESPONSE_BUDGET_BYTES
    assert (toolspec.DESCRIBE_RESPONSE_BUDGET_BYTES
            == response_budget.DESCRIBE_RESPONSE_BUDGET_BYTES)
    assert (toolspec.CAPABILITIES_RESPONSE_BUDGET_BYTES
            == response_budget.CAPABILITIES_RESPONSE_BUDGET_BYTES)


def test_toolspec_uses_canonical_schema_fragments() -> None:
    assert toolspec._CONTEXT_EVENTS_PROPERTY is tool_schema.CONTEXT_EVENTS_PROPERTY
    assert toolspec._COVARIATES_PROPERTY is tool_schema.COVARIATES_PROPERTY
    assert (toolspec._COVARIATE_MAPPING_PROPERTY
            is tool_schema.COVARIATE_MAPPING_PROPERTY)
    assert toolspec._INPUT_PROPERTIES is tool_schema.INPUT_PROPERTIES
    assert toolspec._OBSERVATIONS_PROPERTY is tool_schema.OBSERVATIONS_PROPERTY
    assert toolspec._REPLAY_PROPERTIES is tool_schema.REPLAY_PROPERTIES
    assert (toolspec._TEMPORAL_QUESTIONS_PROPERTY
            is tool_schema.TEMPORAL_QUESTIONS_PROPERTY)
