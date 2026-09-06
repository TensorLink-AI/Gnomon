"""Architectural guards for the canonical tool surface."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

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


def test_legacy_wire_contract_is_unchanged_by_internal_cleanup():
    # Public descriptions, schemas, ordering and profile membership at fa67fbe.
    # Deliberate future wire changes must update compatibility notes and this pin.
    contract = {
        "tools": [{key: value for key, value in tool.items() if key != "runner"}
                  for tool in toolspec.TOOLS],
        "profiles": {name: sorted(members) for name, members in toolspec.PROFILES.items()},
    }
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    assert digest == "6cf70fd294125dd7d17e5b31016ff45dde587678b322c0d4366988f8045006d1"
    assert all(callable(tool["runner"]) for tool in toolspec.TOOLS)


@pytest.mark.parametrize("statement,forbidden", [
    ("from gnomon.tool_response import forecast_summary",
     ("toolspec", "tool_operations", "runtime", "tool_publication")),
    ("from gnomon.tool_catalog import TOOL_SCHEMAS",
     ("toolspec", "tool_operations", "runtime", "tool_publication")),
    ("from gnomon.runtime import capabilities; capabilities()",
     ("toolspec", "tool_operations", "tool_publication")),
])
def test_projection_and_discovery_do_not_load_tool_execution(statement, forbidden):
    root = Path(__file__).resolve().parents[1]
    script = (f"import sys; {statement}; "
              f"assert not set({forbidden!r}) & "
              "{name.removeprefix('gnomon.') for name in sys.modules "
              "if name.startswith('gnomon.')}")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "GNOMON_MCP_PROFILE": "full"},
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_retired_experiment_dispatch_is_not_shipped():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src/gnomon/legacy_experiments.py").exists()


def test_shared_admission_config_preserves_single_and_batch_activation():
    from types import SimpleNamespace
    from gnomon.tool_operations import _forecast_config

    events = [SimpleNamespace(event_type="structural:step")]
    single = _forecast_config({}, events, infer_structural=True)
    assert single.context.structural_events is True
    assert _forecast_config({}, events, infer_structural=False) is None
    batch = _forecast_config({"structural_events": True}, events, infer_structural=False)
    assert batch.context.structural_events is True
    for infer_structural in (False, True):
        future = _forecast_config(
            {}, [SimpleNamespace(event_type="constraint:literal_min")],
            infer_structural=infer_structural,
        )
        assert future.context.future_events is True


def test_shared_admission_config_still_requires_model_evidence():
    from gnomon.contracts import GnomonError
    from gnomon.tool_operations import _forecast_config

    with pytest.raises(GnomonError) as raised:
        _forecast_config({"model_admission": "evidence_weighted"}, [], infer_structural=True)
    assert raised.value.code == "MISSING_MODEL_EVIDENCE_REGISTRY"
    config = _forecast_config(
        {"model_admission": "evidence_weighted", "model_evidence_registry": "evidence.json"},
        [], infer_structural=False,
    )
    assert config.models.admission_policy == "evidence_weighted"
    assert config.models.evidence_registry_path == "evidence.json"
