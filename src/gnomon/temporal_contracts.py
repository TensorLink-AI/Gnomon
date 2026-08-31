"""Shared contracts for extending Gnomon's temporal execution runtime.

Capabilities are registered here before they can be planned or published.
The registry is intentionally small and deterministic: an LLM may compile a
question, but it cannot invent a capability or substitute a nearby operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import re

from .temporal_question import TemporalQuestion


DATASET_CONTRACT_VERSION = "0.1"
EXECUTION_PLAN_VERSION = "0.1"


@dataclass(frozen=True)
class DatasetContract:
    shape: str
    targets: tuple[str, ...]
    observations: dict[str, int]
    frequency: str | None = None
    time_column: str | None = None
    series_column: str | None = None
    label_column: str | None = None
    truncated: bool = False
    confidence: str = "high"
    ambiguities: tuple[str, ...] = ()
    version: str = DATASET_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextAssessment:
    state: str
    provenance: str
    may_influence_primary: bool
    may_influence_conditional: bool
    reason: str
    primary_forecast_unchanged: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_context(effect: dict[str, Any] | None) -> ContextAssessment:
    """Project artifact context into one shared influence vocabulary."""
    payload = effect or {}
    measured = payload.get("measured_effect") or {}
    admitted = bool(measured.get("admitted") or measured.get("applied"))
    conditional = int(payload.get("conditional_forecast_count") or 0) > 0 \
        or int(payload.get("sensitivity_scenario_count") or 0) > 0
    provenance = str(payload.get("provenance") or "unavailable")
    if admitted:
        return ContextAssessment(
            "historically_validated", provenance, True, True,
            "the effect passed the governed historical admission policy")
    if conditional:
        return ContextAssessment(
            "prior_assisted_conditional", provenance, False, True,
            "the prospective effect is available only as a labelled scenario")
    if payload:
        return ContextAssessment(
            "informational_only", provenance, False, False,
            "context was retained but did not earn numerical influence")
    return ContextAssessment(
        "rejected", provenance, False, False,
        "no provenance-bearing context effect was available")


def classify_dataset_contract(
    targets: list[str] | tuple[str, ...], *,
    observations: dict[str, int] | None = None,
    frequency: str | None = None,
    time_column: str | None = None,
    series_column: str | None = None,
    label_column: str | None = None,
    truncated: bool = False,
) -> DatasetContract:
    """Classify the representation without inspecting answer values."""
    names = tuple(str(name) for name in targets)
    counts = dict(observations or {})
    waveform = len(names) >= 8 and all(
        re.fullmatch(r"(?:t|step|lag)_?\d+", name, re.IGNORECASE)
        for name in names)
    if waveform:
        shape = "wide_waveform"
    elif series_column:
        shape = "panel"
    elif label_column:
        shape = "supervised_table"
    elif len(names) > 1:
        shape = "multivariate"
    else:
        shape = "univariate"
    ambiguities: list[str] = []
    if not names:
        ambiguities.append("no numeric target was resolved")
    if truncated:
        ambiguities.append("input is a truncated preview")
    return DatasetContract(
        shape=shape, targets=names, observations=counts,
        frequency=frequency, time_column=time_column,
        series_column=series_column, label_column=label_column,
        truncated=truncated,
        confidence="low" if ambiguities else "high",
        ambiguities=tuple(ambiguities),
    )


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    properties: tuple[str, ...]
    verbs: tuple[str, ...]
    shapes: tuple[str, ...]
    fitted: bool
    backtested: bool
    context_policy: str
    answer_kind: str
    version: str
    methods: tuple[str, ...] = ()


CAPABILITIES: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        "temporal_property", ("level", "trend", "seasonality", "volatility",
                              "regime", "extreme", "dependence"),
        ("describe", "predict", "compare", "detect", "decide"),
        ("univariate", "multivariate", "panel"), True, True,
        "shared_context_assessment", "temporal_property", "0.8"),
    CapabilitySpec(
        "observed_disturbance", ("disturbance",), ("describe", "detect"),
        ("univariate", "multivariate", "panel"), False, False,
        "informational_only", "observed_disturbance", "0.1"),
    CapabilitySpec(
        "stationarity", ("stationarity",), ("test", "describe"),
        ("univariate", "multivariate", "panel"), True, False,
        "informational_only", "stationarity", "0.1", ("adf", "kpss")),
    CapabilitySpec(
        "fixed_period_decomposition", ("decomposition",),
        ("decompose", "describe"),
        ("univariate", "multivariate", "panel"), True, False,
        "informational_only", "decomposition", "0.1",
        ("centered_moving_average",)),
    CapabilitySpec(
        "exogenous_regression", ("regression",), ("regress", "predict"),
        ("multivariate", "panel"), True, True,
        "shared_context_assessment", "regression", "0.1",
        ("ridge_linear",)),
)


def capability_for(question: TemporalQuestion) -> CapabilitySpec | None:
    for capability in CAPABILITIES:
        if (question.property in capability.properties
                and question.verb in capability.verbs):
            return capability
    return None


@dataclass(frozen=True)
class ExecutionPlan:
    status: str
    question: dict[str, Any]
    dataset: dict[str, Any]
    capability: dict[str, Any] | None
    reason: str | None = None
    recovery: tuple[dict[str, Any], ...] = ()
    version: str = EXECUTION_PLAN_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_execution(
    question: TemporalQuestion, dataset: DatasetContract,
) -> ExecutionPlan:
    capability = capability_for(question)
    if capability is None:
        return ExecutionPlan(
            "unsupported", question.to_dict(), dataset.to_dict(), None,
            reason=(f"No capability is registered for {question.verb}/"
                    f"{question.property}."),
            recovery=({"action": "describe_supported_capabilities",
                       "arguments": {"brief": True}},),
        )
    if dataset.truncated:
        return ExecutionPlan(
            "unsupported", question.to_dict(), dataset.to_dict(),
            asdict(capability),
            reason="Exact execution requires the complete dataset, not a preview.",
            recovery=({"action": "provide_data_ref_or_path", "arguments": {}},),
        )
    if dataset.shape not in capability.shapes:
        return ExecutionPlan(
            "unsupported", question.to_dict(), dataset.to_dict(),
            asdict(capability),
            reason=(f"{capability.name} does not accept dataset shape "
                    f"{dataset.shape}."),
            recovery=({"action": "resolve_dataset_roles", "arguments": {
                "shape": dataset.shape, "targets": list(dataset.targets)}},),
        )
    if question.method and capability.methods \
            and question.method not in capability.methods:
        return ExecutionPlan(
            "unsupported", question.to_dict(), dataset.to_dict(),
            asdict(capability),
            reason=(f"{capability.name} does not implement requested method "
                    f"{question.method}; supported methods are "
                    f"{', '.join(capability.methods)}."),
            recovery=({"action": "retry_with_supported_method", "arguments": {
                "method": capability.methods[0]}},),
        )
    return ExecutionPlan(
        "ready", question.to_dict(), dataset.to_dict(), asdict(capability))


def unsupported_answer(question: TemporalQuestion, plan: ExecutionPlan
                       ) -> dict[str, Any]:
    """One terminal, useful response for unsupported operations."""
    return {
        "question": question.to_dict(),
        "best_estimate": {"value": None, "support": "abstained",
                          "automation_eligible": False},
        "support": {"state": "abstained", "automation_eligible": False,
                    "meaning": "the requested operation was not executed"},
        "synthesis_policy": {
            "canonical_immutable": True,
            "canonical_authority": "unavailable",
            "llm_may_synthesize": False,
            "synthesis_must_be_separately_labelled": True,
            "primary_forecast_unchanged": True,
        },
        "answer": {"direction": None, "estimate": None, "interval": None,
                   "support": "abstained", "automation_eligible": False},
        "headline": f"{plan.reason} No substitute operation was run.",
        "limitations": [plan.reason] if plan.reason else [],
        "execution_plan": plan.to_dict(),
        "next_actions": list(plan.recovery),
        "artifact_id": None,
    }
