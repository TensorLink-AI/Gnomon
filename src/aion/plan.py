"""TemporalPlan — the typed intermediate representation for temporal work.

A plan is a DAG of operator calls with declared inputs (literals or
``{"$ref": "step_id.path"}`` references to earlier steps), the artifact
type each step produces, and a failure policy. The IR was extracted from
the four proven macros rather than invented: the macros compile to plans,
and their outputs verify the executor.

The validator is deterministic and returns structured violations that a
host model can repair — at most :data:`MAX_REPAIR_ROUNDS` rounds, by
design. Aion never proposes plans; it validates and executes them.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .contracts import AionError
from .ids import content_id
from .registry import OPERATORS

# Composite operators: whole macros runnable as single plan steps. This is
# the first implementation of "macros as compiled plans" — each macro is a
# verified composite; full decomposition reuses the same IR later.
MACRO_OPERATORS = {"forecast", "investigate_change", "decide", "monitor"}

# The one operator allowed to touch task sources. Everything else receives
# only literals and references — leakage control at the IR level.
LOAD_OPERATOR = "load"

MAX_REPAIR_ROUNDS = 2

_PATH_LIKE = re.compile(r"(^store:|\.csv$|\.parquet$|\.pq$|^/|^~|^\.{1,2}/)")


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    operator: str
    inputs: dict[str, Any] = field(default_factory=dict)
    produces: str = "value"
    failure_policy: str = "abort"  # abort | skip

    def call_fingerprint(self) -> str:
        return content_id("call", {"operator": self.operator, "inputs": self.inputs})


@dataclass(frozen=True)
class TemporalPlan:
    task: dict[str, Any]
    steps: tuple[PlanStep, ...]

    def plan_id(self) -> str:
        return content_id("plan", {
            "task": self.task,
            "steps": [
                {"step_id": step.step_id, "operator": step.operator,
                 "inputs": step.inputs, "produces": step.produces,
                 "failure_policy": step.failure_policy}
                for step in self.steps
            ],
        })


def plan_from_dict(payload: dict[str, Any]) -> TemporalPlan:
    try:
        steps = tuple(
            PlanStep(
                step_id=str(step["step_id"]),
                operator=str(step["operator"]),
                inputs=dict(step.get("inputs", {})),
                produces=str(step.get("produces", "value")),
                failure_policy=str(step.get("failure_policy", "abort")),
            )
            for step in payload.get("steps", [])
        )
    except (KeyError, TypeError) as exc:
        raise AionError("INVALID_PLAN", f"Malformed plan step: {exc}") from exc
    return TemporalPlan(task=dict(payload.get("task", {})), steps=steps)


def _references(value: Any) -> list[str]:
    """Every step_id referenced anywhere inside an input value."""
    found: list[str] = []
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            found.append(str(value["$ref"]).split(".", 1)[0])
        else:
            for item in value.values():
                found.extend(_references(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_references(item))
    return found


def _literal_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            return []
        return [item for entry in value.values() for item in _literal_strings(entry)]
    if isinstance(value, list):
        return [item for entry in value for item in _literal_strings(entry)]
    return []


def _operator_parameters(name: str) -> tuple[set[str], set[str]] | None:
    """(accepted, required) parameter names, from the runner signature."""
    spec = OPERATORS.get(name)
    if spec is None or spec.runner is None:
        return None
    signature = inspect.signature(spec.runner)
    accepted, required = set(), set()
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        accepted.add(parameter.name)
        if parameter.default is parameter.empty:
            required.add(parameter.name)
    # The executor injects the task snapshot; plans never pass it.
    accepted.discard("snapshot")
    required.discard("snapshot")
    return accepted, required


def validate_plan(plan: TemporalPlan) -> list[dict[str, Any]]:
    """Deterministic validation: structured violations, empty when executable."""
    violations: list[dict[str, Any]] = []
    known_operators = set(OPERATORS) | MACRO_OPERATORS | {LOAD_OPERATOR}
    seen_ids: set[str] = set()
    seen_calls: dict[str, str] = {}
    requested = [str(item) for item in plan.task.get("outputs", [])]

    for position, step in enumerate(plan.steps):
        if step.step_id in seen_ids:
            violations.append({
                "code": "DUPLICATE_STEP_ID", "step_id": step.step_id,
                "message": f"Step id {step.step_id!r} is used more than once.",
            })
        if step.operator not in known_operators:
            violations.append({
                "code": "UNKNOWN_OPERATOR", "step_id": step.step_id,
                "message": f"No operator named {step.operator!r}.",
                "available": sorted(known_operators),
            })
            seen_ids.add(step.step_id)
            continue
        if step.failure_policy not in ("abort", "skip"):
            violations.append({
                "code": "INVALID_FAILURE_POLICY", "step_id": step.step_id,
                "message": f"failure_policy must be abort or skip, not {step.failure_policy!r}.",
            })
        # Acyclicity by construction: references may only point backwards.
        for reference in _references(step.inputs):
            if reference == step.step_id:
                violations.append({
                    "code": "SELF_REFERENCE", "step_id": step.step_id,
                    "message": "A step cannot reference its own output.",
                })
            elif reference not in seen_ids:
                violations.append({
                    "code": "FORWARD_OR_MISSING_REFERENCE", "step_id": step.step_id,
                    "message": (
                        f"Step references {reference!r}, which is not an "
                        "earlier step; plans are DAGs executed in order."
                    ),
                })
        # Leakage: only the load operator touches sources.
        if step.operator != LOAD_OPERATOR and step.operator not in MACRO_OPERATORS:
            leaky = [text for text in _literal_strings(step.inputs) if _PATH_LIKE.search(text)]
            if leaky:
                violations.append({
                    "code": "TEMPORAL_LEAKAGE", "step_id": step.step_id,
                    "message": (
                        f"Step passes path-like literals {leaky}; data enters a "
                        "plan only through the load step's snapshot."
                    ),
                })
        # Missing/unknown inputs against the runner signature.
        parameters = _operator_parameters(step.operator)
        if parameters is not None:
            accepted, required = parameters
            unknown = sorted(set(step.inputs) - accepted)
            missing = sorted(required - set(step.inputs))
            if unknown:
                violations.append({
                    "code": "UNKNOWN_INPUT", "step_id": step.step_id,
                    "message": f"Operator {step.operator} does not accept {unknown}.",
                    "accepted": sorted(accepted),
                })
            if missing:
                violations.append({
                    "code": "MISSING_INPUT", "step_id": step.step_id,
                    "message": f"Operator {step.operator} requires {missing}.",
                })
        # Claim-class feasibility.
        spec = OPERATORS.get(step.operator)
        if spec is not None and plan.task.get("claim_class"):
            wanted = str(plan.task["claim_class"])
            if wanted not in spec.claim_classes and wanted != "descriptive":
                violations.append({
                    "code": "CLAIM_CLASS_INFEASIBLE", "step_id": step.step_id,
                    "message": (
                        f"Operator {step.operator} supports claim classes "
                        f"{list(spec.claim_classes)}, not {wanted!r}."
                    ),
                })
        # Duplicate-equivalent calls.
        fingerprint = step.call_fingerprint()
        if fingerprint in seen_calls:
            violations.append({
                "code": "DUPLICATE_EQUIVALENT_CALL", "step_id": step.step_id,
                "message": (
                    f"Step duplicates {seen_calls[fingerprint]!r} exactly; "
                    "reference its output instead of recomputing."
                ),
                "duplicate_of": seen_calls[fingerprint],
            })
        else:
            seen_calls[fingerprint] = step.step_id
        seen_ids.add(step.step_id)

    budget = plan.task.get("budget") or {}
    max_steps = budget.get("max_steps")
    if max_steps is not None and len(plan.steps) > int(max_steps):
        violations.append({
            "code": "BUDGET_EXCEEDED",
            "message": f"Plan has {len(plan.steps)} steps; budget allows {max_steps}.",
        })
    if not plan.steps:
        violations.append({"code": "EMPTY_PLAN", "message": "A plan needs at least one step."})
    # Does the plan actually produce the requested outputs?
    produced = {step.produces for step in plan.steps}
    for wanted in requested:
        if wanted not in produced:
            violations.append({
                "code": "REQUESTED_OUTPUT_NOT_PRODUCED",
                "message": f"No step produces the requested output {wanted!r}.",
                "produced": sorted(produced),
            })
    if plan.task.get("claim_class") in ("causal", "counterfactual"):
        violations.append({
            "code": "CLAIM_CLASS_INFEASIBLE",
            "message": (
                "No available operator can support causal or counterfactual "
                "claims; the causal family is out of scope until it ships "
                "with honest abstention criteria."
            ),
        })
    return violations


def compile_task(task_type: str, params: dict[str, Any]) -> TemporalPlan:
    """Compile one of the four canonical task types into a validated plan.

    Each macro becomes a load-free composite step (the macro owns its own
    snapshot discipline); the plan layer adds identity, validation, and
    replayable execution around it."""
    if task_type not in MACRO_OPERATORS:
        raise AionError(
            "UNKNOWN_TASK_TYPE",
            f"Cannot compile task type {task_type!r}.",
            {"available": sorted(MACRO_OPERATORS)},
        )
    produces = {
        "forecast": "forecast", "investigate_change": "change_investigation",
        "decide": "decision", "monitor": "monitor",
    }[task_type]
    step = PlanStep(
        step_id=f"{task_type}_1", operator=task_type,
        inputs=dict(params), produces=produces,
    )
    plan = TemporalPlan(
        task={"task_type": task_type, "outputs": [produces], "params": dict(params)},
        steps=(step,),
    )
    return plan


def repair_plan(
    plan_payload: dict[str, Any],
    propose_fix: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]],
) -> tuple[TemporalPlan, list[dict[str, Any]], int]:
    """Bounded repair: validate, hand violations to the host-supplied
    ``propose_fix`` (this is where the host's LLM lives — Aion contains no
    model), at most MAX_REPAIR_ROUNDS times. Returns the final plan, its
    remaining violations, and the rounds used. Deliberately thin: this
    layer exists for today's models and should shrink."""
    plan = plan_from_dict(plan_payload)
    violations = validate_plan(plan)
    rounds = 0
    while violations and rounds < MAX_REPAIR_ROUNDS:
        rounds += 1
        plan_payload = propose_fix(plan_payload, violations)
        plan = plan_from_dict(plan_payload)
        violations = validate_plan(plan)
    return plan, violations, rounds
