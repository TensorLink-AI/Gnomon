"""The plan executor.

Executes a validated :class:`TemporalPlan` step by step: reference
resolution, step-level checkpointing, content-addressed caching keyed by
the Phase-0 IDs, isolated step failures, budget enforcement, and full
step-level provenance. Executing the same plan twice replays it — the run
artifact is content-addressed and byte-identical, and cached step outputs
are reused rather than recomputed.

Data enters a plan only through the ``load`` step (which materialises a
snapshot) or through a composite macro step (which owns its own snapshot
discipline); every other step sees only literals and upstream outputs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .artifacts import write_json_artifact
from .contracts import GnomonError
from .ids import SYSTEM_CLOCK, Clock, content_id
from .versioning import RUNTIME_VERSION
from .plan import LOAD_OPERATOR, MACRO_OPERATORS, TemporalPlan, validate_plan
from .registry import OPERATORS


def _resolve(value: Any, outputs: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            path = str(value["$ref"]).split(".")
            current: Any = outputs
            current = current[path[0]]
            for key in path[1:]:
                if isinstance(current, list):
                    current = current[int(key)]
                else:
                    current = current[key]
            return current
        return {key: _resolve(item, outputs) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, outputs) for item in value]
    return value


def _run_load(inputs: dict[str, Any], as_of, store_path) -> dict[str, Any]:
    from .pipeline import load_stage
    loaded = load_stage(
        inputs["input"],
        time_column=inputs["time_column"],
        target_column=inputs["target_column"],
        series_column=inputs.get("series_column"),
        frequency=inputs.get("frequency"),
        as_of=as_of, store_path=store_path,
    )
    return {
        "series": {
            name: {
                "timestamps": [item.timestamp.isoformat() for item in items],
                "values": [item.value for item in items],
            }
            for name, items in sorted(loaded.groups.items())
        },
        "frequency": loaded.frequency,
        "source_fingerprint": loaded.source_fingerprint,
        "snapshot_access": loaded.snapshot.access_summary(),
    }


def _run_macro(operator: str, inputs: dict[str, Any], output: str,
               as_of, store_path, clock: Clock) -> dict[str, Any]:
    common = dict(inputs)
    common.setdefault("output", output)
    if operator == "forecast":
        from .runtime import forecast
        from .toolspec import forecast_summary
        arguments = {key: value for key, value in common.items() if key != "output"}
        artifact, path = forecast(
            arguments.pop("input"), output=common["output"],
            as_of=as_of, store_path=store_path, clock=clock, **arguments,
        )
        return {**forecast_summary(artifact, path), "artifact_path": str(path)}
    from . import macros
    runner = getattr(macros, operator)
    arguments = {key: value for key, value in common.items() if key != "output"}
    payload, path = runner(
        arguments.pop("input"), output=common["output"],
        as_of=as_of, store_path=store_path, clock=clock, **arguments,
    )
    return {**payload, "artifact_path": str(path)}


def execute_plan(
    plan: TemporalPlan,
    *,
    output: str = "gnomon-output",
    as_of=None,
    store_path: str | None = None,
    clock: Clock | None = None,
) -> tuple[dict[str, Any], Path]:
    """Validate and execute a plan. Returns the run payload and its
    immutable artifact directory. Re-executing an already-executed plan
    returns the stored run unchanged (deterministic replay)."""
    clock = clock or SYSTEM_CLOCK
    violations = validate_plan(plan)
    if violations:
        raise GnomonError(
            "INVALID_PLAN", f"{len(violations)} validation violation(s).",
            {"violations": violations},
        )
    run_id = content_id("run", {
        "plan": plan.plan_id(),
        "as_of": as_of.isoformat() if as_of else None,
    })
    parent = Path(output).expanduser().resolve()
    run_dir = parent / run_id
    if (run_dir / "artifact.json").is_file():
        from .artifacts import read_artifact
        return {**read_artifact(run_dir), "replayed": True}, run_dir

    cache_dir = parent / ".step-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    budget = plan.task.get("budget") or {}
    deadline = None
    if budget.get("max_wall_seconds") is not None:
        deadline = time.monotonic() + float(budget["max_wall_seconds"])

    outputs: dict[str, Any] = {}
    step_records: list[dict[str, Any]] = []
    failed: set[str] = set()

    for step in plan.steps:
        if deadline is not None and time.monotonic() > deadline:
            step_records.append({
                "step_id": step.step_id, "operator": step.operator,
                "status": "not_run", "reason": "budget_exhausted",
            })
            continue
        upstream = [ref for value in step.inputs.values() for ref in _iter_refs(value)]
        blocked = sorted({ref for ref in upstream if ref in failed})
        if blocked:
            failed.add(step.step_id)
            step_records.append({
                "step_id": step.step_id, "operator": step.operator,
                "status": "skipped", "reason": f"upstream failed: {blocked}",
            })
            continue
        try:
            resolved = {key: _resolve(value, outputs) for key, value in step.inputs.items()}
            # The visible-data boundary is part of a step's identity: the
            # same operator over the same inputs at a different ``as_of``
            # is a different computation. Omitting it let a replay at an
            # earlier instant return numbers computed from later data.
            cache_key = content_id("step", {
                "operator": step.operator,
                "inputs": resolved,
                # A cache entry computed by a different build is the same
                # stale-answer bug as an artifact id collision: the code is
                # part of the answerer's identity.
                "runtime_version": RUNTIME_VERSION,
                "as_of": as_of.isoformat() if as_of else None,
                "store_path": store_path,
            })
            cache_file = cache_dir / f"{cache_key}.json"
            if cache_file.is_file():
                result = json.loads(cache_file.read_text(encoding="utf-8"))
                cached = True
            else:
                result = _execute_step(step.operator, resolved, str(parent),
                                       as_of, store_path, clock)
                cache_file.write_text(
                    json.dumps(result, allow_nan=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                cached = False
            outputs[step.step_id] = result
            step_records.append({
                "step_id": step.step_id, "operator": step.operator,
                "status": "complete", "cache_key": cache_key, "cached": cached,
                "produces": step.produces,
            })
        except GnomonError as exc:
            failed.add(step.step_id)
            step_records.append({
                "step_id": step.step_id, "operator": step.operator,
                "status": "failed", "error": exc.to_dict()["error"],
            })
            if step.failure_policy == "abort":
                break
        except Exception as exc:  # isolated step failure, never a crashed run
            failed.add(step.step_id)
            step_records.append({
                "step_id": step.step_id, "operator": step.operator,
                "status": "failed",
                "error": {"code": "OPERATOR_ERROR", "message": str(exc)},
            })
            if step.failure_policy == "abort":
                break

    completed = {record["step_id"] for record in step_records if record["status"] == "complete"}
    produced = {
        step.produces: outputs[step.step_id]
        for step in plan.steps if step.step_id in completed
    }
    payload = {
        "schema_version": "0.1",
        "run_id": run_id,
        "plan_id": plan.plan_id(),
        "created_at": clock.now().isoformat(),
        "status": "complete" if len(completed) == len(plan.steps) else "partial",
        "task": plan.task,
        "steps": step_records,
        "outputs": produced,
    }
    return payload, write_json_artifact(run_id, payload, str(parent))


def _iter_refs(value: Any) -> list[str]:
    from .plan import _references
    return _references(value)


def _execute_step(operator: str, resolved: dict[str, Any], output: str,
                  as_of, store_path, clock: Clock) -> Any:
    if operator == LOAD_OPERATOR:
        return _run_load(resolved, as_of, store_path)
    if operator in MACRO_OPERATORS:
        return _run_macro(operator, resolved, output, as_of, store_path, clock)
    spec = OPERATORS[operator]
    if spec.runner is None:
        raise GnomonError("OPERATOR_UNAVAILABLE", f"Operator {operator} has no direct runner.")
    return spec.runner(**resolved)
