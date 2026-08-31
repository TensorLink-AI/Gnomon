"""Run the frozen P9 recovery-boundary probe serially and resumably."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
from typing import Any


RUNNER_VERSION = "p9-recovery-boundary-1"
CASE_TIMEOUT_SECONDS = 30
RETRIES = 1
ROOT = Path(__file__).resolve().parents[2]
SHORT = ROOT / "tests" / "data" / "golden_short.csv"
DAILY = ROOT / "examples" / "daily_requests.csv"


class CaseTimeout(RuntimeError):
    pass


def _timeout(_signum: int, _frame: Any) -> None:
    raise CaseTimeout("case exceeded its timeout")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def cases(run_dir: Path) -> list[dict[str, Any]]:
    common = {"input": str(SHORT), "time_column": "timestamp",
              "target_column": "value"}
    return [
        {"case_id": "floor-h1", "class": "automatic",
         "tool": "gnomon_forecast",
         "arguments": {**common, "horizon": 1,
                       "minimum_support": "supported",
                       "output_dir": str(run_dir / "artifacts" / "floor-h1")}},
        {"case_id": "floor-h7", "class": "automatic",
         "tool": "gnomon_forecast",
         "arguments": {**common, "horizon": 7,
                       "minimum_support": "supported",
                       "output_dir": str(run_dir / "artifacts" / "floor-h7")}},
        {"case_id": "default-h7", "class": "automatic",
         "tool": "gnomon_forecast",
         "arguments": {**common, "horizon": 7,
                       "output_dir": str(run_dir / "artifacts" / "default-h7")}},
        {"case_id": "missing-file", "class": "external_choice",
         "tool": "gnomon_forecast",
         "arguments": {"input": str(run_dir / "absent.csv"), "horizon": 3,
                       "output_dir": str(run_dir / "artifacts" / "missing")}},
        {"case_id": "ambiguous-frequency", "class": "external_choice",
         "tool": "gnomon_forecast", "prepare": "irregular",
         "arguments": {"input": str(run_dir / "irregular.csv"),
                       "time_column": "timestamp", "target_column": "value",
                       "horizon": 2,
                       "output_dir": str(run_dir / "artifacts" / "frequency")}},
        {"case_id": "malformed-actions", "class": "external_choice",
         "tool": "gnomon_decide",
         "arguments": {"input": str(DAILY), "time_column": "timestamp",
                       "target_column": "requests", "horizon": 3,
                       "threshold": 100.0, "actions": ["wait", "act"],
                       "output_dir": str(run_dir / "artifacts" / "decide")}},
    ]


def _prepare(case: dict[str, Any]) -> None:
    if case.get("prepare") == "irregular":
        Path(case["arguments"]["input"]).write_text(
            "timestamp,value\n2026-01-01,1\n2026-01-03,2\n"
            "2026-01-06,3\n2026-01-11,4\n",
            encoding="utf-8",
        )


def _call(case: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    from gnomon.contracts import GnomonError
    from gnomon.toolspec import apply_response_contract, runner_for

    runner = runner_for(case["tool"])
    if runner is None:
        raise RuntimeError(f"tool not visible: {case['tool']}")
    try:
        return runner(dict(case["arguments"])), None
    except GnomonError as error:
        return apply_response_contract(error.to_dict()), None
    except Exception as error:  # baseline makes boundary crashes measurable
        payload = {
            "schema_version": "recoverybench-observation-1",
            "status": "error",
            "error": {
                "code": "UNHANDLED_TOOL_EXCEPTION",
                "message": str(error),
                "retryable": False,
                "details": {"exception_type": type(error).__name__},
                "repair_options": [],
            },
        }
        return apply_response_contract(payload), type(error).__name__


def _source_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    error = payload.get("error")
    if isinstance(error, dict):
        actions = error.get("repair_options") or []
    else:
        actions = payload.get("recovery_actions") or []
    return [item for item in actions if isinstance(item, dict)]


def _strip_additions(payload: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(payload)
    value.pop("recovery_plan", None)
    for field in ("reasoning", "rejection"):
        envelope = value.get(field)
        if isinstance(envelope, dict):
            envelope.pop("recovery_plan_ref", None)
    return value


def _normalise_identity(
        payload: dict[str, Any], run_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    value = deepcopy(_strip_additions(payload))
    for key in ("artifact_id", "artifact_path", "forecast_id", "data_ref",
                "wall_clock_now", "staleness"):
        value.pop(key, None)

    def scrub(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: scrub(item) for key, item in node.items()}
        if isinstance(node, list):
            return [scrub(item) for item in node]
        if isinstance(node, str):
            # Baseline and treatment use separate artifact roots. The path is
            # an execution identity, not a semantic response difference.
            for root in run_roots:
                prefix = str(root.resolve())
                if prefix in node:
                    node = node.replace(prefix, "<RUN_DIR>")
            return node
        return node

    return scrub(value)


def _plan_checks(payload: dict[str, Any], expected_actions: int,
                 case_class: str) -> dict[str, Any]:
    plan = payload.get("recovery_plan") or []
    valid = isinstance(plan, list)
    ranks = [item.get("rank") for item in plan if isinstance(item, dict)]
    recommended = [item for item in plan if isinstance(item, dict)
                   and item.get("recommended") is True]
    executable = [item for item in plan if isinstance(item, dict)
                  and isinstance(item.get("execution"), dict)
                  and item["execution"].get("mode") == "tool"
                  and isinstance(item["execution"].get("argument_patch"), dict)]
    no_fake = all(
        item.get("execution", {}).get("requires_user_input") is True
        and "argument_patch" not in item.get("execution", {})
        for item in plan if isinstance(item, dict)
    ) if case_class == "external_choice" else True
    sources = all(
        isinstance(item.get("source"), str)
        and item["source"].startswith(("/recovery_actions/",
                                       "/error/repair_options/"))
        for item in plan if isinstance(item, dict)
    )
    causal_authority = all(
        bool(item.get("code")) and bool(item.get("expected_effect"))
        and "upgrade" in str(item.get("authority_limit") or "")
        for item in plan if isinstance(item, dict)
    )
    envelope = payload.get("rejection") if payload.get("error") else payload.get(
        "reasoning")
    resolution_ref = (not plan or isinstance(envelope, dict)
                      and envelope.get("recovery_plan_ref") == "/recovery_plan/0")
    return {
        "valid": valid,
        "action_coverage": len(plan) == expected_actions,
        "ranks_consecutive": ranks == list(range(1, len(plan) + 1)),
        "one_recommended": len(recommended) == (1 if plan else 0),
        "first_recommended": not plan or bool(plan[0].get("recommended")),
        "executable_count": len(executable),
        "no_fake_automation": no_fake,
        "sources_valid": sources,
        "causal_authority": causal_authority,
        "resolution_ref": resolution_ref,
    }


def _execute_first(case: dict[str, Any], payload: dict[str, Any],
                   run_dir: Path) -> dict[str, Any] | None:
    plan = payload.get("recovery_plan") or []
    executable = next((item for item in plan
                       if item.get("recommended") is True
                       and item.get("execution", {}).get("mode") == "tool"), None)
    if executable is None:
        return None
    execution = executable["execution"]
    follow = deepcopy(case)
    follow["tool"] = execution["tool"]
    follow["arguments"].update(execution["argument_patch"])
    follow["arguments"]["output_dir"] = str(
        run_dir / "recoveries" / case["case_id"])
    response, unhandled = _call(follow)
    results = response.get("results") or []
    rows = sum(int(item.get("forecast_rows", len(item.get("forecast") or [])))
               for item in results if isinstance(item, dict))
    tiers = sorted({str(row.get("tier")) for item in results
                    for row in (item.get("forecast") or [])
                    if isinstance(row, dict) and row.get("tier")})
    return {"response": response, "unhandled": unhandled,
            "forecast_rows": rows, "tiers": tiers,
            "success": response.get("status") != "error" and rows > 0}


def evaluate(case: dict[str, Any], mode: str, run_dir: Path,
             baseline_dir: Path | None) -> dict[str, Any]:
    _prepare(case)
    response, unhandled = _call(case)
    source_actions = _source_actions(response)
    checks = _plan_checks(response, len(source_actions), case["class"])
    baseline = None
    canonical_equal = True
    baseline_exact_patch = False
    if baseline_dir is not None:
        baseline = json.loads((baseline_dir / "cases" /
                               f"{case['case_id']}.json").read_text())
        baseline_response = baseline["response"]
        canonical_equal = _sha(
            _normalise_identity(response, (run_dir,))) == _sha(
                _normalise_identity(baseline_response, (baseline_dir,)))
        baseline_exact_patch = bool(
            (baseline_response.get("recovery_plan") or []))
    contract_repair_allowed = bool(
        case["case_id"] == "malformed-actions" and baseline is not None
        and baseline.get("unhandled_exception") == "AttributeError"
        and (response.get("error") or {}).get("code") == "INVALID_ACTIONS"
        and unhandled is None)
    if case["case_id"] == "default-h7" and baseline is not None:
        before_reasoning = baseline["response"].get("reasoning") or {}
        after_reasoning = response.get("reasoning") or {}
        contract_repair_allowed = bool(
            before_reasoning.get("sufficiency", {}).get("requires_follow_up")
            is False
            and before_reasoning.get("resolution", {}).get("kind") == "recovery"
            and after_reasoning.get("sufficiency", {}).get("requires_follow_up")
            is False
            and after_reasoning.get("resolution", {}).get("kind") == "complete")
    if case["case_id"] in {"floor-h1", "floor-h7"} and baseline is not None:
        before_reasoning = baseline["response"].get("reasoning") or {}
        after_reasoning = response.get("reasoning") or {}
        contract_repair_allowed = bool(
            baseline["response"].get("tier_floor") == "inconclusive"
            and before_reasoning.get("sufficiency", {}).get("requires_follow_up")
            is False
            and after_reasoning.get("sufficiency", {}).get("requires_follow_up")
            is True
            and after_reasoning.get("resolution", {}).get("kind") == "recovery")
    recovered = (_execute_first(case, response, run_dir)
                 if mode == "candidate" and case["class"] == "automatic"
                 else None)
    return {
        "schema_version": 1, "runner_version": RUNNER_VERSION,
        "case_id": case["case_id"], "class": case["class"],
        "tool": case["tool"], "arguments": case["arguments"],
        "complete": isinstance(response, dict), "unhandled_exception": unhandled,
        "source_action_count": len(source_actions), "response": response,
        "response_sha256": _sha(response), "plan_checks": checks,
        "canonical_equal_to_baseline": canonical_equal,
        "contract_repair_allowed": contract_repair_allowed,
        "baseline_exact_patch": baseline_exact_patch,
        "recovery_execution": recovered,
    }


def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    automatic = [row for row in rows if row["class"] == "automatic"]
    external = [row for row in rows if row["class"] == "external_choice"]
    successful = sum(bool((row.get("recovery_execution") or {}).get("success"))
                     for row in automatic)
    plan_gates = all(
        all(checks[key] for key in ("valid", "action_coverage",
                                    "ranks_consecutive", "one_recommended",
                                    "first_recommended", "sources_valid",
                                    "causal_authority", "resolution_ref"))
        for checks in (row["plan_checks"] for row in rows))
    gates = {
        "completion": len(rows) == 6 and all(row["complete"] for row in rows),
        "canonical_immutability": (mode == "baseline" or all(
            row["canonical_equal_to_baseline"]
            or row["contract_repair_allowed"] for row in rows)),
        "coverage_ranking_traceability": plan_gates,
        "one_call_recovery": (mode == "baseline" or successful == 3),
        "no_fake_automation": (mode == "baseline" or all(
            row["plan_checks"]["no_fake_automation"] for row in external)),
        "no_unhandled_exceptions": all(
            row["unhandled_exception"] is None for row in rows),
        "improvement": (mode == "baseline" or successful >= 2),
        "serial_bounded_resumable": True,
    }
    return {
        "schema_version": 1, "runner_version": RUNNER_VERSION, "mode": mode,
        "cases": len(rows), "automatic_cases": len(automatic),
        "external_choice_cases": len(external),
        "successful_one_call_recoveries": successful,
        "unhandled_exceptions": sum(row["unhandled_exception"] is not None
                                    for row in rows),
        "gates": gates, "all_gates_pass": all(gates.values()),
        "topology": {"jobs": 1, "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
                     "retries": RETRIES, "atomic_case_files": True},
    }


def run(mode: str, output: Path, baseline: Path | None) -> dict[str, Any]:
    os.environ["GNOMON_MCP_PROFILE"] = "full"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    old_handler = signal.signal(signal.SIGALRM, _timeout)
    try:
        for case in cases(output):
            target = output / "cases" / f"{case['case_id']}.json"
            if target.is_file():
                rows.append(json.loads(target.read_text(encoding="utf-8")))
                continue
            last_error: Exception | None = None
            for _attempt in range(RETRIES + 1):
                try:
                    signal.alarm(CASE_TIMEOUT_SECONDS)
                    row = evaluate(case, mode, output, baseline)
                    signal.alarm(0)
                    _atomic_json(target, row)
                    rows.append(row)
                    last_error = None
                    break
                except Exception as error:
                    signal.alarm(0)
                    last_error = error
            if last_error is not None:
                raise last_error
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    summary = summarize(rows, mode)
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.mode == "candidate" and args.baseline is None:
        parser.error("candidate mode requires --baseline")
    summary = run(args.mode, args.output, args.baseline)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if args.mode == "baseline" or summary["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
