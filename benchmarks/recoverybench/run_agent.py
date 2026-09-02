"""Measure one-retry agent recovery with and without Gnomon's executable plan.

The control asks the configured model to turn an ordinary recovery response
into one argument patch.  The treatment applies Gnomon's recommended
``recovery_plan`` patch directly.  Both arms then ask the same model to relay
the recovered answer, so the measured difference is the avoidable planning
turn rather than a different human-facing task.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any

from benchmarks.common.envfile import load_env_file
from benchmarks.common.manifest import code_revision
from benchmarks.common.openrouter import OpenRouterClient, extract_json_objects
from benchmarks.recoverybench.run import _call, cases


SCHEMA_VERSION = 1
CASE_ID = "floor-h7"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _choice_object(text: str) -> dict[str, Any]:
    objects = extract_json_objects(text)
    if len(objects) != 1 or not isinstance(objects[0], dict):
        raise ValueError("agent must return exactly one JSON object")
    return objects[0]


def _recommended_execution(response: dict[str, Any]) -> dict[str, Any]:
    plans = response.get("recovery_plan") or []
    plan = next((item for item in plans if isinstance(item, dict)
                 and item.get("recommended") is True), None)
    execution = (plan or {}).get("execution")
    if (not isinstance(execution, dict)
            or execution.get("mode") != "tool"
            or not isinstance(execution.get("argument_patch"), dict)):
        raise ValueError("response has no recommended executable recovery")
    return execution


def _control_patch(
        client: OpenRouterClient, case: dict[str, Any],
        response: dict[str, Any], request_timeout: int) -> dict[str, Any]:
    compact = {
        "tool": case["tool"],
        "arguments": {
            key: value for key, value in case["arguments"].items()
            if key != "output_dir"
        },
        "tier_floor": response.get("tier_floor"),
        "headline": response.get("headline"),
        "recovery_actions": response.get("recovery_actions") or [],
        "reasoning": response.get("reasoning") or {},
    }
    prompt = (
        "Repair this forecast request once. The goal is the longest "
        "supportable orientation forecast without claiming stronger support. "
        "Do not invent observations or change the input. Return only JSON as "
        "{\"tool\":\"gnomon_forecast\",\"argument_patch\":{...}}.\n"
        + json.dumps(compact, sort_keys=True)
    )
    text = client.completions(
        [{"role": "user", "content": prompt}], n=1, temperature=0,
        reasoning_effort="none", request_timeout=request_timeout,
        transport_retries=0)[0]
    proposed = _choice_object(text)
    if proposed.get("tool") != case["tool"]:
        raise ValueError("agent selected a different tool")
    patch = proposed.get("argument_patch")
    if not isinstance(patch, dict) or not patch:
        raise ValueError("agent did not return an argument patch")
    return patch


def _execute_patch(
        case: dict[str, Any], patch: dict[str, Any], output: Path,
) -> dict[str, Any]:
    follow = deepcopy(case)
    follow["arguments"].update(patch)
    follow["arguments"]["output_dir"] = str(output / "recovered-artifact")
    response, unhandled = _call(follow)
    results = response.get("results") or []
    rows = sum(int(item.get("forecast_rows", len(item.get("forecast") or [])))
               for item in results if isinstance(item, dict))
    if unhandled is not None or response.get("status") == "error" or rows <= 0:
        raise ValueError("the single repaired tool call did not publish rows")
    return response


def _relay(
        client: OpenRouterClient, response: dict[str, Any],
        request_timeout: int) -> dict[str, Any]:
    results = response.get("results") or []
    rows = sum(int(item.get("forecast_rows", len(item.get("forecast") or [])))
               for item in results if isinstance(item, dict))
    compact = {
        "headline": response.get("headline"),
        "tier_floor": response.get("tier_floor"),
        "forecast_rows": rows,
        "automation_eligible": bool(response.get("automation_eligible", False)),
    }
    prompt = (
        "Relay this recovered forecast without strengthening its support or "
        "automation authority. Return only JSON with headline, tier_floor, "
        "forecast_rows, and automation_eligible.\n"
        + json.dumps(compact, sort_keys=True)
    )
    text = client.completions(
        [{"role": "user", "content": prompt}], n=1, temperature=0,
        reasoning_effort="none", request_timeout=request_timeout,
        transport_retries=0)[0]
    relayed = _choice_object(text)
    if (relayed.get("tier_floor") != compact["tier_floor"]
            or relayed.get("forecast_rows") != compact["forecast_rows"]
            or bool(relayed.get("automation_eligible"))
            != compact["automation_eligible"]):
        raise ValueError("agent did not preserve the recovered answer contract")
    return relayed


def run_arm(
        mode: str, client: OpenRouterClient, output: Path,
        request_timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    case = next(item for item in cases(output) if item["case_id"] == CASE_ID)
    initial, unhandled = _call(case)
    if unhandled is not None:
        raise ValueError("initial tool response raised an unhandled exception")
    recommended = _recommended_execution(initial)
    patch = (recommended["argument_patch"] if mode == "treatment" else
             _control_patch(client, case, initial, request_timeout))
    recovered = _execute_patch(case, patch, output)
    relayed = _relay(client, recovered, request_timeout)
    usage = client.usage_summary
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": CASE_ID,
        "mode": mode,
        "patch": patch,
        "recommended_patch": recommended["argument_patch"],
        "patch_matches_recommendation": patch == recommended["argument_patch"],
        "single_retry_succeeded": True,
        "relayed": relayed,
        "support_preserved": relayed.get("tier_floor") == "best_effort",
        "automation_withheld": relayed.get("automation_eligible") is False,
        "llm_usage": usage,
        "latency_seconds": time.monotonic() - started,
    }


def summarize(control: dict[str, Any], treatment: dict[str, Any],
              revision: str) -> dict[str, Any]:
    def metric(row: dict[str, Any], key: str) -> float:
        usage = row["llm_usage"]
        if key == "tokens":
            return float(usage["prompt_tokens"] + usage["completion_tokens"])
        if key == "requests":
            return float(usage["requests"])
        return float(row["latency_seconds"])

    accounting_complete = all(
        row["llm_usage"].get("sample_cache_accounting_complete") is True
        for row in (control, treatment))
    gates = {
        "matched_successful_single_retry": all(
            row.get("single_retry_succeeded") is True
            for row in (control, treatment)),
        "matched_recommended_patch": all(
            row.get("patch_matches_recommendation") is True
            for row in (control, treatment)),
        "support_preserved": all(row.get("support_preserved") is True
                                 for row in (control, treatment)),
        "automation_withheld": all(row.get("automation_withheld") is True
                                   for row in (control, treatment)),
        "usage_accounting_complete": accounting_complete,
        "treatment_saves_a_planning_request": (
            metric(treatment, "requests") < metric(control, "requests")),
    }
    raw = {
        f"{arm}_{name}": metric(row, name)
        for arm, row in (("control", control), ("treatment", treatment))
        for name in ("requests", "tokens", "latency_seconds")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "recovery-agent-efficiency",
        "evaluated_commit": revision,
        "case_id": "efficiency:repair:one-retry",
        "control": control,
        "treatment": treatment,
        "gfr_raw": raw,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                        choices=("OPENROUTER_API_KEY", "ENGY_API_KEY",
                                 "CHUTES_API_KEY"))
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    load_env_file()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    for mode in ("control", "treatment"):
        path = args.output_dir / f"{mode}.json"
        if path.is_file():
            rows[mode] = json.loads(path.read_text(encoding="utf-8"))
            continue
        client = OpenRouterClient(
            args.model, api_key=os.environ.get(args.api_key_env),
            base_url=args.base_url, temperature=0, max_tokens=1200,
            max_retries=args.max_retries, timeout=args.request_timeout,
            reasoning_effort="none", sample_parallelism=1,
            sample_cache_dir=args.output_dir / "sample-cache" / mode,
        )
        row = run_arm(mode, client, args.output_dir / mode,
                      args.request_timeout)
        _atomic_json(path, row)
        rows[mode] = row
    summary = summarize(rows["control"], rows["treatment"], code_revision())
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
