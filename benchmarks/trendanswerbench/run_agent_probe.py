"""Crash-safe Evidence-profile agent probe for frozen Q2 trend cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from benchmarks.common.envfile import load_env_file
from benchmarks.common.manifest import code_revision
from benchmarks.common.openrouter import OpenRouterClient, OpenRouterError
from benchmarks.temporalbench.mcp_agent import run_row
from gnomon.soft_context import content_fingerprint
from gnomon.temporal_intent import INTENT_COMPILER_VERSION
from gnomon.temporal_vocabulary import project_temporal_choice

from .run import _cases, _generate


CASE_IDS = (
    "additive_up-8121",
    "multiplicative_zero-8122",
    "noisy_down-8123",
    "plateau_after_up-8124",
    "unadmitted_season_zero-8125",
    "insufficient_cycle_up-8126",
)
OPTIONS = ("Upward", "Constant", "Downward", "Uncertain")


def _public_id(case_id: str) -> str:
    return "ta-" + hashlib.sha256(case_id.encode()).hexdigest()[:16]


def _selected_cases() -> list[dict[str, Any]]:
    indexed = {case["case_id"]: case for case in _cases(tuple(range(8121, 8129)))}
    return [indexed[case_id] for case_id in CASE_IDS]


def _row(case: dict[str, Any]) -> dict[str, Any]:
    history, _ = _generate(case)
    public_id = _public_id(str(case["case_id"]))
    horizon = int(case["horizon"])
    return {
        "id": public_id,
        "tier": "T2",
        "prompt": (
            f"Forecast the next {horizon} values of `value`. Also answer the "
            "typed question: over that future horizon, what is the underlying "
            "trend after any admitted seasonal adjustment? Choose exactly one "
            "of Upward, Constant, Downward, or Uncertain. Preserve Gnomon's "
            "support and automation limits; the temporal answer does not "
            "change the primary forecast."
        ),
        "input": {"history": {"value": history}},
        # Only the key is consumed by the agent harness. No realized future is
        # supplied to the engine, model, prompt, or scorer.
        "ground_truth": {"value": [0.0] * horizon},
        "meta": {"n_horizon": horizon, "main_key": "value"},
        "mcq": {"trend": {
            "question": (
                "What is the underlying trend of value over the forecast "
                "horizon after any admitted seasonal adjustment?"),
            "options": list(OPTIONS),
        }},
        "_frequency": ("h" if int(case["declared_period"]) == 24
                       else "synthetic"),
        "_time_origin": "2025-01-01T00:00:00+00:00",
        "_time_step_seconds": 3600,
        "_host_compiled_forecast": True,
    }


def _write_temporal_receipt(row: dict[str, Any], receipt_dir: Path) -> Path:
    question = str(row["mcq"]["trend"]["question"])
    horizon = int(row["meta"]["n_horizon"])
    fingerprint = content_fingerprint(json.dumps({
        "text": question,
        "targets": ["value"],
        "default_horizon": horizon,
        "compiler_version": INTENT_COMPILER_VERSION,
    }, sort_keys=True))
    payload = {
        "input_fingerprint": fingerprint,
        "proposed": [{
            "id": "trend", "verb": "predict", "target": "value",
            "property": "trend", "measure": "slope", "horizon": horizon,
        }],
        "questions": [{
            "id": "trend", "verb": "predict", "target": "value",
            "property": "trend", "measure": "slope", "horizon": horizon,
        }],
        "rejected": [],
        "compiler_called": False,
        "host_preregistered": True,
    }
    path = receipt_dir / f"{row['id']}.json"
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError("immutable temporal receipt already differs")
    path.write_text(rendered, encoding="utf-8")
    return path


def _temporal_answer(outcome: dict[str, Any]) -> dict[str, Any] | None:
    paths = ((outcome.get("mcp") or {}).get("artifact_paths") or [])
    for artifact_path in reversed(paths):
        receipt = Path(artifact_path) / "temporal_answers.json"
        if not receipt.is_file():
            continue
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        for answer in payload.get("answers") or []:
            if str((answer.get("question") or {}).get("id")) == "trend":
                return answer
    return None


def _score(case: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    engine = _temporal_answer(outcome)
    best = (engine or {}).get("best_estimate") or {}
    answer = (engine or {}).get("answer") or {}
    choice = project_temporal_choice(best.get("value"), list(OPTIONS))
    expected = (choice or {}).get("display_value") or "Uncertain"
    final = ((outcome.get("answer") or {}).get("mcq") or {}).get("trend")
    contract = (outcome.get("temporal_choice_contracts") or {}).get("trend")
    support = best.get("support")
    automation = best.get("automation_eligible")
    primary = ((answer.get("reasoning") or {}).get(
        "primary_forecast_unchanged"))
    authority = (outcome.get("choice_authority") or {}).get("trend")
    return {
        "case_id": case["case_id"],
        "public_id": _public_id(str(case["case_id"])),
        "status": "answered" if engine is not None and final is not None
                  else "incomplete",
        "engine_complete": engine is not None,
        "engine_direction": best.get("value"),
        "engine_display_value": best.get("display_value"),
        "engine_support": support,
        "engine_automation_eligible": automation,
        "engine_primary_forecast_unchanged": primary,
        "engine_estimate": answer.get("estimate"),
        "engine_interval": answer.get("interval"),
        "expected_final_choice": expected,
        "final_choice": final,
        "agent_choice_preserved": final == expected,
        "choice_authority": authority,
        "authority_not_inflated": (
            authority != "binding" if support != "supported" else True),
        "forecast_route": (outcome.get("channel_route") or {}).get("value"),
        "artifact_paths": (outcome.get("mcp") or {}).get("artifact_paths") or [],
        "host_contract": contract,
        "host_contract_complete": bool(
            isinstance(contract, dict)
            and contract.get("canonical_value") == best.get("value")
            and contract.get("display_value") == best.get("display_value")
            and contract.get("support") == support
            and contract.get("automation_eligible") == automation
            and contract.get("primary_forecast_unchanged") == primary),
        "submit_reasoning": outcome.get("submit_reasoning"),
        "outcome": outcome,
    }


def _atomic_rows(path: Path, cases: list[dict[str, Any]],
                 rows: dict[str, dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            if case["case_id"] in rows:
                handle.write(json.dumps(
                    rows[case["case_id"]], sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "trendanswerbench-agent-preservation",
        "code_revision": code_revision(),
        "case_ids": list(CASE_IDS),
        "profile": "evidence",
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0.0,
        "reasoning_effort": "none",
        "jobs": 1,
        "request_timeout_seconds": args.request_timeout,
        "tool_timeout_seconds": args.tool_timeout,
        "max_retries": args.max_retries,
        "infrastructure_retries": args.infrastructure_retries,
    }


def _summary(rows: list[dict[str, Any]], usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "trendanswerbench-agent-preservation",
        "evaluated_commit": code_revision(),
        "cases": len(rows),
        "answered": sum(row["status"] == "answered" for row in rows),
        "engine_complete": sum(row["engine_complete"] for row in rows),
        "agent_choice_preserved": sum(
            row["agent_choice_preserved"] for row in rows),
        "host_contract_complete": sum(
            row["host_contract_complete"] for row in rows),
        "authority_not_inflated": sum(
            row["authority_not_inflated"] for row in rows),
        "primary_forecast_unchanged": sum(
            row["engine_primary_forecast_unchanged"] is True for row in rows),
        "gnomon_artifact_routes": sum(
            row["forecast_route"] == "gnomon" for row in rows),
        "llm_usage": usage,
        "gates": {
            "all_six_answered": len(rows) == 6
                and all(row["status"] == "answered" for row in rows),
            "all_engine_receipts_complete": all(
                row["engine_complete"] for row in rows),
            "all_agent_choices_preserved": all(
                row["agent_choice_preserved"] for row in rows),
            "all_host_contracts_complete": all(
                row["host_contract_complete"] for row in rows),
            "no_authority_inflation": all(
                row["authority_not_inflated"] for row in rows),
            "all_primary_forecasts_unchanged": all(
                row["engine_primary_forecast_unchanged"] is True
                for row in rows),
            "all_forecasts_from_gnomon_artifacts": all(
                row["forecast_route"] == "gnomon" for row in rows),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--base-url", default="https://api.engy.ai/v1")
    parser.add_argument("--api-key-env", default="ENGY_API_KEY")
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--tool-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--infrastructure-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_env_file()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir = args.output_dir / "question-receipts"
    work_dir = args.output_dir / "work"
    receipt_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)
    identity = _identity(args)
    identity_path = args.output_dir / "run_identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise SystemExit("resume identity mismatch; use a new output directory")
        if not args.resume:
            raise SystemExit("output exists; pass --resume or use a new directory")
    else:
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    cases = _selected_cases()
    checkpoint = args.output_dir / "observations.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[item["case_id"]] = item
    client = OpenRouterClient(
        args.model,
        api_key=os.environ.get(args.api_key_env),
        base_url=args.base_url,
        temperature=0.0,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        max_tokens=8000,
        reasoning_effort="none",
    )
    attempts_path = args.output_dir / "attempts.jsonl"
    for case in cases:
        if case["case_id"] in completed:
            continue
        task = _row(case)
        _write_temporal_receipt(task, receipt_dir)
        last_error: Exception | None = None
        for attempt in range(1, args.infrastructure_retries + 2):
            started = time.perf_counter()
            try:
                outcome = run_row(
                    task, client,
                    work_dir=str(work_dir),
                    profile="evidence",
                    compile_questions=True,
                    question_receipts_dir=str(receipt_dir),
                    mcp_call_timeout=args.tool_timeout,
                )
                row = _score(case, outcome)
                attempt_row = {
                    "case_id": case["case_id"], "attempt": attempt,
                    "status": row["status"],
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
                with attempts_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(attempt_row, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                completed[case["case_id"]] = row
                _atomic_rows(checkpoint, cases, completed)
                break
            except OpenRouterError as error:
                last_error = error
                with attempts_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "case_id": case["case_id"], "attempt": attempt,
                        "status": "provider_failure",
                        "error": f"{type(error).__name__}: {error}",
                        "latency_seconds": round(
                            time.perf_counter() - started, 6),
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    }, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        else:
            completed[case["case_id"]] = {
                "case_id": case["case_id"],
                "public_id": _public_id(str(case["case_id"])),
                "status": "provider_failure",
                "error": f"{type(last_error).__name__}: {last_error}",
                "engine_complete": False,
                "agent_choice_preserved": False,
                "host_contract_complete": False,
                "authority_not_inflated": False,
                "engine_primary_forecast_unchanged": False,
                "forecast_route": None,
            }
            _atomic_rows(checkpoint, cases, completed)
    ordered = [completed[case["case_id"]] for case in cases]
    summary = _summary(ordered, dict(client.usage_summary))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "evaluated_commit", "cases", "answered", "engine_complete",
        "agent_choice_preserved", "host_contract_complete", "gates",
        "llm_usage")}, indent=2, sort_keys=True))
    return 0 if all(summary["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
