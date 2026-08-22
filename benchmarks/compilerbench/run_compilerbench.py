"""Run CompilerBench against an OpenAI-compatible model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.envfile import load_env_file
from benchmarks.common.manifest import code_revision, write_manifest
from benchmarks.common.openrouter import OpenRouterClient
from benchmarks.compilerbench.generate import cases
from gnomon.contracts import GnomonError
from gnomon.temporal_intent import compile_temporal_text


def _sum_usage(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}
    return {
        "model": summaries[0].get("model"),
        "base_url": summaries[0].get("base_url"),
        "requests": sum(int(row.get("requests", 0)) for row in summaries),
        "transport_attempts": sum(
            int(row.get("transport_attempts", 0)) for row in summaries),
        "prompt_tokens": sum(
            int(row.get("prompt_tokens", 0)) for row in summaries),
        "completion_tokens": sum(
            int(row.get("completion_tokens", 0)) for row in summaries),
        "cost_usd": round(sum(
            float(row.get("cost_usd", 0)) for row in summaries), 6),
        "truncation_escalations": sum(
            int(row.get("truncation_escalations", 0)) for row in summaries),
    }


class ChatAdapter:
    def __init__(self, client: OpenRouterClient):
        self.client = client

    def complete(self, prompt: str, response_schema: dict[str, Any]) -> dict[str, Any]:
        tool = {"type": "function", "function": {
            "name": "submit_temporal_intent", "description": "Submit intent only.",
            "parameters": response_schema}}
        response = self.client.chat(
            [{"role": "system", "content": prompt}], tools=[tool],
            tool_choice={"type": "function", "function": {
                "name": "submit_temporal_intent"}}, max_tokens=700)
        call = response.choices[0].message.tool_calls[0]
        return json.loads(call.function.arguments)


def _score(row: dict[str, Any], compiled: list[Any] | None,
           error: str | None, error_details: dict[str, Any] | None = None,
           *, semantic_refusal: bool = False,
           infrastructure_error: bool = False) -> dict[str, Any]:
    expected = row["expected"]
    refused = semantic_refusal
    def matches(item: Any, wanted: dict[str, Any]) -> bool:
        match = item.property == wanted["property"] and item.scope == wanted["scope"]
        if wanted["scope"] == "series":
            match = match and item.target == wanted["target"]
        else:
            match = match and list(item.members) == wanted["members"]
        if wanted.get("aggregation"):
            match = match and item.aggregation == wanted["aggregation"]
        if wanted.get("horizon"):
            match = match and item.horizon == wanted["horizon"]
        return match

    if expected.get("refusal"):
        correct = refused
    elif refused or not compiled:
        correct = False
    elif expected.get("questions"):
        correct = (len(compiled) == len(expected["questions"])
                   and all(matches(item, wanted) for item, wanted in zip(
                       compiled, expected["questions"])))
    else:
        correct = matches(compiled[0], expected)
    return {"id": row["id"], "case_kind": row["case_kind"],
            "expected": expected, "correct": correct,
            "refused": refused, "error": error,
            "error_details": error_details,
            "infrastructure_error": infrastructure_error,
            "compiled": [item.to_dict() for item in compiled or []]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--base-url", default="https://api.engy.ai/v1")
    parser.add_argument("--api-key-env", default="ENGY_API_KEY")
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument("--ids", help="Comma-separated case ids to rerun")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--infrastructure-retries", type=int, default=2)
    parser.add_argument("--output-dir", default="results/compilerbench")
    args = parser.parse_args()
    run_revision = code_revision()
    load_env_file()
    rows = cases(seed=args.seed, count=args.count)
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [row for row in rows if row["id"] in wanted]

    def run(row: dict[str, Any]) -> dict[str, Any]:
        usage: list[dict[str, Any]] = []
        for attempt in range(args.infrastructure_retries + 1):
            client = OpenRouterClient(
                args.model, api_key=os.environ.get(args.api_key_env),
                base_url=args.base_url, temperature=0, max_retries=3, timeout=180)
            try:
                compiled = compile_temporal_text(
                    row["text"], available_targets=row["targets"],
                    adapter=ChatAdapter(client), default_verb="describe")
                result = _score(row, compiled, None)
                result["infrastructure_retries"] = attempt
                usage.append(client.usage_summary)
                result["llm_usage"] = _sum_usage(usage)
                return result
            except GnomonError as exc:
                usage.append(client.usage_summary)
                details = getattr(exc, "details", None) or {}
                proposal = details.get("compiler_proposal") or {}
                # A declared refusal or a well-formed proposal rejected by the
                # deterministic contract is a semantic refusal. Malformed model
                # output is a completed, incorrect compilation—not a refusal and
                # not provider infrastructure failure.
                semantic_refusal = (
                    details.get("compiler_status") == "refused"
                    or isinstance(proposal.get("questions"), list)
                )
                result = _score(row, None, f"{type(exc).__name__}: {exc}",
                                details, semantic_refusal=semantic_refusal)
                result["infrastructure_retries"] = attempt
                result["llm_usage"] = _sum_usage(usage)
                return result
            except Exception as exc:
                usage.append(client.usage_summary)
                if attempt < args.infrastructure_retries:
                    continue
                result = _score(row, None, f"{type(exc).__name__}: {exc}",
                                infrastructure_error=True)
                result["infrastructure_retries"] = attempt
                result["llm_usage"] = _sum_usage(usage)
                return result
        raise AssertionError("unreachable")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, row): row for row in rows}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["id"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completed = [item for item in results if not item["infrastructure_error"]]
    exact = [item for item in completed
             if item["expected"].get("scope") == "series"]
    ambiguous = [item for item in completed if item["case_kind"] == "ambiguity"]
    adversarial = [item for item in completed
                   if item["case_kind"] == "adversarial_target"]
    malformed = [item for item in completed
                 if item["error_details"]
                 and (item["error_details"].get("compiler_proposal") or {}).get(
                     "status") == "compiled"
                 and not isinstance((item["error_details"].get(
                     "compiler_proposal") or {}).get("questions"), list)]
    summary = {"schema_version": "0.1", "model": args.model,
               "code_revision": run_revision,
               "seed": args.seed,
               "cases": len(results),
               "completed": len(completed),
               "infrastructure_errors": len(results) - len(completed),
               "infrastructure_retries": sum(
                   item.get("infrastructure_retries", 0) for item in results),
               "accuracy": (sum(item["correct"] for item in completed) /
                            max(1, len(completed))),
               "refusal_cases": sum(bool(item["expected"].get("refusal")) for item in results),
               "refusal_accuracy": (sum(item["correct"] for item in results
                                         if item["expected"].get("refusal")) /
                                    max(1, sum(bool(item["expected"].get("refusal"))
                                               for item in results))),
               "exact_target_accuracy": (sum(item["correct"] for item in exact) /
                                         max(1, len(exact))),
               "ambiguity_refusal_accuracy": (sum(item["correct"] for item in ambiguous) /
                                              max(1, len(ambiguous))),
               "invented_targets_accepted": sum(not item["refused"]
                                                for item in adversarial),
               "malformed_model_outputs": len(malformed),
               "llm_usage": _sum_usage([
                   item.get("llm_usage", {}) for item in results]),
               "results": results}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(
        output, benchmark="compilerbench", condition="compiled-intent",
        model=args.model, target=f"seed={args.seed};count={args.count}",
        base_url=args.base_url, code_revision=run_revision,
    )
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
