"""Run ContextBench's raw-LLM and LLM-compiled matched arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from benchmarks.common.envfile import load_env_file
from benchmarks.common.openrouter import OpenRouterClient
from benchmarks.temporalbench.mcp_agent import coerce_json_containers
from gnomon.workflows import (
    DocumentRef, build_context_investigation_prompt,
    normalise_context_response_containers, parse_context_response,
)

from .run_contextbench import run_case, smape
from .schema import Case, Oracle, load_cases, load_oracles


FORECAST_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_forecast",
        "description": "Submit exactly the requested numeric trajectory.",
        "parameters": {
            "type": "object",
            "properties": {
                "forecast": {"type": "array", "items": {"type": "number"}},
                "context_used": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["forecast", "context_used"],
        },
    },
}


def safe_payload(case: Case) -> dict[str, Any]:
    """Exactly what an arm may see; the oracle type cannot enter this path."""
    covariate_names = [item["name"] for item in case.covariate_mapping]
    historical_rows = case.covariates[:-case.horizon] if case.covariates else ()
    future_rows = case.covariates[-case.horizon:] if case.covariates else ()
    return {
        "frequency": case.frequency, "horizon": case.horizon,
        "history": list(case.history), "context": case.narrative,
        "historical_covariates": (
            {name: [row[name] for row in historical_rows]
             for name in covariate_names} if historical_rows else {}),
        "future_covariates": (
            {name: [row[name] for row in future_rows]
             for name in covariate_names} if future_rows else {}),
    }


def raw_prompt(case: Case, *, include_context: bool = True) -> str:
    payload = safe_payload(case)
    if not include_context:
        payload.pop("context", None)
        payload.pop("historical_covariates", None)
        payload.pop("future_covariates", None)
    return (
        "Forecast the next horizon values of the value series. Use only the "
        "provided cutoff-safe material. Context may be useful, irrelevant, or "
        "insufficient to size numerically; do not invent an effect. Return "
        "exactly horizon finite numbers through submit_forecast.\n\n" +
        json.dumps(payload, separators=(",", ":"))
    )


def _tool_arguments(response: Any, expected: str) -> dict[str, Any]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("model returned no choices")
    calls = getattr(choices[0].message, "tool_calls", None) or []
    call = next((item for item in calls
                 if getattr(item.function, "name", None) == expected), None)
    if call is None:
        raise ValueError(f"model did not call {expected}")
    value = json.loads(call.function.arguments)
    if not isinstance(value, dict):
        raise ValueError("tool arguments are not an object")
    return value


def _response_usage(response: Any) -> tuple[int, int]:
    """Return request-local usage; global client deltas overlap concurrently."""
    usage = getattr(response, "usage", None)
    return (int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0))


def raw_case(case: Case, oracle: Oracle, client: OpenRouterClient) -> dict[str, Any]:
    started = time.perf_counter()
    request_usage: list[tuple[int, int]] = []
    try:
        def invoke(include_context: bool) -> tuple[list[float], dict[str, Any]]:
            response = client.chat(
                [{"role": "user", "content": raw_prompt(
                    case, include_context=include_context)}],
                tools=[FORECAST_TOOL], tool_choice="required", max_tokens=3000)
            request_usage.append(_response_usage(response))
            arguments = _tool_arguments(response, "submit_forecast")
            values = arguments.get("forecast")
            if not isinstance(values, list) or len(values) != case.horizon:
                raise ValueError(f"forecast must contain {case.horizon} values")
            forecast = [float(value) for value in values]
            if not all(__import__("math").isfinite(value) for value in forecast):
                raise ValueError("forecast contains a non-finite value")
            return forecast, arguments

        history_forecast, _ = invoke(False)
        forecast, arguments = invoke(True)
        history_replicate, _ = invoke(False)
        history_scores = [smape(oracle.actual, history_forecast),
                          smape(oracle.actual, history_replicate)]
        return {
            "case_id": case.case_id, "family": case.family, "status": "answered",
            "history_smape": mean(history_scores),
            "history_smape_replicates": history_scores,
            "context_smape": smape(oracle.actual, forecast),
            "incremental_smape": (mean(history_scores)
                                  - smape(oracle.actual, forecast)),
            "null_replication_delta": history_scores[0] - history_scores[1],
            "null_replication_absolute_delta": abs(
                history_scores[0] - history_scores[1]),
            "primary_changed": any(abs(left - right) > 1e-9 for left, right in
                                   zip(history_forecast, forecast)),
            "history_forecast": history_forecast,
            "history_replicate_forecast": history_replicate,
            "context_forecast": forecast,
            "context_used_claim": bool(arguments.get("context_used")),
            "reason": str(arguments.get("reason", "")),
            "should_influence": oracle.should_influence,
            "prompt_tokens": sum(item[0] for item in request_usage),
            "completion_tokens": sum(item[1] for item in request_usage),
            "usage_accounting_version": 2,
            "latency_seconds": round(time.perf_counter() - started, 6),
        }
    except Exception as error:
        return {"case_id": case.case_id, "family": case.family, "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "should_influence": oracle.should_influence,
                "latency_seconds": round(time.perf_counter() - started, 6)}


def compile_events(case: Case, client: OpenRouterClient) -> dict[str, Any]:
    if not case.context_events:
        return {"events": [], "rejected": [], "hypotheses": [],
                "compiler_called": False}
    source_type = ("calendar" if case.family != "prior_only"
                   else "narrative_assertion")
    lines = case.narrative.splitlines()
    header = lines[0] if lines else ""
    event_lines = [line for line in lines if " affects the value series from " in line]
    footer = lines[-1] if lines else ""
    chunks = [event_lines[index:index + 8]
              for index in range(0, len(event_lines), 8)] or [[]]

    def compile_chunk(item: tuple[int, list[str]]) -> dict[str, Any]:
        chunk_index, chunk_lines = item
        content = "\n".join([header, *chunk_lines, footer])
        document = DocumentRef(
            f"context-chunk-{chunk_index}.txt", content,
            source_type=source_type,
            reference=("contextbench:" + hashlib.sha256(
                case.case_id.encode()).hexdigest()[:16] +
                f":chunk:{chunk_index}"))
        request = build_context_investigation_prompt(
            [document], ["*"], future_events=False)
        tool = {"type": "function", "function": {
            "name": "submit_context", "description": "Submit extracted context.",
            "parameters": request["response_schema"],
        }}
        response = client.chat(
            [{"role": "system", "content": request["instructions"]}],
            tools=[tool], tool_choice="required", max_tokens=5000)
        prompt_tokens, completion_tokens = _response_usage(response)
        raw, coerced = coerce_json_containers(
            _tool_arguments(response, "submit_context"),
            fields=("events", "hypotheses"))
        raw, context_repairs = normalise_context_response_containers(raw)
        coerced.extend(context_repairs)
        parsed = parse_context_response(
            raw, [document], proposer={"proposer_id": f"llm:{client.model}",
                                      "kind": "llm"})
        for event_index, event in enumerate(parsed["events"]):
            event["event_id"] = (
                f"event_llm_chunk_{chunk_index:03d}_{event_index:03d}")
            # The public file column is `value`; the single-target runtime
            # represents it as `__default__`. This host-owned alias mapping is
            # schema resolution, not an LLM judgement. Wildcard remains
            # wildcard, and any genuinely different scope stays different.
            scopes = list(event.get("entity_scope") or [])
            event["entity_scope"] = [
                "*" if scope in {"value", "__default__"} else scope
                for scope in scopes
            ]
        parsed["raw_proposal"] = raw
        parsed["container_coercions"] = coerced
        parsed["chunk_index"] = chunk_index
        parsed["prompt_tokens"] = prompt_tokens
        parsed["completion_tokens"] = completion_tokens
        return parsed

    # Bounded chunks are independent quoted documents. Parallel compilation
    # prevents a 30-occurrence schedule becoming a 5× latency penalty while
    # preserving deterministic result order below.
    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
        compiled_chunks = list(pool.map(
            compile_chunk, list(enumerate(chunks))))
    return {
        "events": [event for chunk in compiled_chunks
                   for event in chunk.get("events") or []],
        "rejected": [item for chunk in compiled_chunks
                     for item in chunk.get("rejected") or []],
        "hypotheses": [item for chunk in compiled_chunks
                       for item in chunk.get("hypotheses") or []],
        "compiler_called": True,
        "compiler_calls": len(compiled_chunks),
        "raw_proposals": [chunk.get("raw_proposal") for chunk in compiled_chunks],
        "container_coercions": [item for chunk in compiled_chunks
                                for item in chunk.get("container_coercions") or []],
        "compiled_chunks": compiled_chunks,
    }


def _window_key(event: dict[str, Any]) -> tuple[str, str]:
    return str(event.get("effective_start")), str(event.get("effective_end"))


def compiled_case(case: Case, oracle: Oracle, client: OpenRouterClient,
                  work_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        compiled = compile_events(case, client)
        row = run_case(case, oracle, work_root,
                       event_override=list(compiled.get("events") or []))
        expected = {_window_key(event) for event in case.context_events}
        produced = {_window_key(event) for event in compiled.get("events") or []}
        row.update({
            "status": "answered", "compiler_called": compiled["compiler_called"],
            "compiler_events_expected": len(expected),
            "compiler_events_accepted": len(produced),
            "compiler_event_recall": (len(expected & produced) / len(expected)
                                      if expected else 1.0),
            "compiler_false_events": len(produced - expected),
            "compiler_rejected": len(compiled.get("rejected") or []),
            "compiler_calls": int(compiled.get("compiler_calls", 0)),
            "compiler_raw_events": sum(len((proposal or {}).get("events") or [])
                                       for proposal in compiled.get("raw_proposals") or []),
            "compiler_raw_proposals": compiled.get("raw_proposals"),
            "compiler_container_coercions": compiled.get("container_coercions"),
            "prompt_tokens": sum(int(chunk.get("prompt_tokens", 0))
                                 for chunk in compiled.get("compiled_chunks", [])),
            "completion_tokens": sum(int(chunk.get("completion_tokens", 0))
                                     for chunk in compiled.get("compiled_chunks", [])),
            "usage_accounting_version": 2,
            "latency_seconds_total": round(time.perf_counter() - started, 6),
        })
        return row
    except Exception as error:
        return {"case_id": case.case_id, "family": case.family, "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "should_influence": oracle.should_influence,
                "latency_seconds_total": round(time.perf_counter() - started, 6)}


def summarize(rows: list[dict[str, Any]], condition: str,
              client: OpenRouterClient,
              manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    answered = [row for row in rows if row.get("status") == "answered"]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answered:
        families[row["family"]].append(row)
    invocation_usage = {**client.usage_summary, "accounting_scope": "this_invocation"}
    summary: dict[str, Any] = {
        "benchmark": "contextbench", "condition": condition,
        "rows": len(rows), "answered": len(answered),
        "errors": len(rows) - len(answered),
        "llm_usage": invocation_usage,
        "llm_usage_this_invocation": invocation_usage,
        "families": {},
        **({"seed": manifest.get("seed"),
            "fresh_seed": manifest.get("fresh_seed"),
            "generator": manifest.get("generator")}
           if manifest else {}),
    }
    usage_complete = all(row.get("usage_accounting_version") == 2
                         for row in answered)
    summary["llm_usage_observations"] = {
        "accounting_complete": usage_complete,
        "prompt_tokens": (sum(int(row.get("prompt_tokens", 0)) for row in answered)
                          if usage_complete else None),
        "completion_tokens": (
            sum(int(row.get("completion_tokens", 0)) for row in answered)
            if usage_complete else None),
        "requests": (
            sum(3 if condition == "raw-llm"
                else int(row.get("compiler_calls", 0)) for row in answered)
            if usage_complete else None),
    }
    for family, members in sorted(families.items()):
        metric = "context_smape"
        summary["families"][family] = {
            "rows": len(members), metric: mean(row[metric] for row in members),
            "history_smape": mean(row["history_smape"] for row in members),
            "incremental_smape": mean(row["incremental_smape"] for row in members),
        }
        if condition == "raw-llm":
            null_noise = mean(
                row["null_replication_absolute_delta"] for row in members)
            summary["families"][family].update({
                "null_replication_absolute_delta": null_noise,
                "incremental_smape_beyond_noise_floor": (
                    mean(row["incremental_smape"] for row in members)
                    - null_noise),
            })
        if condition == "compiled-context":
            family_event_rows = [row for row in members
                                 if row.get("compiler_events_expected", 0) > 0]
            summary["families"][family].update({
                "compiler_event_recall": (
                    mean(row["compiler_event_recall"] for row in family_event_rows)
                    if family_event_rows else None),
                "compiler_false_events": sum(row["compiler_false_events"] for row in members),
                "applied": sum(row["applied"] for row in members),
            })
    if condition == "raw-llm":
        summary["context_claim_accuracy"] = (
            mean(row["context_used_claim"] == row["should_influence"]
                 for row in answered) if answered else None)
        summary["mean_null_replication_absolute_delta"] = (
            mean(row["null_replication_absolute_delta"] for row in answered)
            if answered else None)
    else:
        event_rows = [row for row in answered
                      if row.get("compiler_events_expected", 0) > 0]
        summary["compiler_event_recall_macro"] = (
            mean(row["compiler_event_recall"] for row in event_rows)
            if event_rows else None)
        summary["compiler_false_events"] = sum(
            row["compiler_false_events"] for row in answered)
        accepted = sum(int(row.get("compiler_events_accepted", 0))
                       for row in event_rows)
        expected = sum(int(row.get("compiler_events_expected", 0))
                       for row in event_rows)
        false = sum(int(row.get("compiler_false_events", 0))
                    for row in event_rows)
        matched = sum(
            float(row["compiler_event_recall"])
            * int(row.get("compiler_events_expected", 0))
            for row in event_rows)
        summary["compiler_event_recall"] = (
            matched / expected if expected else None)
        summary["compiler_event_precision"] = (
            (accepted - false) / accepted if accepted else 1.0)
        summary["compiler_calls"] = sum(
            int(row.get("compiler_calls", 0)) for row in answered)
        summary["compiler_container_coercions"] = sum(
            len(row.get("compiler_container_coercions") or [])
            for row in answered)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", choices=["raw-llm", "compiled-context"],
                        required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                        choices=["OPENROUTER_API_KEY", "ENGY_API_KEY", "CHUTES_API_KEY"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true",
                        help="retain terminal observations and run only missing IDs")
    parser.add_argument("--retry-errors", action="store_true",
                        help="with --resume, rerun prior errors; ordinary resume retains all terminal rows")
    args = parser.parse_args()
    load_env_file()
    client = OpenRouterClient(
        args.model, api_key=os.environ.get(args.api_key_env),
        base_url=args.base_url, temperature=args.temperature,
        max_tokens=5000, timeout=args.request_timeout,
        max_retries=args.max_retries)
    cases = load_cases(Path(args.corpus_dir) / "cases.jsonl")
    oracles = load_oracles(Path(args.corpus_dir) / "oracle.jsonl")
    corpus = Path(args.corpus_dir)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256((corpus / "cases.jsonl").read_bytes()).hexdigest() \
            != manifest.get("cases_sha256"):
        raise SystemExit("cases.jsonl does not match its manifest hash")
    if hashlib.sha256((corpus / "oracle.jsonl").read_bytes()).hexdigest() \
            != manifest.get("oracle_sha256"):
        raise SystemExit("oracle.jsonl does not match its manifest hash")
    if len(cases) != int(manifest.get("cases", -1)):
        raise SystemExit("case count does not match the corpus manifest")
    if args.limit:
        grouped: dict[str, list[Case]] = defaultdict(list)
        for case in cases:
            grouped[case.family].append(case)
        cases = []
        while len(cases) < args.limit and any(grouped.values()):
            for family in sorted(grouped):
                if grouped[family] and len(cases) < args.limit:
                    cases.append(grouped[family].pop(0))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contextbench-llm-", dir=str(output)))
    retained: dict[str, dict[str, Any]] = {}
    observation_path = output / "observations.jsonl"
    if args.resume and observation_path.exists():
        for line in observation_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "answered" or not args.retry_errors:
                retained[str(row["case_id"])] = row
    pending = [case for case in cases if case.case_id not in retained]

    def one(case: Case) -> dict[str, Any]:
        oracle = oracles[case.case_id]
        return (raw_case(case, oracle, client) if args.condition == "raw-llm"
                else compiled_case(case, oracle, client, work))

    rows_by_id = dict(retained)

    def checkpoint() -> None:
        temporary = observation_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for case in cases:
                row = rows_by_id.get(case.case_id)
                if row is not None:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.replace(observation_path)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(one, case): case for case in pending}
        for future in as_completed(futures):
            row = future.result()
            rows_by_id[str(row["case_id"])] = row
            checkpoint()
    rows = [rows_by_id[case.case_id] for case in cases]
    summary = summarize(rows, args.condition, client, manifest)
    summary["resumed_answered_rows"] = sum(
        row.get("status") == "answered" for row in retained.values())
    summary["resumed_terminal_rows"] = len(retained)
    summary["rows_executed_this_invocation"] = len(pending)
    summary["corpus_manifest_sha256"] = hashlib.sha256(
        (corpus / "manifest.json").read_bytes()).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
