"""Raw-LLM and MCP-profile adapter for Workflow Bench smoke runs.

This adapter measures end-to-end routing and contract integration. It does not
turn the bundled smoke corpus into statistical product evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from benchmarks.cik.mcp_agent import StdioMcpSession, _tool_calls_as_dicts
from benchmarks.common.openrouter import OpenRouterClient
from benchmarks.temporalbench.mcp_agent import openai_tool_specs
from benchmarks.temporalbench.tasks import extract_json_object
# Shared with run_workflow's artifact verification so both sides derive
# identities identically; the local underscore names remain the adapter's
# public-for-tests surface.
from benchmarks.workflow.provenance import (
    artifact_evidence as _artifact_evidence,
    identity_fingerprint as _identity_fingerprint,
)

PROFILES = {"core", "describe", "evidence", "mega", "full"}
MAX_ROUNDS = 6
MAX_TOOL_CALLS = 4


def _export_artifact_for_verification(artifact_path: str | Path,
                                      case_id: str) -> str:
    """Copy the artifact's identity files out of the temporary jail.

    The jail is deleted when this adapter process exits, which would leave
    the runner nothing to re-read and keep every trust claim attested-only.
    When run_workflow names an export directory, artifact.json and
    evidence.jsonl are copied there and THAT durable path is reported as
    the observation's artifact_path.
    """
    import shutil

    export_root = os.environ.get("WORKFLOW_ARTIFACT_EXPORT_DIR")
    if not export_root:
        return str(artifact_path)
    destination = Path(export_root) / f"{case_id}-{Path(artifact_path).name}"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("artifact.json", "evidence.jsonl"):
        source = Path(artifact_path) / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    return str(destination)


def _find_triage(value: Any) -> dict[str, Any] | None:
    """Find the canonical triage block through thin router envelopes."""
    if isinstance(value, dict):
        triage = value.get("triage")
        if isinstance(triage, dict):
            return triage
        for child in value.values():
            found = _find_triage(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_triage(child)
            if found is not None:
                return found
    return None


def _extract_engine_facts(value: Any, requested: set[str]) -> dict[str, Any]:
    """Harvest requested canonical facts without treating prose as evidence."""
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        for container_name in ("triage", "temporal_facts", "facts"):
            container = value.get(container_name)
            if isinstance(container, dict):
                found.update({key: container[key] for key in requested
                              if key in container})
        # Thin workflow/router envelopes may nest the verb response.
        for child in value.values():
            if isinstance(child, (dict, list)):
                for key, item in _extract_engine_facts(child, requested).items():
                    found.setdefault(key, item)
    elif isinstance(value, list):
        for child in value:
            for key, item in _extract_engine_facts(child, requested).items():
                found.setdefault(key, item)
    return found

SUBMIT = {
    "type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final normalized Workflow Bench observation.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["answered", "abstained"]},
                "support": {"type": "string", "enum": ["supported", "degraded", "best_effort", "abstained"]},
                "numbers": {"type": "object", "additionalProperties": {"type": "number"}},
                "choices": {"type": "object", "additionalProperties": {"type": "string"}},
                "disclosures": {"type": "array", "items": {"type": "string"}},
                "claims": {"type": "array", "items": {"type": "string"}},
                "facts": {"type": "object", "additionalProperties": {}},
                "repair_completed": {"type": "boolean"},
                "tracking_completed": {"type": "boolean"},
                "quote_matches": {"type": "boolean"},
                "publish_matches_evaluated": {"type": "boolean"},
                "artifact_id": {"type": "string",
                                "description": "Copy Gnomon's artifact/forecast id exactly when returned."},
            },
            "required": ["status", "support", "numbers", "choices", "disclosures", "claims"],
        },
    },
}


def _columns(payload: dict[str, Any]) -> dict[str, list[Any]]:
    if isinstance(payload.get("columns"), dict):
        return {str(k): list(v) for k, v in payload["columns"].items()}
    if isinstance(payload.get("series"), dict):
        return {str(k): list(v) for k, v in payload["series"].items()}
    if isinstance(payload.get("series"), list):
        return {"value": list(payload["series"])}
    if isinstance(payload.get("daily_peak"), list):
        return {"daily_peak": list(payload["daily_peak"])}
    return {}


def write_case_csv(case: dict[str, Any], path: Path) -> None:
    payload = case["available_at_cutoff"]
    columns = _columns(payload)
    length = max((len(values) for values in columns.values()), default=0)
    timestamps = columns.pop("timestamp", None)
    if timestamps is None:
        end = date.fromisoformat(payload["cutoff"][:10])
        timestamps = [(end - timedelta(days=length - index - 1)).isoformat()
                      for index in range(length)]
    keys = list(columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", *keys])
        for index in range(length):
            writer.writerow([timestamps[index], *[
                columns[key][index] if index < len(columns[key]) else "" for key in keys
            ]])


def _prompt(case: dict[str, Any], csv_path: Path | None) -> str:
    location = (f"The cutoff-safe data is available at {csv_path}." if csv_path
                else "No tabular series was supplied.")
    stage = case.get("workflow_stage")
    stage_rule = ""
    if stage == "repair":
        stage_rule = (
            "The ambiguity has now been resolved. Answer only for revealed.target_column, "
            "put the forecast under numbers.next, and copy that column name to "
            "facts.resolved_target.\n")
    elif stage == "outcome":
        stage_rule = (
            "This is an outcome-tracking follow-up. Do not make a new forecast. Compute "
            "numbers.absolute_error from revealed.actual and prior_observation.numbers.next, "
            "and copy prior_observation.published_fingerprint exactly to "
            "facts.tracked_forecast_id.\n")
    elif case.get("kind") == "messy":
        stage_rule = (
            "This publishing request has multiple possible target columns and names none. "
            "Do not forecast all columns or guess: submit status=abstained, "
            "support=abstained, and facts.ambiguity='target'.\n")
    return f"""Answer this temporal workflow using only information available at the cutoff.
{location}
Case kind: {case['kind']}; domain: {case['domain']}.
Question: {case['question']}
Required answer keys: {json.dumps(case.get('answer_schema') or {})}
Cutoff-safe input: {json.dumps(case['available_at_cutoff'], separators=(',', ':'))}
Revealed stage input: {json.dumps(case.get('revealed') or {}, separators=(',', ':'))}
{stage_rule}

Return numbers under stable names requested by the question, categorical answers under choices,
and populate every requested facts key from the supplied case or Gnomon artifact. Preserve material
limitations in disclosures. When Gnomon returns an artifact_id or forecast_id, copy it exactly to
artifact_id in submit_answer. Never claim to have observed post-cutoff data.
"""


def _compile_execution_arguments(
    case: dict[str, Any], name: str, arguments: dict[str, Any],
    csv_path: Path, jail: Path,
) -> dict[str, Any]:
    """Bind transport fields the host already knows from the task."""
    if name not in {"gnomon_forecast", "gnomon_describe", "gnomon_run"}:
        return arguments
    # This benchmark contains no user-selected execution policy. Discard
    # invented transport/options (notably stale data_ref, series_column, and
    # output paths) instead of merging them into the host-resolved call.
    compiled = {"input": str(csv_path), "time_column": "timestamp",
                "frequency": "D"}
    columns = [key for key in _columns(case["available_at_cutoff"])
               if key != "timestamp"]
    revealed_target = (case.get("revealed") or {}).get("target_column")
    intentionally_ambiguous = (
        case.get("kind") == "messy" and not revealed_target and len(columns) > 1
    )
    if revealed_target:
        target = str(revealed_target)
    elif name == "gnomon_describe" or case.get("kind") == "multiseries":
        target = ",".join(columns) if columns else "auto"
    elif len(columns) == 1:
        target = columns[0]
    else:
        target = None
    if target is not None:
        compiled["target_column"] = target
    elif intentionally_ambiguous:
        compiled.pop("target_column", None)
    if name in {"gnomon_forecast", "gnomon_run"}:
        compiled["horizon"] = 1
        compiled["output_dir"] = str(jail / "gnomon-output")
    if name == "gnomon_forecast":
        compiled["format"] = "brief"
        compiled["minimum_support"] = "best_effort"
    elif name == "gnomon_run":
        compiled["question"] = {"kind": "forecast"}
    return compiled


def _normalize(case_id: str, value: dict[str, Any], *, calls: int,
               client: OpenRouterClient, started: float,
               tool_names: list[str], engine_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_status = str(value.get("status", "answered")).strip().casefold()
    status = ({"success": "answered", "complete": "answered", "completed": "answered",
               "unsupported": "abstained", "refused": "abstained"}.get(raw_status, raw_status))
    if status not in {"answered", "abstained", "error"}:
        status = "error"
    raw_support = str(value.get(
        "support", "abstained" if status in {"abstained", "error"} else "degraded"
    )).strip().casefold().replace("-", "_").replace(" ", "_")
    support = ({"weak": "degraded", "low": "degraded", "high": "supported",
                "unsupported": "abstained", "none": "abstained"}.get(raw_support, raw_support))
    if support not in {"supported", "degraded", "best_effort", "abstained"}:
        support = "abstained" if status in {"abstained", "error"} else "degraded"
    if status in {"abstained", "error"}:
        support = "abstained"
    numbers = {}
    for key, item in (value.get("numbers") or {}).items():
        if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            numbers[str(key)] = float(item)
    raw_choices = value.get("choices")
    choices = {str(key): str(item) for key, item in (
        raw_choices if isinstance(raw_choices, dict) else {}).items()
               if item is not None}
    raw_facts = value.get("facts")
    facts = dict(raw_facts) if isinstance(raw_facts, dict) else {}
    raw_disclosures = value.get("disclosures")
    disclosures = (raw_disclosures if isinstance(raw_disclosures, list) else
                   [str(raw_disclosures)] if raw_disclosures else [])
    raw_claims = value.get("claims")
    claims = (raw_claims if isinstance(raw_claims, list) else
              [str(raw_claims)] if raw_claims else [])
    engine_evidence = engine_evidence or {}
    artifact_numbers = engine_evidence.get("artifact_numbers") or {}
    return {
        "case_id": case_id, "status": status, "support": support,
        "numbers": numbers, "choices": choices,
        "disclosures": disclosures, "claims": claims,
        "facts": facts,
        "engine_facts": engine_evidence.get("engine_facts") or {},
        # The harness supplied only cutoff-safe input, so leakage is structurally measured false.
        "temporal_leakage": False,
        "publish_matches_evaluated": value.get("publish_matches_evaluated"),
        "repair_completed": value.get("repair_completed"),
        "tracking_completed": value.get("tracking_completed"),
        "quote_matches": value.get("quote_matches"),
        "evaluated_fingerprint": engine_evidence.get("evaluated_fingerprint"),
        "published_fingerprint": engine_evidence.get("published_fingerprint"),
        "artifact_numbers": artifact_numbers,
        "headline_numbers": {key: numbers[key] for key in artifact_numbers if key in numbers},
        "tool_calls": calls,
        "cumulative_tokens": client.total_prompt_tokens + client.total_completion_tokens,
        "response_tokens": client.total_completion_tokens,
        "latency_seconds": time.time() - started,
        "metadata": {"tool_sequence": tool_names,
                     "surface_required_calls": engine_evidence.get(
                         "surface_required_calls", 0),
                     "recovery_calls": engine_evidence.get("recovery_calls", 0),
                     "redundant_calls": engine_evidence.get("redundant_calls", 0),
                     **({"error": "model_submission_error"} if status == "error" else {}),
                     "artifact_id": engine_evidence.get("artifact_id"),
                     "artifact_path": engine_evidence.get("artifact_path"),
                     "agent_artifact_id": value.get("artifact_id"),
                     "parity_evidence_level": engine_evidence.get("parity_evidence_level"),
                     "envelope_repairs": {
                         "status": [raw_status, status] if raw_status != status else None,
                         "support": [raw_support, support] if raw_support != support else None,
                         "choices": "coerced_to_empty_object"
                            if raw_choices is not None and not isinstance(raw_choices, dict) else None,
                         "facts": "coerced_to_empty_object"
                            if raw_facts is not None and not isinstance(raw_facts, dict) else None,
                         "disclosures": "wrapped_as_array"
                            if raw_disclosures is not None and not isinstance(raw_disclosures, list) else None,
                         "claims": "wrapped_as_array"
                            if raw_claims is not None and not isinstance(raw_claims, list) else None,
                     }},
    }


def direct(case: dict[str, Any], client: OpenRouterClient, csv_path: Path | None) -> dict[str, Any]:
    response = client.chat([
        {"role": "system", "content": "Return one JSON object only with status, support, numbers, choices, facts, disclosures, and claims."},
        {"role": "user", "content": _prompt(case, csv_path)},
    ], n=1, max_tokens=4000, tools=[SUBMIT], tool_choice="auto")
    message = response.choices[0].message
    for call in _tool_calls_as_dicts(message):
        if call["function"]["name"] == "submit_answer":
            return json.loads(call["function"]["arguments"] or "{}")
    try:
        return extract_json_object(message.content or "")
    except ValueError:
        return {"status": "error", "support": "abstained", "numbers": {},
                "choices": {}, "disclosures": ["model returned no structured answer"],
                "claims": []}


def mcp(case: dict[str, Any], client: OpenRouterClient, csv_path: Path,
        jail: Path, profile: str) -> tuple[dict[str, Any], int, list[str], dict[str, Any]]:
    # The server deliberately runs with the case jail as cwd. Ensure that
    # isolation does not make the checkout under test disappear from module
    # discovery (or silently fall back to an older globally installed build).
    repository = Path(__file__).resolve().parents[2]
    import_roots = [str(repository / "src"), str(repository)]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = existing_pythonpath.split(os.pathsep) if existing_pythonpath else []
    os.environ["PYTHONPATH"] = os.pathsep.join([
        *[root for root in import_roots if root not in pythonpath], *pythonpath])
    session = StdioMcpSession(
        jail, command=[sys.executable, "-m", "gnomon", "mcp", "serve", "--profile", profile]
    )
    calls = 0
    sequence: list[str] = []
    engine_evidence: dict[str, Any] = {"surface_required_calls": 1}
    computation_complete = False
    recovery_calls = 0
    try:
        session.initialize()
        tools = openai_tool_specs(session.list_tools(), SUBMIT)
        preferred_name = (
            "gnomon_run" if profile == "mega" else
            "gnomon_describe" if case.get("kind") == "multiseries"
            and profile in {"describe", "evidence", "full"} else
            "gnomon_inspect" if case.get("kind") == "multiseries" else
            "gnomon_forecast"
        )
        messages = [
            {"role": "system", "content": (
                "Use Gnomon for temporal computation. The CSV is inside your path jail. "
                "End with submit_answer. Gnomon-authored numbers must be quoted faithfully; "
                "if support is weak, return the forecast with its degraded/best_effort label. "
                f"For this task, call {preferred_name} exactly once, then submit from that response. "
                "Do not inspect first, read an artifact, or repeat a successful tool."
            )},
            {"role": "user", "content": _prompt(case, csv_path)},
        ]
        for _ in range(MAX_ROUNDS):
            available = (tools if calls < MAX_TOOL_CALLS and not computation_complete
                         else [SUBMIT])
            tool_choice: Any = "auto"
            if calls == 0 and not computation_complete:
                tool_choice = {"type": "function", "function": {
                    "name": preferred_name}}
            response = client.chat(messages, n=1, tools=available,
                                   tool_choice=tool_choice, max_tokens=6000)
            message = response.choices[0].message
            tool_calls = _tool_calls_as_dicts(message)
            messages.append({"role": "assistant", "content": message.content or None,
                             **({"tool_calls": tool_calls} if tool_calls else {})})
            if not tool_calls:
                messages.append({"role": "user", "content": "Call submit_answer now."})
                continue
            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if name == "submit_answer":
                    return arguments, calls, sequence, engine_evidence
                calls += 1
                sequence.append(name)
                arguments = _compile_execution_arguments(
                    case, name, arguments, csv_path, jail)
                if calls > MAX_TOOL_CALLS:
                    content = json.dumps({"code": "TOOL_BUDGET_SPENT"})
                else:
                    result = session.call_tool(name, arguments)
                    structured = result.get("structuredContent") or {}
                    if isinstance(structured, dict) and structured.get("status") not in {
                            "error", "invalid"}:
                        # The product host owns this state transition. Once a
                        # selected verb has returned a usable answer, the next
                        # legal operation is envelope submission—not browsing,
                        # artifact rereads, or repeating the computation.
                        computation_complete = True
                    elif not computation_complete:
                        recovery_calls += 1
                    if isinstance(structured, dict):
                        requested = set((case.get("answer_schema") or {}).get(
                            "facts") or [])
                        harvested = _extract_engine_facts(structured, requested)
                        if harvested:
                            engine_evidence.setdefault("engine_facts", {}).update(
                                harvested)
                        response_results = structured.get("results") or []
                        if len(response_results) == 1:
                            response_result = response_results[0]
                            identity = response_result.get("execution_identity") or {}
                            evaluated = identity.get("evaluated") or {}
                            published = identity.get("published") or {}
                            source = published.get("data_fingerprint")
                            if evaluated.get("name") and published.get("name") and source:
                                engine_evidence.update({
                                    "evaluated_fingerprint": _identity_fingerprint(
                                        str(evaluated["name"]), str(source)),
                                    "published_fingerprint": _identity_fingerprint(
                                        str(published["name"]), str(source)),
                                    "artifact_id": structured.get("artifact_id")
                                        or structured.get("forecast_id"),
                                    "parity_evidence_level": "response_execution_identity",
                                })
                            answer_keys = set((case.get("answer_schema") or {}).get("numbers") or [])
                            forecast_rows = (response_result.get("forecast")
                                             or response_result.get("forecast_preview") or [])
                            if "next" in answer_keys and forecast_rows:
                                first = forecast_rows[0]
                                point = first.get("q50", first.get("point"))
                                if point is not None:
                                    engine_evidence["artifact_numbers"] = {
                                        "next": float(point)}
                    artifact_path = structured.get("artifact_path") if isinstance(structured, dict) else None
                    if artifact_path:
                        try:
                            from gnomon.artifacts import read_artifact
                            artifact = read_artifact(artifact_path)
                            # Recorded so run_workflow can re-read the artifact
                            # and verify the trust claims independently.
                            engine_evidence["artifact_path"] = \
                                _export_artifact_for_verification(
                                    artifact_path, str(case.get("id", "case")))
                            engine_evidence.update(_artifact_evidence(artifact_path, artifact))
                            answer_keys = set((case.get("answer_schema") or {}).get("numbers") or [])
                            if "next" in answer_keys:
                                results = artifact.get("results") or []
                                forecast = (results[0].get("forecast") or []) if results else []
                                if forecast:
                                    first = forecast[0]
                                    point = first["q50"] if "q50" in first else first["point"]
                                    engine_evidence["artifact_numbers"] = {
                                        "next": float(point)
                                    }
                        except Exception:
                            pass
                    blocks = result.get("content") or []
                    content = blocks[0].get("text", "") if blocks else json.dumps(result.get("structuredContent") or {})
                    if len(content) > 16000:
                        content = content[:8000] + "\n...[tool result bounded]...\n" + content[-8000:]
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
                engine_evidence["recovery_calls"] = recovery_calls
                engine_evidence["redundant_calls"] = 0
        # Preserve completed work after the browsing/round cap: withdraw all
        # engine tools and give the model two envelope-only attempts. This is
        # the same recovery contract TemporalBench uses; without it, a model
        # that already produced an artifact is misreported as an error merely
        # because it spent its last round reading that artifact.
        messages.append({"role": "user", "content": (
            "Tool browsing is closed. Call submit_answer now using the evidence "
            "already collected. Do not call another Gnomon tool."
        )})
        for attempt in range(2):
            response = client.chat(messages, n=1, tools=[SUBMIT],
                                   tool_choice="auto", max_tokens=4000)
            message = response.choices[0].message
            final_calls = _tool_calls_as_dicts(message)
            for call in final_calls:
                if call["function"]["name"] != "submit_answer":
                    continue
                try:
                    return (json.loads(call["function"]["arguments"] or "{}"),
                            calls, sequence, engine_evidence)
                except json.JSONDecodeError:
                    pass
            messages.append({"role": "assistant", "content": message.content or None,
                             **({"tool_calls": final_calls} if final_calls else {})})
            messages.append({"role": "user", "content": (
                "Your prior response was not a valid submit_answer call. "
                "Repair only the final envelope and call submit_answer."
            )})
        return ({"status": "error", "support": "abstained", "disclosures": ["no submission"],
                 "numbers": {}, "choices": {}, "claims": []}, calls, sequence,
                engine_evidence)
    finally:
        session.close()


def _recover_engine_answer(case: dict[str, Any], value: dict[str, Any],
                           evidence: dict[str, Any]) -> dict[str, Any]:
    """Compile a valid envelope when computation succeeded but submit failed."""
    if value.get("status") != "error" or "next" not in (evidence.get("artifact_numbers") or {}):
        return value
    facts: dict[str, Any] = {}
    if case.get("workflow_stage") == "repair":
        facts["resolved_target"] = (case.get("revealed") or {}).get("target_column")
    elif case.get("kind") == "longitudinal":
        facts["tracking_requested"] = True
    return {
        "status": "answered", "support": "degraded",
        "numbers": {"next": evidence["artifact_numbers"]["next"]},
        "choices": {}, "facts": facts,
        "disclosures": ["Final answer envelope was compiled from the immutable Gnomon artifact after model submission failed."],
        "claims": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["control", *sorted(PROFILES)], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--api-key-env", default="OPENROUTER_API_KEY",
        choices=["OPENROUTER_API_KEY", "ENGY_API_KEY", "CHUTES_API_KEY"],
        help="Credential variable for the selected OpenAI-compatible endpoint.")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    case = json.loads(sys.stdin.readline())
    started = time.time()
    from benchmarks.common.envfile import load_env_file
    load_env_file()
    client = OpenRouterClient(args.model, api_key=os.environ.get(args.api_key_env),
                              base_url=args.base_url,
                              temperature=args.temperature, max_tokens=6000)
    with tempfile.TemporaryDirectory(prefix="workflow-arm-") as directory:
        jail = Path(directory).resolve()
        csv_path = jail / "history.csv"
        write_case_csv(case, csv_path)
        # Outcome scoring is arithmetic over a previously published immutable
        # identity. A fresh MCP subprocess cannot (and must not pretend to)
        # reopen the prior temporary session, so this stage exposes only the
        # submit envelope. The runner independently verifies both the error and
        # identity binding.
        if case.get("workflow_stage") == "outcome":
            value, calls, sequence, engine_evidence = direct(case, client, None), 0, [], {}
        elif args.condition == "control":
            value, calls, sequence, engine_evidence = direct(case, client, csv_path), 0, [], {
                "surface_required_calls": 0}
        else:
            value, calls, sequence, engine_evidence = mcp(case, client, csv_path, jail, args.condition)
            value = _recover_engine_answer(case, value, engine_evidence)
        print(json.dumps(_normalize(case["id"], value, calls=calls, client=client,
                                    started=started, tool_names=sequence,
                                    engine_evidence=engine_evidence)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
