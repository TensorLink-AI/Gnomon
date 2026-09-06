"""One unsteered agent loop for operator-supplied ordinary/lean/full backends.

Tool/round ceilings are dispatch limits. Total tokens are provider measurements:
an unknown count prevents more work; a measured overrun is disclosed, not described
as a preempted expense. Factories/backends are trusted operator code, not sandboxes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import time

from benchmarks.workflow.agent_metrics import _decode_record
from .matched import fingerprint
from .schema import Observation
from .accounting import reported_cost_limit

WIRE_BYTES = 1_048_576
CONTEXT_BYTES = 4_194_304
ANSWER_FIELDS = {"status", "support", "numbers", "choices", "facts", "disclosures", "claims"}
SUBMIT = {"type": "function", "function": {
    "name": "submit_answer", "description": "Submit the final answer as the only tool call in this message.",
    "parameters": {"type": "object", "additionalProperties": False,
        "properties": {"status": {"enum": ["answered", "abstained"]},
                       "support": {"enum": ["supported", "degraded", "best_effort", "abstained"]},
                       "numbers": {"type": "object", "additionalProperties": {"type": "number"}},
                       "choices": {"type": "object", "additionalProperties": {"type": "string"}},
                       "facts": {"type": "object"},
                       "disclosures": {"type": "array", "items": {"type": "string"}},
                       "claims": {"type": "array", "items": {"type": "string"}}},
        "required": ["status", "support"]}}}


@dataclass
class ToolReply:
    value: object
    # Operator-reported service charge, not machine/energy cost or attestation.
    cost_usd: float | None = None


def _encode(value, limit=WIRE_BYTES):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) > limit:
        raise ValueError("agent message byte limit exceeded")
    return text


def _charge(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid reported service cost")
    return value


def _usage(client):
    usage = client.usage_summary
    complete = usage["resource_fields_complete"]
    counts = [usage[key] for key in ("prompt_tokens", "completion_tokens")]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("invalid client token counters")
    known = all(complete.get(key) is True for key in ("prompt_tokens", "completion_tokens"))
    cost = _charge(usage["observed_cost_usd"]) if complete.get("cost") is True else None
    return sum(counts), counts[1], known, cost


def run_agent(case, *, prompt, budget, client_factory, backend_factory, generation=None, clock=time.monotonic,
              episode=None, checkpoint=None):
    """Return a normalized Observation without choosing tools or repairing answers.

    client_factory creates a fresh OpenRouterClient-compatible accounting client.
    backend_factory creates an object with tools(), call(name,args,timeout=...),
    close(), and optional startup_cost_usd. tools() returns actual MCP tool specs;
    call returns ToolReply. Factories must not make hidden unaccounted model calls.
    No service calls occur until the caller explicitly supplies these factories.
    """
    generation = dict(generation or {})
    if set(generation) - {"temperature", "reasoning_effort", "max_output_tokens"}:
        raise ValueError("unsupported generation settings; do not silently ignore model controls")
    limits = {key: budget[key] for key in ("max_rounds", "max_tool_calls", "max_tokens", "timeout_seconds")}
    cost_limit = reported_cost_limit(budget)
    if cost_limit is not None:
        limits["max_reported_cost_usd"] = cost_limit
    for key in ("max_rounds", "max_tool_calls", "max_tokens"):
        if type(limits[key]) is not int or limits[key] < (0 if key == "max_tool_calls" else 1):
            raise ValueError(f"invalid {key}")
    if type(limits["timeout_seconds"]) not in (int, float) or not 0 < limits["timeout_seconds"] <= 3600:
        raise ValueError("invalid timeout_seconds")
    output_limit = generation.pop("max_output_tokens", limits["max_tokens"])
    if type(output_limit) is not int or output_limit < 1:
        raise ValueError("invalid max_output_tokens")
    if not isinstance(prompt, str) or not prompt.strip() or any(key in case for key in ("oracle", "stages", "_private_episode", "_episode_journal", "_spending_allowance_usd")):
        raise ValueError("supply a public single-stage case and common prompt")
    from .episodes import Episode
    journey = Episode(episode, checkpoint) if episode else None
    submit_tool = ({**SUBMIT, "function": {**SUBMIT["function"],
                   "description": "Commit the current phase as the only tool call; the next phase is revealed afterward."}}
                   if journey else SUBMIT)
    case_id = case["id"]
    started = clock()
    deadline = started + limits["timeout_seconds"]
    calls, dispatched, rounds = 0, 0, 0
    client = backend = None
    inventory, trace, tool_costs = [], [], []
    environment_costs, environment_events = [], []
    backend_provenance = {}
    tokens, response_tokens, tokens_known, llm_cost = 0, 0, False, None
    answer = None
    termination = "driver_error"
    cleanup_errors = []

    def remaining():
        return max(0.0, deadline - clock())

    def spending_stop():
        if cost_limit is None:
            return None
        values = [llm_cost, *tool_costs, *environment_costs]
        if any(value is None for value in values):
            return "cost_usage_unmeasured"
        return "cap:cost" if math.fsum(values) >= cost_limit else None

    try:
        client = client_factory()
        if getattr(client, "sample_cache_dir", None) is not None:
            raise ValueError("matched agent client must not restore or reuse sample caches")
        tokens, response_tokens, tokens_known, llm_cost = _usage(client)
        if tokens or llm_cost not in (0, None) or getattr(client, "total_transport_attempts", 0):
            raise ValueError("matched agent requires a fresh accounting client")
        backend = backend_factory()
        tool_costs.append(_charge(getattr(backend, "startup_cost_usd", None)))
        discovered = backend.tools()
        if not isinstance(discovered, list) or len(discovered) > 64:
            raise ValueError("invalid/beyond-limit tool inventory")
        inventory = json.loads(_encode(discovered))
        backend_provenance = json.loads(_encode(getattr(backend, "provenance", {})))
        specs = {}
        for tool in inventory:
            name = tool["name"]
            if not isinstance(name, str) or not name or name in specs or name == "submit_answer":
                raise ValueError("duplicate/invalid/reserved tool name")
            specs[name] = {"type": "function", "function": {
                "name": name, "description": tool.get("description", ""), "parameters": tool["inputSchema"]}}
        messages = [{"role": "system", "content": prompt + "\nSubmit with submit_answer as a sole tool call."},
                    {"role": "user", "content": _encode(case)}]
        if journey:
            messages[0]["content"] += (f"\nThis task has {len(journey.phases)} ordered phases. "
                                       "submit_answer commits the current phase before revealing the next; "
                                       "the last submission ends the task. Earlier submissions cannot be replaced.")
        termination = "cap:rounds"
        for _ in range(limits["max_rounds"]):
            if spending_stop():
                termination = spending_stop()
                break
            if remaining() <= 0:
                termination = "cap:time"
                break
            if tokens >= limits["max_tokens"]:
                termination = "cap:tokens"
                break
            tools = [*specs.values(), submit_tool] if calls < limits["max_tool_calls"] else [submit_tool]
            _encode({"messages": messages, "tools": tools}, CONTEXT_BYTES)
            rounds += 1
            response = client.chat(messages, tools=tools, tool_choice="auto",
                                   max_tokens=min(output_limit, limits["max_tokens"] - tokens),
                                   request_timeout=remaining(), **generation)
            tokens, response_tokens, tokens_known, llm_cost = _usage(client)
            if cost_limit is not None and math.fsum(value for value in [llm_cost, *tool_costs, *environment_costs] if value is not None) > cost_limit:
                termination = "cap:cost"
                break
            if remaining() <= 0 or tokens > limits["max_tokens"]:
                termination = "cap:time" if remaining() <= 0 else "cap:tokens"
                break
            if len(response.choices) != 1:
                termination = "model_response_error"
                break
            message = response.choices[0].message
            raw_calls = getattr(message, "tool_calls", None) or []
            if not isinstance(raw_calls, list) or len(raw_calls) > 100:
                raise ValueError("invalid/beyond-limit model tool calls")
            proposed = [{"id": call.id, "type": "function", "function": {
                "name": call.function.name, "arguments": call.function.arguments}} for call in raw_calls]
            _encode(proposed)
            ids = [item["id"] for item in proposed]
            if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
                raise ValueError("invalid/duplicate model tool call IDs")
            content = getattr(message, "content", None)
            if content is not None and not isinstance(content, str):
                raise ValueError("model message content must be text or null")
            _encode(content)
            messages.append({"role": "assistant", "content": content,
                             **({"tool_calls": proposed} if proposed else {})})
            submitted = len(proposed) == 1 and proposed[0]["function"]["name"] == "submit_answer"
            if spending_stop() and not submitted:
                termination = spending_stop()
                break
            if not tokens_known and not submitted:
                termination = "usage_unmeasured"
                break
            if tokens >= limits["max_tokens"] and not submitted:
                termination = "cap:tokens"
                break
            if not proposed:
                messages.append({"role": "user", "content": "Use submit_answer for the final answer envelope."})
            mixed_submit = any(call["function"]["name"] == "submit_answer" for call in proposed) and not submitted
            for call in proposed:
                name = call["function"]["name"]
                if name != "submit_answer":
                    if calls >= limits["max_tool_calls"]:
                        termination = "cap:tools"
                        break
                    calls += 1  # Malformed/unknown requests consume an attempt too.
                entry = {"name": name, "dispatched": False}
                trace.append(entry)
                try:
                    arguments = _decode_record(call["function"]["arguments"])
                    if mixed_submit:
                        raise ValueError("submit_answer must be the only tool call")
                    if name == "submit_answer":
                        if set(arguments) - ANSWER_FIELDS or arguments.get("status") not in {"answered", "abstained"}:
                            raise ValueError("invalid answer envelope")
                        candidate = Observation.from_dict({"case_id": case_id, **arguments})
                        if journey:
                            observed_costs = [value for value in [llm_cost, *tool_costs, *environment_costs] if value is not None]
                            try:
                                following = journey.commit(arguments, {"tool_calls": calls, "cumulative_tokens": tokens,
                                    "response_tokens": response_tokens, "latency_seconds": clock() - started,
                                    "cost_usd": math.fsum(observed_costs) if observed_costs else None})
                            except Exception as error:
                                termination = "episode_commit_error"
                                entry.update(status="error", error_type=type(error).__name__)
                                break
                            if following is not None:
                                answer = None
                                if spending_stop():
                                    termination = spending_stop()
                                    break
                                if not tokens_known or tokens >= limits["max_tokens"]:
                                    termination = "usage_unmeasured" if not tokens_known else "cap:tokens"
                                    break
                                if remaining() <= 0:
                                    termination = "cap:time"
                                    break
                                if callable(getattr(backend, "reveal", None)):
                                    environment_costs.append(None)
                                    try:
                                        event = backend.reveal(following["revealed"], timeout=remaining())
                                        if not isinstance(event, ToolReply):
                                            raise ValueError("reveal must return ToolReply")
                                        environment_costs[-1] = _charge(event.cost_usd)
                                        environment_events.append({"phase": following["name"], "input_sha256": fingerprint(following["revealed"]),
                                                                   "result": json.loads(_encode(event.value))})
                                    except Exception as error:
                                        termination = "episode_reveal_error"
                                        entry.update(status="error", error_type=type(error).__name__)
                                        break
                                if remaining() <= 0:
                                    termination = "cap:time"
                                    break
                                messages.append({"role": "tool", "tool_call_id": call["id"], "content": _encode({
                                    "phase_committed": journey.records[-1]["phase"], "next_phase": following["name"],
                                    "revealed": following["revealed"], "answer_schema": following["answer_schema"]})})
                                entry["status"] = "phase_committed"
                                break
                        answer = candidate
                        termination = ("submitted_cost_unmeasured" if cost_limit is not None and spending_stop() == "cost_usage_unmeasured"
                                       else "submitted" if tokens_known else "submitted_usage_unmeasured")
                        entry["status"] = "submitted"
                        break
                    if name not in specs:
                        raise ValueError("unknown tool")
                    if spending_stop():
                        termination = spending_stop()
                        break
                    if remaining() <= 0:
                        termination = "cap:time"
                        break
                    dispatched += 1
                    entry["dispatched"] = True
                    tool_costs.append(None)  # Failed dispatch does not prove zero service charge.
                    reply = backend.call(name, arguments, timeout=remaining())
                    if not isinstance(reply, ToolReply):
                        raise ValueError("backend must return ToolReply")
                    tool_costs[-1] = _charge(reply.cost_usd)
                    body = _encode(reply.value)
                    entry.update(status="returned", result_sha256=fingerprint(reply.value), response_bytes=len(body.encode()))
                except Exception as error:
                    entry.update(status="error", error_type=type(error).__name__)
                    body = _encode({"status": "error", "error_type": type(error).__name__,
                                    "instruction": "Repair the request or submit an answer; no automatic tool retry occurred."})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": body})
            if answer is not None or termination in {"cap:tools", "cap:time", "cap:tokens", "cap:cost", "cost_usage_unmeasured", "usage_unmeasured", "episode_commit_error", "episode_reveal_error"}:
                break
            if not tokens_known:
                termination = "usage_unmeasured"
                break
    except Exception as error:
        termination = "driver_error"
        trace.append({"status": "error", "error_type": type(error).__name__})
    finally:
        if client is not None:
            try:
                tokens, response_tokens, tokens_known, llm_cost = _usage(client)
            except Exception:
                tokens_known, llm_cost = False, None
        for resource in (backend, client):
            if callable(getattr(resource, "close", None)):
                try:
                    resource.close()
                except Exception as error:
                    cleanup_errors.append(type(error).__name__)

    elapsed = clock() - started
    cost = None
    if llm_cost is not None and tool_costs and not cleanup_errors and all(value is not None for value in [*tool_costs, *environment_costs]):
        try:
            cost = math.fsum([llm_cost, *tool_costs, *environment_costs])
        except OverflowError:
            pass
    known = ["tool_calls", "latency_seconds"]
    if tokens_known:
        known.extend(["cumulative_tokens", "response_tokens"])
    if cost is not None:
        known.append("cost_usd")
    if elapsed > limits["timeout_seconds"]:
        answer, termination = None, "cap:time"
    if cost_limit is not None and cost is not None and cost > cost_limit:
        answer, termination = None, "cap:cost"
    answer = answer or Observation(case_id=case_id, status="error", support="abstained")
    # A normal submission with complete token counters establishes no cap hit.
    # Other exits cannot attest compliance of work whose usage was not reported.
    budget_exceeded = (True if termination.startswith("cap:") else False
                       if termination == "submitted" and not cleanup_errors else None)
    metadata = {"resource_fields": known, "termination": termination,
                       **({"episode_checkpoints": journey.records} if journey else {}),
                       "budget_exceeded": budget_exceeded,
                       **({"error": termination} if answer.status == "error" else {}),
                       "model_rounds": rounds, "dispatched_tool_calls": dispatched,
                       "tool_inventory": inventory, "tool_inventory_sha256": fingerprint(inventory),
                       "backend_provenance": backend_provenance,
                       "trace": trace, "cleanup_errors": cleanup_errors, "limits": limits,
                       "token_limit_basis": "provider_reported_postresponse_stop_not_preemptive_prompt_quota",
                       "token_overrun": max(0, tokens - limits["max_tokens"]),
                       "wall_overrun_seconds": max(0, elapsed - limits["timeout_seconds"]),
                       **({"reported_cost_overrun_usd": max(0, cost - cost_limit) if cost is not None else None,
                           "cost_limit_basis": "reported_service_cost_stop_not_provider_enforced_dollar_ceiling"} if cost_limit is not None else {}),
                       "llm_cost_usd": llm_cost, "tool_service_costs_usd": tool_costs,
                       "environment_service_costs_usd": environment_costs, "environment_events": environment_events,
                       "cost_scope": "reported_llm_and_tool_service_charges_excludes_infrastructure",
                       "leakage_audit": "not_provided"}
    return replace(answer, tool_calls=calls, cumulative_tokens=tokens, response_tokens=response_tokens,
                   cost_usd=cost, latency_seconds=elapsed, metadata=metadata)
