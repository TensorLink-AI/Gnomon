"""Scripted model responses test the loop; they are not agent uplift evidence."""

from dataclasses import asdict
import io
import json

import pytest

from benchmarks.common.openrouter import OpenRouterClient
from benchmarks.workflow.bounded_agent import ToolReply, run_agent
from benchmarks.workflow.schema import Observation

CASE = {"id": "calculation", "question": "Calculate the requested value.",
        "available_at_cutoff": {"series": [3, 7]}, "answer_schema": {"numbers": ["next"]}}
BUDGET = {"max_rounds": 4, "max_tool_calls": 3, "max_tokens": 1000, "timeout_seconds": 10}


def call(name, arguments=None, identifier="call"):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments or {})}}


def submit(number=17):
    return call("submit_answer", {"status": "answered", "support": "supported", "numbers": {"next": number}})


def response(*calls, usage=True, choices=True, finish="stop"):
    value = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": list(calls)},
                           "finish_reason": finish}] if choices else []}
    if usage:
        value["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01}
    return value


class Backend:
    startup_cost_usd = 0

    def __init__(self):
        self.calls = []
        self.closed = False

    def tools(self):
        return [{"name": name, "description": "Example local arithmetic", "inputSchema": {"type": "object"}}
                for name in ("first", "second")]

    def call(self, name, arguments, *, timeout):
        self.calls.append((name, arguments, timeout))
        return ToolReply({"number": 17}, 0)

    def close(self):
        self.closed = True


def execute(monkeypatch, replies, backend=None, **options):
    requests = []
    replies = iter(replies)

    def http(request, **kwargs):
        payload = json.loads(request.data)
        requests.append(payload)
        value = next(replies)
        value = value(payload) if callable(value) else value
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr("urllib.request.urlopen", http)
    client = OpenRouterClient("scripted/model", api_key="unused", max_retries=5)
    backend = backend or Backend()
    result = run_agent(CASE, prompt="Answer only from the supplied case.", budget={**BUDGET, **options.pop("budget", {})},
                       client_factory=lambda: client, backend_factory=lambda: backend, **options)
    assert backend.closed
    Observation.from_dict(asdict(result))
    termination = result.metadata["termination"]
    expected_cap = (True if termination.startswith("cap:") else False
                    if termination == "submitted" and not result.metadata["cleanup_errors"] else None)
    assert result.metadata["budget_exceeded"] is expected_cap
    return result, requests, backend


def test_agent_selects_tools_and_numbers_without_host_rewriting(monkeypatch):
    result, requests, backend = execute(monkeypatch, [response(call("second", {"original": 4})), response(submit(99))])
    assert backend.calls[0][:2] == ("second", {"original": 4})
    assert result.numbers == {"next": 99}  # Tool returned17; the driver must not silently correct the model.
    assert all(request["tool_choice"] == "auto" for request in requests)
    assert result.tool_calls == 1 and result.cumulative_tokens == 30
    assert result.cost_usd == pytest.approx(0.02)
    assert result.temporal_leakage is None
    assert result.metadata["trace"][0]["result_sha256"]


def test_tool_failure_spend_and_error_survive_model_recovery(monkeypatch):
    class Failing(Backend):
        def call(self, name, arguments, **kwargs):
            if name == "first":
                raise RuntimeError("sensitive error text must not enter a trace")
            return super().call(name, arguments, **kwargs)
    result, requests, _ = execute(monkeypatch, [response(call("first")), response(call("second")), response(submit())], Failing())
    assert result.status == "answered" and len(requests) == 3
    assert result.tool_calls == 2 and result.cumulative_tokens == 45
    assert result.cost_usd is None  # The failed dispatch's service charge is unknown.
    assert result.metadata["llm_cost_usd"] == pytest.approx(0.03)
    assert result.metadata["trace"][0]["error_type"] == "RuntimeError"
    assert "sensitive error" not in json.dumps(asdict(result))


@pytest.mark.parametrize("proposed", [call("unknown"), {**call("first"), "function": {
    "name": "first", "arguments": "not JSON"}}])
def test_invalid_requests_consume_attempts_without_dispatch(monkeypatch, proposed):
    result, _, backend = execute(monkeypatch, [response(proposed), response(submit())])
    assert result.tool_calls == 1 and backend.calls == []
    assert result.metadata["dispatched_tool_calls"] == 0
    assert result.status == "answered"


def test_tool_limit_stops_a_batched_response_before_extra_dispatch(monkeypatch):
    result, requests, backend = execute(monkeypatch, [response(call("first", identifier="a"), call("second", identifier="b"))],
                                         budget={"max_tool_calls": 1})
    assert result.status == "error" and result.metadata["termination"] == "cap:tools"
    assert len(backend.calls) == len(requests) == result.tool_calls == 1
    assert result.cost_usd == pytest.approx(0.01)


def test_last_allowed_tool_can_be_followed_by_submission(monkeypatch):
    result, requests, _ = execute(monkeypatch, [response(call("first")), response(submit())], budget={"max_tool_calls": 1})
    assert result.status == "answered"
    assert [tool["function"]["name"] for tool in requests[1]["tools"]] == ["submit_answer"]


def test_round_limit_preserves_charged_unfinished_work(monkeypatch):
    result, requests, backend = execute(monkeypatch, [response(call("first"))], budget={"max_rounds": 1})
    assert result.status == "error" and result.metadata["termination"] == "cap:rounds"
    assert len(backend.calls) == len(requests) == 1 and result.cumulative_tokens == 15


def test_provider_reported_token_overrun_is_not_claimed_preempted(monkeypatch):
    result, requests, backend = execute(monkeypatch, [response(call("first"))], budget={"max_tokens": 14})
    assert requests[0]["max_tokens"] == 14
    assert not backend.calls and result.metadata["token_overrun"] == 1
    assert result.cumulative_tokens == 15 and result.cost_usd == pytest.approx(0.01)
    assert result.metadata["termination"] == "cap:tokens"


def test_exhausted_tokens_do_not_dispatch_work_that_cannot_be_submitted(monkeypatch):
    result, requests, backend = execute(monkeypatch, [response(call("first"))], budget={"max_tokens": 15})
    assert not backend.calls and len(requests) == 1
    assert result.metadata["termination"] == "cap:tokens" and result.metadata["token_overrun"] == 0


@pytest.mark.parametrize("terminal", [False, True])
def test_unknown_usage_stops_more_work_but_does_not_erase_a_delivered_answer(monkeypatch, terminal):
    result, requests, backend = execute(monkeypatch, [response(submit() if terminal else call("first"), usage=False)])
    assert len(requests) == 1 and not backend.calls
    assert result.status == ("answered" if terminal else "error")
    assert "cumulative_tokens" not in result.metadata["resource_fields"] and result.cost_usd is None


def test_answer_cannot_forge_host_accounting_or_leakage_audit(monkeypatch):
    forged = call("submit_answer", {"status": "answered", "support": "supported", "tool_calls": 999,
                                    "temporal_leakage": False})
    result, _, _ = execute(monkeypatch, [response(forged), response(submit())])
    assert result.status == "answered" and result.tool_calls == 0 and result.cumulative_tokens == 30
    assert result.temporal_leakage is None


def test_mixed_submission_never_executes_trailing_actions(monkeypatch):
    result, _, backend = execute(monkeypatch, [response(submit(), call("first", identifier="other")), response(submit())])
    assert not backend.calls and result.tool_calls == 1 and result.status == "answered"


@pytest.mark.parametrize("reply", [response(choices=False), response(finish="length")])
def test_missing_or_truncated_reply_is_one_transport_not_hidden_retries(monkeypatch, reply):
    result, requests, _ = execute(monkeypatch, [reply], budget={"max_rounds": 1})
    assert len(requests) == 1 and result.status == "error" and result.cumulative_tokens == 15


def test_time_overrun_after_tool_return_is_disclosed(monkeypatch):
    now = [0.0]
    class Slow(Backend):
        def call(self, *args, **kwargs):
            result = super().call(*args, **kwargs)
            now[0] = 11
            return result
    result, requests, _ = execute(monkeypatch, [response(call("first"))], Slow(), clock=lambda: now[0], budget={"max_rounds": 1})
    assert result.status == "error" and result.metadata["termination"] == "cap:time"
    assert result.metadata["wall_overrun_seconds"] == 1 and len(requests) == 1


def test_cleanup_failure_is_reported_without_erasing_the_answer(monkeypatch):
    class Closing(Backend):
        def close(self):
            self.closed = True
            raise RuntimeError("private")
    result, _, _ = execute(monkeypatch, [response(submit())], Closing())
    assert result.status == "answered" and result.cost_usd is None
    assert result.metadata["cleanup_errors"] == ["RuntimeError"]


@pytest.mark.parametrize("inventory", [[{"name": "submit_answer"}], [float("nan")], [None], "bad"])
def test_invalid_inventory_fails_before_a_model_request(monkeypatch, inventory):
    backend = Backend()
    backend.tools = lambda: inventory
    result, requests, _ = execute(monkeypatch, [], backend)
    assert requests == [] and result.status == "error"


def test_backend_factory_failure_still_closes_the_created_client():
    client = OpenRouterClient("scripted/model", api_key="unused")
    closed = []
    client.close = lambda: closed.append(True)
    def failure():
        raise RuntimeError("private")
    result = run_agent(CASE, prompt="Answer.", budget=BUDGET, client_factory=lambda: client, backend_factory=failure)
    assert closed == [True] and result.status == "error" and result.cost_usd is None


def test_same_loop_discovers_and_calls_real_mcp_session(monkeypatch, tmp_path):
    from benchmarks.common.mcp import StdioMcpSession
    class Mcp(Backend):
        def __init__(self):
            super().__init__()
            self.session = StdioMcpSession(tmp_path, profile="execution", call_timeout=5)
            self.session.initialize()
        def tools(self):
            return self.session.list_tools()
        def call(self, name, arguments, *, timeout):
            self.session.call_timeout = min(5, timeout)
            return ToolReply(self.session.call_tool(name, arguments), 0)
        def close(self):
            super().close()
            self.session.close()
    result, requests, _ = execute(monkeypatch, [response(call("gnomon_capabilities")), response(submit())], Mcp())
    assert result.status == "answered" and result.metadata["trace"][0]["status"] == "returned"
    forecast = next(tool for tool in result.metadata["tool_inventory"] if tool["name"] == "gnomon_forecast")
    variants = forecast["inputSchema"]["oneOf"]
    assert all("provider" in variant["required"] for variant in variants)
    assert all("input" not in variant["properties"] for variant in variants)
    assert "oracle" not in requests[0]["messages"][1]["content"]


def test_loop_can_quote_a_real_current_provider_result_without_host_recovery(monkeypatch):
    from gnomon import GnomonSession
    class Local(Backend):
        def __init__(self):
            super().__init__()
            self.session = GnomonSession.from_config()
        def tools(self):
            return self.session.tools()
        def call(self, name, arguments, *, timeout):
            return ToolReply(self.session.call(name, arguments), 0)
        def close(self):
            self.closed = True
            self.session.close()
    def read_response(payload):
        tool_result = json.loads(payload["messages"][-1]["content"])
        return response(submit(tool_result["result"]["point"][0]))
    result, _, _ = execute(monkeypatch, [response(call("gnomon_forecast", {
        "provider": "last_value", "request": {"history": [3, 7], "horizon": 1}})), read_response], Local())
    assert result.numbers["next"] == 7 and result.metadata["dispatched_tool_calls"] == 1
