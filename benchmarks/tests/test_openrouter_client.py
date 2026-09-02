"""Tests for the OpenRouter client's truncation handling.

Reasoning models spend the completion budget on hidden reasoning
tokens; when the budget runs out first, OpenRouter returns a choice
with empty content and ``finish_reason: "length"``. Scoring that as an
answer would record a wrong answer the model never gave.
"""

import json
import sys
import threading
import time
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.openrouter import (  # noqa: E402
    MAX_TOKENS_CEILING,
    OpenRouterClient,
    OpenRouterError,
)


def _response(content, finish_reason="stop"):
    return {
        "choices": [{
            "message": {"content": content, "role": "assistant"},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.001},
    }


class _Transport:
    """Records the budget of every request and replays canned replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.budgets = []
        self.calls = []

    def __call__(self, client, messages, **kwargs):
        self.budgets.append(kwargs["max_tokens"])
        self.calls.append(dict(kwargs))
        payload = self.replies.pop(0) if self.replies else _response("late")
        client._account(payload)
        from benchmarks.common.openrouter import _to_namespace

        return _to_namespace(payload)


def _client(monkeypatch, replies, **kwargs):
    client = OpenRouterClient("test/model", api_key="k", **kwargs)
    transport = _Transport(replies)
    monkeypatch.setattr(
        OpenRouterClient, "_request",
        lambda self, messages, **kw: transport(self, messages, **kw),
    )
    return client, transport


MESSAGES = [{"role": "user", "content": "forecast"}]


def test_complete_response_is_returned_without_escalation(monkeypatch):
    client, transport = _client(monkeypatch, [_response('{"answer": 1}')],
                                max_tokens=8000)
    assert client.completions(MESSAGES) == ['{"answer": 1}']
    assert transport.budgets == [8000]
    assert client.truncation_escalations == 0


def test_per_call_transport_policy_is_forwarded(monkeypatch):
    client, transport = _client(monkeypatch, [_response("ok")])
    assert client.completions(
        MESSAGES, request_timeout=12.5, transport_retries=0) == ["ok"]
    # The transport double receives the exact caller policy.  The lower-level
    # tests exercise the real retry loop; this pins the public API seam.
    assert transport.budgets == [client.max_tokens]
    assert transport.calls[0]["request_timeout"] == 12.5
    assert transport.calls[0]["transport_retries"] == 0


def test_empty_truncated_reply_escalates_the_budget(monkeypatch):
    client, transport = _client(
        monkeypatch,
        [_response(None, "length"), _response('{"answer": 2}')],
        max_tokens=8000,
    )
    assert client.completions(MESSAGES) == ['{"answer": 2}']
    assert transport.budgets == [8000, 32000]
    assert client.truncation_escalations == 1


def test_escalated_budget_persists_for_later_calls(monkeypatch):
    client, transport = _client(
        monkeypatch,
        [_response(None, "length"), _response("a"), _response("b")],
        max_tokens=8000,
    )
    client.completions(MESSAGES)
    client.completions(MESSAGES)
    # First call discovers the need; the second starts at the raised budget.
    assert transport.budgets == [8000, 32000, 32000]
    assert client.truncation_escalations == 1


def test_caller_supplied_budget_is_not_persisted(monkeypatch):
    client, transport = _client(
        monkeypatch,
        [_response(None, "length"), _response("a"), _response("b")],
        max_tokens=8000,
    )
    client.chat(MESSAGES, max_tokens=8000)
    client.chat(MESSAGES)
    assert transport.budgets == [8000, 32000, 8000]


def test_empty_reply_at_the_ceiling_raises(monkeypatch):
    client, transport = _client(
        monkeypatch, [_response(None, "length")] * 3, max_tokens=8000,
    )
    with pytest.raises(OpenRouterError) as error:
        client.completions(MESSAGES)
    # Escalates to the ceiling, then stops re-asking.
    assert transport.budgets == [8000, 32000, MAX_TOKENS_CEILING]
    assert "empty completion" in str(error.value)
    assert str(MAX_TOKENS_CEILING) in str(error.value)


def test_truncated_but_non_empty_reply_is_left_alone(monkeypatch):
    # Partial content is the benchmark's business to score, not the
    # client's to re-request.
    client, transport = _client(monkeypatch, [_response("partial", "length")],
                                max_tokens=8000)
    assert client.completions(MESSAGES) == ["partial"]
    assert transport.budgets == [8000]


def test_usage_summary_discloses_escalations(monkeypatch):
    client, _ = _client(
        monkeypatch, [_response(None, "length"), _response("ok")],
        max_tokens=8000,
    )
    client.completions(MESSAGES)
    summary = client.usage_summary
    assert summary["truncation_escalations"] == 1
    assert summary["requests"] == 2


def test_provider_ignoring_n_is_fanned_out_to_singles(monkeypatch):
    # Measured against OpenRouter: providers may return one choice for
    # any requested n. The client must make up the shortfall with
    # concurrent single-sample requests, not hand back a short batch
    # for DirectPrompt to collect one retry at a time.
    replies = [_response(f"sample-{i}") for i in range(5)]
    client, transport = _client(monkeypatch, replies, max_tokens=8000)
    response = client.chat(MESSAGES, n=5)
    contents = sorted(c.message.content for c in response.choices)
    assert contents == sorted(f"sample-{i}" for i in range(5))
    assert [c.index for c in response.choices] == [0, 1, 2, 3, 4]
    assert client.usage_summary["requests"] == 5


def test_provider_top_up_respects_sample_parallelism(monkeypatch):
    active = 0
    peak = 0
    calls = 0
    lock = threading.Lock()

    def request(client, messages, **kwargs):
        nonlocal active, peak, calls
        with lock:
            call = calls
            calls += 1
            active += 1
            peak = max(peak, active)
        try:
            # Every response deliberately contains one choice to exercise
            # the provider-ignores-n top-up path.
            time.sleep(.03 if call else .005)
            payload = _response(f"sample-{call}")
            client._account(payload)
            from benchmarks.common.openrouter import _to_namespace
            return _to_namespace(payload)
        finally:
            with lock:
                active -= 1

    client = OpenRouterClient(
        "test/model", api_key="k", sample_parallelism=2)
    monkeypatch.setattr(OpenRouterClient, "_request", request)

    response = client.chat(MESSAGES, n=7)

    assert peak == 2
    assert len(response.choices) == 7
    assert [choice.index for choice in response.choices] == list(range(7))
    assert client.usage_summary["sample_parallelism"] == 2


def test_sample_parallelism_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        OpenRouterClient("test/model", api_key="k", sample_parallelism=0)


def test_shared_rate_limit_gate_cools_and_staggers_siblings(monkeypatch):
    clock = [100.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.sleep", sleep)
    client = OpenRouterClient(
        "test/model", api_key="k",
        rate_limit_cooldown_seconds=60,
        rate_limit_spacing_seconds=2,
    )

    client._register_rate_limit()
    client._wait_for_provider_slot()
    client._wait_for_provider_slot()

    assert sleeps == [60.0, 2.0]
    assert client.usage_summary["rate_limit_events"] == 1
    assert client.usage_summary["rate_limit_wait_seconds"] == 62.0


def test_sample_cache_replays_without_new_requests(tmp_path, monkeypatch):
    client, _ = _client(
        monkeypatch, [_response(f"sample-{i}") for i in range(5)],
        max_tokens=8000, sample_parallelism=2,
        sample_cache_dir=tmp_path,
    )
    original = client.chat(MESSAGES, n=5)

    resumed = OpenRouterClient(
        "test/model", api_key="k", max_tokens=8000,
        sample_parallelism=2, sample_cache_dir=tmp_path)

    def unexpected_request(*args, **kwargs):
        raise AssertionError("a complete sample bank must not call the provider")

    monkeypatch.setattr(OpenRouterClient, "_request", unexpected_request)
    replayed = resumed.chat(MESSAGES, n=5)

    assert sorted(choice.message.content for choice in replayed.choices) == \
        sorted(choice.message.content for choice in original.choices)
    assert resumed.usage_summary["requests"] == 0
    assert resumed.usage_summary["sample_cache_hits"] == 5


def test_cached_samples_are_consumed_once_per_process(tmp_path, monkeypatch):
    producer, _ = _client(
        monkeypatch, [_response(f"sample-{i}") for i in range(3)],
        max_tokens=8000, sample_parallelism=2,
        sample_cache_dir=tmp_path,
    )
    producer.chat(MESSAGES, n=3)

    resumed = OpenRouterClient(
        "test/model", api_key="k", max_tokens=8000,
        sample_cache_dir=tmp_path)
    monkeypatch.setattr(
        OpenRouterClient, "_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached samples unexpectedly exhausted")),
    )
    first = resumed.chat(MESSAGES, n=2)
    second = resumed.chat(MESSAGES, n=1)

    first_contents = {choice.message.content for choice in first.choices}
    second_contents = {choice.message.content for choice in second.choices}
    assert not first_contents & second_contents
    assert first_contents | second_contents == {
        "sample-0", "sample-1", "sample-2"}


def test_successful_siblings_survive_one_top_up_failure(
        tmp_path, monkeypatch):
    calls = 0
    lock = threading.Lock()

    def flaky(client, messages, **kwargs):
        nonlocal calls
        with lock:
            call = calls
            calls += 1
        if call == 2:
            raise OpenRouterError("one sibling failed")
        payload = _response(f"sample-{call}")
        client._account(payload)
        from benchmarks.common.openrouter import _to_namespace
        return _to_namespace(payload)

    client = OpenRouterClient(
        "test/model", api_key="k", sample_parallelism=4,
        sample_cache_dir=tmp_path)
    monkeypatch.setattr(OpenRouterClient, "_request", flaky)

    with pytest.raises(OpenRouterError, match="one sibling failed"):
        client.chat(MESSAGES, n=5)

    assert len(list(tmp_path.glob("*/choice-*.json"))) == 4

    resumed = OpenRouterClient(
        "test/model", api_key="k", sample_cache_dir=tmp_path)
    transport = _Transport([_response("replacement")])
    monkeypatch.setattr(
        OpenRouterClient, "_request",
        lambda self, messages, **kwargs: transport(self, messages, **kwargs),
    )
    response = resumed.chat(MESSAGES, n=5)
    assert len(response.choices) == 5
    assert transport.calls and len(transport.calls) == 1


def test_full_batch_from_provider_is_not_fanned_out(monkeypatch):
    batch = {
        "choices": [
            {"message": {"content": f"s{i}", "role": "assistant"},
             "finish_reason": "stop"}
            for i in range(3)
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    client, transport = _client(monkeypatch, [batch], max_tokens=8000)
    response = client.chat(MESSAGES, n=3)
    assert len(response.choices) == 3
    assert client.usage_summary["requests"] == 1


def test_client_survives_pickling_for_the_worker_pool():
    # evaluate_all_tasks pickles the baseline to reach its pool; the
    # accounting lock must not travel with it.
    import pickle

    client = OpenRouterClient("test/model", api_key="k")
    clone = pickle.loads(pickle.dumps(client))
    clone._account({"usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    assert clone.total_requests == 1


def test_zero_retries_still_performs_the_initial_request(monkeypatch):
    attempts = []

    def unavailable(*args, **kwargs):
        attempts.append((args, kwargs))
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    client = OpenRouterClient(
        "test/model", api_key="k", max_retries=0, timeout=1)
    with pytest.raises(OpenRouterError, match="after 1 attempts"):
        client._request(MESSAGES, n=1, temperature=None, max_tokens=10,
                        tools=None, tool_choice=None)
    assert len(attempts) == 1
    assert client.usage_summary["requests"] == 0
    assert client.usage_summary["transport_attempts"] == 1


def test_retryable_error_inside_success_body_is_retried(monkeypatch):
    replies = [
        {"error": {"message": "miner timeout or disconnect", "code": 504}},
        _response("recovered"),
    ]

    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps(self.payload).encode()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: Response(replies.pop(0)))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = OpenRouterClient(
        "test/model", api_key="k", max_retries=1, timeout=1)

    response = client._request(
        MESSAGES, n=1, temperature=None, max_tokens=10,
        tools=None, tool_choice=None)

    assert response.choices[0].message.content == "recovered"
    assert client.usage_summary["transport_attempts"] == 2
    assert client.usage_summary["requests"] == 1


def test_reasoning_effort_is_sent_only_when_explicit(monkeypatch):
    payloads = []

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps(_response("ok")).encode()

    def capture(request, **kwargs):
        payloads.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", capture)
    explicit = OpenRouterClient(
        "test/model", api_key="k", max_retries=0, reasoning_effort="low")
    explicit._request(MESSAGES, n=1, temperature=None, max_tokens=10,
                      tools=None, tool_choice=None)
    default = OpenRouterClient("test/model", api_key="k", max_retries=0)
    default._request(MESSAGES, n=1, temperature=None, max_tokens=10,
                     tools=None, tool_choice=None)
    explicit.completions(MESSAGES, reasoning_effort="none")
    assert payloads[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in payloads[1]
    assert payloads[2]["reasoning_effort"] == "none"


def test_absolute_deadline_bounds_trickling_transport(monkeypatch):
    import time

    class SlowResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            time.sleep(.2)
            return b'{"choices": []}'

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *args, **kwargs: SlowResponse())
    client = OpenRouterClient(
        "test/model", api_key="k", max_retries=0, timeout=.02)
    started = time.monotonic()
    with pytest.raises(OpenRouterError, match="absolute request deadline"):
        client._request(MESSAGES, n=1, temperature=None, max_tokens=10,
                        tools=None, tool_choice=None)
    assert time.monotonic() - started < .15
