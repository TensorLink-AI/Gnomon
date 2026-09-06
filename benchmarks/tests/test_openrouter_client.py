"""Independent transport limits and conservative usage accounting."""
import io
import json
import urllib.error

import pytest

from benchmarks.common.openrouter import OpenRouterClient, OpenRouterError


def test_single_attempt_disables_truncation_and_missing_choice_retries(monkeypatch):
    requests = []
    replies = [{"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "cost": 0.1}},
               {"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "cost": 0.1}}]
    def http(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps(replies.pop(0)).encode())
    monkeypatch.setattr("urllib.request.urlopen", http)
    client = OpenRouterClient("test", api_key="unused")
    assert client.chat([], max_tokens=3).choices == []
    assert client.chat([], max_tokens=3).choices[0].finish_reason == "length"
    assert [request["max_tokens"] for request in requests] == [3, 3]
    assert client.total_transport_attempts == 2


def test_single_attempt_never_retries_transient_http_failure(monkeypatch):
    requests = []
    def http(request, **kwargs):
        requests.append(request)
        raise urllib.error.HTTPError("https://example.invalid", 503, "unavailable", {}, io.BytesIO(b"unavailable"))
    monkeypatch.setattr("urllib.request.urlopen", http)
    client = OpenRouterClient("test", api_key="unused")
    with pytest.raises(OpenRouterError):
        client.chat([])
    assert len(requests) == client.total_transport_attempts == 1
    assert client.usage_summary["resource_fields_complete"]["cost"] is False


def test_single_attempt_rejects_multiple_samples_and_bounds_response_bytes(monkeypatch):
    client = OpenRouterClient("test", api_key="unused")
    with pytest.raises(TypeError):
        client.chat([], n=2)
    requests = []
    def http(request, **kwargs):
        requests.append(request)
        return io.BytesIO(b" " * 8_388_609)
    monkeypatch.setattr("urllib.request.urlopen", http)
    with pytest.raises(OpenRouterError, match="byte limit"):
        client.chat([])
    assert len(requests) == 1


@pytest.mark.parametrize("body", [b'{"choices":[],"usage":{},"usage":{}}', b'{"choices":[],"cost":NaN}', b'[]'])
def test_single_attempt_refuses_ambiguous_or_nonfinite_response_json(monkeypatch, body):
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: io.BytesIO(body))
    client = OpenRouterClient("test", api_key="unused")
    with pytest.raises(OpenRouterError, match="strict JSON"):
        client.chat([])
    assert client.total_transport_attempts == 1


def test_missing_usage_is_unknown_not_zero_and_extra_transports_invalidate_totals():
    client = OpenRouterClient("test/model", api_key="unused")
    client.total_transport_attempts = 1
    client._account({"usage": {"prompt_tokens": 10}})
    summary = client.usage_summary
    assert summary["resource_fields_complete"]["prompt_tokens"] is True
    assert summary["resource_fields_complete"]["completion_tokens"] is False
    assert summary["cost_usd"] is None
    assert summary["observed_cost_usd"] == 0
    client.total_transport_attempts = 2  # A failed transport may have consumed resources.
    assert client.usage_summary["resource_fields_complete"]["prompt_tokens"] is False


def test_invalid_usage_values_do_not_reduce_or_poison_observed_totals():
    client = OpenRouterClient("test/model", api_key="unused")
    client.total_transport_attempts = 1
    client._account({"usage": {"prompt_tokens": -1, "completion_tokens": True, "cost": float("nan")}})
    assert client.total_prompt_tokens == client.total_completion_tokens == 0
    assert client.total_cost_usd == 0
    assert not any(client.usage_summary["resource_fields_complete"].values())
    json.dumps(client.usage_summary, allow_nan=False)


def test_small_known_costs_are_not_rounded_to_free():
    client = OpenRouterClient("test/model", api_key="unused")
    client.total_transport_attempts = 1
    client._account({"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 1e-8}})
    assert client.usage_summary["cost_usd"] == 1e-8


def test_usage_has_one_set_of_counters_across_multiple_responses():
    client = OpenRouterClient("test/model", api_key="unused")
    for _ in range(3):
        client.total_transport_attempts += 1
        client._account({"usage": {"prompt_tokens": 2, "completion_tokens": 3, "cost": 0.125}})
    usage = client.usage_summary
    assert usage["prompt_tokens"] == 6 and usage["completion_tokens"] == 9
    assert usage["cost_usd"] == usage["observed_cost_usd"] == 0.375
    assert all(usage["resource_fields_complete"].values())
    assert "current_process_usage" not in usage
    assert not any(key.startswith("current_") for key in vars(client))


def test_overflowed_cost_totals_are_explicitly_unknown_not_nonfinite_json():
    client = OpenRouterClient("test/model", api_key="unused")
    client.total_transport_attempts = 2
    for _ in range(2):
        client._account({"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 1e308}})
    assert client.usage_summary["cost_usd"] is None
    assert client.usage_summary["cost_total_overflow"] is True
    json.dumps(client.usage_summary, allow_nan=False)


def test_reasoning_effort_is_sent_only_when_explicit(monkeypatch):
    payloads = []

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, _limit):
            return b'{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"cost":0.1}}'

    def capture(request, **kwargs):
        payloads.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", capture)
    explicit = OpenRouterClient(
        "test/model", api_key="k", reasoning_effort="low")
    explicit.chat([], max_tokens=10)
    default = OpenRouterClient("test/model", api_key="k")
    default.chat([], max_tokens=10)
    explicit.chat([], reasoning_effort="none")
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
        def read(self, _limit):
            time.sleep(.2)
            return b'{"choices": []}'

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *args, **kwargs: SlowResponse())
    client = OpenRouterClient(
        "test/model", api_key="k", timeout=.02)
    started = time.monotonic()
    with pytest.raises(OpenRouterError, match="absolute request deadline"):
        client.chat([], max_tokens=10)
    assert time.monotonic() - started < .15


@pytest.mark.parametrize("option", ["max_retries", "sample_cache_dir", "sample_parallelism"])
def test_retired_client_options_are_rejected(option):
    with pytest.raises(TypeError):
        OpenRouterClient("test", api_key="unused", **{option: 1})


def test_credentials_are_explicit_even_when_environment_has_a_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-used")
    with pytest.raises(TypeError):
        OpenRouterClient("test")
    with pytest.raises(ValueError, match="explicit API key"):
        OpenRouterClient("test", api_key="")


def test_error_body_never_retries_or_discloses_provider_body(monkeypatch):
    requests = []
    def http(request, **kwargs):
        requests.append(request)
        return io.BytesIO(b'{"error":{"code":504,"message":"private-provider-details"}}')
    monkeypatch.setattr("urllib.request.urlopen", http)
    client = OpenRouterClient("test", api_key="unused")
    with pytest.raises(OpenRouterError, match="^provider returned an error$"):
        client.chat([])
    assert len(requests) == client.total_transport_attempts == 1
    assert client.usage_summary["cost_usd"] is None


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_deadlines_fail_before_transport(timeout):
    client = OpenRouterClient("test", api_key="unused")
    with pytest.raises(ValueError, match="timeout"):
        client.chat([], request_timeout=timeout)
    assert client.total_transport_attempts == 0
