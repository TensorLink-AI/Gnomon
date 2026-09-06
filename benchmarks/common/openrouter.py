"""Single-request chat transport for the matched agent evaluation.

Budgets, retries and durable receipts belong to the agent loop and runner.
This client never replays cached answers, fans out samples or increases budgets.
Credentials and endpoint selection belong to the caller.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RESPONSE_BYTES = 1_048_576


def _measured_usage(usage):
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "cost"):
        value = usage.get(key) if isinstance(usage, dict) else None
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if key != "cost":
            valid = valid and type(value) is int
        if valid:
            result[key] = value
    return result


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


class OpenRouterError(RuntimeError):
    """The single transport attempt failed."""


class OpenRouterClient:
    """A bounded, explicitly configured OpenAI-compatible chat client."""

    def __init__(self, model: str, *, api_key: str,
                 base_url: str = DEFAULT_BASE_URL, temperature: float = 1.0,
                 max_tokens: int = 10000, timeout: float = 600,
                 reasoning_effort: str | None = None, request_opener: Any = None):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("an explicit API key is required")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.request_opener = request_opener
        self._usage_lock = threading.Lock()
        self.unmeasured_usage_fields: set[str] = set()
        self.total_prompt_tokens = self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.total_requests = self.total_transport_attempts = 0
        self.total_request_latency_seconds = 0.0

    def chat(self, messages: list[dict[str, Any]], *,
             temperature: float | None = None, max_tokens: int | None = None,
             reasoning_effort: str | None = None, tools: list | None = None,
             tool_choice: str | None = None, request_timeout: float | None = None):
        timeout = self.timeout if request_timeout is None else request_timeout
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("request timeout must be finite and positive")
        payload = {
            "model": self.model, "messages": messages, "n": 1,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "usage": {"include": True},
        }
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        if effort is not None:
            payload["reasoning_effort"] = effort
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        # Socket timeouts alone allow a trickling server to run indefinitely.
        # The daemon owns its response and closes it even after our deadline.
        result: list[object] = []

        def transport():
            try:
                opener = self.request_opener.open if self.request_opener is not None else urllib.request.urlopen
                with opener(request, timeout=timeout) as raw:
                    body = raw.read(RESPONSE_BYTES + 1)
                    if len(body) > RESPONSE_BYTES:
                        raise OpenRouterError("LLM response exceeded the byte limit")
                    from benchmarks.workflow.agent_metrics import _decode_record
                    try:
                        result.append(_decode_record(body.decode("utf-8")))
                    except (ValueError, UnicodeError) as error:
                        raise OpenRouterError("LLM response is not a bounded strict JSON object") from error
            except BaseException as error:
                result.append(error)

        started = time.monotonic()
        self.total_transport_attempts += 1
        try:
            worker = threading.Thread(target=transport, name="gnomon-llm-transport", daemon=True)
            worker.start()
            worker.join(timeout)
            if worker.is_alive():
                raise OpenRouterError(f"absolute request deadline exceeded after {timeout}s")
            if not result:
                raise OpenRouterError("transport ended without a result")
            if isinstance(result[0], BaseException):
                raise result[0]
            parsed = result[0]
            if "error" in parsed:
                raise OpenRouterError("provider returned an error")
            self._account(parsed)
            response = _to_namespace(parsed)
            if not getattr(response, "provider", None):
                response.provider = "unknown"
            return response
        except urllib.error.HTTPError as error:
            error.close()
            raise OpenRouterError(f"LLM HTTP error {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise OpenRouterError("LLM transport failed") from error
        finally:
            self.total_request_latency_seconds += time.monotonic() - started

    def _account(self, parsed: dict[str, Any]) -> None:
        usage = parsed.get("usage") or {}
        measured = _measured_usage(usage)
        with self._usage_lock:
            self.unmeasured_usage_fields.update({"prompt_tokens", "completion_tokens", "cost"} - set(measured))
            self.total_prompt_tokens += measured.get("prompt_tokens", 0)
            self.total_completion_tokens += measured.get("completion_tokens", 0)
            self.total_cost_usd += measured.get("cost", 0.0)
            self.total_requests += 1

    @property
    def usage_summary(self) -> dict[str, Any]:
        complete = {key: key not in self.unmeasured_usage_fields
                    and self.total_requests == self.total_transport_attempts
                    for key in ("prompt_tokens", "completion_tokens", "cost")}
        cost_finite = math.isfinite(self.total_cost_usd)
        complete["cost"] = complete["cost"] and cost_finite
        return {
            "resource_fields_complete": complete,
            "resource_accounting_basis": "counters_are_observed_lower_bounds_when_incomplete",
            "observed_cost_usd": self.total_cost_usd if cost_finite else None,
            "cost_total_overflow": not cost_finite,
            "model": self.model,
            # Provenance, not decoration: the same model id served from a
            # different endpoint is a different measurement.
            "base_url": self.base_url,
            "requests": self.total_requests,
            "transport_attempts": self.total_transport_attempts,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cost_usd": self.total_cost_usd if complete["cost"] else None,
            "request_latency_seconds": (self.total_request_latency_seconds
                                        if math.isfinite(self.total_request_latency_seconds) else None),
        }
