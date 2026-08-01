"""A minimal OpenRouter chat client built on the standard library.

All benchmark adapters source their LLM completions from OpenRouter so a
single ``OPENROUTER_API_KEY`` covers every model under evaluation. The
client intentionally mirrors the response surface the official CiK
``DirectPrompt`` baseline expects (``choices[i].message.content``,
``usage.prompt_tokens``, ``usage.completion_tokens``, ``provider``), so it
can be dropped in as that baseline's client without touching the
benchmark's own prompting or rejection-sampling code.

No third-party dependency is required: requests go through
``urllib.request`` and honour the standard proxy environment variables.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 600
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    """A request to OpenRouter failed after all retries."""


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


class OpenRouterClient:
    """Chat-completion client for any model hosted on OpenRouter.

    Parameters
    ----------
    model:
        Full OpenRouter model identifier, e.g. ``openai/gpt-4o``,
        ``anthropic/claude-sonnet-4`` or ``qwen/qwen-2.5-72b-instruct``.
    api_key:
        Defaults to the ``OPENROUTER_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float = 1.0,
        max_tokens: int = 10000,
        max_retries: int = 5,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.total_requests = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> SimpleNamespace:
        """Send one chat-completion request and return the parsed response.

        The response mimics the OpenAI SDK object shape used by the
        official benchmarks: ``.choices[i].message.content``,
        ``.usage.prompt_tokens`` / ``.completion_tokens`` and
        ``.provider``. Retries with exponential backoff on transient
        HTTP failures; raises :class:`OpenRouterError` once exhausted.
        """
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set. Export it before running a "
                "benchmark that queries an LLM."
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            # Ask OpenRouter to report token accounting and cost.
            "usage": {"include": True},
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/TensorLink-AI/Aion",
                    "X-Title": "Aion benchmarks",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as raw:
                    parsed = json.loads(raw.read().decode("utf-8"))
                if "error" in parsed and "choices" not in parsed:
                    raise OpenRouterError(str(parsed["error"]))
                self._account(parsed)
                response = _to_namespace(parsed)
                if not getattr(response, "provider", None):
                    response.provider = "unknown"
                return response
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code not in RETRYABLE_STATUS:
                    detail = error.read().decode("utf-8", errors="replace")[:500]
                    raise OpenRouterError(
                        f"OpenRouter returned HTTP {error.code}: {detail}"
                    ) from error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
            time.sleep(2**attempt)
        raise OpenRouterError(
            f"OpenRouter request failed after {self.max_retries} attempts: {last_error}"
        )

    def completions(self, messages: list[dict[str, Any]], *, n: int = 1) -> list[str]:
        """Convenience wrapper returning just the completion texts."""
        response = self.chat(messages, n=n)
        return [choice.message.content for choice in response.choices]

    def _account(self, parsed: dict[str, Any]) -> None:
        usage = parsed.get("usage") or {}
        self.total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.total_completion_tokens += int(usage.get("completion_tokens") or 0)
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            self.total_cost_usd += float(cost)
        self.total_requests += 1

    @property
    def usage_summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "requests": self.total_requests,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
        }


def extract_json_array(text: str) -> list[Any]:
    """Extract the first JSON array embedded in free-form LLM output.

    Benchmark adapters ask models for JSON, but models wrap answers in
    prose or code fences; this pulls out the first parseable top-level
    array. Raises ``ValueError`` when none exists.
    """
    depth = 0
    start = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, list):
                    return parsed
                start = None
    raise ValueError("No JSON array found in model output.")
