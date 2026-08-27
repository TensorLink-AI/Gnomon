"""A minimal OpenRouter chat client built on the standard library.

All benchmark adapters source their LLM completions from OpenRouter so a
single ``OPENROUTER_API_KEY`` covers every model under evaluation. The
endpoint is not hard-wired, though: it speaks the plain
chat-completions wire format, so any OpenAI-compatible server answers
it — set ``OPENROUTER_BASE_URL`` (or pass ``base_url=``) to evaluate a
model that OpenRouter does not host. The resolved endpoint travels with
the numbers it produced, in ``usage_summary`` and the run manifests.
The
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
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any

#: OpenRouter's own endpoint, and the default. Any OpenAI-compatible
#: endpoint works instead — the wire format this client speaks is the
#: chat-completions one, not an OpenRouter dialect — so the base URL is
#: overridable per client, or process-wide through
#: ``OPENROUTER_BASE_URL`` for adapters that construct their own client.
#: A custom endpoint is provenance, not a detail: every runner records
#: the resolved base URL in its manifest, because "model X scored Y"
#: means something different when X was served from somewhere else.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def resolved_base_url(base_url: str | None = None) -> str:
    """The endpoint to call: explicit argument, else the environment,
    else OpenRouter. Read per call, not once at import, so a test or a
    runner can set the variable after this module is imported."""
    if base_url:
        return base_url
    from_env = os.environ.get("OPENROUTER_BASE_URL", "").strip()
    return from_env or DEFAULT_BASE_URL


#: Backwards-compatible module constant (the import-time resolution).
OPENROUTER_BASE_URL = resolved_base_url()
DEFAULT_TIMEOUT_SECONDS = 600
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Reasoning models spend the completion budget on hidden reasoning
# tokens before writing an answer: when the budget runs out first the
# response is a truncated `finish_reason: "length"` with *empty*
# content. Retry such a call with a larger budget, up to this ceiling,
# instead of handing the caller an answer that is not there.
TRUNCATION_ESCALATION_FACTOR = 4
# Well under the completion limit of current reasoning models (GLM-5.2
# allows 128k), but high enough that a model which reasons for ~20k
# tokens before answering still gets to answer.
MAX_TOKENS_CEILING = 64000


class OpenRouterError(RuntimeError):
    """A request to OpenRouter failed after all retries."""


def _truncated_empty(response: SimpleNamespace) -> bool:
    """True when every choice ran out of budget before writing anything."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    for choice in choices:
        content = getattr(getattr(choice, "message", None), "content", None)
        if content:
            return False
        if getattr(choice, "finish_reason", None) != "length":
            return False
    return True


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
    base_url:
        Any OpenAI-compatible chat-completions endpoint. Defaults to
        ``OPENROUTER_BASE_URL`` from the environment, else OpenRouter's
        own. Whatever it resolves to is reported in ``usage_summary``
        and the runners' manifests: the endpoint that served a model is
        part of what a score means.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 10000,
        max_retries: int = 5,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        if not api_key and "OPENROUTER_API_KEY" not in os.environ:
            from benchmarks.common.envfile import load_env_file

            load_env_file()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = resolved_base_url(base_url).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        # chat() may fan out concurrent single-sample requests when a
        # provider ignores ``n``; accounting must not lose updates.
        self._usage_lock = threading.Lock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.total_requests = 0
        self.total_transport_attempts = 0
        self.truncation_escalations = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        reasoning_effort: str | None = None,
        request_timeout: float | None = None,
        transport_retries: int | None = None,
    ) -> SimpleNamespace:
        """Send one chat-completion request and return the parsed response.

        The response mimics the OpenAI SDK object shape used by the
        official benchmarks: ``.choices[i].message.content``,
        ``.usage.prompt_tokens`` / ``.completion_tokens`` and
        ``.provider``. Retries with exponential backoff on transient
        HTTP failures; raises :class:`OpenRouterError` once exhausted.

        A reasoning model that exhausts the completion budget on hidden
        reasoning returns empty content with ``finish_reason:
        "length"``; that request is retried with a budget escalated up
        to :data:`MAX_TOKENS_CEILING` before the response is returned.
        """
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set. Export it, or put it in a "
                ".env file in the working directory or repository root, "
                "before running a benchmark that queries an LLM."
            )
        caller_budget = max_tokens is not None
        budget = self.max_tokens if max_tokens is None else max_tokens
        while True:
            response = self._request(
                messages, n=n, temperature=temperature, max_tokens=budget,
                tools=tools, tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                request_timeout=request_timeout,
                transport_retries=transport_retries,
            )
            if not _truncated_empty(response) or budget >= MAX_TOKENS_CEILING:
                break
            budget = min(budget * TRUNCATION_ESCALATION_FACTOR,
                         MAX_TOKENS_CEILING)
            self.truncation_escalations += 1
            if not caller_budget:
                # A model that reasons past one budget will do it again:
                # keep the larger budget so only the first call pays for
                # the discovery.
                self.max_tokens = budget
        missing = n - len(response.choices)
        if missing > 0:
            # OpenRouter providers may ignore ``n`` and return a single
            # choice (measured: n=3 -> 1 choice on both BaseTen and
            # DeepInfra for deepseek-v4-flash). DirectPrompt's rejection
            # sampling then collects samples one request at a time and
            # spends its whole retry budget doing it — a 25x slowdown
            # that presents as endpoint degradation. Independent
            # single-sample requests at the same temperature are the
            # same sampling protocol as one n-sample request; issue the
            # shortfall concurrently and merge.
            from concurrent.futures import ThreadPoolExecutor

            def one_more(_: int) -> Any:
                return self.chat(
                    messages, n=1, temperature=temperature,
                    max_tokens=max_tokens, tools=tools,
                    tool_choice=tool_choice,
                    reasoning_effort=reasoning_effort,
                    request_timeout=request_timeout,
                    transport_retries=transport_retries,
                )

            # All singles at once: a wave of 24 multi-minute requests
            # serialised 8 at a time triples the batch latency for no
            # protection — 429s are retryable with backoff.
            with ThreadPoolExecutor(max_workers=min(32, missing)) as pool:
                extras = list(pool.map(one_more, range(missing)))
            merged = list(response.choices)
            for extra in extras:
                merged.extend(extra.choices)
            for index, choice in enumerate(merged):
                choice.index = index
            response.choices = merged
        return response

    def _request(
        self,
        messages: list[dict[str, Any]],
        *,
        n: int,
        temperature: float | None,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        reasoning_effort: str | None = None,
        request_timeout: float | None = None,
        transport_retries: int | None = None,
    ) -> SimpleNamespace:
        """Perform one request, retrying transient HTTP failures."""
        payload = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            # Ask OpenRouter to report token accounting and cost.
            "usage": {"include": True},
        }
        effective_reasoning = (self.reasoning_effort
                               if reasoning_effort is None
                               else reasoning_effort)
        if effective_reasoning is not None:
            payload["reasoning_effort"] = effective_reasoning
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        attempts = (self.max_retries if transport_retries is None
                    else max(0, int(transport_retries))) + 1
        effective_timeout = (self.timeout if request_timeout is None
                             else max(.001, float(request_timeout)))
        for attempt in range(attempts):
            with self._usage_lock:
                self.total_transport_attempts += 1
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/TensorLink-AI/Gnomon",
                    "X-Title": "Gnomon benchmarks",
                },
                method="POST",
            )
            try:
                # ``urlopen(timeout=...)`` is an inactivity timeout, not a
                # request deadline. A chunked provider can trickle bytes and
                # keep resetting it forever. Run the complete transport read
                # in a daemon and impose the benchmark's wall-clock budget on
                # the attempt. The abandoned daemon owns and eventually
                # closes its response; benchmark retry/resume owns recovery.
                result: list[object] = []

                def transport() -> None:
                    try:
                        with urllib.request.urlopen(
                                request, timeout=effective_timeout) as raw:
                            result.append(json.loads(
                                raw.read().decode("utf-8")))
                    except BaseException as error:  # handed back to caller
                        result.append(error)

                worker = threading.Thread(
                    target=transport, name="gnomon-llm-transport", daemon=True)
                worker.start()
                worker.join(effective_timeout)
                if worker.is_alive():
                    raise TimeoutError(
                        f"absolute request deadline exceeded after "
                        f"{effective_timeout}s")
                if not result:
                    raise TimeoutError("transport ended without a result")
                if isinstance(result[0], BaseException):
                    raise result[0]
                parsed = result[0]
                if not isinstance(parsed, dict):
                    raise json.JSONDecodeError(
                        "response must be a JSON object", repr(parsed), 0)
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
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
        raise OpenRouterError(
            f"OpenRouter request failed after {attempts} attempts: {last_error}"
        )

    def completions(self, messages: list[dict[str, Any]], *, n: int = 1,
                    temperature: float | None = None,
                    max_tokens: int | None = None,
                    reasoning_effort: str | None = None,
                    request_timeout: float | None = None,
                    transport_retries: int | None = None) -> list[str]:
        """Convenience wrapper returning just the completion texts.

        An empty completion is an error, not an answer: returning it
        would reach a scorer as a missing or unparseable response and be
        recorded as a wrong answer the model never gave.
        """
        response = self.chat(messages, n=n, temperature=temperature,
                             max_tokens=max_tokens,
                             reasoning_effort=reasoning_effort,
                             request_timeout=request_timeout,
                             transport_retries=transport_retries)
        texts = [choice.message.content for choice in response.choices]
        if any(not text for text in texts):
            reasons = [getattr(choice, "finish_reason", None)
                       for choice in response.choices]
            raise OpenRouterError(
                f"{self.model} returned an empty completion "
                f"(finish_reason={reasons}). Reasoning models can spend the "
                f"whole budget on reasoning tokens; the request was already "
                f"retried up to max_tokens={MAX_TOKENS_CEILING}."
            )
        return texts

    def __getstate__(self) -> dict[str, Any]:
        # evaluate_all_tasks pickles the baseline (and this client with
        # it) to reach its worker pool; a lock cannot cross the process
        # boundary. Each worker gets a fresh lock, which is also
        # correct: accounting is per-process.
        state = self.__dict__.copy()
        del state["_usage_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._usage_lock = threading.Lock()

    def _account(self, parsed: dict[str, Any]) -> None:
        usage = parsed.get("usage") or {}
        with self._usage_lock:
            self.total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.total_completion_tokens += int(
                usage.get("completion_tokens") or 0)
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                self.total_cost_usd += float(cost)
            self.total_requests += 1

    @property
    def usage_summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            # Provenance, not decoration: the same model id served from a
            # different endpoint is a different measurement.
            "base_url": self.base_url,
            "requests": self.total_requests,
            "transport_attempts": self.total_transport_attempts,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            # Disclosed, not hidden: requests that had to be re-sent with
            # a larger budget because the model reasoned past the first.
            "truncation_escalations": self.truncation_escalations,
        }


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Every top-level JSON object embedded in free-form LLM output, in
    order.

    A greedy ``\\{.*\\}`` regex mis-parses the common case of a valid
    JSON answer followed by prose that happens to contain a brace (or
    preceded by an echoed packet): the span from the first ``{`` to the
    last ``}`` is not JSON, and a correct answer gets scored as invalid.
    Scanning balanced top-level spans lets a caller validate each
    candidate and keep the first that has the expected shape."""
    found: list[dict[str, Any]] = []
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
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    found.append(parsed)
                start = None
    return found


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
