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
import hashlib
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
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


class _TransientProviderError(OpenRouterError):
    """A retryable error returned inside an otherwise successful HTTP body."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _provider_error_is_retryable(value: Any) -> bool:
    """Recognize OpenAI-wire transient errors without retrying semantics.

    Some compatible gateways return HTTP 200 with an ``error`` object whose
    nested code is the real upstream status. Treat only standard transport
    statuses as retryable; invalid requests and content errors remain loud.
    """
    if not isinstance(value, dict):
        return False
    try:
        code = int(value.get("code"))
    except (TypeError, ValueError):
        return False
    return code in RETRYABLE_STATUS


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


def _to_plain(value: Any) -> Any:
    """Convert an OpenAI-shaped namespace back to JSON-safe values."""
    if isinstance(value, SimpleNamespace):
        return {key: _to_plain(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
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
    sample_parallelism:
        Maximum concurrent single-sample requests used when a provider
        ignores ``n``. This is deliberately independent of a benchmark's
        task-level parallelism and defaults to a conservative four.
    sample_cache_dir:
        Optional per-run crash-safe sample bank. Completed choices are
        atomically retained and consumed only once per client process, so a
        killed benchmark case resumes by requesting only its missing samples.
    rate_limit_cooldown_seconds:
        Shared cooldown applied after any HTTP 429 (default: 60 seconds).
    rate_limit_spacing_seconds:
        Minimum spacing between sibling request starts after a 429, preventing
        deterministic retry backoff from creating another synchronized burst.
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
        sample_parallelism: int = 4,
        sample_cache_dir: str | Path | None = None,
        rate_limit_cooldown_seconds: float = 60.0,
        rate_limit_spacing_seconds: float = 2.0,
    ) -> None:
        if int(sample_parallelism) < 1:
            raise ValueError("sample_parallelism must be at least 1")
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
        self.sample_parallelism = int(sample_parallelism)
        self.sample_cache_dir = (
            Path(sample_cache_dir) if sample_cache_dir is not None else None)
        self.rate_limit_cooldown_seconds = max(
            0.0, float(rate_limit_cooldown_seconds))
        self.rate_limit_spacing_seconds = max(
            0.0, float(rate_limit_spacing_seconds))
        # chat() may fan out concurrent single-sample requests when a
        # provider ignores ``n``; accounting must not lose updates.
        self._usage_lock = threading.Lock()
        self._sample_cache_lock = threading.Lock()
        self._sample_cache_consumed: set[str] = set()
        self._sample_usage_consumed: set[str] = set()
        self._sample_cache_next_sequence: dict[str, int] = {}
        self._rate_limit_lock = threading.Lock()
        self._rate_limit_not_before = 0.0
        self._rate_limit_next_start = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.total_requests = 0
        self.total_transport_attempts = 0
        self.total_request_latency_seconds = 0.0
        self.restored_requests = 0
        self.truncation_escalations = 0
        self.sample_cache_hits = 0
        self.sample_cache_writes = 0
        self.rate_limit_events = 0
        self.rate_limit_wait_seconds = 0.0
        self.sample_cache_accounting_complete = True
        self._restore_all_cached_usage()

    def _register_rate_limit(self, retry_after: float | None = None) -> None:
        """Apply one shared cooldown to all sibling request workers."""
        cooldown = self.rate_limit_cooldown_seconds
        if retry_after is not None:
            cooldown = max(cooldown, max(0.0, retry_after))
        now = time.monotonic()
        with self._rate_limit_lock:
            self._rate_limit_not_before = max(
                self._rate_limit_not_before, now + cooldown)
            self._rate_limit_next_start = max(
                self._rate_limit_next_start, self._rate_limit_not_before)
        with self._usage_lock:
            self.rate_limit_events += 1

    def _wait_for_provider_slot(self) -> None:
        """After any 429, stagger every sibling's next request start."""
        while True:
            with self._rate_limit_lock:
                now = time.monotonic()
                ready = max(
                    self._rate_limit_not_before, self._rate_limit_next_start)
                delay = ready - now
                if delay <= 0:
                    if self._rate_limit_next_start > 0:
                        self._rate_limit_next_start = (
                            now + self.rate_limit_spacing_seconds)
                    return
            time.sleep(delay)
            with self._usage_lock:
                self.rate_limit_wait_seconds += delay

    def _sample_key(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        reasoning_effort: str | None,
    ) -> str:
        """Identity of one exchangeable sample distribution."""
        effective_reasoning = (self.reasoning_effort
                               if reasoning_effort is None
                               else reasoning_effort)
        payload = {
            "schema_version": 1,
            "model": self.model,
            "base_url": self.base_url,
            "messages": messages,
            "temperature": (self.temperature if temperature is None
                            else temperature),
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
            "reasoning_effort": effective_reasoning,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _claim_cached_samples(
            self, key: str, count: int) -> tuple[list[Any], list[str]]:
        if self.sample_cache_dir is None or count <= 0:
            return [], []
        directory = self.sample_cache_dir / key
        choices: list[Any] = []
        providers: list[str] = []
        with self._sample_cache_lock:
            records = []
            for path in directory.glob("choice-*.json"):
                identity = str(path)
                if identity in self._sample_cache_consumed:
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    choice = record["choice"]
                    if not isinstance(choice, dict):
                        continue
                except (OSError, KeyError, json.JSONDecodeError):
                    continue
                sequence = record.get("sequence")
                if (isinstance(sequence, bool)
                        or not isinstance(sequence, int) or sequence < 0):
                    # Legacy caches predate durable provider-return order.
                    # Keep their historical filename order ahead of any
                    # later top-up. New caches use the exact live merge order.
                    order = (0, path.name)
                else:
                    order = (1, sequence, path.name)
                records.append((order, path, record, choice))
            for _, path, record, choice in sorted(
                    records, key=lambda item: item[0]):
                identity = str(path)
                self._sample_cache_consumed.add(identity)
                choices.append(_to_namespace(choice))
                providers.append(str(record.get("provider") or "cache"))
                if len(choices) >= count:
                    break
            self.sample_cache_hits += len(choices)
        return choices, providers

    def _reserve_sample_sequences(self, key: str, count: int) -> int:
        """Reserve stable choice positions for one provider response."""
        if count <= 0:
            return 0
        with self._sample_cache_lock:
            if key not in self._sample_cache_next_sequence:
                maximum = -1
                directory = self.sample_cache_dir / key
                for path in directory.glob("choice-*.json"):
                    try:
                        sequence = json.loads(path.read_text(
                            encoding="utf-8")).get("sequence")
                    except (OSError, json.JSONDecodeError):
                        continue
                    if (isinstance(sequence, int)
                            and not isinstance(sequence, bool)
                            and sequence >= 0):
                        maximum = max(maximum, sequence)
                self._sample_cache_next_sequence[key] = maximum + 1
            start = self._sample_cache_next_sequence[key]
            self._sample_cache_next_sequence[key] += count
            return start

    def _restore_all_cached_usage(self) -> None:
        """Restore cumulative request economics for this case cache.

        A resumed benchmark must not look cheaper merely because its earlier
        completions came from disk. Request ledgers are written before their
        choices, so every reusable sample has durable provider accounting.
        Legacy choice-only caches remain usable for recovery, but are marked
        incomplete so score assemblers can fail closed on their economics.
        """
        if self.sample_cache_dir is None or not self.sample_cache_dir.exists():
            return
        with self._sample_cache_lock:
            request_paths = {
                path.stem.removeprefix("request-"): path
                for path in self.sample_cache_dir.glob("*/request-*.json")
            }
            for path in sorted(request_paths.values()):
                identity = str(path)
                if identity in self._sample_usage_consumed:
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    successful = record["successful"]
                    integer_fields = (
                        record["requests"], record["transport_attempts"],
                        record["prompt_tokens"], record["completion_tokens"],
                    )
                    if (not isinstance(successful, bool)
                            or any(isinstance(value, bool)
                                   or not isinstance(value, int)
                                   for value in integer_fields)):
                        raise ValueError("invalid request accounting types")
                    (requests, transport_attempts, prompt_tokens,
                     completion_tokens) = integer_fields
                    cost_usd = float(record["cost_usd"])
                    latency = float(record["request_latency_seconds"])
                    if min(requests, transport_attempts, prompt_tokens,
                           completion_tokens) < 0 or cost_usd < 0 or latency < 0:
                        raise ValueError("negative request accounting")
                    if requests != int(successful):
                        raise ValueError("request success accounting mismatch")
                except (OSError, KeyError, TypeError, ValueError,
                        json.JSONDecodeError):
                    self.sample_cache_accounting_complete = False
                    continue
                self._sample_usage_consumed.add(identity)
                with self._usage_lock:
                    self.total_requests += requests
                    self.restored_requests += requests
                    self.total_transport_attempts += transport_attempts
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_cost_usd += cost_usd
                    self.total_request_latency_seconds += latency
            for path in self.sample_cache_dir.glob("*/choice-*.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    request_id = str(record["request_id"])
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    self.sample_cache_accounting_complete = False
                    continue
                if request_id not in request_paths:
                    self.sample_cache_accounting_complete = False

    def _persist_request_record(
        self,
        key: str | None,
        *,
        successful: bool,
        usage: Any = None,
        provider: str = "unknown",
        transport_attempts: int = 1,
        request_latency_seconds: float = 0.0,
    ) -> str | None:
        """Durably record one logical provider request before its choices."""
        if self.sample_cache_dir is None or key is None:
            return None
        plain_usage = _to_plain(usage) if usage is not None else {}
        if not isinstance(plain_usage, dict):
            plain_usage = {}
        request_id = uuid.uuid4().hex
        record = {
            "schema_version": 1,
            "successful": bool(successful),
            "requests": int(bool(successful)),
            "transport_attempts": max(0, int(transport_attempts)),
            "prompt_tokens": max(0, int(plain_usage.get("prompt_tokens") or 0)),
            "completion_tokens": max(
                0, int(plain_usage.get("completion_tokens") or 0)),
            "cost_usd": max(0.0, float(plain_usage.get("cost") or 0.0)),
            "request_latency_seconds": max(
                0.0, float(request_latency_seconds)),
            "provider": provider,
        }
        directory = self.sample_cache_dir / key
        path = directory / f"request-{request_id}.json"
        temporary = directory / f".request-{request_id}.tmp"
        with self._sample_cache_lock:
            directory.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
            # The live client already counted this request. Marking the ledger
            # consumed prevents a later cache claim in the same process from
            # counting it twice.
            self._sample_usage_consumed.add(str(path))
        return request_id

    def _store_cached_samples(self, key: str, response: SimpleNamespace,
                              *, sequence_start: int) -> None:
        if self.sample_cache_dir is None:
            return
        directory = self.sample_cache_dir / key
        provider = str(getattr(response, "provider", None) or "unknown")
        request_id = getattr(response, "_gnomon_cache_request_id", None)
        if request_id is None:
            request_id = self._persist_request_record(
                key, successful=True, usage=getattr(response, "usage", None),
                provider=provider,
                transport_attempts=int(getattr(
                    response, "_gnomon_transport_attempts", 1)),
                request_latency_seconds=float(getattr(
                    response, "_gnomon_request_latency_seconds", 0.0)),
            )
        for offset, choice in enumerate(
                list(getattr(response, "choices", None) or [])):
            record = {"choice": _to_plain(choice), "provider": provider,
                      "request_id": request_id,
                      "sequence": sequence_start + offset}
            with self._sample_cache_lock:
                directory.mkdir(parents=True, exist_ok=True)
                suffix = uuid.uuid4().hex
                path = directory / f"choice-{suffix}.json"
                temporary = directory / f".choice-{suffix}.tmp"
                temporary.write_text(
                    json.dumps(record, sort_keys=True) + "\n",
                    encoding="utf-8")
                temporary.replace(path)
                self._sample_cache_consumed.add(str(path))
                self.sample_cache_writes += 1

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
        _skip_sample_cache: bool = False,
        _sample_cache_key: str | None = None,
        _sample_cache_sequence_start: int | None = None,
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
        cache_key = _sample_cache_key
        cached_choices: list[Any] = []
        cached_providers: list[str] = []
        if self.sample_cache_dir is not None and not _skip_sample_cache:
            cache_key = self._sample_key(
                messages, temperature=temperature, max_tokens=budget,
                tools=tools, tool_choice=tool_choice,
                reasoning_effort=reasoning_effort)
            cached_choices, cached_providers = self._claim_cached_samples(
                cache_key, n)
        request_n = n - len(cached_choices)
        if request_n <= 0:
            for index, choice in enumerate(cached_choices):
                choice.index = index
            return SimpleNamespace(
                choices=cached_choices,
                usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                provider=(cached_providers[0] if cached_providers else "cache"),
            )
        sequence_start = _sample_cache_sequence_start
        if cache_key is not None and sequence_start is None:
            sequence_start = self._reserve_sample_sequences(
                cache_key, request_n)
        while True:
            response = self._request(
                messages, n=request_n, temperature=temperature,
                max_tokens=budget,
                tools=tools, tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                request_timeout=request_timeout,
                transport_retries=transport_retries,
                sample_cache_key=cache_key,
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
        if cache_key is not None:
            self._store_cached_samples(
                cache_key, response, sequence_start=int(sequence_start or 0))
        missing = request_n - len(response.choices)
        if missing > 0:
            # OpenRouter providers may ignore ``n`` and return a single
            # choice (measured: n=3 -> 1 choice on both BaseTen and
            # DeepInfra for deepseek-v4-flash). DirectPrompt's rejection
            # sampling then collects samples one request at a time and
            # spends its whole retry budget doing it — a 25x slowdown
            # that presents as endpoint degradation. Independent
            # single-sample requests at the same temperature are the
            # same sampling protocol as one n-sample request; issue the
            # shortfall with explicit bounded concurrency and merge.
            from concurrent.futures import ThreadPoolExecutor

            def one_more(index: int) -> Any:
                extra = self.chat(
                    messages, n=1, temperature=temperature,
                    max_tokens=max_tokens, tools=tools,
                    tool_choice=tool_choice,
                    reasoning_effort=reasoning_effort,
                    request_timeout=request_timeout,
                    transport_retries=transport_retries,
                    _skip_sample_cache=True,
                    _sample_cache_key=cache_key,
                    _sample_cache_sequence_start=(
                        int(sequence_start or 0)
                        + len(response.choices) + index),
                )
                # Recursive chat() persists this successful sibling before
                # returning. A later sibling may exhaust retries, or the case
                # supervisor may kill the worker at its deadline; neither
                # discards samples that already completed.
                return extra

            # Task-level ``jobs=1`` does not constrain this inner fan-out.
            # Keep it independently bounded: large bursts can trip provider
            # limits or exhaust a machine even though the outer benchmark is
            # nominally serial. ``pool.map`` preserves request order.
            workers = min(self.sample_parallelism, missing)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                extras = list(pool.map(one_more, range(missing)))
            merged = cached_choices + list(response.choices)
            for extra in extras:
                merged.extend(extra.choices)
            for index, choice in enumerate(merged):
                choice.index = index
            response.choices = merged
        elif cached_choices:
            response.choices = cached_choices + list(response.choices)
            for index, choice in enumerate(response.choices):
                choice.index = index
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
        sample_cache_key: str | None = None,
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
        request_started = time.monotonic()
        attempts_used = 0
        for attempt in range(attempts):
            attempts_used = attempt + 1
            self._wait_for_provider_slot()
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
                    error = parsed["error"]
                    if _provider_error_is_retryable(error):
                        raise _TransientProviderError(
                            str(error), status_code=int(error["code"]))
                    raise OpenRouterError(str(error))
                self._account(parsed)
                response = _to_namespace(parsed)
                if not getattr(response, "provider", None):
                    response.provider = "unknown"
                request_latency = time.monotonic() - request_started
                response._gnomon_request_latency_seconds = request_latency
                response._gnomon_transport_attempts = attempts_used
                response._gnomon_cache_request_id = self._persist_request_record(
                    sample_cache_key, successful=True,
                    usage=getattr(response, "usage", None),
                    provider=str(response.provider),
                    transport_attempts=attempts_used,
                    request_latency_seconds=request_latency,
                )
                with self._usage_lock:
                    self.total_request_latency_seconds += request_latency
                return response
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code not in RETRYABLE_STATUS:
                    detail = error.read().decode("utf-8", errors="replace")[:500]
                    raise OpenRouterError(
                        f"OpenRouter returned HTTP {error.code}: {detail}"
                    ) from error
                if error.code == 429:
                    retry_after = None
                    try:
                        retry_after = float(error.headers.get("Retry-After"))
                    except (AttributeError, TypeError, ValueError):
                        pass
                    self._register_rate_limit(retry_after)
            except _TransientProviderError as error:
                last_error = error
                if error.status_code == 429:
                    self._register_rate_limit()
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
        request_latency = time.monotonic() - request_started
        self._persist_request_record(
            sample_cache_key, successful=False,
            transport_attempts=attempts_used,
            request_latency_seconds=request_latency,
        )
        with self._usage_lock:
            self.total_request_latency_seconds += request_latency
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
        del state["_sample_cache_lock"]
        del state["_rate_limit_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._usage_lock = threading.Lock()
        self._sample_cache_lock = threading.Lock()
        self._rate_limit_lock = threading.Lock()

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
            "sample_parallelism": self.sample_parallelism,
            "sample_cache_hits": self.sample_cache_hits,
            "sample_cache_writes": self.sample_cache_writes,
            "rate_limit_events": self.rate_limit_events,
            "rate_limit_wait_seconds": round(
                self.rate_limit_wait_seconds, 3),
            "requests": self.total_requests,
            "restored_requests": self.restored_requests,
            "transport_attempts": self.total_transport_attempts,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            "request_latency_seconds": round(
                self.total_request_latency_seconds, 6),
            "sample_cache_accounting_complete": (
                self.sample_cache_accounting_complete),
            # Disclosed, not hidden: requests that had to be re-sent with
            # a larger budget because the model reasoned past the first.
            "truncation_escalations": self.truncation_escalations,
        }


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Every JSON object embedded in free-form LLM output, in order.

    A greedy ``\\{.*\\}`` regex mis-parses the common case of a valid
    JSON answer followed by prose that happens to contain a brace (or
    preceded by an echoed packet): the span from the first ``{`` to the
    last ``}`` is not JSON, and a correct answer gets scored as invalid.
    Attempting a real decode at each opening brace — and, on failure,
    resuming at the next brace — also survives an *unmatched* ``{`` in
    the prose before the answer, which a balanced-span scan never
    recovers from (depth stays positive and the valid answer that
    follows is dropped). Each successful parse skips its own span, so
    nested objects are not extracted twice; callers validate each
    candidate and keep the first with the expected shape."""
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        try:
            parsed, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(parsed, dict):
            found.append(parsed)
        index = end
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
