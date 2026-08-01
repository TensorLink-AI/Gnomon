"""OpenRouter client — one code path for open- and closed-weight models.

Deliberately stdlib-only (``urllib``), matching ``aion.api_inference``: a
benchmark harness that drags in an HTTP stack is a benchmark harness that
stops installing cleanly six months from now.

Three things beyond a plain POST matter here:

- **Cost is measured, not estimated.** ``usage.include`` asks OpenRouter to
  return the actual credits spent for the call, upstream markup included.
- **Responses are cached on disk.** A benchmark run is expensive and long;
  resuming one must not re-spend on completed items. The cache key covers
  every input that could change the output.
- **Retries are bounded and typed.** 429/5xx back off; 4xx fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import BenchError

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

# Transient upstream conditions. Everything else surfaces immediately.
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524})


@dataclass
class ChatResponse:
    """One completion, with everything the harness needs to grade and bill it."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    cached: bool = False
    raw_message: dict[str, Any] = field(default_factory=dict)


class ResponseCache:
    """Content-addressed cache of provider responses.

    Keyed by the full request body, so changing a prompt, a tool schema, the
    temperature, or the model produces a different key. Corrupt entries are
    treated as misses rather than raising: a damaged cache must never be able
    to fail a run.
    """

    def __init__(self, directory: str | os.PathLike[str] | None):
        self.directory = Path(directory).expanduser() if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: Mapping[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path | None:
        if not self.directory:
            return None
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path or not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: Mapping[str, Any]) -> None:
        path = self._path(key)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(value, default=str), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass


class OpenRouterClient:
    """Minimal OpenRouter chat-completions client with tool-calling support."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 300,
        max_retries: int = 4,
        cache: ResponseCache | None = None,
        referer: str = "https://github.com/TensorLink-AI/Aion",
        title: str = "AionBench",
        seed: int | None = None,
    ):
        self.api_key = api_key or os.environ.get(API_KEY_ENV, "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = cache or ResponseCache(None)
        self.referer = referer
        self.title = title
        self.seed = seed
        self._sleep = time.sleep
        self._random = random.Random(1729)

    # -- transport ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise BenchError(
                "OPENROUTER_KEY_MISSING",
                f"Set {API_KEY_ENV} to run against OpenRouter.",
                {"env": API_KEY_ENV, "signup": "https://openrouter.ai/keys"},
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }

    def _request(self, path: str, payload: Mapping[str, Any] | None, method: str = "POST") -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        last_error = ""

        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=self._headers(), method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in _RETRY_STATUS or attempt == self.max_retries:
                    raise BenchError(
                        "OPENROUTER_HTTP_ERROR", last_error,
                        {"status": exc.code, "url": url},
                    ) from exc
                self._sleep(self._backoff(attempt, exc.headers.get("Retry-After")))
            except urllib.error.URLError as exc:
                last_error = f"Connection error: {exc.reason}"
                if attempt == self.max_retries:
                    raise BenchError("OPENROUTER_UNREACHABLE", last_error, {"url": url}) from exc
                self._sleep(self._backoff(attempt, None))
            except json.JSONDecodeError as exc:
                raise BenchError("OPENROUTER_BAD_JSON", f"Malformed response body: {exc}") from exc

        raise BenchError("OPENROUTER_EXHAUSTED", last_error, {"url": url})

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(60.0, float(retry_after))
            except ValueError:
                pass
        return min(32.0, (2 ** attempt)) * (0.5 + self._random.random())

    # -- API ---------------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        """Every model slug OpenRouter currently serves, with pricing."""
        return list(self._request("/models", None, method="GET").get("data", []))

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        max_tokens: int = 2048,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Ask for real spend rather than inferring it from a price table
            # that may lag the provider's.
            "usage": {"include": True},
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if self.seed is not None:
            payload["seed"] = self.seed
        if tools:
            payload["tools"] = list(tools)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if extra_body:
            payload.update(dict(extra_body))

        key = ResponseCache.key(payload)
        hit = self.cache.get(key)
        if hit is not None:
            response = _parse_chat(hit)
            response.cached = True
            response.latency_seconds = 0.0
            return response

        started = time.monotonic()
        body = self._request("/chat/completions", payload)
        elapsed = time.monotonic() - started

        if "error" in body and not body.get("choices"):
            error = body["error"]
            raise BenchError(
                "OPENROUTER_ERROR",
                str(error.get("message", error)),
                {"model": model, "code": error.get("code")},
            )

        self.cache.put(key, body)
        response = _parse_chat(body)
        response.latency_seconds = elapsed
        return response


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - body already consumed
        return exc.reason or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:400]
    error = parsed.get("error", parsed)
    if isinstance(error, Mapping):
        return str(error.get("message", error))[:400]
    return str(error)[:400]


def _parse_chat(body: Mapping[str, Any]) -> ChatResponse:
    choices = body.get("choices") or [{}]
    message = dict(choices[0].get("message") or {})
    usage = body.get("usage") or {}
    raw_tool_calls = message.get("tool_calls") or []
    return ChatResponse(
        text=(message.get("content") or "").strip(),
        tool_calls=[dict(call) for call in raw_tool_calls],
        finish_reason=str(choices[0].get("finish_reason") or ""),
        model=str(body.get("model") or ""),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cost_usd=float(usage.get("cost") or 0.0),
        raw_message=message,
    )


class EchoClient:
    """A scripted stand-in for ``OpenRouterClient``.

    Used by ``--dry-run`` and by the tests: it exercises every code path in
    the runner, the suites, and the scorers without a key, a network, or a
    cent of spend. Responses are looked up by task id when supplied, else the
    default is returned.
    """

    def __init__(
        self,
        default: str = "Answer: A",
        by_task: Mapping[str, str] | None = None,
        tool_script: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ):
        self.default = default
        self.by_task = dict(by_task or {})
        self.tool_script = {key: list(value) for key, value in (tool_script or {}).items()}
        self.calls: list[dict[str, Any]] = []

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": "echo/echo", "name": "Echo", "pricing": {"prompt": "0", "completion": "0"}}]

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        max_tokens: int = 2048,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatResponse:
        self.calls.append({"model": model, "messages": list(messages), "tools": list(tools or [])})
        prompt = "\n".join(str(item.get("content") or "") for item in messages)
        task_id = _task_id_from_prompt(prompt)
        pending = self.tool_script.get(task_id or "")
        if pending:
            call = pending.pop(0)
            return ChatResponse(text="", tool_calls=[dict(call)], model=model, finish_reason="tool_calls")
        text = self.by_task.get(task_id or "", self.default)
        return ChatResponse(text=text, model=model, finish_reason="stop", completion_tokens=len(text.split()))


def _task_id_from_prompt(prompt: str) -> str | None:
    marker = "[task_id: "
    start = prompt.find(marker)
    if start < 0:
        return None
    end = prompt.find("]", start)
    return prompt[start + len(marker):end] if end > 0 else None
