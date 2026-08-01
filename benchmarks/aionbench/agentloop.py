"""Running one task, once, under one condition.

The control conditions are a single completion. The treatment conditions are
a bounded tool-calling loop over Aion's toolbelt. Both return the same
``Attempt``, so every downstream comparison is like-for-like.

The loop is deliberately unhelpful to the model: it does not repair malformed
tool arguments, retry a failed call on the model's behalf, or nudge it toward
a tool. A model that cannot drive Aion is a finding, not a harness bug.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .contracts import Attempt, BenchError, BenchTask
from .prompting import SYSTEM_PROMPTS, build_user_prompt
from .router import ChatResponse
from .tools import AionToolbelt, materialise_series

DEFAULT_MAX_TOOL_STEPS = 8


class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = ...,
        tool_choice: str | Mapping[str, Any] | None = ...,
        temperature: float = ...,
        top_p: float | None = ...,
        max_tokens: int = ...,
        extra_body: Mapping[str, Any] | None = ...,
    ) -> ChatResponse:
        ...


@dataclass
class AttemptOptions:
    """Everything that can change a completion, in one place.

    Anything added here must also feed the response-cache key (it does, via
    the request body) or resumed runs will silently mix configurations.
    """

    temperature: float = 0.0
    max_tokens: int = 4096
    max_tool_steps: int = DEFAULT_MAX_TOOL_STEPS
    tool_profile: str = "analysis"
    include_hint: bool = False
    include_concepts: bool = False
    decimals: int = 1
    max_points: int | None = None
    system_prompt_override: str | None = None


def run_attempt(
    client: ChatClient,
    task: BenchTask,
    *,
    model: str,
    condition: str,
    workdir: str | Path,
    options: AttemptOptions | None = None,
    repeat: int = 0,
) -> Attempt:
    """Produce one graded-ready attempt for ``task``."""
    options = options or AttemptOptions()
    attempt = Attempt(task_id=task.task_id, model=model, condition=condition, repeat=repeat)
    work = Path(workdir)

    csv_paths: dict[str, str] = {}
    toolbelt: AionToolbelt | None = None
    if condition in ("aion", "aion_code") and task.series:
        try:
            csv_paths = materialise_series(task.series, work / "series")
        except BenchError as exc:
            attempt.error = exc.message
            attempt.failure = "transport"
            return attempt

    if condition == "aion":
        try:
            toolbelt = AionToolbelt(profile=options.tool_profile, workdir=work)
        except BenchError as exc:
            attempt.error = exc.message
            attempt.failure = "transport"
            return attempt

    system = options.system_prompt_override or SYSTEM_PROMPTS.get(condition)
    if system is None:
        raise BenchError(
            "UNKNOWN_CONDITION",
            f"No system prompt for condition {condition!r}.",
            {"supported": sorted(SYSTEM_PROMPTS)},
        )

    user = build_user_prompt(
        task,
        condition=condition,
        csv_paths=csv_paths,
        include_hint=options.include_hint,
        include_concepts=options.include_concepts,
        decimals=options.decimals,
        max_points=options.max_points,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    started = time.monotonic()
    try:
        if toolbelt is None:
            response = client.chat(
                model, messages,
                temperature=options.temperature, max_tokens=options.max_tokens,
            )
            _absorb(attempt, response)
            attempt.text = response.text
        else:
            _run_tool_loop(client, attempt, messages, toolbelt, model, options)
    except BenchError as exc:
        attempt.error = f"{exc.code}: {exc.message}"
        attempt.failure = "transport"
    except Exception as exc:  # noqa: BLE001 - provider libraries raise broadly
        attempt.error = f"{type(exc).__name__}: {exc}"
        attempt.failure = "transport"

    attempt.latency_seconds = round(time.monotonic() - started, 4)
    return attempt


def _run_tool_loop(
    client: ChatClient,
    attempt: Attempt,
    messages: list[dict[str, Any]],
    toolbelt: AionToolbelt,
    model: str,
    options: AttemptOptions,
) -> None:
    definitions = toolbelt.definitions()

    for step in range(options.max_tool_steps + 1):
        final_turn = step == options.max_tool_steps
        response = client.chat(
            model, messages,
            tools=None if final_turn else definitions,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
        )
        _absorb(attempt, response)

        if not response.tool_calls:
            attempt.text = response.text
            return

        messages.append(_assistant_turn(response))
        for raw_call in response.tool_calls:
            call, rendered = _dispatch(toolbelt, raw_call)
            attempt.tool_calls.append(call)
            messages.append({
                "role": "tool",
                "tool_call_id": raw_call.get("id", ""),
                "name": call.name,
                "content": rendered,
            })

    # Budget exhausted with tool calls still outstanding.
    attempt.text = messages[-1].get("content", "") if messages else ""
    attempt.failure = attempt.failure or "structural"
    attempt.error = attempt.error or (
        f"Tool-call budget of {options.max_tool_steps} steps exhausted without a final answer."
    )


def _dispatch(toolbelt: AionToolbelt, raw_call: Mapping[str, Any]) -> tuple[Any, str]:
    function = raw_call.get("function") or {}
    name = str(function.get("name") or "")
    raw_arguments = function.get("arguments")

    if isinstance(raw_arguments, Mapping):
        arguments: dict[str, Any] = dict(raw_arguments)
    else:
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            from .contracts import ToolCall
            call = ToolCall(
                name=name, arguments={"_raw": str(raw_arguments)[:400]}, ok=False,
                error=f"Tool arguments were not valid JSON: {exc}",
            )
            return call, json.dumps({"error": call.error})
        if not isinstance(arguments, dict):
            from .contracts import ToolCall
            call = ToolCall(
                name=name, arguments={"_raw": str(raw_arguments)[:400]}, ok=False,
                error="Tool arguments must be a JSON object.",
            )
            return call, json.dumps({"error": call.error})

    call = toolbelt.call(name, arguments)
    rendered = getattr(call, "rendered", None) or json.dumps({"error": call.error})
    return call, rendered


def _assistant_turn(response: ChatResponse) -> dict[str, Any]:
    turn: dict[str, Any] = {"role": "assistant", "content": response.text or None}
    if response.tool_calls:
        turn["tool_calls"] = response.tool_calls
    return turn


def _absorb(attempt: Attempt, response: ChatResponse) -> None:
    attempt.prompt_tokens += response.prompt_tokens
    attempt.completion_tokens += response.completion_tokens
    attempt.cost_usd = round(attempt.cost_usd + response.cost_usd, 8)
    attempt.provider_calls += 1
    attempt.cache_hits += 1 if response.cached else 0
