"""TSAIA's CodeAct agent loop, reproduced.

This is a behavioural port of ``tools/codeact_agent.py`` from
USC-Melady/TSAIA. Their implementation is built on AgentScope, which reaches
models through a fixed list of provider configs; routing through OpenRouter
instead is the one deliberate substitution, because comparing many models is
the point. Everything that can move a score is kept identical:

- ``SYSTEM_MESSAGE`` and ``EXAMPLE_MESSAGE`` verbatim, and — as in the
  original — both are sent with ``role="user"``, not ``role="system"``.
  That looks like a bug in their code. It is not ours to fix: changing it
  changes the numbers, and this file exists to reproduce their numbers.
- ``example_code`` is ``",".join(executor_variables.keys())``.
- ``n_max_executions = 5``, counted the same way.
- The loop continues while the last message is from the user, so a reply
  containing no ``<execute>`` block ends the episode.
- Execution feedback strings, the 4096→2048 truncation, and the failure
  side-note are byte-identical.
- The answer is read from the kernel as ``predictions``. It is never parsed
  out of prose.

Divergences are listed in ``PROTOCOL_NOTES`` and copied into the run
manifest, so no result can be quoted without them.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contracts import Attempt, BenchError
from .kernel import KernelConfig, PersistentKernel

# Verbatim from tools/codeact_agent.py (SYSTEM_MESSAGE). Includes their
# original typos ("recieve", "abailable") — this string is part of the
# experimental condition, not prose we are free to tidy.
SYSTEM_MESSAGE = """
You are a helpful assistant that gives helpful, detailed, and polite answers to the user's questions.
The code written by assistant should be enclosed using <execute> tag, for example: <execute> print('Hello World!') </execute>.
You should provide the solution in a single <execute> block instead of taking many turns. You'll recieve feedback from your code execution. You should always import packages and define variables before starting to use them.
You should stop <execute> and provide an answer when they have already obtained the answer from the execution result. Whenever possible, execute the code for the user using <execute> instead of providing it.
Your response should be concise, but do express their thoughts. Always write the code in <execute> block to execute them.
You should not ask for the user's input unless necessary. Solve the task on your own and leave no unanswered questions behind.
You should do every thing by your self. You are not allowed to install any new packages or overwrite abailable variables provided to you in the question.
"""

# Verbatim from tools/codeact_agent.py (EXAMPLE_MESSAGE).
EXAMPLE_MESSAGE = """
Additionally, you are provided with the following variables available:
{example_code}
The above variables is already available in your interactive Python (Jupyter Notebook) environment, allowing you to directly use them without needing to redeclare them.
"""

# The single sentence that distinguishes the Aion arm. Appended to
# EXAMPLE_MESSAGE and nothing else — no rewritten system prompt, no worked
# examples, no hints about which tool to reach for. If the uplift needs a
# better prompt to appear, that is a finding about the prompt.
AION_MESSAGE = """
The `aion` package is also installed and importable in your environment. It provides evaluated forecasting, anomaly detection, and change investigation over time-series data. See `aion.toolspec` for the available operations and their arguments.
"""

N_MAX_EXECUTIONS = 5
MAX_FEEDBACK_CHARS = 4096
TRUNCATED_FEEDBACK_CHARS = 2048

TRUNCATION_NOTICE = (
    "... [truncated due to length, do not print too much output as it exceeds your context length]"
)
FAILURE_SIDE_NOTE = (
    "\nSide note: Remember to enclose your code in <execute> </execute> tag and do not "
    "overwrite any available variables provided to you in the question, especially that "
    "they contain the data."
)
MAX_EXECUTIONS_MESSAGE = (
    "I have reached the maximum number "
    f"of executions (self.n_max_executions={N_MAX_EXECUTIONS}). "
    "Can you assist me or ask me another question?"
)

PROTOCOL_NOTES = (
    "Model access is routed through OpenRouter rather than AgentScope; "
    "prompts, roles, turn budget, feedback strings, and answer retrieval are unchanged.",
    "Execution uses a persistent subprocess kernel rather than a Jupyter kernel; "
    "variable injection, cross-turn state, and pickled retrieval of `predictions` match.",
    "temperature=0 and top_p=1, matching TSAIA's non-Gemini model configurations.",
)

# RegexTaggedContentParser in AgentScope matches <tag>...</tag>; this is the
# `execute` tag it extracts. Non-greedy, dot-matches-newline, and the LAST
# block wins if a model emits several — matching a re-scan of the full reply.
_EXECUTE_RE = re.compile(r"<execute>(.*?)</execute>", re.DOTALL | re.IGNORECASE)


@dataclass
class CodeActOptions:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096
    n_max_executions: int = N_MAX_EXECUTIONS
    turn_timeout_seconds: int = 120
    memory_mb: int = 4096
    with_aion: bool = False
    extra_syspath: tuple[str, ...] = ()


@dataclass
class CodeActEpisode:
    """Everything one episode produced, for grading and for the trace."""

    answer: Any = None
    answer_found: bool = False
    executions: int = 0
    execution_failures: int = 0
    hit_execution_cap: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    used_aion: bool = False
    error: str | None = None


def parse_execute(text: str) -> str | None:
    """Extract the ``<execute>`` block from a model reply, or ``None``."""
    matches = _EXECUTE_RE.findall(text or "")
    if not matches:
        return None
    code = matches[-1].strip()
    return code or None


def format_feedback(ok: bool, content: str) -> str:
    """Build the execution-result message exactly as TSAIA does.

    Mirrors ``CodeActAgent.handle_code_result``. The truncation is
    deliberately asymmetric — their code tests ``len > 4096`` but keeps only
    the first 2048 characters — and the side-note is appended *after*
    truncation, so it always survives.
    """
    body = "Execution Successful:\n" if ok else "Execution Failed:\n"
    body += "Execution Output:\n" + str(content)
    if len(body) > MAX_FEEDBACK_CHARS:
        body = body[:TRUNCATED_FEEDBACK_CHARS] + TRUNCATION_NOTICE
    if not ok:
        body += FAILURE_SIDE_NOTE
    return body


def build_messages(
    prompt: str, executor_variables: Mapping[str, Any], *, with_aion: bool = False,
) -> list[dict[str, Any]]:
    """The three opening messages, in TSAIA's order and with their roles."""
    example_code = ",".join(executor_variables.keys())
    example = EXAMPLE_MESSAGE.format(example_code=example_code)
    if with_aion:
        example = example.rstrip() + "\n" + AION_MESSAGE.strip() + "\n"
    return [
        {"role": "user", "content": SYSTEM_MESSAGE.strip()},
        {"role": "user", "content": example.strip()},
        {"role": "user", "content": prompt},
    ]


def run_episode(
    client: Any,
    *,
    model: str,
    prompt: str,
    executor_variables: Mapping[str, Any],
    attempt: Attempt,
    workdir: Any = None,
    options: CodeActOptions | None = None,
) -> CodeActEpisode:
    """Run one CodeAct episode and return its outcome.

    This **executes model-written Python**. The ``--allow-code-execution``
    gate is enforced upstream, by the suite, before any episode is started;
    reaching this function means execution was already authorised for the
    run. Do not call it from a path that has not checked.

    Provider and kernel failures are recorded on ``attempt``/``episode``
    rather than raised: a benchmark cell that dies is a datum, and the run
    must continue.
    """
    options = options or CodeActOptions()
    episode = CodeActEpisode()
    messages = build_messages(prompt, executor_variables, with_aion=options.with_aion)
    episode.messages = list(messages)

    kernel_config = KernelConfig(
        enabled=True,
        turn_timeout_seconds=options.turn_timeout_seconds,
        memory_mb=options.memory_mb,
        extra_syspath=options.extra_syspath if options.with_aion else (),
    )

    kernel: PersistentKernel | None = None
    try:
        kernel = PersistentKernel(kernel_config, workdir=workdir)
        kernel.inject(executor_variables)

        while episode.executions < options.n_max_executions:
            response = client.chat(
                model, messages,
                temperature=options.temperature,
                top_p=options.top_p,
                max_tokens=options.max_tokens,
            )
            _absorb(attempt, response)
            reply = response.text or ""
            messages.append({"role": "assistant", "content": reply})

            code = parse_execute(reply)
            if code is None:
                # No <execute> block: the last message is now the assistant's,
                # so TSAIA's `while last.role == "user"` loop terminates.
                attempt.text = reply
                break

            if "aion" in code:
                episode.used_aion = True
            result = kernel.run(code)
            episode.executions += 1
            if not result.ok:
                episode.execution_failures += 1
            messages.append({"role": "user", "content": format_feedback(result.ok, result.content)})
            attempt.text = reply
        else:
            # Loop fell through with the budget spent, which in TSAIA appends
            # a fixed assistant message and returns it as the reply.
            episode.hit_execution_cap = True
            messages.append({"role": "assistant", "content": MAX_EXECUTIONS_MESSAGE})
            attempt.text = MAX_EXECUTIONS_MESSAGE

        episode.answer_found, episode.answer = kernel.fetch("predictions")

    except BenchError as exc:
        episode.error = f"{exc.code}: {exc.message}"
        attempt.failure = "transport" if exc.code.startswith("KERNEL") else "execution"
        attempt.error = episode.error
    except Exception as exc:  # noqa: BLE001 - provider libraries raise broadly
        episode.error = f"{type(exc).__name__}: {exc}"
        attempt.failure = "transport"
        attempt.error = episode.error
    finally:
        if kernel is not None:
            kernel.close()

    episode.messages = messages
    attempt.codeact = {
        "executions": episode.executions,
        "execution_failures": episode.execution_failures,
        "hit_execution_cap": episode.hit_execution_cap,
        "used_aion": episode.used_aion,
        "answer_found": episode.answer_found,
    }
    return episode


def _absorb(attempt: Attempt, response: Any) -> None:
    attempt.prompt_tokens += getattr(response, "prompt_tokens", 0)
    attempt.completion_tokens += getattr(response, "completion_tokens", 0)
    attempt.cost_usd = round(attempt.cost_usd + getattr(response, "cost_usd", 0.0), 8)
    attempt.provider_calls += 1
    attempt.cache_hits += 1 if getattr(response, "cached", False) else 0


def describe_protocol() -> dict[str, Any]:
    """Protocol fingerprint for the run manifest."""
    import hashlib

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    return {
        "source": "USC-Melady/TSAIA tools/codeact_agent.py",
        "n_max_executions": N_MAX_EXECUTIONS,
        "system_message_sha256_16": digest(SYSTEM_MESSAGE),
        "example_message_sha256_16": digest(EXAMPLE_MESSAGE),
        "aion_message_sha256_16": digest(AION_MESSAGE),
        "system_message_role": "user",
        "divergences": list(PROTOCOL_NOTES),
    }
