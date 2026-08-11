"""The two non-Gnomon arms, and the tool loop they share with the MCP arm.

Three conditions, one submission contract (``grading.validate_submission``
— shared so no arm is easier to answer than another):

- ``llm-direct`` — the floor: one completion, the history inline, the
  model answers with a JSON object. No tools.
- ``llm-sandbox`` — the honest competitor: the same model with a
  ``run_python`` tool over the task file in a jail. This is what an
  agent actually has today, so it is the arm the treatment must beat —
  a naked-LLM control alone would flatter Gnomon.
- ``gnomon-mcp`` (in ``mcp_arm.py``) — the treatment: the same loop with
  the real ``gnomon mcp serve`` surface instead of an interpreter.

The loop, caps, nudge, and last-call protocol follow the measured
lessons of the TemporalBench MCP arm: a spent budget ends tool use and
asks for the submission rather than voiding the row, a run that never
submits is an abstention (kept out of cost denominators), and long tool
results are bounded per message because every round re-sends them.

Sandbox containment is disclosed, not oversold: ``run_python`` executes
in a subprocess with ``-I``, the jail as its working directory, and a
minimal environment (no proxy variables, so outbound HTTP through the
usual env-configured routes fails closed). That is best-effort isolation
for a benchmark harness, not a security boundary.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cik.mcp_agent import _tool_calls_as_dicts  # noqa: E402
from benchmarks.decisionbench.grading import validate_submission  # noqa: E402
from benchmarks.decisionbench.tasks import DecisionTask  # noqa: E402

MAX_ROUNDS = 10
MAX_TOOL_CALLS = 12
MAX_RUN_TOKENS = 200_000
MAX_TOOL_RESULT_CHARS = 16_000
PYTHON_TIMEOUT_SECONDS = 60

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": (
            "End the run with your decision. Give decision.capacity (a "
            "number) OR decision.escalate=true, plus peak_forecast "
            "{q10,q50,q90} — your percentile estimates of the maximum "
            "daily demand over the window, given even when escalating."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "object",
                    "properties": {
                        "capacity": {"type": "number"},
                        "escalate": {"type": "boolean"},
                    },
                },
                "peak_forecast": {
                    "type": "object",
                    "properties": {"q10": {"type": "number"},
                                   "q50": {"type": "number"},
                                   "q90": {"type": "number"}},
                    "required": ["q10", "q50", "q90"],
                },
                "rationale": {"type": "string"},
            },
            "required": ["decision"],
        },
    },
}

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute Python code in a fresh subprocess (same interpreter "
            "as the harness; the standard library is guaranteed, other "
            "packages may or may not be importable). Working directory "
            "is the run's jail, where the task data file lives. stdout "
            "and stderr come back as the result. State does NOT persist "
            "between calls — each call is a fresh process."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}

LAST_CALL = (
    "This run is ending now ({cap}). This is the final message: call "
    "submit_decision with the best decision you have — escalating is a "
    "valid answer, an unsubmitted run is not. No other tool is available."
)

NUDGE = "Finish by calling submit_decision."

#: How a tool-holding arm delivers the answer. The task statement is
#: shared verbatim across arms; only this block and the data reference
#: differ, so a formatting mismatch can never masquerade as a decision
#: difference between conditions.
TOOL_ANSWER_FORMAT = (
    "Deliver your answer by calling the submit_decision tool: "
    "decision.capacity (a number) or decision.escalate=true, plus "
    "peak_forecast {q10, q50, q90} and a short rationale."
)

DIRECT_ANSWER_FORMAT = "\n".join([
    "Answer with a single JSON object, no code fences:",
    '{"decision": {"capacity": <number>} or {"escalate": true},',
    ' "peak_forecast": {"q10": <number>, "q50": <number>, '
    '"q90": <number>},',
    ' "rationale": "<one or two sentences>"}',
])


class ToolLoopOutcome(dict):
    """The arm result: submission (or None), abstention reason, trace."""


class ToolLoopArm:
    """One task's conversation for any tool-holding arm.

    Subclasses provide the system prompt, the tool specs, and the
    dispatch for non-submit tools. Submission validation, caps, the
    nudge, and last call are common — measured behaviour, kept
    identical so arm differences are the treatment, not the harness.
    """

    def __init__(self, task: DecisionTask, client: Any,
                 work_dir: str | None = None):
        self.task = task
        self.client = client
        self.jail = Path(tempfile.mkdtemp(prefix="decisionbench-",
                                          dir=work_dir)).resolve()
        self.csv_path = task.write_csv(self.jail / "demand.csv")
        self.trace: list[dict[str, Any]] = []
        self.tool_calls = 0
        self.submission: dict[str, Any] | None = None
        self.tokens_at_start = (getattr(client, "total_prompt_tokens", 0)
                                + getattr(client, "total_completion_tokens", 0))

    # -- arm contract ------------------------------------------------------
    def system_prompt(self) -> str:
        raise NotImplementedError

    def tool_specs(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def screen_tool(self, name: str,
                    arguments: dict[str, Any]) -> str | None:
        """A harness rejection for this call, or ``None`` to dispatch.
        Screened-out calls cost no tool budget — a path typo is a
        round-trip, not a spent call."""
        return None

    def dispatch_tool(self, name: str, arguments: dict[str, Any]) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # -- caps --------------------------------------------------------------
    def _run_tokens(self) -> int:
        return (getattr(self.client, "total_prompt_tokens", 0)
                + getattr(self.client, "total_completion_tokens", 0)
                - self.tokens_at_start)

    def _cap_breach(self) -> str | None:
        if self._run_tokens() > MAX_RUN_TOKENS:
            return f"cap:tokens exceeded {MAX_RUN_TOKENS}"
        return None

    # -- loop --------------------------------------------------------------
    def drive(self) -> ToolLoopOutcome:
        try:
            return self._drive()
        finally:
            self.close()

    def _drive(self) -> ToolLoopOutcome:
        tools = self.tool_specs()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user",
             "content": self.task.prompt(self._data_reference(),
                                         TOOL_ANSWER_FORMAT)},
        ]
        nudged = False
        for _round in range(MAX_ROUNDS):
            breach = self._cap_breach()
            if breach:
                return self._last_call(messages, breach)
            response = self.client.chat(messages, n=1, tools=tools,
                                        tool_choice="auto")
            message = response.choices[0].message
            tool_calls = _tool_calls_as_dicts(message)
            if not tool_calls:
                messages.append({"role": "assistant",
                                 "content": message.content or ""})
                if self.submission:
                    break
                if nudged:
                    return self._last_call(
                        messages, "no submission: prose after the nudge")
                nudged = True
                messages.append({"role": "user", "content": NUDGE})
                continue
            messages.append({"role": "assistant",
                             "content": message.content or None,
                             "tool_calls": tool_calls})
            for call in tool_calls:
                try:
                    arguments = json.loads(call["function"]["arguments"]
                                           or "{}")
                except json.JSONDecodeError:
                    arguments = None
                text = self._route(call["function"]["name"], arguments)
                messages.append({"role": "tool",
                                 "tool_call_id": call["id"],
                                 "content": text})
            if self.submission:
                break
        if not self.submission:
            return self._last_call(
                messages, f"cap:rounds {MAX_ROUNDS} without a submission")
        return self._outcome()

    def _data_reference(self) -> str:
        return (f"The demand history is on disk at {self.csv_path} "
                f"(columns: {', '.join(self.task.columns)}).")

    def _route(self, name: str, arguments: dict[str, Any] | None) -> str:
        entry: dict[str, Any] = {"tool": name}
        self.trace.append(entry)
        if arguments is None:
            entry["error"] = "unparseable arguments"
            return json.dumps({"code": "INVALID_ARGUMENTS",
                               "message": "Tool arguments were not valid "
                                          "JSON.",
                               "authored_by": "harness"})
        if name == "submit_decision":
            if self.submission is not None:
                entry["problem"] = "already submitted"
                return json.dumps({"accepted": False,
                                   "message": "A decision was already "
                                              "submitted; the first "
                                              "accepted submission stands.",
                                   "authored_by": "harness"})
            submission, problem = validate_submission(arguments)
            if problem:
                entry["problem"] = problem
                return json.dumps({"accepted": False, "problems": [problem],
                                   "authored_by": "harness"})
            self.submission = submission
            entry["accepted"] = True
            return json.dumps({"accepted": True})
        screened = self.screen_tool(name, arguments)
        if screened is not None:
            return screened
        if self.tool_calls >= MAX_TOOL_CALLS:
            entry["code"] = "TOOL_BUDGET_SPENT"
            return json.dumps({
                "code": "TOOL_BUDGET_SPENT",
                "message": f"This run's {MAX_TOOL_CALLS} tool calls are "
                           f"spent. Call submit_decision with your best "
                           f"decision.",
                "authored_by": "harness"})
        self.tool_calls += 1
        text = self.dispatch_tool(name, arguments)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            entry["truncated"] = True
            text = (text[:MAX_TOOL_RESULT_CHARS]
                    + f"\n[harness: result cut at {MAX_TOOL_RESULT_CHARS} "
                      f"of {len(text)} characters]")
        return text

    def _last_call(self, messages: list[dict[str, Any]],
                   cap: str) -> ToolLoopOutcome:
        self.trace.append({"last_call": cap})
        messages = messages + [{"role": "user",
                                "content": LAST_CALL.format(cap=cap)}]
        try:
            response = self.client.chat(messages, n=1, tools=[SUBMIT_TOOL],
                                        tool_choice="auto")
        except Exception as error:
            return self._abstained(f"{cap}; last call failed: {error}")
        for call in _tool_calls_as_dicts(response.choices[0].message):
            if call["function"]["name"] != "submit_decision":
                continue
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                continue
            submission, _problem = validate_submission(arguments)
            if submission:
                self.submission = submission
                outcome = self._outcome()
                outcome["last_call"] = cap
                return outcome
        return self._abstained(cap)

    def _abstained(self, reason: str) -> ToolLoopOutcome:
        self.trace.append({"abstained": reason})
        return ToolLoopOutcome(submission=None, abstained=reason,
                               trace=self.trace, tool_calls=self.tool_calls,
                               run_tokens=self._run_tokens())

    def _outcome(self) -> ToolLoopOutcome:
        return ToolLoopOutcome(submission=self.submission, abstained=None,
                               trace=self.trace, tool_calls=self.tool_calls,
                               run_tokens=self._run_tokens())


class SandboxArm(ToolLoopArm):
    """``llm-sandbox``: the same model with an interpreter over the file."""

    SYSTEM = (
        "You are making one operational capacity decision. You have a "
        "run_python tool over the task's data file — use it to load, "
        "clean, and analyse the history however you see fit, or answer "
        "without it. Each call is a fresh process; print what you need "
        "to see. Finish by calling submit_decision. Tool budget: "
        f"{MAX_TOOL_CALLS} calls, {MAX_ROUNDS} rounds."
    )

    def system_prompt(self) -> str:
        return self.SYSTEM

    def tool_specs(self) -> list[dict[str, Any]]:
        return [RUN_PYTHON_TOOL, SUBMIT_TOOL]

    def dispatch_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name != "run_python":
            return json.dumps({"code": "UNKNOWN_TOOL", "message": name,
                               "authored_by": "harness"})
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return json.dumps({"code": "INVALID_ARGUMENTS",
                               "message": "code must be a non-empty string",
                               "authored_by": "harness"})
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=str(self.jail),
                env={"PATH": "/usr/bin:/bin"},
                capture_output=True, text=True,
                timeout=PYTHON_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return json.dumps({
                "code": "TIMEOUT",
                "message": f"code did not finish within "
                           f"{PYTHON_TIMEOUT_SECONDS}s",
                "authored_by": "harness"})
        parts = []
        if completed.stdout:
            parts.append(completed.stdout)
        if completed.stderr:
            parts.append("[stderr]\n" + completed.stderr)
        if completed.returncode != 0:
            parts.append(f"[exit code {completed.returncode}]")
        return "\n".join(parts) or "[no output]"


def extract_json_object(text: str) -> dict[str, Any]:
    """The first parseable top-level JSON object in free-form output —
    the object-shaped sibling of ``openrouter.extract_json_array``."""
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
                candidate = text[start:index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = None
    raise ValueError("No JSON object found in model output.")


DIRECT_SYSTEM = (
    "You are making one operational capacity decision from the data in "
    "the message. Answer with the single JSON object the task specifies "
    "— no code fences, no surrounding prose."
)

#: One repair round for a malformed direct answer, then the row stands
#: as unanswered: a stricter loop would turn formatting into the
#: measurement.
DIRECT_RETRY = (
    "Your reply was not a valid answer object ({problem}). Reply with "
    "exactly one JSON object in the specified format."
)


def run_direct(task: DecisionTask, client: Any) -> ToolLoopOutcome:
    """``llm-direct``: one completion (plus one repair round), the
    history inline."""
    reference = ("The demand history follows as CSV (columns: "
                 f"{', '.join(task.columns)}):\n\n{task.csv_text()}")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user",
         "content": task.prompt(reference, DIRECT_ANSWER_FORMAT)},
    ]
    trace: list[dict[str, Any]] = []
    tokens_at_start = (getattr(client, "total_prompt_tokens", 0)
                       + getattr(client, "total_completion_tokens", 0))

    def tokens() -> int:
        return (getattr(client, "total_prompt_tokens", 0)
                + getattr(client, "total_completion_tokens", 0)
                - tokens_at_start)

    for attempt in range(2):
        text = client.completions(messages)[0]
        trace.append({"attempt": attempt, "reply_chars": len(text)})
        try:
            parsed = extract_json_object(text)
        except ValueError:
            problem = "no JSON object found"
            parsed = None
        if parsed is not None:
            submission, problem = validate_submission(parsed)
            if submission:
                return ToolLoopOutcome(submission=submission, abstained=None,
                                       trace=trace, tool_calls=0,
                                       run_tokens=tokens())
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",
                         "content": DIRECT_RETRY.format(problem=problem)})
    trace.append({"abstained": "unparseable after one repair round"})
    return ToolLoopOutcome(submission=None,
                           abstained="unparseable after one repair round",
                           trace=trace, tool_calls=0, run_tokens=tokens())
