"""The CodeAct loop, checked against TSAIA's reference implementation.

Several assertions here look pedantic — exact feedback strings, a typo in a
prompt, a system message sent with ``role="user"``. They are the point. This
module exists to reproduce someone else's numbers, and every one of those
details is part of the condition that produced them.
"""

from __future__ import annotations

import pytest

from aionbench.codeact import (
    AION_MESSAGE,
    EXAMPLE_MESSAGE,
    FAILURE_SIDE_NOTE,
    MAX_EXECUTIONS_MESSAGE,
    N_MAX_EXECUTIONS,
    SYSTEM_MESSAGE,
    TRUNCATION_NOTICE,
    CodeActOptions,
    build_messages,
    describe_protocol,
    format_feedback,
    parse_execute,
    run_episode,
)
from aionbench.contracts import Attempt
from aionbench.router import ChatResponse


class ScriptedClient:
    """Returns queued replies; records exactly what it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, model, messages, **kwargs):
        self.calls.append({"messages": [dict(m) for m in messages], **kwargs})
        text = self.replies.pop(0) if self.replies else "done, no more code"
        return ChatResponse(text=text, model=model, finish_reason="stop", completion_tokens=5)


def _attempt():
    return Attempt(task_id="t", model="m", condition="code")


# -- prompt fidelity -------------------------------------------------------

def test_system_message_matches_tsaia_verbatim():
    # Including their typos. Fixing these would change the condition.
    assert "You'll recieve feedback from your code execution." in SYSTEM_MESSAGE
    assert "overwrite abailable variables" in SYSTEM_MESSAGE
    assert "<execute> print('Hello World!') </execute>" in SYSTEM_MESSAGE


def test_opening_messages_use_tsaias_roles_and_order():
    messages = build_messages("What is the forecast?", {"VAL": 1, "LOOKBACK": 2})
    assert len(messages) == 3
    # TSAIA sends the system message as a *user* message. Reproduced, not fixed.
    assert [m["role"] for m in messages] == ["user", "user", "user"]
    assert messages[0]["content"] == SYSTEM_MESSAGE.strip()
    assert "VAL,LOOKBACK" in messages[1]["content"]
    assert messages[2]["content"] == "What is the forecast?"


def test_example_code_is_the_joined_variable_names():
    messages = build_messages("q", {"a": 1, "b": 2, "c": 3})
    assert EXAMPLE_MESSAGE.format(example_code="a,b,c").strip() == messages[1]["content"]


def test_aion_arm_adds_exactly_one_paragraph_and_nothing_else():
    baseline = build_messages("q", {"VAL": 1})
    treatment = build_messages("q", {"VAL": 1}, with_aion=True)

    assert treatment[0] == baseline[0]      # system message untouched
    assert treatment[2] == baseline[2]      # question untouched
    added = treatment[1]["content"].replace(baseline[1]["content"], "").strip()
    assert added == AION_MESSAGE.strip()


# -- <execute> parsing -----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("<execute>predictions = 1</execute>", "predictions = 1"),
    ("thinking...\n<execute>\n  x = 2\n</execute>\nmore", "x = 2"),
    ("<EXECUTE>y = 3</EXECUTE>", "y = 3"),
    ("no code at all", None),
    ("<execute>   </execute>", None),
])
def test_execute_parsing(text, expected):
    assert parse_execute(text) == expected


def test_last_execute_block_wins():
    assert parse_execute("<execute>a = 1</execute> then <execute>a = 2</execute>") == "a = 2"


def test_markdown_fences_are_not_accepted():
    # TSAIA's parser matches the tag only. A model that emits a ```python
    # fence gets no execution, which is exactly what happens upstream.
    assert parse_execute("```python\npredictions = 1\n```") is None


# -- execution feedback ----------------------------------------------------

def test_success_feedback_is_byte_identical_to_tsaia():
    assert format_feedback(True, "42\n") == "Execution Successful:\nExecution Output:\n42\n"


def test_failure_feedback_appends_the_side_note():
    body = format_feedback(False, "Traceback...")
    assert body.startswith("Execution Failed:\nExecution Output:\nTraceback...")
    assert body.endswith(FAILURE_SIDE_NOTE)


def test_truncation_reproduces_their_asymmetric_rule():
    # Their code tests len > 4096 but keeps only the first 2048 characters.
    body = format_feedback(True, "x" * 5000)
    assert body.endswith(TRUNCATION_NOTICE)
    assert len(body) == 2048 + len(TRUNCATION_NOTICE)


def test_short_output_is_not_truncated():
    body = format_feedback(True, "y" * 100)
    assert TRUNCATION_NOTICE not in body


def test_side_note_survives_truncation():
    body = format_feedback(False, "z" * 5000)
    assert TRUNCATION_NOTICE in body
    assert body.endswith(FAILURE_SIDE_NOTE)


# -- the loop --------------------------------------------------------------

def test_answer_is_read_from_the_kernel_not_the_text(tmp_path):
    client = ScriptedClient([
        "<execute>predictions = [1, 2, 3]</execute>",
        "The answer is [99, 99, 99].",       # prose disagrees with the kernel
    ])
    attempt = _attempt()
    episode = run_episode(
        client, model="m", prompt="q", executor_variables={"VAL": [1]},
        attempt=attempt, workdir=tmp_path,
    )
    assert episode.answer_found is True
    assert episode.answer == [1, 2, 3]       # the kernel wins
    assert episode.executions == 1


def test_state_accumulates_across_turns(tmp_path):
    client = ScriptedClient([
        "<execute>parts = [1, 2]</execute>",
        "<execute>parts.append(3)</execute>",
        "<execute>predictions = sum(parts)</execute>",
        "Done.",
    ])
    attempt = _attempt()
    episode = run_episode(
        client, model="m", prompt="q", executor_variables={},
        attempt=attempt, workdir=tmp_path,
    )
    assert episode.executions == 3
    assert episode.answer == 6


def test_injected_variables_are_available_to_the_model(tmp_path):
    client = ScriptedClient(["<execute>predictions = sum(VAL) * LOOKBACK</execute>", "Done."])
    attempt = _attempt()
    episode = run_episode(
        client, model="m", prompt="q",
        executor_variables={"VAL": [1, 2, 3], "LOOKBACK": 10},
        attempt=attempt, workdir=tmp_path,
    )
    assert episode.answer == 60


def test_execution_failure_is_fed_back_and_the_model_can_recover(tmp_path):
    client = ScriptedClient([
        "<execute>predictions = undefined_name</execute>",
        "<execute>predictions = 7</execute>",
        "Recovered.",
    ])
    attempt = _attempt()
    episode = run_episode(
        client, model="m", prompt="q", executor_variables={},
        attempt=attempt, workdir=tmp_path,
    )
    assert episode.executions == 2
    assert episode.execution_failures == 1
    assert episode.answer == 7

    feedback = client.calls[1]["messages"][-1]
    assert feedback["role"] == "user"
    assert feedback["content"].startswith("Execution Failed:")
    assert "NameError" in feedback["content"]


def test_loop_stops_when_the_model_stops_writing_code(tmp_path):
    client = ScriptedClient(["I do not need to run anything.", "unused"])
    attempt = _attempt()
    episode = run_episode(
        client, model="m", prompt="q", executor_variables={},
        attempt=attempt, workdir=tmp_path,
    )
    assert episode.executions == 0
    assert episode.answer_found is False
    assert len(client.calls) == 1


def test_execution_cap_is_five_and_emits_their_terminal_message(tmp_path):
    client = ScriptedClient(["<execute>x = 1</execute>"] * 10)
    attempt = _attempt()
    episode = run_episode(
        client, model="m", prompt="q", executor_variables={},
        attempt=attempt, workdir=tmp_path,
    )
    assert episode.executions == N_MAX_EXECUTIONS == 5
    assert episode.hit_execution_cap is True
    assert attempt.text == MAX_EXECUTIONS_MESSAGE
    assert episode.messages[-1]["content"] == MAX_EXECUTIONS_MESSAGE


def test_never_binding_predictions_is_not_found(tmp_path):
    client = ScriptedClient(["<execute>answer = 5</execute>", "The answer is 5."])
    attempt = _attempt()
    episode = run_episode(
        client, model="m", prompt="q", executor_variables={},
        attempt=attempt, workdir=tmp_path,
    )
    assert episode.answer_found is False
    assert episode.answer is None


def test_decoding_parameters_match_tsaia(tmp_path):
    client = ScriptedClient(["no code"])
    run_episode(
        client, model="m", prompt="q", executor_variables={},
        attempt=_attempt(), workdir=tmp_path,
    )
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["top_p"] == 1.0


# -- the Aion arm ----------------------------------------------------------

def test_baseline_kernel_cannot_import_aion(tmp_path):
    client = ScriptedClient(["<execute>import aion\npredictions = 1</execute>", "Failed."])
    attempt = _attempt()
    episode = run_episode(
        client, model="m", prompt="q", executor_variables={},
        attempt=attempt, workdir=tmp_path,
    )
    assert episode.execution_failures == 1
    assert episode.answer_found is False


def test_aion_arm_can_import_aion_and_is_marked_grounded(tmp_path):
    from aionbench.runner import _aion_syspath

    client = ScriptedClient([
        "<execute>import aion\npredictions = [aion.__version__]</execute>",
        "Done.",
    ])
    attempt = Attempt(task_id="t", model="m", condition="aion_code")
    episode = run_episode(
        client, model="m", prompt="q", executor_variables={},
        attempt=attempt, workdir=tmp_path,
        options=CodeActOptions(with_aion=True, extra_syspath=_aion_syspath()),
    )
    assert episode.execution_failures == 0
    assert episode.answer_found is True
    assert episode.used_aion is True
    assert attempt.grounded is True


def test_aion_arm_that_never_touches_aion_is_not_grounded(tmp_path):
    from aionbench.runner import _aion_syspath

    client = ScriptedClient(["<execute>predictions = 42</execute>", "Done."])
    attempt = Attempt(task_id="t", model="m", condition="aion_code")
    run_episode(
        client, model="m", prompt="q", executor_variables={},
        attempt=attempt, workdir=tmp_path,
        options=CodeActOptions(with_aion=True, extra_syspath=_aion_syspath()),
    )
    # Right answer, Aion never used. The report must be able to see that.
    assert attempt.codeact["answer_found"] is True
    assert attempt.grounded is False


# -- manifest --------------------------------------------------------------

def test_protocol_fingerprint_records_the_divergences():
    described = describe_protocol()
    assert described["n_max_executions"] == 5
    assert described["system_message_role"] == "user"
    assert described["source"].startswith("USC-Melady/TSAIA")
    assert any("OpenRouter" in note for note in described["divergences"])
    assert any("persistent subprocess kernel" in note for note in described["divergences"])
