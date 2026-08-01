"""The code-execution sandbox.

These spawn real subprocesses. They stay fast because every case is trivial
code with a short timeout.
"""

from __future__ import annotations

import pickle

import pytest

from aionbench.sandbox import SandboxConfig, SandboxDisabled, run_code

ENABLED = SandboxConfig(enabled=True, timeout_seconds=30)


def test_execution_is_refused_unless_explicitly_enabled():
    with pytest.raises(SandboxDisabled, match="--allow-code-execution"):
        run_code("predictions = 1", config=SandboxConfig())


def test_result_variable_is_returned():
    result = run_code("predictions = [1, 2, 3]", config=ENABLED)
    assert result.ok is True
    assert result.value == [1, 2, 3]


def test_injected_variables_are_visible_to_the_code():
    result = run_code(
        "predictions = [value * 2 for value in VAL]",
        variables={"VAL": [1, 2, 3]}, config=ENABLED,
    )
    assert result.ok is True and result.value == [2, 4, 6]


def test_a_prepickled_variables_file_is_used_verbatim(tmp_path):
    path = tmp_path / "vars.pkl"
    path.write_bytes(pickle.dumps({"VAL": [10, 20]}))
    result = run_code("predictions = sum(VAL)", variables_path=str(path), config=ENABLED)
    assert result.ok is True and result.value == 30


def test_an_exception_is_reported_with_its_traceback_not_raised():
    result = run_code("predictions = 1 / 0", config=ENABLED)
    assert result.ok is False
    assert "ZeroDivisionError" in (result.error or "")


def test_code_that_never_assigns_the_result_variable_fails_clearly():
    result = run_code("total = 1 + 1", config=ENABLED)
    assert result.ok is False
    assert "never assigned `predictions`" in (result.error or "")


def test_empty_code_is_rejected_without_spawning_anything():
    assert run_code("   ", config=ENABLED).ok is False


def test_a_runaway_loop_is_killed_by_the_timeout():
    result = run_code(
        "while True:\n    pass\npredictions = 1",
        config=SandboxConfig(enabled=True, timeout_seconds=2),
    )
    assert result.ok is False and result.timed_out is True
    assert "exceeded 2s" in (result.error or "")


def test_stdout_is_captured_and_kept_out_of_the_result():
    result = run_code("print('chatter')\npredictions = 5", config=ENABLED)
    assert result.ok is True and result.value == 5
    assert "chatter" in result.stdout


def test_provider_and_hub_credentials_are_stripped_from_the_child(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    result = run_code(
        "import os\npredictions = [os.environ.get('OPENROUTER_API_KEY'), os.environ.get('HF_TOKEN')]",
        config=ENABLED,
    )
    assert result.ok is True
    assert result.value == [None, None]


def test_an_unpicklable_result_is_reported_rather_than_crashing_the_harness():
    result = run_code("predictions = lambda x: x", config=ENABLED)
    assert result.ok is False
    assert "could not be serialised" in (result.error or "")


def test_aion_is_importable_inside_the_sandbox_for_the_aion_code_condition():
    from aionbench.runner import _aion_syspath

    result = run_code(
        "import aion\npredictions = [aion.__version__]",
        config=SandboxConfig(enabled=True, timeout_seconds=30, extra_syspath=_aion_syspath()),
    )
    assert result.ok is True
    assert isinstance(result.value[0], str)
