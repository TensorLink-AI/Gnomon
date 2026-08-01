"""The persistent kernel.

The property that matters is **persistence**: TSAIA gives the model five
execution turns against one live namespace. If state didn't survive between
turns, a model that builds its answer incrementally would score zero for
reasons that have nothing to do with its reasoning.
"""

from __future__ import annotations

import numpy  # noqa: F401 - proves third-party imports survive in the child
import pytest

from aionbench.kernel import KernelConfig, KernelDisabled, PersistentKernel

ENABLED = KernelConfig(enabled=True, turn_timeout_seconds=30)


@pytest.fixture
def kernel():
    with PersistentKernel(ENABLED) as instance:
        yield instance


def test_execution_is_refused_unless_enabled():
    with pytest.raises(KernelDisabled, match="--allow-code-execution"):
        PersistentKernel(KernelConfig())


def test_state_persists_across_turns(kernel):
    assert kernel.run("total = 0").ok
    for _ in range(4):
        assert kernel.run("total += 10").ok
    found, value = kernel.fetch("total")
    assert (found, value) == (True, 40)


def test_a_function_defined_early_is_callable_later(kernel):
    assert kernel.run("def double(x):\n    return x * 2").ok
    assert kernel.run("import math\nbase = math.floor(3.7)").ok
    assert kernel.run("predictions = double(base)").ok
    assert kernel.fetch("predictions") == (True, 6)


def test_imports_persist_between_turns(kernel):
    assert kernel.run("import numpy as np").ok
    result = kernel.run("predictions = np.arange(3).tolist()")
    assert result.ok, result.error
    assert kernel.fetch("predictions") == (True, [0, 1, 2])


def test_injected_variables_land_in_globals(kernel):
    kernel.inject({"VAL": [1, 2, 3], "LOOKBACK": 2})
    assert kernel.run("predictions = sum(VAL) * LOOKBACK").ok
    assert kernel.fetch("predictions") == (True, 12)


def test_injected_variables_survive_later_turns(kernel):
    kernel.inject({"VAL": [1, 2, 3]})
    kernel.run("scratch = 1")
    kernel.run("scratch = 2")
    assert kernel.run("predictions = list(VAL)").ok
    assert kernel.fetch("predictions") == (True, [1, 2, 3])


def test_a_failing_turn_does_not_destroy_earlier_state(kernel):
    assert kernel.run("keep = 'here'").ok
    broken = kernel.run("raise ValueError('boom')")
    assert broken.ok is False
    assert "ValueError: boom" in broken.error
    # The model gets to see the traceback and try again — with its state.
    assert kernel.run("predictions = keep").ok
    assert kernel.fetch("predictions") == (True, "here")


def test_syntax_errors_are_returned_not_raised(kernel):
    result = kernel.run("def oops(:")
    assert result.ok is False
    assert "SyntaxError" in result.error


def test_stdout_is_captured_as_execution_output(kernel):
    result = kernel.run("print('turn one')")
    assert result.ok and "turn one" in result.output
    assert result.content == result.output


def test_failed_turn_content_carries_the_traceback(kernel):
    result = kernel.run("print('before')\n1 / 0")
    assert result.ok is False
    assert "before" in result.content
    assert "ZeroDivisionError" in result.content


def test_unbound_predictions_reads_as_not_found(kernel):
    kernel.run("something_else = 5")
    assert kernel.fetch("predictions") == (False, None)


def test_an_unpicklable_answer_is_found_but_carries_no_value(kernel):
    assert kernel.run("predictions = lambda x: x").ok
    found, value = kernel.fetch("predictions")
    assert found is True and value is None


def test_a_runaway_turn_is_killed_by_the_timeout():
    with PersistentKernel(KernelConfig(enabled=True, turn_timeout_seconds=2)) as instance:
        result = instance.run("while True:\n    pass")
        assert result.ok is False and result.timed_out is True
        assert instance.alive is False
        # A dead kernel reports rather than hanging the run.
        assert instance.run("x = 1").ok is False
        assert instance.fetch("predictions") == (False, None)


def test_credentials_are_stripped_from_the_kernel(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    with PersistentKernel(ENABLED) as instance:
        instance.run(
            "import os\n"
            "predictions = [os.environ.get(k) for k in "
            "('OPENROUTER_API_KEY', 'HF_TOKEN', 'HTTPS_PROXY')]"
        )
        assert instance.fetch("predictions") == (True, [None, None, None])


def test_turn_count_is_tracked(kernel):
    for _ in range(3):
        kernel.run("pass")
    assert kernel.turns == 3


def test_aion_is_importable_when_syspath_is_supplied():
    from aionbench.runner import _aion_syspath

    config = KernelConfig(enabled=True, turn_timeout_seconds=30, extra_syspath=_aion_syspath())
    with PersistentKernel(config) as instance:
        result = instance.run("import aion\npredictions = aion.__version__")
        assert result.ok, result.error
        found, value = instance.fetch("predictions")
        assert found and isinstance(value, str)


def test_aion_is_absent_from_the_baseline_kernel():
    """The control arm must not have Aion available by accident."""
    with PersistentKernel(KernelConfig(enabled=True, turn_timeout_seconds=30)) as instance:
        result = instance.run("import aion")
        assert result.ok is False
        assert "ModuleNotFoundError" in result.error or "ImportError" in result.error
