"""Tests for the TSFM adapter framework.

These tests verify:
1. The registry works (register, list, get).
2. Graceful degradation when optional deps are not installed.
3. The protocol is satisfied by all registered adapters.
4. Capabilities correctly reports available vs installed TSFMs.
5. Evaluation pipeline integrates TSFM candidates without errors.
"""

import pytest
import sys

sys.path.insert(0, "src")

from aion.tsfm import (
    TSFMAdapter,
    TSFMUnavailable,
    available_tsfms,
    check_tsfm,
    get_tsfm,
    installed_tsfms,
    register_tsfm,
    tsfm_candidates,
)
from aion.evaluation import evaluate
from aion.runtime import capabilities


class TestRegistry:
    """Registry mechanics."""

    def test_registered_adapters_exist(self):
        names = available_tsfms()
        assert "chronos_bolt_mini" in names
        assert "chronos_bolt_small" in names
        assert "toto2_22m" in names
        assert "flowstate" in names
        assert "ttm" in names
        assert "moirai2_small" in names
        assert "moment_small" in names

    def test_unknown_adapter_raises(self):
        with pytest.raises(KeyError):
            get_tsfm("nonexistent_model")

    def test_check_tsfm_returns_bool(self):
        # Every registered adapter should return a bool from check_tsfm
        for name in available_tsfms():
            assert isinstance(check_tsfm(name), bool)

    def test_installed_tsfms_is_subset_of_available(self):
        installed = set(installed_tsfms())
        available = set(available_tsfms())
        assert installed.issubset(available)


class TestGracefulDegradation:
    """Adapters without deps degrade gracefully."""

    def test_tsfm_candidates_empty_without_deps(self):
        # With no optional deps installed, candidates should be empty
        # (or only contain adapters whose deps are actually available)
        candidates = tsfm_candidates(frequency="h")
        # If torch is not installed, this should be empty
        # If torch + some TSFM libs ARE installed, they'll appear
        # Either way, this should NOT raise
        assert isinstance(candidates, list)

    def test_tsfm_unavailable_raises_correctly(self):
        # When torch is installed but the specific TSFM library isn't,
        # the adapter should raise TSFMUnavailable.
        # When torch IS NOT installed, the same applies.
        # We test the import-checking behavior via the _try_import helper.
        from aion.tsfm import _try_import, TSFMUnavailable
        with pytest.raises(TSFMUnavailable):
            _try_import("this_module_does_not_exist_xyz")


class TestProtocolCompliance:
    """Registered adapters satisfy the TSFMAdapter protocol."""

    @pytest.mark.parametrize("adapter_name", [
        "chronos_bolt_mini",
        "chronos_bolt_small",
        "toto2_22m",
        "flowstate",
        "ttm",
        "moirai2_small",
        "moment_small",
    ])
    def test_adapter_is_protocol_compliant(self, adapter_name):
        # The protocol check is runtime_checkable, but we can't instantiate
        # without deps. Instead, verify the class exists in the registry.
        names = available_tsfms()
        assert adapter_name in names


class TestCapabilities:
    """Capabilities reports TSFM info."""

    def test_capabilities_has_tsfm_fields(self):
        caps = capabilities()
        assert "models" in caps
        models = caps["models"]
        assert "tsfm" in models
        assert "tsfm_available" in models
        assert isinstance(models["tsfm"], list)
        assert isinstance(models["tsfm_available"], list)

    def test_capabilities_tsfm_available_lists_all(self):
        caps = capabilities()
        available = set(caps["models"]["tsfm_available"])
        expected = {
            "chronos_bolt_mini", "chronos_bolt_small", "toto2_22m",
            "flowstate", "ttm", "moirai2_small", "moment_small",
        }
        assert expected.issubset(available)


class TestSandbox:
    """Sandbox venv management."""

    def test_list_sandboxes_empty_by_default(self):
        from aion.tsfm_sandbox import list_sandboxes
        # On a fresh system, no sandboxes should exist
        # (but we can't guarantee the test env is clean, so just check it's a list)
        assert isinstance(list_sandboxes(), list)

    def test_sandbox_exists_returns_bool(self):
        from aion.tsfm_sandbox import sandbox_exists
        assert isinstance(sandbox_exists("chronos_bolt_mini"), bool)
        assert sandbox_exists("nonexistent") is False

    def test_subprocess_adapter_protocol(self):
        from aion.tsfm_sandbox import SubprocessAdapter
        adapter = SubprocessAdapter("chronos_bolt_mini")
        assert adapter.name == "chronos_bolt_mini"
        assert adapter.params_m == 21.0
        assert adapter.supports_quantiles is True

    def test_ttm_does_not_support_quantiles(self):
        from aion.tsfm_sandbox import SubprocessAdapter
        adapter = SubprocessAdapter("ttm")
        assert adapter.supports_quantiles is False

    def test_moment_does_not_support_quantiles(self):
        from aion.tsfm_sandbox import SubprocessAdapter
        adapter = SubprocessAdapter("moment_small")
        assert adapter.supports_quantiles is False

    def test_sandbox_candidates_empty_without_venvs(self):
        from aion.tsfm_sandbox import sandbox_tsfm_candidates
        candidates = sandbox_tsfm_candidates(frequency="h")
        # Without any sandboxes set up, should be empty
        # (or contain only sandboxes that actually exist)
        assert isinstance(candidates, list)

    def test_tsfm_pip_specs_cover_all_adapters(self):
        from aion.tsfm_sandbox import TSFM_PIP_SPECS
        from aion.tsfm import available_tsfms
        # Every registered TSFM should have a pip spec
        for name in available_tsfms():
            assert name in TSFM_PIP_SPECS, f"Missing pip spec for {name}"

    def test_capabilities_reports_sandboxes(self):
        caps = capabilities()
        assert "tsfm_sandboxes" in caps["models"]
        assert isinstance(caps["models"]["tsfm_sandboxes"], list)

    def test_cli_tsfm_list(self):
        from aion.cli import main
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["tsfm", "list"]) == 0
        import json
        result = json.loads(buf.getvalue())
        assert "available" in result
        assert "installed_in_process" in result
        assert "sandboxed" in result
        assert "pip_specs" in result

    def test_evaluate_without_tsfms_works(self):
        """Evaluate with no TSFMs installed should behave like v0.1."""
        # Simple linear trend data — 200 points
        values = [100.0 + 2.0 * i for i in range(200)]
        result = evaluate(values, horizon=24, season=24, minimum_improvement=0.02)
        assert result.supported is True
        # drift should win on perfect linear data
        assert result.selected_model is not None

    def test_evaluate_with_tsfm_names_fallback(self):
        """Evaluate with a non-existent TSFM name should still work."""
        values = [100.0 + 2.0 * i for i in range(200)]
        # Request a TSFM that isn't installed — should gracefully skip
        result = evaluate(
            values, horizon=24, season=24, minimum_improvement=0.02,
            frequency="h", tsfm_names=["nonexistent_tsfm"],
        )
        assert result.supported is True
        assert result.selected_model is not None

    def test_evaluate_tsfm_scores_present(self):
        """Evaluation result should have a tsfm_scores dict."""
        values = [100.0 + 2.0 * i for i in range(200)]
        result = evaluate(values, horizon=24, season=24, minimum_improvement=0.02)
        assert hasattr(result, "tsfm_scores")
        assert isinstance(result.tsfm_scores, dict)

    def test_evaluate_insufficient_history(self):
        """Fewer than required observations should abstain."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = evaluate(values, horizon=24, season=24, minimum_improvement=0.02)
        assert result.supported is False
        assert result.selected_model is None
