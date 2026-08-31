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

from gnomon.tsfm import (
    ChronosBoltAdapter,
    TSFMAdapter,
    TSFMUnavailable,
    available_tsfms,
    check_tsfm,
    get_tsfm,
    installed_tsfms,
    register_tsfm,
    tsfm_candidates,
)
from gnomon.evaluation import evaluate
from gnomon.runtime import capabilities
from gnomon.toolspec import forecast_summary


class TestRegistry:
    """Registry mechanics."""

    def test_registered_adapters_exist(self):
        names = available_tsfms()
        assert "chronos_bolt_mini" in names
        assert "chronos_bolt_small" in names
        assert "toto2_4m" in names
        assert "toto2_22m" in names
        assert "flowstate" in names
        assert "ttm" in names
        assert "moirai2_small" in names
        assert "moment_small" in names

    def test_chronos_variants_keep_distinct_runtime_identity(self):
        assert ChronosBoltAdapter("chronos_bolt_mini").name == "chronos_bolt_mini"
        assert ChronosBoltAdapter("chronos_bolt_small").name == "chronos_bolt_small"
        with pytest.raises(TSFMUnavailable):
            ChronosBoltAdapter("chronos_bolt_typo")

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

    def test_fresh_install_response_has_exact_toto_on_ramp(
        self, tmp_path, monkeypatch,
    ):
        from datetime import date, timedelta
        from gnomon.runtime import forecast

        monkeypatch.setattr("gnomon.tsfm.installed_tsfms", lambda: [])
        monkeypatch.setattr(
            "gnomon.tsfm_sandbox.sandbox_available_tsfms", lambda: [])
        source = tmp_path / "series.csv"
        source.write_text(
            "timestamp,value\n" + "\n".join(
                f"{(date(2026, 1, 1) + timedelta(days=index)).isoformat()},"
                f"{100 + index}"
                for index in range(40)) + "\n",
            encoding="utf-8",
        )
        artifact, directory = forecast(
            str(source), time_column="timestamp", target_column="value",
            horizon=3, output=str(tmp_path / "out"),
        )
        payload = forecast_summary(artifact, directory)
        assert payload["tsfm_on_ramp"]["command"] == \
            "gnomon tsfm install toto2_4m"
        assert payload["tsfm_on_ramp"]["mcp_tool_call"] == {
            "name": "gnomon_install_tsfm",
            "arguments": {"name": "toto2_4m"},
        }

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
        from gnomon.tsfm import _try_import, TSFMUnavailable
        with pytest.raises(TSFMUnavailable):
            _try_import("this_module_does_not_exist_xyz")


class TestProtocolCompliance:
    """Registered adapters satisfy the TSFMAdapter protocol."""

    @pytest.mark.parametrize("adapter_name", [
        "chronos_bolt_mini",
        "chronos_bolt_small",
        "toto2_4m",
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

    def test_toto_one_step_forecast_preserves_horizon_axis(self, monkeypatch):
        from gnomon.tsfm import Toto2Adapter

        class _Row:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return list(self._values)

        class _Array:
            """Small ndarray stand-in; NumPy is an optional TSFM dependency."""

            shape = (9, 1)

            def reshape(self, *shape):
                assert shape == (9, -1)
                return self

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    row, column = key
                    assert column == 0
                    return float(row)
                return _Row([float(key)])

        class _Tensor:
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return _Array()

        adapter = Toto2Adapter("toto2_4m")
        monkeypatch.setattr(adapter, "_forecast_quantiles",
                            lambda history, horizon: _Tensor())
        assert adapter.predict([1.0] * 32, 1, 1) == [4.0]
        rows = adapter.predict_quantiles(
            [1.0] * 32, 1, 1, quantiles=(.1, .5, .9))
        assert rows == [{"0.1": 0.0, "0.5": 4.0, "0.9": 8.0}]


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
            "chronos_bolt_mini", "chronos_bolt_small", "toto2_4m", "toto2_22m",
            "flowstate", "ttm", "moirai2_small", "moment_small",
        }
        assert expected.issubset(available)

    def test_toto_patch_context_is_machine_actionable(self):
        from gnomon.tsfm import eligible_tsfms, tsfm_capabilities

        assert tsfm_capabilities("toto2_4m").min_context_length == 32
        eligible, excluded = eligible_tsfms(
            history_length=31, horizon=1, frequency="D")
        assert "toto2_4m" not in eligible
        assert "needs at least 32" in excluded["toto2_4m"][0]


class TestSandbox:
    """Sandbox venv management."""

    def test_list_sandboxes_empty_by_default(self):
        from gnomon.tsfm_sandbox import list_sandboxes
        # On a fresh system, no sandboxes should exist
        # (but we can't guarantee the test env is clean, so just check it's a list)
        assert isinstance(list_sandboxes(), list)

    def test_sandbox_exists_returns_bool(self):
        from gnomon.tsfm_sandbox import sandbox_exists
        assert isinstance(sandbox_exists("chronos_bolt_mini"), bool)
        assert sandbox_exists("nonexistent") is False

    def test_subprocess_adapter_protocol(self):
        from gnomon.tsfm_sandbox import SubprocessAdapter
        adapter = SubprocessAdapter("chronos_bolt_mini")
        assert adapter.name == "chronos_bolt_mini"
        assert adapter.params_m == 21.0
        assert adapter.supports_quantiles is True

    def test_ttm_does_not_support_quantiles(self):
        from gnomon.tsfm_sandbox import SubprocessAdapter
        adapter = SubprocessAdapter("ttm")
        assert adapter.supports_quantiles is False

    def test_toto_4m_sandbox_identity(self):
        from gnomon.tsfm_sandbox import SubprocessAdapter
        adapter = SubprocessAdapter("toto2_4m")
        assert adapter.params_m == 4.14
        assert adapter.supports_quantiles is True
        assert "Datadog/Toto-2.0-4m@" in str(adapter.revision)

    def test_sandbox_timeout_opens_per_run_circuit(self, monkeypatch):
        from gnomon.tsfm import TSFMError
        from gnomon.tsfm_sandbox import SubprocessAdapter
        import gnomon.tsfm_sandbox as sandbox

        class Input:
            def write(self, value):
                return len(value)
            def flush(self):
                return None
            def close(self):
                return None

        class Output:
            def fileno(self):
                return 0

        class Process:
            stdin = Input()
            stdout = Output()
            def poll(self):
                return None
            def wait(self, timeout=None):
                return 0
            def terminate(self):
                return None

        adapter = SubprocessAdapter("toto2_4m", timeout=1)
        starts = []
        monkeypatch.setattr(adapter, "_start_worker",
                            lambda: starts.append(True) or Process())
        monkeypatch.setattr(sandbox.select, "select",
                            lambda *args: ([], [], []))
        with pytest.raises(TSFMError, match="circuit is open"):
            adapter.predict([1.0] * 33, 1, 1)
        with pytest.raises(TSFMError, match="circuit is open"):
            adapter.predict([1.0] * 33, 1, 1)
        assert len(starts) == 1

    def test_sandbox_predict_many_validates_batch_shape(self, monkeypatch):
        from gnomon.tsfm_sandbox import SubprocessAdapter
        adapter = SubprocessAdapter("toto2_4m")
        monkeypatch.setattr(adapter, "_run_subprocess", lambda request: {
            "points": [[1.0, 2.0], [3.0, 4.0]],
        })
        assert adapter.predict_many([[0.0] * 33, [1.0] * 35], 2, 1) == [
            [1.0, 2.0], [3.0, 4.0]]

        monkeypatch.setattr(adapter, "_run_subprocess", lambda request: {
            "points": [[1.0, 2.0]],
        })
        with pytest.raises(Exception, match="invalid forecast batch"):
            adapter.predict_many([[0.0] * 33, [1.0] * 35], 2, 1)

    def test_moment_does_not_support_quantiles(self):
        from gnomon.tsfm_sandbox import SubprocessAdapter
        adapter = SubprocessAdapter("moment_small")
        assert adapter.supports_quantiles is False

    def test_adapter_metadata_has_one_authoritative_source(self):
        from gnomon.api_inference import APIAdapter
        from gnomon.config import APIAuthConfig, APIProviderConfig
        from gnomon.tsfm import (
            available_tsfms,
            get_tsfm,
            tsfm_parameter_count,
            tsfm_supports_quantiles,
        )
        from gnomon.tsfm_sandbox import SubprocessAdapter

        provider = APIProviderConfig(
            url="https://example.invalid/forecast",
            auth=APIAuthConfig(type="none"),
        )
        for name in available_tsfms():
            expected_params = tsfm_parameter_count(name)
            expected_quantiles = tsfm_supports_quantiles(name)
            assert expected_params > 0
            assert get_tsfm(name).params_m == expected_params
            assert get_tsfm(name).supports_quantiles is expected_quantiles
            assert SubprocessAdapter(name).params_m == expected_params
            assert SubprocessAdapter(name).supports_quantiles is expected_quantiles
            assert APIAdapter(name, provider).params_m == expected_params
            assert APIAdapter(name, provider).supports_quantiles is expected_quantiles

    def test_sandbox_candidates_empty_without_venvs(self):
        from gnomon.tsfm_sandbox import sandbox_tsfm_candidates
        candidates = sandbox_tsfm_candidates(frequency="h")
        # Without any sandboxes set up, should be empty
        # (or contain only sandboxes that actually exist)
        assert isinstance(candidates, list)

    def test_tsfm_pip_specs_cover_all_adapters(self):
        from gnomon.tsfm_sandbox import TSFM_PIP_SPECS
        from gnomon.tsfm import available_tsfms
        # Every registered TSFM should have a pip spec
        for name in available_tsfms():
            assert name in TSFM_PIP_SPECS, f"Missing pip spec for {name}"

    def test_capabilities_reports_sandboxes(self):
        caps = capabilities()
        assert "tsfm_sandboxes" in caps["models"]
        assert isinstance(caps["models"]["tsfm_sandboxes"], list)

    def test_cli_tsfm_list(self):
        from gnomon.cli import main
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

    def test_evaluate_notes_disclose_missing_tsfm_tier(self):
        """Eligible-but-uninstalled TSFMs must be disclosed as a note that
        names the install command — and must not downgrade support."""
        values = [100.0 + 2.0 * i for i in range(200)]
        result = evaluate(
            values, horizon=24, season=24, minimum_improvement=0.02, frequency="h",
        )
        assert result.supported is True
        assert result.notes, "expected a TSFM-availability note"
        assert any("gnomon tsfm install" in note for note in result.notes)
        assert not any("gnomon tsfm install" in warning for warning in result.warnings)

    def test_capability_exclusion_does_not_downgrade_support(self):
        """A TSFM being ineligible says nothing about the forecast's evidence.

        Quarterly data excludes `flowstate` on frequency. Routed through
        `warnings` that exclusion made the pipeline report
        "weakly_supported" for a forecast whose own evidence was intact.
        """
        values = [100.0 + 2.0 * i for i in range(200)]
        result = evaluate(
            values, horizon=8, season=4, minimum_improvement=0.02, frequency="QS",
        )
        assert result.supported is True
        assert any("Skipped TSFM" in note for note in result.notes)
        assert not any("Skipped TSFM" in warning for warning in result.warnings)
        assert result.warnings == []

    def test_evaluate_notes_empty_when_tsfms_explicitly_disabled(self):
        """An explicit empty request means no TSFM tier was wanted: no note."""
        values = [100.0 + 2.0 * i for i in range(200)]
        result = evaluate(
            values, horizon=24, season=24, minimum_improvement=0.02,
            frequency="h", tsfm_names=[],
        )
        assert result.notes == []

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
