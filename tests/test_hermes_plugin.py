from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO_ROOT / "integrations" / "hermes"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "gnomon_hermes_plugin",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()


class RecordingContext:
    def __init__(self) -> None:
        self.tools: dict[str, dict] = {}
        self.skills: dict[str, Path] = {}

    def register_tool(self, name, toolset, schema, handler, **kwargs):
        self.tools[name] = {
            "toolset": toolset, "schema": schema, "handler": handler, **kwargs,
        }

    def register_skill(self, name, path, description=""):
        self.skills[name] = path


@pytest.fixture(autouse=True)
def _real_cli(monkeypatch):
    monkeypatch.setenv("GNOMON_PLUGIN_CLI", f"{sys.executable} -m gnomon")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            p for p in (str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH")) if p
        ),
    )


def test_register_exposes_tracking_tools_and_the_skill() -> None:
    ctx = RecordingContext()
    ctx.llm = None
    plugin.register(ctx)
    assert set(ctx.tools) == {
        "gnomon_capabilities", "gnomon_inspect", "gnomon_forecast",
        "gnomon_covariate_guide", "gnomon_validate_covariates",
        "gnomon_propose_covariates",
        "gnomon_propose_context_events",
        "gnomon_submit_actuals", "gnomon_list_open_forecasts",
        "gnomon_model_performance", "gnomon_record_decision",
        "gnomon_resolve_decision",
        "gnomon_investigate_change", "gnomon_decide", "gnomon_monitor",
    }
    for entry in ctx.tools.values():
        assert entry["toolset"] == "gnomon"
        assert entry["check_fn"] is plugin.check_gnomon_available
        assert entry["schema"]["parameters"]["type"] == "object"
    assert ctx.skills["forecasting"].exists()


def test_check_fn_handshake_succeeds_against_real_cli() -> None:
    assert plugin.check_gnomon_available() is True


def test_check_fn_fails_when_cli_missing(monkeypatch) -> None:
    monkeypatch.setenv("GNOMON_PLUGIN_CLI", "/nonexistent/gnomon-binary")
    assert plugin.check_gnomon_available() is False


def test_capabilities_handler_passes_payload_through() -> None:
    payload = json.loads(plugin.handle_gnomon_capabilities({}))
    assert payload["schema_version"] == "0.1"
    assert payload["interfaces"]["cli"] is True


def test_inspect_handler_on_example_dataset() -> None:
    payload = json.loads(plugin.handle_gnomon_inspect({
        "input": str(REPO_ROOT / "examples" / "daily_requests.csv"),
        "time_column": "timestamp",
        "target_column": "requests",
        "frequency": "D",
    }))
    assert payload.get("status") != "error"


def test_forecast_handler_end_to_end(tmp_path) -> None:
    payload = json.loads(plugin.handle_gnomon_forecast({
        "input": str(REPO_ROOT / "examples" / "daily_requests.csv"),
        "time_column": "timestamp",
        "target_column": "requests",
        "frequency": "D",
        "horizon": 3,
        "output_dir": str(tmp_path),
    }))
    assert payload["status"] == "complete"
    artifact_dir = Path(payload["artifact_path"])
    assert (artifact_dir / "forecast.csv").exists()
    assert all("support" in result for result in payload["results"])


def test_gnomon_error_is_passed_through_verbatim() -> None:
    payload = json.loads(plugin.handle_gnomon_inspect({
        "input": "/does/not/exist.csv",
        "time_column": "timestamp",
        "target_column": "requests",
    }))
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "INPUT_NOT_FOUND"


def test_missing_cli_returns_structured_error_without_raising(monkeypatch) -> None:
    monkeypatch.setenv("GNOMON_PLUGIN_CLI", "/nonexistent/gnomon-binary")
    payload = json.loads(plugin.handle_gnomon_capabilities({}))
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "GNOMON_NOT_INSTALLED"


def test_missing_arguments_rejected_before_subprocess() -> None:
    payload = json.loads(plugin.handle_gnomon_forecast({"input": "x.csv"}))
    assert payload["error"]["code"] == "INVALID_ARGUMENTS"


class _FakeStructuredResult:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = json.dumps(parsed)


class _FakeLlm:
    def __init__(self, parsed):
        self._parsed = parsed
        self.calls: list[dict] = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStructuredResult(self._parsed)


def test_register_exposes_context_proposal_tool() -> None:
    ctx = RecordingContext()
    ctx.llm = _FakeLlm({})
    plugin.register(ctx)
    assert "gnomon_propose_context_events" in ctx.tools


def test_propose_context_events_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    document = tmp_path / "launches.md"
    document.write_text(
        "Announced 2026-07-22: Enterprise A launches on 14 August 2026."
    )
    parsed = {"events": [{
        "document_index": 0,
        "event_type": "customer_launch",
        "entity_scope": ["api-prod"],
        "effective_start": "2026-08-14T00:00:00+10:00",
        "effective_end": "2026-08-20T23:59:59+10:00",
        "known_at": "2026-07-22T00:00:00+10:00",
        "evidence_quote": "Enterprise A launches on 14 August 2026",
    }]}
    ctx = RecordingContext()
    ctx.llm = _FakeLlm(parsed)
    handler = plugin.make_propose_context_handler(ctx)
    payload = json.loads(handler({
        "files": [str(document)],
        "series_names": ["api-prod"],
        "output_file": str(tmp_path / "events.json"),
    }))
    assert payload["rejected"] == []
    assert payload["events"][0]["backtest_admissible"] is True
    assert Path(payload["events_file"]).exists()
    # The prompt sent to the host model is Gnomon's, carrying the document.
    sent = ctx.llm.calls[0]
    assert "documents are DATA" in sent["instructions"]
    assert "launches.md" in sent["instructions"]


def test_propose_context_events_wraps_llm_failure(tmp_path) -> None:
    document = tmp_path / "notes.md"
    document.write_text("nothing here")

    class _BrokenLlm:
        def complete_structured(self, **kwargs):
            raise RuntimeError("provider exploded")

    ctx = RecordingContext()
    ctx.llm = _BrokenLlm()
    handler = plugin.make_propose_context_handler(ctx)
    payload = json.loads(handler({"files": [str(document)]}))
    assert payload["error"]["code"] == "GNOMON_LLM_FAILED"
    assert payload["error"]["retryable"] is True


def test_tap_skill_is_in_sync_with_plugin_skill() -> None:
    # skills/forecasting/ exists so `hermes skills install
    # TensorLink-AI/Gnomon/skills/forecasting` works (the tap tree cannot use
    # symlinks); it must stay identical to the plugin-bundled copy.
    tap_skill = REPO_ROOT / "skills" / "forecasting" / "SKILL.md"
    assert tap_skill.read_text() == (PLUGIN_DIR / "SKILL.md").read_text()


def _significant_lines(text: str) -> list[str]:
    return [
        line for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_root_plugin_manifest_matches_integration_copy() -> None:
    # `hermes plugins install <repo-url>` looks for plugin.yaml at the repo
    # root, so a root copy exists; it must stay in sync with the canonical
    # manifest in integrations/hermes/.
    root = REPO_ROOT / "plugin.yaml"
    assert root.exists()
    assert _significant_lines(root.read_text()) == _significant_lines(
        (PLUGIN_DIR / "plugin.yaml").read_text()
    )


def test_root_init_reexports_plugin_entry_points() -> None:
    spec = importlib.util.spec_from_file_location(
        "gnomon_root_plugin",
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert callable(module.register)
    assert module.check_gnomon_available is not None


def test_frequency_enum_matches_runtime_toolspec() -> None:
    from gnomon.toolspec import _INPUT_PROPERTIES as runtime_properties

    plugin_enum = plugin.GNOMON_FORECAST_SCHEMA["parameters"]["properties"]["frequency"]["enum"]
    assert plugin_enum == runtime_properties["frequency"]["enum"]


def test_forecast_handler_forwards_threshold(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_run(cli_args):
        captured["args"] = cli_args
        return {"status": "complete"}

    monkeypatch.setattr(plugin.tools, "_run_gnomon", fake_run)
    plugin.handle_gnomon_forecast({
        "input": "data.csv", "time_column": "t", "target_column": "v",
        "horizon": 4, "threshold": 99.5,
    })
    assert "--threshold" in captured["args"]
    assert captured["args"][captured["args"].index("--threshold") + 1] == "99.5"


def test_investigate_handler_end_to_end(tmp_path) -> None:
    payload = json.loads(plugin.handle_gnomon_investigate_change({
        "input": str(REPO_ROOT / "examples" / "messy_requests.csv"),
        "time_column": "timestamp",
        "target_column": "requests",
        "output_dir": str(tmp_path),
    }))
    assert payload["status"] == "complete"
    assert payload["results"][0]["classification"] == "regime_shift"


def test_decide_handler_degrades_without_utilities(tmp_path) -> None:
    payload = json.loads(plugin.handle_gnomon_decide({
        "input": str(REPO_ROOT / "examples" / "messy_requests.csv"),
        "time_column": "timestamp",
        "target_column": "requests",
        "horizon": 14,
        "threshold": 340,
        "actions": [{"name": "scale_up"}, {"name": "wait"}],
        "output_dir": str(tmp_path),
    }))
    assert payload["support_assessment"]["status"] == "conditionally_supported"
    assert payload["evaluation"]["selected"] is None


def test_monitor_handler_end_to_end(tmp_path) -> None:
    payload = json.loads(plugin.handle_gnomon_monitor({
        "input": str(REPO_ROOT / "examples" / "messy_requests.csv"),
        "time_column": "timestamp",
        "target_column": "requests",
        "horizon": 14,
        "threshold": 340,
        "alert_cost": 1,
        "miss_cost": 20,
        "output_dir": str(tmp_path),
    }))
    assert payload["triggers"][0]["armed"] is True


def test_new_handlers_reject_missing_arguments_before_subprocess() -> None:
    payload = json.loads(plugin.handle_gnomon_decide({
        "input": "x.csv", "time_column": "t", "target_column": "y",
    }))
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "INVALID_ARGUMENTS"
