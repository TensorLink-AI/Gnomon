"""TSFM tier truthfulness: what the artifact and capabilities attest must
be what the runtime does.

Four defects fixed together (audited in docs/design/news-regime.md §1.4):
the sandbox root resolved to the cwd because ``Path("")`` is truthy; the
sandbox worker loaded Hub weights unpinned while the forecast id recorded
the pin; ``capabilities()["models"]["tsfm"]`` reported ``[]`` for a
working sandbox install; and ``models.tsfm.candidates`` was parsed but
never passed to ``evaluate``.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from gnomon import tsfm_sandbox
from gnomon.tsfm import TSFM_REVISIONS, resolved_weights


class TestSandboxRoot:
    def test_default_is_home_cache_not_cwd(self, monkeypatch):
        original = os.environ.get("GNOMON_TSFM_SANDBOX_ROOT")
        monkeypatch.delenv("GNOMON_TSFM_SANDBOX_ROOT", raising=False)
        module = importlib.reload(tsfm_sandbox)
        try:
            assert module.SANDBOX_ROOT == Path.home() / ".cache" / "gnomon-tsfm-venvs"
            assert module.SANDBOX_ROOT != Path(".")
        finally:
            if original is None:
                monkeypatch.delenv(
                    "GNOMON_TSFM_SANDBOX_ROOT", raising=False)
            else:
                monkeypatch.setenv("GNOMON_TSFM_SANDBOX_ROOT", original)
            importlib.reload(tsfm_sandbox)

    def test_env_override_is_honoured(self, monkeypatch, tmp_path):
        original = os.environ.get("GNOMON_TSFM_SANDBOX_ROOT")
        monkeypatch.setenv("GNOMON_TSFM_SANDBOX_ROOT", str(tmp_path / "roots"))
        module = importlib.reload(tsfm_sandbox)
        try:
            assert module.SANDBOX_ROOT == tmp_path / "roots"
        finally:
            if original is None:
                monkeypatch.delenv(
                    "GNOMON_TSFM_SANDBOX_ROOT", raising=False)
            else:
                monkeypatch.setenv("GNOMON_TSFM_SANDBOX_ROOT", original)
            importlib.reload(tsfm_sandbox)


class TestWorkerRevisionPinning:
    def test_every_worker_load_is_pinned(self):
        # Static invariant on the worker source: each from_pretrained call
        # carries revision=pinned(...) — an unpinned load would produce
        # numbers the forecast id's `model_weights` payload never covered.
        script = tsfm_sandbox.WORKER_SCRIPT
        assert script.count("from_pretrained(") == script.count("revision=pinned(")
        assert script.count("from_pretrained(") >= 8
        # The worker refuses, rather than falls back, when no pin arrives.
        assert "refusing an unpinned" in script

    def test_request_carries_the_pinned_revisions(self, monkeypatch, tmp_path):
        adapter = tsfm_sandbox.SubprocessAdapter("chronos_bolt_mini",
                                                 frequency="D")
        sandbox = tmp_path / "chronos_bolt_mini"
        (sandbox / "venv" / "bin").mkdir(parents=True)
        (sandbox / "venv" / "bin" / "python").touch()
        monkeypatch.setattr(tsfm_sandbox, "ensure_sandbox",
                            lambda name, force=False: sandbox)

        captured: dict = {}

        class FakeStdin:
            def write(self, value):
                captured["request"] = json.loads(value)
            def flush(self):
                pass

        class FakeStdout:
            def fileno(self):
                return 42
            def readline(self):
                return json.dumps({"point": [1.0, 1.0, 1.0]}) + "\n"

        class FakeProc:
            stdin = FakeStdin()
            stdout = FakeStdout()
            stderr = None
            def poll(self):
                return None

        monkeypatch.setattr(adapter, "_start_worker", lambda: FakeProc())
        monkeypatch.setattr(tsfm_sandbox.select, "select",
                            lambda *args: ([42], [], []))
        adapter.predict([1.0] * 30, 3, 7)
        expected = resolved_weights("chronos_bolt_mini")
        assert expected  # the adapter has pinned weights to send
        assert captured["request"]["revisions"] == expected
        sha = TSFM_REVISIONS["amazon/chronos-bolt-mini"]
        assert captured["request"]["revisions"]["amazon/chronos-bolt-mini"] == sha


class TestCapabilitiesTruthfulness:
    def test_sandbox_install_appears_under_models_tsfm(self, monkeypatch):
        import gnomon.runtime as runtime

        monkeypatch.setattr("gnomon.tsfm_sandbox.list_sandboxes",
                            lambda: ["chronos_bolt_mini"])
        caps = runtime.capabilities()
        assert "chronos_bolt_mini" in caps["models"]["tsfm"]
        assert caps["models"]["tsfm_sandboxes"] == ["chronos_bolt_mini"]


class TestConfigCandidatesReachSelection:
    def test_configured_candidates_restrict_the_eligible_note(self, tmp_path):
        # Without sandboxes, the "no foundation-model candidate competed"
        # note names the requested pool. Configuring one candidate must
        # narrow that pool — before the fix the config key never reached
        # `evaluate` and all seven adapters were named regardless.
        import csv
        from datetime import datetime, timedelta, timezone

        from gnomon.config import load_config
        from gnomon.runtime import forecast

        config_path = tmp_path / "gnomon.yaml"
        config_path.write_text(
            "models:\n  tsfm:\n    candidates: [chronos_bolt_mini]\n"
        )
        csv_path = tmp_path / "series.csv"
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "value"])
            for i in range(60):
                writer.writerow([(start + timedelta(days=i)).isoformat(),
                                 100.0 + (i % 5)])
        artifact, _ = forecast(
            str(csv_path), time_column="timestamp", target_column="value",
            horizon=7, frequency="D", output=str(tmp_path / "out"),
            config=load_config(str(config_path)),
        )
        notes = " ".join(artifact.results[0].notes)
        assert "chronos_bolt_mini" in notes
        assert "moment_small" not in notes
