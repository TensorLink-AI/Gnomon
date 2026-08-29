"""TSFM installation is reachable from the tool surface.

The 2026-08 MCP evaluation's highest-priority open item:
`gnomon_capabilities` disclosed seven eligible foundation models, but
installing one required a shell — unreachable for the agents the MCP
server exists to serve. The constraint is that `ensure_sandbox` can
legitimately run for minutes (torch dominates), and an MCP call blocking
that long looks like a hang to every interactive host. So the tool
starts a *detached* install and reports state on each call; the
`.gnomon-sandbox-ready` marker stays the single source of truth for
success, exactly as it is for the CLI path.
"""

from __future__ import annotations

import os

import pytest

from gnomon.contracts import GnomonError


def _any_tsfm() -> str:
    from gnomon.tsfm_sandbox import TSFM_PIP_SPECS

    return sorted(TSFM_PIP_SPECS)[0]


@pytest.fixture
def sandbox_root(monkeypatch, tmp_path):
    import gnomon.tsfm_sandbox as sandbox

    root = tmp_path / "sandboxes"
    monkeypatch.setattr(sandbox, "SANDBOX_ROOT", root)
    return root


def test_unknown_name_refuses_with_the_available_list(sandbox_root) -> None:
    from gnomon.toolspec import _run_install_tsfm

    with pytest.raises(GnomonError) as raised:
        _run_install_tsfm({"name": "chronos_bolt_titan"})
    assert raised.value.code == "UNKNOWN_TSFM"
    assert raised.value.details["available"]


def test_remove_unknown_name_never_resolves_a_filesystem_target(
        sandbox_root) -> None:
    from gnomon.tsfm import TSFMUnavailable
    from gnomon.tsfm_sandbox import remove_sandbox

    sentinel = sandbox_root.parent / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(TSFMUnavailable):
        remove_sandbox("..")
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_absent_sandbox_reports_absent_without_side_effects(sandbox_root) -> None:
    from gnomon.toolspec import _run_install_tsfm

    payload = _run_install_tsfm({"name": _any_tsfm(), "status_only": True})
    assert payload["state"] == "absent"
    assert not sandbox_root.exists()


def test_install_starts_detached_and_reports_installing(
        sandbox_root, monkeypatch) -> None:
    import gnomon.tsfm_sandbox as sandbox
    from gnomon.toolspec import _run_install_tsfm

    launched: list[list[str]] = []

    class FakeProcess:
        pid = os.getpid()  # alive, so a re-poll must say "installing"

    def fake_popen(argv, **kwargs):
        launched.append(argv)
        assert kwargs.get("start_new_session") is True
        return FakeProcess()

    monkeypatch.setattr(sandbox.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sandbox, "_uv_available", lambda: True)

    name = _any_tsfm()
    payload = _run_install_tsfm({"name": name})
    assert payload["state"] == "installing"
    assert payload["pid"] == os.getpid()
    assert launched and launched[0][-1] == name
    assert "tsfm" in launched[0] and "install" in launched[0]

    # Polling reports the running install; a second start does not spawn
    # a second process.
    assert _run_install_tsfm({"name": name, "status_only": True})["state"] \
        == "installing"
    assert _run_install_tsfm({"name": name})["state"] == "installing"
    assert len(launched) == 1


def test_dead_install_without_marker_reports_failed_with_log(
        sandbox_root) -> None:
    import gnomon.tsfm_sandbox as sandbox
    from gnomon.toolspec import _run_install_tsfm

    name = _any_tsfm()
    directory = sandbox._sandbox_dir(name)
    directory.mkdir(parents=True)
    (directory / "install.pid").write_text("999999999")
    (directory / "install.log").write_text("resolution failed: no torch wheel")

    payload = _run_install_tsfm({"name": name, "status_only": True})
    assert payload["state"] == "failed"
    assert "no torch wheel" in payload["log_tail"]


def test_ready_sandbox_short_circuits_the_install(sandbox_root, monkeypatch) -> None:
    import gnomon.tsfm_sandbox as sandbox
    from gnomon.toolspec import _run_install_tsfm

    def exploding_popen(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("a ready sandbox must never be rebuilt")

    monkeypatch.setattr(sandbox.subprocess, "Popen", exploding_popen)
    name = _any_tsfm()
    directory = sandbox._sandbox_dir(name)
    directory.mkdir(parents=True)
    (directory / ".gnomon-sandbox-ready").write_text(f"tsfm={name}\n")

    payload = _run_install_tsfm({"name": name})
    assert payload["state"] == "ready"
    assert payload["sandbox_path"] == str(directory)


def test_ensure_sandbox_materializes_worker_before_ready_marker(
        sandbox_root, monkeypatch) -> None:
    """A ready sandbox must remain executable in a read-only deployment;
    first inference may not be responsible for creating its worker."""
    import gnomon.tsfm_sandbox as sandbox

    class Result:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(sandbox, "_uv_available", lambda: True)
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: Result())

    name = _any_tsfm()
    directory = sandbox.ensure_sandbox(name)
    worker = directory / "worker.py"
    marker = directory / ".gnomon-sandbox-ready"
    assert worker.read_text(encoding="utf-8") == sandbox.WORKER_SCRIPT
    assert marker.is_file()

    # A legacy ready marker is repaired by an explicit install/check rather
    # than deferring the write to the model's first forecast.
    worker.unlink()
    assert sandbox.ensure_sandbox(name) == directory
    assert worker.read_text(encoding="utf-8") == sandbox.WORKER_SCRIPT


def test_missing_uv_refuses_with_repair_options(sandbox_root, monkeypatch) -> None:
    import gnomon.tsfm_sandbox as sandbox
    from gnomon.toolspec import _run_install_tsfm

    monkeypatch.setattr(sandbox, "_uv_available", lambda: False)
    with pytest.raises(GnomonError) as raised:
        _run_install_tsfm({"name": _any_tsfm()})
    assert raised.value.code == "SANDBOX_UNAVAILABLE"


def test_tool_is_on_the_full_surface_only(monkeypatch) -> None:
    """Install belongs to the full profile: the narrowed profiles exist to
    shrink context pressure, and none of them advertised TSFMs before."""
    from gnomon.toolspec import PROFILES, visible_tools

    monkeypatch.setenv("GNOMON_MCP_PROFILE", "full")
    assert "gnomon_install_tsfm" in {tool["name"] for tool in visible_tools()}
    for profile in PROFILES:
        monkeypatch.setenv("GNOMON_MCP_PROFILE", profile)
        assert "gnomon_install_tsfm" not in {
            tool["name"] for tool in visible_tools()}, profile


def test_capabilities_point_at_the_tool(monkeypatch) -> None:
    from gnomon.runtime import capabilities

    models = capabilities()["models"]
    assert models["tsfm_install_tool"] == "gnomon_install_tsfm"
    assert "gnomon_install_tsfm" in models["tsfm_install_note"]
