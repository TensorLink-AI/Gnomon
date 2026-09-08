from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gnomon import installation
from gnomon.build_info import build_info, source_build_info
from gnomon.contracts import GnomonError
from test_cli_workflow import invoke, process, source


def test_temporal_schema_and_errors_explain_interval_arguments():
    code, schema = invoke("temporal", "--schema")
    assert code == 0
    operations = {item["properties"]["operation"]["const"]: item for item in schema["oneOf"]}
    assert set(operations) == {"normalize", "duration", "shift", "interval", "order_events"}
    assert operations["interval"]["properties"]["left"]["required"] == ["start", "end"]
    for arguments, expected in (({}, "operation must be one of"), ({"operation": "interval"}, "left, right")):
        code, error = invoke("temporal", "--arguments", json.dumps(arguments))
        assert code == 2 and expected in error["error"]["message"]
        assert error["error"]["details"]["schema_command"] == "gnomon temporal --schema"
        example = error["error"]["details"]["example_arguments"]
        assert invoke("temporal", "--arguments", json.dumps(example))[0] == 0


@pytest.mark.parametrize("repair", ["off", "safe"])
@pytest.mark.parametrize("command", [("inspect",), ("infer", "--provider", "last_value", "--horizon", "2")])
def test_safe_gap_error_explains_why_interpolation_needs_an_opt_in(tmp_path, repair, command):
    code, result = invoke(*command, "--input", source(tmp_path, gap=5), "--repair", repair, "--frequency", "D")
    assert code == 2
    error = result["error"]
    assert error["code"] == "IRREGULAR_TIME_GRID"
    assert "Safe repair never fills missing values" in error["message"]
    assert error["details"]["repair_mode"] == repair
    assert "--repair aggressive" in error["repair_options"][1]["description"]


def test_python_passthrough_runs_in_the_reported_environment():
    code, env = process("environment")
    assert code == 0 and env["import_name"] == "gnomon" and env["distribution"] == "gnomon-forecast"
    assert env["build"] == build_info()
    result = subprocess.run([sys.executable, "-m", "gnomon", "python", "-c",
                             "import gnomon,json,sys; print(json.dumps({'python':sys.executable,'module':gnomon.__name__})); sys.exit(7)"],
                            text=True, capture_output=True)
    assert result.returncode == 7
    assert json.loads(result.stdout) == {"python": env["python"], "module": "gnomon"}


def test_build_identity_distinguishes_source_and_preserves_archived_commit(tmp_path, monkeypatch):
    package = tmp_path / "src" / "gnomon"
    package.mkdir(parents=True)
    (package / "product_contract.py").write_text('__version__ = "1.1.0.dev0"\n')
    (package / "module.py").write_text("VALUE = 1\n")
    monkeypatch.setenv("GNOMON_BUILD_COMMIT", "a" * 40)
    first = source_build_info(package)
    assert first["commit"] == "a" * 40 and "+gaaaaaaaaaaaa.s" in first["build_id"]
    (package / "_build_info.json").write_text(json.dumps(first))
    monkeypatch.delenv("GNOMON_BUILD_COMMIT")
    assert source_build_info(package) == first
    (package / "module.py").write_text("VALUE = 2\n")
    second = source_build_info(package)
    assert second["commit"] == first["commit"]
    assert second["source_sha256"] != first["source_sha256"] and second["dirty"] is True
    assert second["build_id"] != first["build_id"]


@pytest.fixture
def managed(tmp_path, monkeypatch):
    root, bins = tmp_path / "installs", tmp_path / "bin"
    bins.mkdir()
    releases = root / "releases"
    entries = []
    for index in range(1, 4):
        entry = releases / f"2026010{index}T000000Z-1"
        (entry / "bin").mkdir(parents=True)
        executable = entry / "bin" / "gnomon"
        executable.write_text("#!/bin/sh\nexit 19\n")
        executable.chmod(0o755)
        package = entry / "lib/python3.12/site-packages/gnomon"
        package.mkdir(parents=True)
        (package / "product_contract.py").write_text('__version__ = "1.0.1"\n')
        (package / "__init__.py").write_text(f"VALUE = {index}\n")
        entries.append(entry)
    command = bins / "gnomon"
    command.symlink_to(entries[-1] / "bin/gnomon")
    receipt = {"schema_version": 1, "release_id": entries[-1].name, "install_root": str(root),
               "command": str(command), "repository": "TensorLink-AI/Gnomon", "requested_ref": "main",
               "installed_at": "2026-01-03T00:00:00Z", "build": build_info()}
    (entries[-1] / installation.RECEIPT).write_text(json.dumps(receipt))
    monkeypatch.setattr(sys, "prefix", str(entries[-1]))
    return root, command, entries


def test_releases_distinguish_legacy_builds_without_executing_them(managed):
    _, _, entries = managed
    result = installation.releases()
    by_id = {entry["release_id"]: entry for entry in result["releases"]}
    a, b = (by_id[entry.name]["build"] for entry in entries[:2])
    assert a["package_version"] == b["package_version"] == "1.0.1"
    assert a["source_sha256"] != b["source_sha256"]
    assert a["commit"] is None and a["provenance"] == "legacy_install_commit_unknown"
    assert by_id[entries[-1].name]["active"] is True


def test_pruning_requires_apply_and_protects_active_running_and_installing(managed):
    root, command, entries = managed
    installation.rollback(entries[0].name)
    assert command.resolve() == entries[0] / "bin/gnomon"
    installing = root / "releases" / "20260104T000000Z-1"
    installing.mkdir()
    (installing / ".installing").touch()
    preview = installation.releases(prune=True, keep=0)
    assert preview["prune_candidates"] == [entries[1].name]
    assert preview["removed"] == [] and entries[1].exists()
    applied = installation.releases(prune=True, keep=0, apply=True)
    assert applied["removed"] == [entries[1].name]
    assert entries[0].exists() and entries[-1].exists() and installing.exists()
    assert not entries[1].exists()


def test_default_pruning_keeps_one_inactive_release(managed):
    _, _, entries = managed
    result = installation.releases(prune=True)
    assert result["prune_candidates"] == [entries[0].name]


@pytest.mark.parametrize("release_id", ["../escape", "/tmp/escape", ".", "..", "missing"])
def test_rollback_rejects_invalid_targets_without_changing_active(managed, release_id):
    _, command, entries = managed
    with pytest.raises(GnomonError):
        installation.rollback(release_id)
    assert command.resolve() == entries[-1] / "bin/gnomon"


def test_release_symlinks_cannot_escape_management(managed, tmp_path):
    root, _, _ = managed
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "releases" / "escape").symlink_to(elsewhere, target_is_directory=True)
    assert "escape" not in [entry["release_id"] for entry in installation.releases()["releases"]]
    with pytest.raises(GnomonError):
        installation.rollback("escape")
    installation.releases(prune=True, apply=True, keep=0)
    assert elsewhere.is_dir()


def test_failed_update_keeps_active_release_and_clears_local_override(managed, monkeypatch):
    root, command, entries = managed
    (entries[-1] / "install.sh").write_text("unused")
    monkeypatch.setenv("GNOMON_LOCAL", "1")
    calls = []
    def fail(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(installation.subprocess, "run", fail)
    with pytest.raises(GnomonError, match="previous command remains active"):
        installation.update("main")
    assert command.resolve() == entries[-1] / "bin/gnomon"
    assert calls[0][0][-3:] == [str(root), "--bin-dir", str(command.parent)]
    assert "GNOMON_LOCAL" not in calls[0][1]["env"]


def test_update_to_legacy_build_returns_a_working_management_command(managed, monkeypatch):
    _, command, entries = managed
    (entries[-1] / "install.sh").write_text("unused")
    def succeed(*args, **kwargs):
        command.unlink()
        command.symlink_to(entries[0] / "bin/gnomon")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(installation.subprocess, "run", succeed)
    result = installation.update("v1.0.1")
    assert result["active_release"] == entries[0].name
    assert Path(result["management_command"]) == entries[-1] / "bin/gnomon"


def test_release_management_on_pip_install_explains_how_to_find_python(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    with pytest.raises(GnomonError, match="gnomon environment") as exc:
        installation.releases()
    assert exc.value.code == "NOT_MANAGED_INSTALL"


@pytest.mark.parametrize("receipt", ["{", "[]", "{}", '{"install_root":false}'])
def test_malformed_install_receipt_refuses_management_with_a_specific_error(managed, receipt):
    _, command, entries = managed
    (entries[-1] / installation.RECEIPT).write_text(receipt)
    with pytest.raises(GnomonError) as exc:
        installation.releases(prune=True, apply=True, keep=0)
    assert exc.value.code == "INVALID_INSTALL_RECEIPT"
    assert all(entry.is_dir() for entry in entries)
    assert command.resolve() == entries[-1] / "bin/gnomon"


def test_failed_installer_removes_incomplete_environment_and_keeps_active(tmp_path):
    root, bins, fakes = (tmp_path / name for name in ("installs", "bin", "fake-bin"))
    previous = root / "releases" / "previous"
    (previous / "bin").mkdir(parents=True)
    (previous / "bin/gnomon").write_text("previous")
    bins.mkdir()
    (bins / "gnomon").symlink_to(previous / "bin/gnomon")
    fakes.mkdir()
    python = fakes / "python3.13"
    python.write_text('''#!/bin/sh
if [ "$1" = "-c" ]; then exit 0; fi
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
  mkdir -p "$3/bin"
  printf '#!/bin/sh\\nexit 1\\n' > "$3/bin/python"
  chmod +x "$3/bin/python"
  exit 0
fi
exit 1
''')
    python.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "install.sh"
    env = {**os.environ, "PATH": str(fakes) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(["bash", str(script), "--local", "--install-root", str(root), "--bin-dir", str(bins)],
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert (bins / "gnomon").resolve() == previous / "bin/gnomon"
    assert list((root / "releases").iterdir()) == [previous]
