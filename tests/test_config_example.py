"""The shipped configuration example must load from a bare install, exactly as a user runs it."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "gnomon.toml.example"

# A user's shell: no Ephemeris variables, no PYTHONPATH, no example plugin.
_SCRUBBED = {k: v for k, v in os.environ.items()
             if k not in {"EPHEMERIS_BASE_URL", "EPHEMERIS_API_TOKEN", "PYTHONPATH"}}


def _capabilities(config, environment=_SCRUBBED, cwd=None):
    completed = subprocess.run(
        [sys.executable, "-I", "-m", "gnomon", "capabilities", "--providers-config", str(config)],
        capture_output=True, text=True, env=environment, cwd=cwd,
    )
    return completed.returncode, json.loads(completed.stdout)


def test_the_example_plugin_is_not_installed_in_the_test_interpreter():
    probe = subprocess.run([sys.executable, "-I", "-c", "import gnomon_example_provider"],
                           capture_output=True, text=True, env=_SCRUBBED)
    assert probe.returncode != 0, "this suite must prove the example loads without the separate plugin"


def test_every_example_key_is_accepted_by_the_configuration_schema():
    from gnomon.session import configuration_schema
    config = tomllib.loads(EXAMPLE.read_text())
    assert set(config) <= set(configuration_schema()["properties"]), sorted(config)
    assert config["schema_version"] == 1
    assert {spec["kind"] for spec in config["providers"].values()} == {"callable"}
    entrypoint = next(iter(config["providers"].values()))["entrypoint"]
    assert entrypoint.startswith("gnomon."), "the callable example must ship inside the wheel"


def test_copied_example_loads_clean_on_first_run_with_no_env_vars_and_no_plugin(tmp_path):
    # Copy to the documented name in an empty directory, like a fresh user.
    config = tmp_path / "providers.toml"
    config.write_text(EXAMPLE.read_text())
    code, result = _capabilities(config, cwd=tmp_path)
    assert code == 0 and result["status"] == "ok", result
    assert result["providers"]["my-model"]["revision"] == "my-model/0.0.0"
    assert result["ephemeris"]["configured"] is False and "setup" in result["ephemeris"]
    assert not [name for name in result["providers"] if name.startswith("ephemeris")]
    # The example names a ledger without creating one during discovery.
    assert result["ledger"]["configured"] is True and result["ledger"]["opened"] is False
    assert not (tmp_path / "gnomon-ledger.db").exists()


def test_uncommented_ephemeris_block_is_valid_and_registers_without_network(tmp_path):
    lines = EXAMPLE.read_text().splitlines()
    block = {"# [providers.ephemeris]", '# kind = "ephemeris"', '# base_url_env = "EPHEMERIS_BASE_URL"',
             '# token_env = "EPHEMERIS_API_TOKEN"', "# discover = false"}
    assert block <= set(lines), "the commented Ephemeris block must stay verbatim so users can uncomment it"
    config = tmp_path / "providers.toml"
    config.write_text("\n".join(line[2:] if line in block else line for line in lines) + "\n")
    # A loopback port with nothing listening: configured, never contacted.
    code, result = _capabilities(config, {**_SCRUBBED, "EPHEMERIS_BASE_URL": "http://127.0.0.1:9",
                                          "EPHEMERIS_API_TOKEN": "unused-in-tests"}, cwd=tmp_path)
    assert code == 0 and result["status"] == "ok", result
    assert result["ephemeris"] == {"configured": True, "base_url_env": "EPHEMERIS_BASE_URL", "discovered_models": 0}
    assert result["providers"]["ephemeris"]["lifecycle"] == "pretrained"
    code, error = _capabilities(config, cwd=tmp_path)
    assert code == 2 and "environment variable is not set" in error["error"]["message"]


@pytest.mark.parametrize("name", ["gnomon.yaml.example"])
def test_unsupported_configuration_formats_are_not_shipped(name):
    assert not (REPO / name).exists(), f"{name}: YAML is not a supported 1.x configuration format"
