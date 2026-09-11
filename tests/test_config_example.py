"""The shipped configuration example must load under the current operator schema."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "gnomon.toml.example"
PLUGIN_SRC = REPO / "examples" / "provider_plugin" / "src"


def _capabilities(environment):
    completed = subprocess.run(
        [sys.executable, "-m", "gnomon", "capabilities", "--providers-config", str(EXAMPLE)],
        capture_output=True, text=True, env={**os.environ, **environment},
    )
    return completed.returncode, json.loads(completed.stdout)


def test_every_example_key_is_accepted_by_the_configuration_schema():
    from gnomon.session import configuration_schema
    config = tomllib.loads(EXAMPLE.read_text())
    assert set(config) <= set(configuration_schema()["properties"]), sorted(config)
    assert config["schema_version"] == 1
    kinds = {name: spec["kind"] for name, spec in config["providers"].items()}
    assert sorted(kinds.values()) == ["callable", "ephemeris"]
    ephemeris = next(spec for spec in config["providers"].values() if spec["kind"] == "ephemeris")
    assert ephemeris["base_url_env"] == "EPHEMERIS_BASE_URL"
    assert ephemeris["token_env"] == "EPHEMERIS_API_TOKEN"
    assert ephemeris.get("discover", False) is False, "the shipped example must not need network access"


def test_example_loads_with_callable_registered_and_ephemeris_configured_but_unreachable(tmp_path):
    # A loopback port with nothing listening: configured, never contacted.
    code, result = _capabilities({
        "EPHEMERIS_BASE_URL": "http://127.0.0.1:9", "EPHEMERIS_API_TOKEN": "unused-in-tests",
        "PYTHONPATH": os.pathsep.join(p for p in (str(PLUGIN_SRC), os.environ.get("PYTHONPATH")) if p),
    })
    assert code == 0 and result["status"] == "ok"
    assert result["providers"]["my-model"]["revision"] == "my-model/0.0.0"
    ephemeris = result["providers"]["ephemeris"]
    assert ephemeris["lifecycle"] == "pretrained" and ephemeris["revision"] is None
    assert not [name for name in result["providers"] if name.startswith("ephemeris/")], "no discovery without network"
    # The example names a ledger without creating one during discovery.
    assert result["ledger"]["configured"] is True and result["ledger"]["opened"] is False


def test_example_without_ephemeris_environment_is_a_structured_configuration_error():
    environment = {k: v for k, v in os.environ.items() if k not in {"EPHEMERIS_BASE_URL", "EPHEMERIS_API_TOKEN"}}
    completed = subprocess.run(
        [sys.executable, "-m", "gnomon", "capabilities", "--providers-config", str(EXAMPLE)],
        capture_output=True, text=True, env=environment,
    )
    assert completed.returncode == 2
    error = json.loads(completed.stdout)["error"]
    assert error["code"] == "INVALID_ARGUMENTS"
    assert "environment variable is not set" in error["message"]


@pytest.mark.parametrize("name", ["gnomon.yaml.example"])
def test_unsupported_configuration_formats_are_not_shipped(name):
    assert not (REPO / name).exists(), f"{name}: YAML is not a supported 1.x configuration format"
