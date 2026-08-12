"""Configuration cannot lie: honoured, refused-with-reason, or unknown.

The denylist chased historically inert options one at a time and its own
reasons drifted false (`ridge_alpha is honoured` while nothing read it).
Validation is now an allowlist — an unknown or unimplementable key fails
at load — the config format prefers TOML (stdlib `tomllib`), and a
discovered config that cannot be read is an error, never a silent
fall-back to defaults.
"""

import sys

import pytest

from gnomon.config import GnomonConfig, load_config
from gnomon.contracts import GnomonError


def _write_yaml(tmp_path, text):
    path = tmp_path / "gnomon.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_unknown_key_is_refused(tmp_path):
    pytest.importorskip("yaml")
    path = _write_yaml(tmp_path, "ensemble:\n  strateggy: median\n")
    with pytest.raises(GnomonError) as caught:
        load_config(path)
    assert caught.value.code == "UNSUPPORTED_CONFIG_KEY"
    assert "ensemble.strateggy" in caught.value.details["keys"]


def test_unknown_section_is_refused(tmp_path):
    pytest.importorskip("yaml")
    path = _write_yaml(tmp_path, "modles:\n  statistical:\n    enabled: true\n")
    with pytest.raises(GnomonError) as caught:
        load_config(path)
    assert caught.value.code == "UNSUPPORTED_CONFIG_KEY"


def test_llm_section_is_refused_with_reason(tmp_path):
    pytest.importorskip("yaml")
    path = _write_yaml(tmp_path, "llm:\n  enabled: false\n")
    with pytest.raises(GnomonError) as caught:
        load_config(path)
    assert caught.value.code == "UNSUPPORTED_CONFIG_KEY"
    assert "llm.enabled" in caught.value.details["reasons"]
    assert "MCP host" in caught.value.details["reasons"]["llm.enabled"]


def test_override_backend_is_honoured_and_device_is_refused(tmp_path):
    pytest.importorskip("yaml")
    good = _write_yaml(tmp_path, (
        "models:\n  tsfm:\n    overrides:\n      chronos_bolt_mini:\n"
        "        backend: sandbox\n"
    ))
    assert load_config(good).models.tsfm_overrides == {
        "chronos_bolt_mini": {"backend": "sandbox"}
    }
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "models:\n  tsfm:\n    overrides:\n      chronos_bolt_mini:\n"
        "        device: cuda\n", encoding="utf-8",
    )
    with pytest.raises(GnomonError) as caught:
        load_config(str(bad))
    assert "models.tsfm.overrides.chronos_bolt_mini.device" in (
        caught.value.details["keys"]
    )


def test_stacking_fails_at_load(tmp_path):
    pytest.importorskip("yaml")
    path = _write_yaml(tmp_path, "ensemble:\n  enabled: true\n  strategy: stacking\n")
    with pytest.raises(GnomonError) as caught:
        load_config(path)
    assert caught.value.code == "UNSUPPORTED_CONFIG_KEY"
    assert "meta_model" in caught.value.message


def test_toml_config_loads_via_stdlib(tmp_path):
    path = tmp_path / "gnomon.toml"
    path.write_text(
        '[ensemble]\nenabled = true\nstrategy = "median"\n', encoding="utf-8",
    )
    cfg = load_config(str(path))
    assert cfg.ensemble.enabled is True
    assert cfg.ensemble.strategy == "median"


def test_invalid_toml_is_a_loud_error(tmp_path):
    path = tmp_path / "gnomon.toml"
    path.write_text("[ensemble\nenabled = true\n", encoding="utf-8")
    with pytest.raises(GnomonError) as caught:
        load_config(str(path))
    assert caught.value.code == "CONFIG_UNREADABLE"


def test_yaml_without_pyyaml_is_a_loud_error(tmp_path, monkeypatch):
    """A present config silently discarded meant the run was configured
    differently from what the operator wrote down."""
    path = _write_yaml(tmp_path, "ensemble:\n  enabled: true\n")
    monkeypatch.setitem(sys.modules, "yaml", None)  # import yaml -> ImportError
    with pytest.raises(GnomonError) as caught:
        load_config(path)
    assert caught.value.code == "CONFIG_UNREADABLE"
    assert "gnomon.toml" in caught.value.message


def test_both_formats_in_one_directory_is_ambiguous(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    (tmp_path / "gnomon.toml").write_text("[ensemble]\n", encoding="utf-8")
    (tmp_path / "gnomon.yaml").write_text("ensemble:\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GNOMON_CONFIG_PATH", raising=False)
    with pytest.raises(GnomonError) as caught:
        load_config()
    assert caught.value.code == "CONFIG_AMBIGUOUS"


def test_defaults_are_fresh_per_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GNOMON_CONFIG_PATH", raising=False)
    first = load_config()
    second = load_config()
    assert first is not second
    first.ensemble.enabled = True
    assert second.ensemble.enabled is False, (
        "mutating one caller's config leaked into another's defaults"
    )
    assert isinstance(first, GnomonConfig)


def test_ridge_alpha_is_actually_honoured():
    """The denylist claimed ridge_alpha was honoured while the solver
    hardcoded its epsilon; the setting must now change the fit."""
    from gnomon.meta_model import train_meta_model

    fold_forecasts = {
        "a": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        "b": [[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]],
    }
    fold_actuals = [[1.2, 1.8], [2.2, 2.8], [3.2, 3.8]]
    light = train_meta_model(fold_forecasts, fold_actuals, ridge_alpha=1e-6)
    heavy = train_meta_model(fold_forecasts, fold_actuals, ridge_alpha=1e3)
    assert light and heavy
    assert light != heavy, "ridge_alpha changed nothing"
    # Heavy shrinkage pulls every weight toward zero.
    assert sum(heavy.values()) < sum(light.values())
