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


def test_remote_adapter_revision_is_parsed_and_preserved(tmp_path):
    path = tmp_path / "gnomon.toml"
    path.write_text(
        '[backends.api]\nenabled = true\n'
        '[backends.api.providers.remote]\n'
        'url = "https://example.invalid/forecast"\n'
        'model = "remote-model"\nrevision = "sha256:abc"\n',
        encoding="utf-8")
    provider = load_config(str(path)).backends.api.providers["remote"]
    assert provider.revision == "sha256:abc"


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


#: Every allowlisted key, mapped to the config attribute it must land in
#: and a non-default value to set it to. The mapping is the behavioural
#: claim: a key that parses into nothing is exactly the defect Phase 1E
#: exists to remove, and this table is what stops a new key being added
#: to ALLOWED_KEYS without being wired.
KEY_EFFECTS: dict[str, tuple[str, object]] = {
    "models.statistical.enabled": ("models.statistical_enabled", False),
    "models.statistical.candidates": ("models.statistical_candidates", ["drift"]),
    "models.tsfm.candidates": ("models.tsfm_candidates", ["chronos_bolt_mini"]),
    "backends.sandbox.enabled": ("backends.sandbox.enabled", False),
    "backends.api.enabled": ("backends.api.enabled", True),
    "backends.api.timeout": ("backends.api.timeout", 42),
    "backends.api.retry": ("backends.api.retry", 5),
    "ensemble.enabled": ("ensemble.enabled", True),
    "ensemble.strategy": ("ensemble.strategy", "median"),
    "ensemble.weighted_mean.min_models": ("ensemble.min_models", 4),
    "ensemble.weighted_mean.max_weight_ratio": ("ensemble.max_weight_ratio", 0.55),
    "meta_model.enabled": ("meta_model.enabled", True),
    "meta_model.min_models": ("meta_model.min_models", 3),
    "meta_model.non_negative": ("meta_model.non_negative", False),
    "meta_model.linear_regression.ridge_alpha": ("meta_model.ridge_alpha", 0.25),
    "evaluation.minimum_baseline_improvement": (
        "evaluation.minimum_baseline_improvement", 0.1),
    "evaluation.folds.min_observations": ("evaluation.min_observations", 144),
    "evaluation.uncertainty.pool_residuals": ("evaluation.pool_residuals", False),
    "evaluation.uncertainty.target_coverage": ("evaluation.target_coverage", 0.9),
    "context.future_events": ("context.future_events", True),
    "context.structural_events": ("context.structural_events", True),
    "output.write_forecast_csv": ("output.write_forecast_csv", False),
    "output.write_summary": ("output.write_summary", False),
    "output.write_evidence": ("output.write_evidence", False),
}

#: Keys whose effect is structural rather than a single attribute:
#: `evaluation.selection` only accepts its one implemented value, and the
#: strategy-specific min_models are alternative spellings of one field.
STRUCTURAL_KEYS = {
    "evaluation.selection",
    "models.admission.policy",
    "models.admission.evidence_registry_path",
    "ensemble.median.min_models",
    "ensemble.voting.min_models",
    "ensemble.voting.threshold",
}


def test_evidence_weighted_admission_requires_explicit_registry(tmp_path):
    path = tmp_path / "gnomon.toml"
    path.write_text('[models.admission]\npolicy = "evidence_weighted"\n')
    with pytest.raises(GnomonError, match="evidence_registry_path"):
        load_config(str(path))
    path.write_text(
        '[models.admission]\npolicy = "evidence_weighted"\n'
        'evidence_registry_path = "forge.json"\n')
    cfg = load_config(str(path))
    assert cfg.models.admission_policy == "evidence_weighted"
    assert cfg.models.evidence_registry_path == "forge.json"


def test_every_allowlisted_key_is_accounted_for():
    """No key may be allowed without a stated effect: that combination is
    precisely the parsed-never-read defect Phase 1E removes."""
    from gnomon.config import ALLOWED_KEYS

    accounted = set(KEY_EFFECTS) | STRUCTURAL_KEYS
    assert set(ALLOWED_KEYS) == accounted, (
        "allowlist and effect table disagree; unaccounted: "
        f"{sorted(set(ALLOWED_KEYS) - accounted)}, stale: "
        f"{sorted(accounted - set(ALLOWED_KEYS))}"
    )


@pytest.mark.parametrize("key", sorted(KEY_EFFECTS))
def test_each_accepted_key_reaches_the_runtime(tmp_path, key):
    pytest.importorskip("yaml")
    import yaml as yaml_module

    attribute, value = KEY_EFFECTS[key]
    nested: dict = {}
    cursor = nested
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value

    path = tmp_path / "gnomon.yaml"
    path.write_text(yaml_module.safe_dump(nested), encoding="utf-8")
    config = load_config(str(path))

    resolved = config
    for attr in attribute.split("."):
        resolved = getattr(resolved, attr)
    assert resolved == value, (
        f"{key} parsed but did not reach {attribute}"
    )


def test_interval_shaping_options_change_the_forecast_id(tmp_path):
    """A key that changes the published numbers must change identity.
    target_coverage and pool_residuals both reshape the interval, and
    both used to leave the content-addressed id untouched — so two runs
    with materially different bands collided on one id."""
    from gnomon.runtime import _config_fingerprint

    default = _config_fingerprint(GnomonConfig())

    coverage = GnomonConfig()
    coverage.evaluation.target_coverage = 0.5
    pooled = GnomonConfig()
    pooled.evaluation.pool_residuals = False
    context = GnomonConfig()
    context.context.future_events = True

    assert _config_fingerprint(coverage) != default
    assert _config_fingerprint(pooled) != default
    assert _config_fingerprint(context) != default
    # Defaults still fingerprint as absent, so existing ids do not churn.
    assert default is None


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
