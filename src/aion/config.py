"""Aion configuration system.

Loads, validates, and provides defaults for ``aion.yaml`` config files.

Config is loaded from (in order of priority):
  1. Explicit ``--config path`` CLI flag
  2. ``AION_CONFIG_PATH`` environment variable
  3. ``./aion.yaml`` in the current directory
  4. ``~/.config/aion/aion.yaml``
  5. Built-in defaults

Every section is optional. With no config file, Aion uses sensible defaults
that match v0.1 behaviour: baselines + drift, no TSFMs, no ensemble, no LLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses for typed config
# ---------------------------------------------------------------------------

@dataclass
class TSFMModelConfig:
    name: str = ""
    backend: str = "auto"           # sandbox | api | auto
    device: str = "cpu"
    timeout: int = 300


@dataclass
class ModelsConfig:
    baselines_enabled: bool = True
    statistical_enabled: bool = True
    statistical_candidates: list[str] = field(default_factory=lambda: ["drift"])
    tsfm_candidates: list[str] = field(default_factory=list)
    tsfm_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SandboxBackendConfig:
    enabled: bool = True
    venv_root: str | None = None
    auto_install: bool = True


@dataclass
class APIAuthConfig:
    type: str = "none"              # none | bearer | header
    token_env: str = ""
    header: str = "Authorization"


@dataclass
class APIProviderConfig:
    url: str = ""
    auth: APIAuthConfig = field(default_factory=APIAuthConfig)
    model: str = ""
    timeout: int = 60
    retry: int = 2


@dataclass
class APIBackendConfig:
    enabled: bool = False
    timeout: int = 60
    retry: int = 2
    providers: dict[str, APIProviderConfig] = field(default_factory=dict)


@dataclass
class BackendsConfig:
    sandbox: SandboxBackendConfig = field(default_factory=SandboxBackendConfig)
    api: APIBackendConfig = field(default_factory=APIBackendConfig)


@dataclass
class EnsembleConfig:
    enabled: bool = False
    strategy: str = "weighted_mean"    # weighted_mean | median | voting | stacking
    min_models: int = 2
    max_weight_ratio: float = 0.7
    fallback: str = "strongest_baseline"
    eligible: list[str] | str = "all_candidates"
    quantile_strategy: str = "union"   # union | intersection | weighted
    # strategy-specific fields
    weighted_mean: dict[str, Any] = field(default_factory=dict)
    median: dict[str, Any] = field(default_factory=dict)
    voting: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaModelConfig:
    enabled: bool = False
    type: str = "linear_regression"
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.1
    min_models: int = 2
    min_folds: int = 3
    non_negative: bool = True
    fallback: str = "weighted_mean"


@dataclass
class LLMConfig:
    enabled: bool = False
    mode: str = "interpret"            # interpret | compare | challenge
    adapter: dict[str, Any] | None = None
    max_tokens: int = 2000
    temperature: float = 0.3


@dataclass
class EvaluationConfig:
    minimum_baseline_improvement: float = 0.02
    min_observations: int = 144
    degraded_mode_enabled: bool = False
    degraded_min_observations: int = 48
    selection: str = "per_series"
    pool_residuals: bool = True
    temporal_scaling: bool = True
    target_coverage: float = 0.70


@dataclass
class OutputConfig:
    write_artifact: bool = True
    write_forecast_csv: bool = True
    write_summary: bool = True
    write_evidence: bool = True
    include_model_comparison: bool = False
    include_ensemble_details: bool = False


@dataclass
class AionConfig:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    backends: BackendsConfig = field(default_factory=BackendsConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    meta_model: MetaModelConfig = field(default_factory=MetaModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    _source_path: str | None = None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = AionConfig()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

CONFIG_SEARCH_PATHS = [
    Path("aion.yaml"),
    Path("aion.yml"),
    Path.home() / ".config" / "aion" / "aion.yaml",
]


def find_config(explicit_path: str | None = None) -> Path | None:
    """Find the config file to load."""
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Config file not found: {p}")

    env_path = os.environ.get("AION_CONFIG_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return p

    for candidate in CONFIG_SEARCH_PATHS:
        if candidate.is_file():
            return candidate

    return None


def load_config(explicit_path: str | None = None) -> AionConfig:
    """Load Aion configuration from file or defaults."""
    path = find_config(explicit_path)
    if path is None:
        return DEFAULT_CONFIG

    try:
        import yaml
    except ImportError:
        # No PyYAML — can't parse config, use defaults
        return DEFAULT_CONFIG

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = _parse_config(raw)
    cfg._source_path = str(path)
    return cfg


def _parse_config(raw: dict[str, Any]) -> AionConfig:
    """Parse a raw dict into a typed AionConfig."""
    cfg = AionConfig()

    # Models
    models_raw = raw.get("models", {})
    cfg.models = ModelsConfig(
        baselines_enabled=models_raw.get("baselines", {}).get("enabled", True),
        statistical_enabled=models_raw.get("statistical", {}).get("enabled", True),
        statistical_candidates=models_raw.get("statistical", {}).get("candidates", ["drift"]),
        tsfm_candidates=models_raw.get("tsfm", {}).get("candidates", []),
        tsfm_overrides=models_raw.get("tsfm", {}).get("overrides", {}),
    )

    # Backends
    backends_raw = raw.get("backends", {})
    sandbox_raw = backends_raw.get("sandbox", {})
    api_raw = backends_raw.get("api", {})
    cfg.backends = BackendsConfig(
        sandbox=SandboxBackendConfig(
            enabled=sandbox_raw.get("enabled", True),
            venv_root=sandbox_raw.get("venv_root"),
            auto_install=sandbox_raw.get("auto_install", True),
        ),
        api=APIBackendConfig(
            enabled=api_raw.get("enabled", False),
            timeout=api_raw.get("timeout", 60),
            retry=api_raw.get("retry", 2),
            providers=_parse_api_providers(api_raw.get("providers", {})),
        ),
    )

    # Ensemble
    ens_raw = raw.get("ensemble", {})
    cfg.ensemble = EnsembleConfig(
        enabled=ens_raw.get("enabled", False),
        strategy=ens_raw.get("strategy", "weighted_mean"),
        min_models=ens_raw.get("weighted_mean", {}).get("min_models",
                          ens_raw.get("median", {}).get("min_models",
                          ens_raw.get("voting", {}).get("min_models", 2))),
        max_weight_ratio=ens_raw.get("weighted_mean", {}).get("max_weight_ratio", 0.7),
        fallback=ens_raw.get("weighted_mean", {}).get("fallback", "strongest_baseline"),
        eligible=ens_raw.get("eligible", "all_candidates"),
        quantile_strategy=ens_raw.get("quantile_strategy", "union"),
        weighted_mean=ens_raw.get("weighted_mean", {}),
        median=ens_raw.get("median", {}),
        voting=ens_raw.get("voting", {}),
    )

    # Meta-model
    mm_raw = raw.get("meta_model", {})
    cfg.meta_model = MetaModelConfig(
        enabled=mm_raw.get("enabled", False),
        type=mm_raw.get("type", "linear_regression"),
        ridge_alpha=mm_raw.get("linear_regression", {}).get("ridge_alpha", 1.0),
        lasso_alpha=mm_raw.get("linear_regression", {}).get("lasso_alpha", 0.1),
        min_models=mm_raw.get("min_models", 2),
        min_folds=mm_raw.get("min_folds", 3),
        non_negative=mm_raw.get("non_negative", True),
        fallback=mm_raw.get("fallback", "weighted_mean"),
    )

    # LLM
    llm_raw = raw.get("llm", {})
    cfg.llm = LLMConfig(
        enabled=llm_raw.get("enabled", False),
        mode=llm_raw.get("mode", "interpret"),
        adapter=llm_raw.get("adapter"),
        max_tokens=llm_raw.get("max_tokens", 2000),
        temperature=llm_raw.get("temperature", 0.3),
    )

    # Evaluation
    eval_raw = raw.get("evaluation", {})
    folds_raw = eval_raw.get("folds", {})
    unc_raw = eval_raw.get("uncertainty", {})
    cfg.evaluation = EvaluationConfig(
        minimum_baseline_improvement=eval_raw.get("minimum_baseline_improvement", 0.02),
        min_observations=folds_raw.get("min_observations", 144),
        degraded_mode_enabled=folds_raw.get("degraded_mode", {}).get("enabled", False),
        degraded_min_observations=folds_raw.get("degraded_mode", {}).get("min_observations", 48),
        selection=eval_raw.get("selection", "per_series"),
        pool_residuals=unc_raw.get("pool_residuals", True),
        temporal_scaling=unc_raw.get("temporal_scaling", True),
        target_coverage=unc_raw.get("target_coverage", 0.70),
    )

    # Output
    out_raw = raw.get("output", {})
    cfg.output = OutputConfig(
        write_artifact=out_raw.get("write_artifact", True),
        write_forecast_csv=out_raw.get("write_forecast_csv", True),
        write_summary=out_raw.get("write_summary", True),
        write_evidence=out_raw.get("write_evidence", True),
        include_model_comparison=out_raw.get("include_model_comparison", False),
        include_ensemble_details=out_raw.get("include_ensemble_details", False),
    )

    return cfg


def _parse_api_providers(
    providers_raw: dict[str, Any],
) -> dict[str, APIProviderConfig]:
    """Parse the backends.api.providers section."""
    result = {}
    for name, provider_raw in providers_raw.items():
        auth_raw = provider_raw.get("auth", {})
        result[name] = APIProviderConfig(
            url=provider_raw.get("url", ""),
            auth=APIAuthConfig(
                type=auth_raw.get("type", "none"),
                token_env=auth_raw.get("token_env", ""),
                header=auth_raw.get("header", "Authorization"),
            ),
            model=provider_raw.get("model", ""),
            timeout=provider_raw.get("timeout", 60),
            retry=provider_raw.get("retry", 2),
        )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_tsfm_backend(
    name: str,
    config: AionConfig,
) -> str:
    """Determine the effective backend for a TSFM: 'sandbox', 'api', or 'skip'."""
    override = config.models.tsfm_overrides.get(name, {})
    backend = override.get("backend", "auto")

    if backend == "sandbox":
        return "sandbox" if config.backends.sandbox.enabled else "skip"
    elif backend == "api":
        return "api" if config.backends.api.enabled else "skip"
    elif backend == "auto":
        # Prefer sandbox if enabled, then API
        if config.backends.sandbox.enabled:
            return "sandbox"
        elif config.backends.api.enabled and name in config.backends.api.providers:
            return "api"
        return "skip"
    return "skip"
