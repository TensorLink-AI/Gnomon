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
    #: Whether the statistical candidates compete at all. False leaves the
    #: mandatory baselines, which is a coherent request: it asks whether
    #: anything beats the naive answer.
    statistical_enabled: bool = True
    #: Which statistical models compete. `None` means all of them, which is
    #: the default; a list restricts the pool to those names.
    statistical_candidates: list[str] | None = None
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
    #: An abstention floor a caller can raise above Aion's own derived
    #: minimum. `None` (the default) uses the derived one; a number refuses
    #: any series with fewer observations, naming this setting.
    min_observations: int | None = None
    #: `per_series` is the only implemented mode; `global` is rejected at
    #: load rather than silently ignored.
    selection: str = "per_series"
    pool_residuals: bool = True
    #: Nominal coverage of the published central interval. The default 0.80
    #: is what q10/q90 have always meant; changing it changes which residual
    #: order statistics are emitted as the interval bounds.
    target_coverage: float = 0.80


@dataclass
class OutputConfig:
    #: `artifact.json` and `lineage.json` have no switch: the artifact is
    #: the run's identity and the lineage is what the verifier checked, so
    #: a run without them is not a run.
    write_forecast_csv: bool = True
    write_summary: bool = True
    write_evidence: bool = True


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


#: Keys Aion parses but cannot honour, each with the reason. Supplying one
#: raises rather than being silently ignored.
#:
#: Roughly thirty documented options were parsed and never read, so a user
#: who disabled statistical models still got all five and a user who set
#: `target_coverage` still got 80% intervals. Everything that could be
#: honoured now is; what cannot be says so at load time, because a setting
#: that is quietly ignored is worse than one that does not exist.
INERT_KEYS: dict[str, str] = {
    "models.baselines.enabled": (
        "Baselines are mandatory by design: every candidate is selected by "
        "beating them, so disabling them would remove the comparison that "
        "makes a selection meaningful. Raise "
        "`evaluation.minimum_baseline_improvement` instead if you want a "
        "stricter bar, or lower it to 0 for a looser one."
    ),
    "evaluation.uncertainty.temporal_scaling": (
        "Intervals are not scaled by sqrt(step). Each lead time's spread is "
        "measured at that lead time, and a lead with too few residuals "
        "borrows the pooled set rather than being stretched by an assumed "
        "shape — see `conformal_spreads`. The flag would have described a "
        "method Aion deliberately does not use."
    ),
    "ensemble.quantile_strategy": (
        "The ensemble's intervals come from its own fold residuals, not from "
        "combining member quantiles, so there is no union/intersection choice "
        "to make. Combining member intervals would produce a band no fold "
        "ever measured."
    ),
    "meta_model.linear_regression.lasso_alpha": (
        "Only ridge regularisation is implemented; `ridge_alpha` is honoured. "
        "L1 selection over a handful of models on a handful of folds would "
        "drop members on noise."
    ),
    "evaluation.folds.degraded_mode.enabled": (
        "Degraded evaluation is not a switch: Aion enters it automatically "
        "when fewer than four folds are available, and says so in a warning "
        "and in the support assessment. Turning it on cannot create folds, "
        "and turning it off would replace a disclosed degradation with an "
        "abstention that hides why."
    ),
    "evaluation.folds.degraded_mode.min_observations": (
        "The floor is derived from the horizon and the seasonal period, not "
        "configured: `minimum_train = max(2 * season, 2 * horizon, 8)`. A "
        "fixed number cannot be right across frequencies."
    ),
    "output.write_artifact": (
        "artifact.json is the run's identity — a content-addressed record "
        "that everything else references. A run that does not write it is "
        "not a run. forecast.csv, summary.md, and evidence.jsonl are "
        "switchable."
    ),
    "output.include_model_comparison": (
        "Every candidate's fold score is already in artifact.json under "
        "`selection_scores`, and in the `rolling_evaluation` evidence "
        "record. There is nothing to include or omit."
    ),
    "output.include_ensemble_details": (
        "The ensemble's composition is already in the evidence when the "
        "ensemble runs, and absent when it does not."
    ),
    "models.baselines": (
        "Baselines are mandatory by design: every candidate is selected by "
        "beating them. Use evaluation.minimum_baseline_improvement to move "
        "the bar."
    ),
}


def _flatten(raw: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def check_inert_keys(raw: dict[str, Any]) -> None:
    """Refuse config keys that cannot take effect."""
    from .contracts import AionError

    supplied = _flatten(raw)
    offending = [key for key in supplied if key in INERT_KEYS]
    if not offending:
        return
    raise AionError(
        "UNSUPPORTED_CONFIG_KEY",
        "The config sets options Aion cannot honour: "
        + "; ".join(f"{key} — {INERT_KEYS[key]}" for key in sorted(offending)),
        {"keys": sorted(offending),
         "reasons": {key: INERT_KEYS[key] for key in sorted(offending)}},
    )


def _section(raw: dict[str, Any], *path: str) -> dict[str, Any]:
    """A nested config section, treating an empty one as absent.

    A YAML key whose body is only comments parses as ``None``, not ``{}`` —
    which is exactly what `backends.api.providers` looks like in the shipped
    example. Without this, `aion.yaml.example` could not be loaded at all.
    """
    cursor: Any = raw
    for key in path:
        if not isinstance(cursor, dict):
            return {}
        cursor = cursor.get(key)
    return cursor if isinstance(cursor, dict) else {}


def _parse_config(raw: dict[str, Any]) -> AionConfig:
    """Parse a raw dict into a typed AionConfig."""
    raw = raw or {}
    check_inert_keys(raw)
    cfg = AionConfig()

    # Models
    models_raw = _section(raw, "models")
    cfg.models = ModelsConfig(
        statistical_enabled=_section(models_raw, "statistical").get("enabled", True),
        statistical_candidates=_section(models_raw, "statistical").get("candidates"),
        tsfm_candidates=_section(models_raw, "tsfm").get("candidates", []),
        tsfm_overrides=_section(models_raw, "tsfm").get("overrides", {}),
    )

    # Backends
    backends_raw = _section(raw, "backends")
    sandbox_raw = _section(backends_raw, "sandbox")
    api_raw = _section(backends_raw, "api")
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
            providers=_parse_api_providers(_section(api_raw, "providers")),
        ),
    )

    # Ensemble
    ens_raw = _section(raw, "ensemble")
    cfg.ensemble = EnsembleConfig(
        enabled=ens_raw.get("enabled", False),
        strategy=ens_raw.get("strategy", "weighted_mean"),
        min_models=_section(ens_raw, "weighted_mean").get("min_models",
                          _section(ens_raw, "median").get("min_models",
                          _section(ens_raw, "voting").get("min_models", 2))),
        max_weight_ratio=_section(ens_raw, "weighted_mean").get("max_weight_ratio", 0.7),
        fallback=_section(ens_raw, "weighted_mean").get("fallback", "strongest_baseline"),
        eligible=ens_raw.get("eligible", "all_candidates"),
        quantile_strategy=ens_raw.get("quantile_strategy", "union"),
        weighted_mean=_section(ens_raw, "weighted_mean"),
        median=_section(ens_raw, "median"),
        voting=_section(ens_raw, "voting"),
    )

    # Meta-model
    mm_raw = _section(raw, "meta_model")
    cfg.meta_model = MetaModelConfig(
        enabled=mm_raw.get("enabled", False),
        type=mm_raw.get("type", "linear_regression"),
        ridge_alpha=_section(mm_raw, "linear_regression").get("ridge_alpha", 1.0),
        lasso_alpha=_section(mm_raw, "linear_regression").get("lasso_alpha", 0.1),
        min_models=mm_raw.get("min_models", 2),
        min_folds=mm_raw.get("min_folds", 3),
        non_negative=mm_raw.get("non_negative", True),
        fallback=mm_raw.get("fallback", "weighted_mean"),
    )

    # LLM
    llm_raw = _section(raw, "llm")
    cfg.llm = LLMConfig(
        enabled=llm_raw.get("enabled", False),
        mode=llm_raw.get("mode", "interpret"),
        adapter=llm_raw.get("adapter"),
        max_tokens=llm_raw.get("max_tokens", 2000),
        temperature=llm_raw.get("temperature", 0.3),
    )

    # Evaluation
    eval_raw = _section(raw, "evaluation")
    folds_raw = _section(eval_raw, "folds")
    unc_raw = _section(eval_raw, "uncertainty")
    selection = eval_raw.get("selection", "per_series")
    if selection != "per_series":
        from .contracts import AionError
        raise AionError(
            "UNSUPPORTED_CONFIG_KEY",
            f"evaluation.selection={selection!r} is not implemented; models "
            f"are selected per series. A global selection would impose one "
            f"model on series whose backtests disagree.",
            {"keys": ["evaluation.selection"], "supported": ["per_series"]},
        )
    target_coverage = float(unc_raw.get("target_coverage", 0.80))
    if not 0.5 <= target_coverage < 1.0:
        from .contracts import AionError
        raise AionError(
            "UNSUPPORTED_CONFIG_KEY",
            f"evaluation.uncertainty.target_coverage must be in [0.5, 1.0); "
            f"got {target_coverage}.",
            {"keys": ["evaluation.uncertainty.target_coverage"],
             "supplied": target_coverage},
        )
    cfg.evaluation = EvaluationConfig(
        minimum_baseline_improvement=eval_raw.get("minimum_baseline_improvement", 0.02),
        min_observations=folds_raw.get("min_observations"),
        selection=selection,
        pool_residuals=unc_raw.get("pool_residuals", True),
        target_coverage=target_coverage,
    )

    # Output
    out_raw = _section(raw, "output")
    cfg.output = OutputConfig(
        write_forecast_csv=out_raw.get("write_forecast_csv", True),
        write_summary=out_raw.get("write_summary", True),
        write_evidence=out_raw.get("write_evidence", True),
    )

    return cfg


def _parse_api_providers(
    providers_raw: dict[str, Any],
) -> dict[str, APIProviderConfig]:
    """Parse the backends.api.providers section."""
    result = {}
    for name, provider_raw in providers_raw.items():
        auth_raw = _section(provider_raw, "auth")
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
