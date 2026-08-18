"""Gnomon configuration system.

Loads and validates ``gnomon.toml`` (preferred; parsed by the stdlib) or
``gnomon.yaml`` (transitional; requires PyYAML — and a present YAML file
without PyYAML is an *error*, never a silent fall-back to defaults).

Config is loaded from (in order of priority):
  1. Explicit ``--config path`` CLI flag
  2. ``GNOMON_CONFIG_PATH`` environment variable
  3. ``./gnomon.toml`` / ``./gnomon.yaml`` (both present = error)
  4. ``~/.config/gnomon/gnomon.toml`` / ``gnomon.yaml``
  5. Built-in defaults (a fresh object per load)

Every section is optional. Validation is an allowlist: every supplied key
is honoured, refused with a reason, or unknown — and unknown fails at
load, because a silently ignored setting is worse than one that does not
exist.
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
    # Immutable server/model revision when the provider supports one. An
    # empty value remains usable, but is disclosed as unversioned in the
    # candidate receipt rather than being mistaken for a reproducible pin.
    revision: str = ""
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
    strategy: str = "weighted_mean"    # weighted_mean | median | voting
    min_models: int = 2
    max_weight_ratio: float = 0.7
    # strategy-specific fields
    weighted_mean: dict[str, Any] = field(default_factory=dict)
    median: dict[str, Any] = field(default_factory=dict)
    voting: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaModelConfig:
    enabled: bool = False
    #: Ridge regularisation strength. The historical default was a
    #: hardcoded numerical-stability epsilon, kept as the default so that
    #: wiring the option changed no existing fit; setting it now has the
    #: effect the documentation always claimed.
    ridge_alpha: float = 1e-6
    min_models: int = 2
    non_negative: bool = True


@dataclass
class EvaluationConfig:
    minimum_baseline_improvement: float = 0.02
    #: An abstention floor a caller can raise above Gnomon's own derived
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
class ContextConfig:
    #: Admit future-dated constraint/override context events by textual
    #: verifiability (`gnomon.future_context`). Off by default: with the
    #: flag off the lane does not run, no artifact gains an ID-payload
    #: key, and every existing ID is byte-identical.
    future_events: bool = False
    #: Admit LLM-classified structural events (`structural:<label>`,
    #: menu: trend_ceases, level_matches_seasonal_high/_low) — the model
    #: classifies from a closed menu and never supplies a number; every
    #: applied quantity is derived from Gnomon's own data. Same
    #: byte-identity guarantee when off. Experimental:
    #: results/structural-effects/HYPOTHESIS.md and
    #: results/seasonal-regime-effects/HYPOTHESIS.md.
    structural_events: bool = False


@dataclass
class OutputConfig:
    #: `artifact.json` and `lineage.json` have no switch: the artifact is
    #: the run's identity and the lineage is what the verifier checked, so
    #: a run without them is not a run.
    write_forecast_csv: bool = True
    write_summary: bool = True
    write_evidence: bool = True


@dataclass
class GnomonConfig:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    backends: BackendsConfig = field(default_factory=BackendsConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    meta_model: MetaModelConfig = field(default_factory=MetaModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    _source_path: str | None = None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = GnomonConfig()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

#: Search locations, one *directory* at a time: TOML is the preferred
#: format (parsed by stdlib `tomllib`, present on every install), YAML the
#: transitional one. Both formats present in the same directory is an
#: error, never a silent preference.
CONFIG_SEARCH_DIRS = [
    Path("."),
    Path.home() / ".config" / "gnomon",
]
TOML_NAMES = ("gnomon.toml",)
YAML_NAMES = ("gnomon.yaml", "gnomon.yml")


def find_config(explicit_path: str | None = None) -> Path | None:
    """Find the config file to load.

    Priority: explicit path, ``GNOMON_CONFIG_PATH``, then each search
    directory. Within a directory, a TOML and a YAML config side by side
    is ambiguous and refuses — a machine that resolved the tie silently
    would leave one file lying about what runs.
    """
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Config file not found: {p}")

    env_path = os.environ.get("GNOMON_CONFIG_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return p

    from .contracts import GnomonError
    for directory in CONFIG_SEARCH_DIRS:
        toml_hits = [directory / name for name in TOML_NAMES
                     if (directory / name).is_file()]
        yaml_hits = [directory / name for name in YAML_NAMES
                     if (directory / name).is_file()]
        if toml_hits and yaml_hits:
            raise GnomonError(
                "CONFIG_AMBIGUOUS",
                f"Both {toml_hits[0]} and {yaml_hits[0]} exist; delete one "
                "so there is a single source of truth (gnomon.toml is the "
                "preferred format).",
                {"toml": str(toml_hits[0]), "yaml": str(yaml_hits[0])},
            )
        if toml_hits:
            return toml_hits[0]
        if yaml_hits:
            return yaml_hits[0]

    return None


def load_config(explicit_path: str | None = None) -> GnomonConfig:
    """Load Gnomon configuration from file or defaults.

    A discovered config that cannot be read is an *error*, never a
    silent fall-back to defaults: a present-but-ignored file means the
    run is configured differently from what the operator wrote down.
    Returns a fresh object per call — defaults included — so no caller
    mutates state shared with another.
    """
    path = find_config(explicit_path)
    if path is None:
        return GnomonConfig()

    from .contracts import GnomonError
    if path.suffix.lower() == ".toml":
        import tomllib
        try:
            with open(path, "rb") as f:
                raw: dict[str, Any] = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise GnomonError(
                "CONFIG_UNREADABLE",
                f"{path} is not valid TOML: {exc}",
                {"path": str(path)},
            ) from exc
    else:
        try:
            import yaml
        except ImportError as exc:
            raise GnomonError(
                "CONFIG_UNREADABLE",
                f"{path} exists but PyYAML is not installed, so it would "
                "have been silently ignored. Convert it to gnomon.toml "
                "(parsed by the standard library) or install pyyaml.",
                {"path": str(path)},
            ) from exc
        try:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise GnomonError(
                "CONFIG_UNREADABLE",
                f"{path} is not valid YAML: {exc}",
                {"path": str(path)},
            ) from exc
    if not isinstance(raw, dict):
        raise GnomonError(
            "CONFIG_UNREADABLE",
            f"{path} must contain a mapping of sections, not "
            f"{type(raw).__name__}.",
            {"path": str(path)},
        )

    cfg = _parse_config(raw)
    cfg._source_path = str(path)
    return cfg


#: Keys Gnomon parses but cannot honour, each with the reason. Supplying one
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
        "method Gnomon deliberately does not use."
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
        "Degraded evaluation is not a switch: Gnomon enters it automatically "
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
    "backends.sandbox.venv_root": (
        "The sandbox root is not configurable: sandboxes live in the "
        "auto-managed cache and the `.gnomon-sandbox-ready` marker is their "
        "single source of truth. The option was parsed and read by nothing."
    ),
    "backends.sandbox.auto_install": (
        "Installation is always explicit — `gnomon tsfm install` or the "
        "gnomon_install_tsfm tool — because an implicit multi-minute "
        "download inside a forecast call reads as a hang in every "
        "interactive host. The option was parsed and read by nothing."
    ),
    "ensemble.eligible": (
        "The candidate pool is set by models.statistical.candidates and "
        "models.tsfm.candidates; the ensemble combines whatever competed. "
        "A second eligibility list was parsed and read by nothing."
    ),
    "ensemble.weighted_mean.fallback": (
        "When the ensemble cannot run, the selected model publishes and "
        "the response says so — that behaviour is not configurable. The "
        "option was parsed and read by nothing."
    ),
    "meta_model.type": (
        "Only linear_regression exists. The option was parsed and read by "
        "nothing, so `type: ridge` silently ran linear_regression anyway."
    ),
    "meta_model.min_folds": (
        "Fold sufficiency is governed by the evaluation's own fold rules; "
        "the meta-model trains when at least meta_model.min_models members "
        "have fold forecasts. The option was parsed and read by nothing."
    ),
    "meta_model.fallback": (
        "A meta-model that produces no weights simply does not compete; "
        "selection proceeds among the models that did. The option was "
        "parsed and read by nothing."
    ),
}

#: Whole sections Gnomon refuses, with the reason. A prefix refuses every
#: key beneath it.
INERT_PREFIXES: dict[str, str] = {
    "llm": (
        "The llm section was parsed and read by nothing: LLM integration "
        "belongs to the MCP host, not the engine, which computes every "
        "number deterministically. "
        "Remove the section."
    ),
}

#: Every key the runtime honours. Validation is an allowlist: a key that
#: is neither here nor in the refusal tables above is unknown, and an
#: unknown key fails at load — the denylist approach chased historically
#: inert options one at a time and its own reasons drifted false.
ALLOWED_KEYS: frozenset[str] = frozenset({
    "models.statistical.enabled",
    "models.statistical.candidates",
    "models.tsfm.candidates",
    "backends.sandbox.enabled",
    "backends.api.enabled",
    "backends.api.timeout",
    "backends.api.retry",
    "ensemble.enabled",
    "ensemble.strategy",
    "ensemble.weighted_mean.min_models",
    "ensemble.weighted_mean.max_weight_ratio",
    "ensemble.median.min_models",
    "ensemble.voting.min_models",
    "ensemble.voting.threshold",
    "meta_model.enabled",
    "meta_model.min_models",
    "meta_model.non_negative",
    "meta_model.linear_regression.ridge_alpha",
    "evaluation.minimum_baseline_improvement",
    "evaluation.folds.min_observations",
    "evaluation.selection",
    "evaluation.uncertainty.pool_residuals",
    "evaluation.uncertainty.target_coverage",
    "context.future_events",
    "context.structural_events",
    "output.write_forecast_csv",
    "output.write_summary",
    "output.write_evidence",
})

#: Dynamic namespaces: (prefix, allowed tail keys). The middle segment is
#: caller-chosen (a provider or model name); the tail is typed, not open.
ALLOWED_WILDCARDS: tuple[tuple[str, frozenset[str]], ...] = (
    ("backends.api.providers.", frozenset({
        "url", "model", "revision", "timeout", "retry",
        "auth.type", "auth.token_env", "auth.header",
    })),
    ("models.tsfm.overrides.", frozenset({"backend"})),
)


def _flatten(raw: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _wildcard_allows(key: str) -> bool:
    """Whether ``key`` fits a typed dynamic namespace."""
    for prefix, tails in ALLOWED_WILDCARDS:
        if key.startswith(prefix):
            remainder = key[len(prefix):]
            # One caller-chosen segment (the provider/model name), then a
            # typed tail.
            _, _, tail = remainder.partition(".")
            if tail in tails:
                return True
    return False


def check_inert_keys(raw: dict[str, Any]) -> None:
    """Validate every supplied key: honoured, refused-with-reason, or unknown.

    Allowlist semantics — the three outcomes are exhaustive, so a key
    cannot be silently ignored. Refusals carry their reasons; unknown
    keys fail with the nearest documented section so a typo is a
    one-line fix rather than an archaeology project.
    """
    from .contracts import GnomonError

    supplied = _flatten(raw)
    inert: dict[str, str] = {}
    unknown: list[str] = []
    section_prefixes = {key.rsplit(".", 1)[0] for key in ALLOWED_KEYS}
    section_prefixes.update(
        prefix.rstrip(".") for prefix, _ in ALLOWED_WILDCARDS
    )
    for key, value in supplied.items():
        if value is None and any(
            name == key or name.startswith(key + ".")
            for name in section_prefixes
        ):
            # A section written with only comments parses as an explicit
            # null (`providers:` in the shipped example) — an empty
            # section, not a setting.
            continue
        if key in INERT_KEYS:
            inert[key] = INERT_KEYS[key]
            continue
        prefix_hit = next(
            (reason for prefix, reason in INERT_PREFIXES.items()
             if key == prefix or key.startswith(prefix + ".")), None,
        )
        if prefix_hit is not None:
            inert[key] = prefix_hit
            continue
        if key in ALLOWED_KEYS or _wildcard_allows(key):
            continue
        unknown.append(key)

    if inert:
        raise GnomonError(
            "UNSUPPORTED_CONFIG_KEY",
            "The config sets options Gnomon cannot honour: "
            + "; ".join(f"{key} — {inert[key]}" for key in sorted(inert)),
            {"keys": sorted(inert), "reasons": dict(sorted(inert.items()))},
        )
    if unknown:
        raise GnomonError(
            "UNSUPPORTED_CONFIG_KEY",
            "The config sets options Gnomon does not recognise: "
            + ", ".join(sorted(unknown))
            + ". Every honoured key is documented in gnomon.toml.example; "
            "an unrecognised key would otherwise be silently ignored.",
            {"keys": sorted(unknown),
             "reasons": {key: "unknown key" for key in sorted(unknown)}},
        )


def _section(raw: dict[str, Any], *path: str) -> dict[str, Any]:
    """A nested config section, treating an empty one as absent.

    A YAML key whose body is only comments parses as ``None``, not ``{}`` —
    which is exactly what `backends.api.providers` looks like in the shipped
    example. Without this, `gnomon.yaml.example` could not be loaded at all.
    """
    cursor: Any = raw
    for key in path:
        if not isinstance(cursor, dict):
            return {}
        cursor = cursor.get(key)
    return cursor if isinstance(cursor, dict) else {}


def _parse_config(raw: dict[str, Any]) -> GnomonConfig:
    """Parse a raw dict into a typed GnomonConfig."""
    raw = raw or {}
    check_inert_keys(raw)
    cfg = GnomonConfig()

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
    strategy = ens_raw.get("strategy", "weighted_mean")
    if strategy == "stacking":
        from .contracts import GnomonError
        raise GnomonError(
            "UNSUPPORTED_CONFIG_KEY",
            "ensemble.strategy=stacking has no executable path: the "
            "documented delegation to the meta-model raised at runtime and "
            "was swallowed into silent per-fold failures. Enable "
            "meta_model.enabled=true for stacked weights; the ensemble "
            "strategies are weighted_mean, median, and voting.",
            {"keys": ["ensemble.strategy"],
             "supported": ["weighted_mean", "median", "voting"]},
        )
    if strategy not in ("weighted_mean", "median", "voting"):
        from .contracts import GnomonError
        raise GnomonError(
            "UNSUPPORTED_CONFIG_KEY",
            f"ensemble.strategy={strategy!r} is not implemented.",
            {"keys": ["ensemble.strategy"],
             "supported": ["weighted_mean", "median", "voting"]},
        )
    cfg.ensemble = EnsembleConfig(
        enabled=ens_raw.get("enabled", False),
        strategy=strategy,
        min_models=_section(ens_raw, "weighted_mean").get("min_models",
                          _section(ens_raw, "median").get("min_models",
                          _section(ens_raw, "voting").get("min_models", 2))),
        max_weight_ratio=_section(ens_raw, "weighted_mean").get("max_weight_ratio", 0.7),
        weighted_mean=_section(ens_raw, "weighted_mean"),
        median=_section(ens_raw, "median"),
        voting=_section(ens_raw, "voting"),
    )

    # Meta-model
    mm_raw = _section(raw, "meta_model")
    cfg.meta_model = MetaModelConfig(
        enabled=mm_raw.get("enabled", False),
        ridge_alpha=_section(mm_raw, "linear_regression").get("ridge_alpha", 1e-6),
        min_models=mm_raw.get("min_models", 2),
        non_negative=mm_raw.get("non_negative", True),
    )

    # Evaluation
    eval_raw = _section(raw, "evaluation")
    folds_raw = _section(eval_raw, "folds")
    unc_raw = _section(eval_raw, "uncertainty")
    selection = eval_raw.get("selection", "per_series")
    if selection != "per_series":
        from .contracts import GnomonError
        raise GnomonError(
            "UNSUPPORTED_CONFIG_KEY",
            f"evaluation.selection={selection!r} is not implemented; models "
            f"are selected per series. A global selection would impose one "
            f"model on series whose backtests disagree.",
            {"keys": ["evaluation.selection"], "supported": ["per_series"]},
        )
    target_coverage = float(unc_raw.get("target_coverage", 0.80))
    if not 0.5 <= target_coverage < 1.0:
        from .contracts import GnomonError
        raise GnomonError(
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

    # Context
    context_raw = _section(raw, "context")
    cfg.context = ContextConfig(
        future_events=_parse_on_off(
            context_raw.get("future_events", False), "context.future_events",
        ),
        structural_events=_parse_on_off(
            context_raw.get("structural_events", False),
            "context.structural_events",
        ),
    )

    # Output
    out_raw = _section(raw, "output")
    cfg.output = OutputConfig(
        write_forecast_csv=out_raw.get("write_forecast_csv", True),
        write_summary=out_raw.get("write_summary", True),
        write_evidence=out_raw.get("write_evidence", True),
    )

    return cfg


def _parse_on_off(raw: Any, key: str) -> bool:
    """A flag written as on/off (or a plain boolean). Anything else is an
    error, not a silent default — a mistyped value that quietly meant
    "off" would hide the very behaviour change the flag discloses."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() in ("on", "true", "yes"):
        return True
    if isinstance(raw, str) and raw.strip().lower() in ("off", "false", "no"):
        return False
    from .contracts import GnomonError
    raise GnomonError(
        "UNSUPPORTED_CONFIG_KEY",
        f"{key} must be on or off; got {raw!r}.",
        {"keys": [key], "supplied": raw, "supported": ["on", "off"]},
    )


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
            revision=provider_raw.get("revision", ""),
            timeout=provider_raw.get("timeout", 60),
            retry=provider_raw.get("retry", 2),
        )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_tsfm_backend(
    name: str,
    config: GnomonConfig,
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
