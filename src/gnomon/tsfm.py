"""Time Series Foundation Model (TSFM) adapter framework for Gnomon.

Each adapter wraps a pre-trained foundation model behind a uniform protocol
that Gnomon's evaluation pipeline can call without knowing the underlying
library. Adapters are **optional** — Gnomon installs with zero TSFM dependencies
and degrades gracefully to baselines + statistical candidates when none are
present.

Design invariants (from the system design):
  - TSFMs are candidates, not oracles. They compete against mandatory baselines
    on identical rolling-origin folds. If a TSFM cannot beat the strongest
    baseline by the configured margin, Gnomon retains the baseline or abstains.
  - Adapters never receive or return forecast values that bypass evaluation.
    They produce *predictions*; the evaluation layer owns selection.
  - Lazy loading: model weights are downloaded / loaded only on first use,
    not at import time. A missing optional dependency is a soft skip, not an
    error.
  - Quantile support: adapters that provide native quantile forecasts expose
    them via ``predict_quantiles``; those that only produce point forecasts
    return ``None`` from that method, and Gnomon falls back to residual-based
    intervals.

Supported models (with optional extras):
  - chronos: Amazon Chronos-Bolt (mini 21M, small 48M) — ``pip install chronos-forecasting``
  - toto: Datadog Toto-2.0 (4M/22M) — ``pip install toto-models``
  - flowstate: IBM Granite FlowState R1.1 (18.5M) — ``pip install tsfm_public``
  - ttm: IBM Granite TinyTimeMixer R2 (1-3M) — ``pip install tsfm_public``
  - moirai: Salesforce Moirai-2.0-R-Small (~14M) — ``pip install uni2ts``
  - moment: CMU MOMENT-1-Small (~38M) — ``pip install momentfm``
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight pinning
# ---------------------------------------------------------------------------
#
# A ``forecast_id`` records the model *name*. Without a pinned revision the
# same name denotes whatever weights the Hub served that day, so two runs
# with the same id could publish different numbers — and the artifact store
# is first-write-wins, which would silently discard the second. Every
# adapter therefore loads at an explicit commit, and the commit actually
# resolved is reported through ``resolved_weights`` so it can reach the id
# payload and the evidence record.
#
# Resolved 2026-08-02 via ``HfApi().model_info(model_id).sha``. To move a
# pin, re-resolve and re-run the TSFM benchmark: changing a revision changes
# every forecast_id that selected the model, which is the intended effect.
TSFM_REVISIONS: dict[str, str] = {
    "amazon/chronos-bolt-mini": "251268337516a88e253628c43e1d26ec577b376b",
    "amazon/chronos-bolt-small": "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
    "Datadog/Toto-2.0-4m": "8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9",
    "Datadog/Toto-2.0-22m": "685e4ae3e2be8d8998025e53dd98e7fdcb296a89",
    "ibm-granite/granite-timeseries-ttm-r2": "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f",
    "Salesforce/moirai-2.0-R-small": "30f43ff08c8494f4943ae1521e9d4e94a0fbb389",
    "AutonLab/MOMENT-1-small": "411e288267f82cce86296dbe4d6c8bc533cc162f",
    "ibm-granite/granite-timeseries-flowstate-r1": "05effc6cb39ee16dce9dd0064ed1a76e4b8ff464",
}


class UnpinnedWeights(RuntimeError):
    """Raised when an adapter would load weights at an unpinned revision."""


def pinned_revision(model_id: str) -> str:
    """The commit an adapter must load ``model_id`` at.

    Failing loudly beats loading whatever is current: an unpinned load
    produces numbers that no id can honestly cover.
    """
    try:
        return TSFM_REVISIONS[model_id]
    except KeyError:
        raise UnpinnedWeights(
            f"No pinned revision for {model_id!r}. Add its commit sha to "
            "TSFM_REVISIONS before the adapter may load it."
        ) from None


#: Adapters whose weights live on disk rather than on the Hub. A Hub commit
#: cannot pin a local checkpoint, so these pin on content instead: the digest
#: of the files the wrapper actually loads. The guarantee is the same one
#: TSFM_REVISIONS provides -- two runs that report the same revision loaded
#: the same weights -- which is what a content-addressed forecast_id needs.
_LOCAL_CHECKPOINT_ENV: dict[str, str] = {
    "cascade": "GNOMON_CASCADE_CHECKPOINT",
}

_LOCAL_CHECKPOINT_FILES: dict[str, tuple[str, ...]] = {
    "cascade": (
        "config.json",
        "forecast_wrapper.py",
        "model.py",
        "weights.safetensors",
    ),
}

# path/name -> (file metadata signature, content digest).  The metadata is
# only a cache invalidator; the published identity remains the SHA-256 over
# every byte the adapter executes or loads.
_LOCAL_PIN_CACHE: dict[
    tuple[str, str], tuple[tuple[tuple[str, int, int, int, int], ...], str]
] = {}


def local_checkpoint_dir(name: str) -> "Any":
    """The configured checkpoint directory for a local adapter."""
    import os
    from pathlib import Path

    env = _LOCAL_CHECKPOINT_ENV.get(name)
    if env is None:
        raise KeyError(f"{name} is not a local-checkpoint adapter")
    raw = os.environ.get(env, "")
    if not raw:
        raise TSFMUnavailable(
            f"{name} needs a checkpoint directory in ${env}. It is a locally "
            f"trained model, so there is no default to fall back to."
        )
    path = Path(raw).expanduser()
    if not (path / "config.json").exists():
        raise TSFMUnavailable(f"${env}={raw} is not an exported checkpoint directory")
    return path


def local_pin(name: str) -> dict[str, str]:
    """``{local:<dir>: sha256}`` over the files that determine the forecast."""
    import hashlib

    path = local_checkpoint_dir(name)
    filenames = _LOCAL_CHECKPOINT_FILES[name]
    candidates = [(filename, path / filename) for filename in filenames]

    def signature() -> tuple[tuple[str, int, int, int, int], ...]:
        try:
            return tuple(
                (filename, stat.st_size, stat.st_mtime_ns,
                 stat.st_ctime_ns, stat.st_ino)
                for filename, candidate in candidates
                for stat in (candidate.stat(),)
            )
        except FileNotFoundError as exc:
            raise TSFMUnavailable(
                f"checkpoint at {path} is missing {exc.filename}"
            ) from None

    before = signature()
    key = (name, str(path.resolve()))
    cached = _LOCAL_PIN_CACHE.get(key)
    if cached is not None and cached[0] == before:
        return {f"local:{path.name}": cached[1]}

    digest = hashlib.sha256()
    # File names and boundaries are part of the identity: moving the same
    # bytes between config, executable wrapper, architecture and tensors must
    # not preserve a revision accidentally.
    for filename, candidate in candidates:
        label = filename.encode("utf-8")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    after = signature()
    if after != before:
        raise TSFMUnavailable(
            f"checkpoint at {path} changed while its identity was being computed"
        )
    revision = digest.hexdigest()
    _LOCAL_PIN_CACHE[key] = (after, revision)
    return {f"local:{path.name}": revision}


def resolved_weights(name: str) -> dict[str, str]:
    """``{model_id: revision}`` for an adapter, for ids and evidence."""
    if name in _LOCAL_CHECKPOINT_ENV:
        try:
            return local_pin(name)
        except TSFMUnavailable:
            return {}
    model_ids = _ADAPTER_MODEL_IDS.get(name, ())
    return {
        model_id: TSFM_REVISIONS[model_id]
        for model_id in model_ids if model_id in TSFM_REVISIONS
    }


#: Which Hub repos each adapter loads. Kept beside the revisions so a new
#: adapter cannot quietly skip the pin.
_ADAPTER_MODEL_IDS: dict[str, tuple[str, ...]] = {
    "chronos_bolt_mini": ("amazon/chronos-bolt-mini",),
    "chronos_bolt_small": ("amazon/chronos-bolt-small",),
    "toto2_4m": ("Datadog/Toto-2.0-4m",),
    "toto2_22m": ("Datadog/Toto-2.0-22m",),
    "flowstate": ("ibm-granite/granite-timeseries-flowstate-r1",),
    "ttm": ("ibm-granite/granite-timeseries-ttm-r2",),
    "moirai2_small": ("Salesforce/moirai-2.0-R-small",),
    "moment_small": ("AutonLab/MOMENT-1-small",),
}


@dataclass(frozen=True)
class TSFMCapabilities:
    """Capabilities implemented and verified by an Gnomon adapter.

    Upstream model claims are deliberately not enough to set these flags.
    Selection may only rely on behavior exposed through the adapter protocol.
    """

    past_observed_covariates: bool = False
    future_known_covariates: bool = False
    static_covariates: bool = False
    multivariate_targets: bool = False
    native_quantiles: bool = False
    #: Tasks this adapter implements and Gnomon has verified — a task appears
    #: here only once the adapter method behind it exists and is tested.
    tasks: tuple[str, ...] = ("forecast",)
    min_context_length: int = 1
    max_context_length: int | None = None
    max_horizon: int | None = None
    supported_frequencies: tuple[str, ...] | None = None
    fine_tuning: bool = False
    adapter_implemented: bool = True
    source: str | None = None
    verified_on: str = "2026-07-29"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CAPABILITIES: dict[str, TSFMCapabilities] = {
    "chronos_bolt_mini": TSFMCapabilities(
        native_quantiles=True, max_context_length=2048,
        source="https://github.com/amazon-science/chronos-forecasting",
    ),
    "chronos_bolt_small": TSFMCapabilities(
        native_quantiles=True, max_context_length=2048,
        source="https://github.com/amazon-science/chronos-forecasting",
    ),
    "toto2_4m": TSFMCapabilities(
        native_quantiles=True,
        min_context_length=32,
        source="https://github.com/DataDog/toto",
        verified_on="2026-08-19",
    ),
    "toto2_22m": TSFMCapabilities(
        native_quantiles=True,
        min_context_length=32,
        source="https://github.com/DataDog/toto",
    ),
    "flowstate": TSFMCapabilities(
        native_quantiles=True,
        supported_frequencies=("min", "5min", "15min", "30min", "h", "D", "W", "MS"),
        source="https://github.com/ibm-granite/granite-tsfm",
    ),
    "ttm": TSFMCapabilities(
        native_quantiles=False, max_context_length=512,
        source="https://github.com/ibm-granite/granite-tsfm",
    ),
    "moirai2_small": TSFMCapabilities(
        native_quantiles=True,
        source="https://github.com/SalesforceAIResearch/uni2ts",
    ),
    "moment_small": TSFMCapabilities(
        native_quantiles=False, max_context_length=512,
        tasks=("forecast", "detect_anomalies", "impute", "embed"),
        source="https://github.com/moment-timeseries-foundation-model/moment",
    ),
    # Locally trained, not a public release. The horizon ceiling is the
    # checkpoint's trained decode length, not a wrapper limitation: past it
    # the model extrapolates outside anything it was fit on, so Gnomon should
    # decline the candidate rather than quietly accept a worse forecast.
    "cascade": TSFMCapabilities(
        native_quantiles=True,
        min_context_length=64,
        max_context_length=4096,
        max_horizon=64,
        source="local checkpoint (cascade-model)",
        verified_on="2026-09-03",
    ),
}

_PARAMETER_COUNTS_M: dict[str, float] = {
    "chronos_bolt_mini": 21.0,
    "chronos_bolt_small": 48.0,
    "toto2_4m": 4.14,
    "toto2_22m": 22.0,
    "flowstate": 18.5,
    "ttm": 3.0,
    "moirai2_small": 14.0,
    "moment_small": 38.0,
    "cascade": 17.7,
}


def tsfm_capabilities(name: str) -> TSFMCapabilities:
    """Return adapter-level capabilities for a registered model."""
    if name not in _CAPABILITIES:
        raise KeyError(f"Unknown TSFM adapter: {name}")
    return _CAPABILITIES[name]


def tsfm_parameter_count(name: str) -> float:
    """Return the registered parameter count in millions, or zero if unknown."""
    return _PARAMETER_COUNTS_M.get(name, 0.0)


def tsfm_supports_quantiles(name: str) -> bool:
    """Return verified native-quantile support.

    Unregistered names are allowed by the generic HTTP adapter, whose remote
    protocol historically assumes quantile support unless told otherwise.
    """
    capabilities = _CAPABILITIES.get(name)
    return capabilities.native_quantiles if capabilities is not None else True


def capability_matrix() -> dict[str, dict[str, Any]]:
    return {name: _CAPABILITIES[name].to_dict() for name in sorted(_CAPABILITIES)}


def eligible_tsfms(
    *, history_length: int, horizon: int, frequency: str,
    require_future_covariates: bool = False,
    task: str = "forecast",
) -> tuple[list[str], dict[str, list[str]]]:
    """Filter registered adapters using verified, machine-actionable limits."""
    eligible: list[str] = []
    excluded: dict[str, list[str]] = {}
    for name in available_tsfms():
        caps = tsfm_capabilities(name)
        reasons: list[str] = []
        if task not in caps.tasks:
            reasons.append(f"Gnomon adapter does not implement task {task!r}")
        if require_future_covariates and not caps.future_known_covariates:
            reasons.append("Gnomon adapter does not implement future-known covariates")
        if caps.min_context_length > history_length:
            reasons.append(f"needs at least {caps.min_context_length} history points")
        if caps.max_horizon is not None and horizon > caps.max_horizon:
            reasons.append(f"horizon {horizon} exceeds verified maximum {caps.max_horizon}")
        if caps.supported_frequencies is not None and frequency not in caps.supported_frequencies:
            reasons.append(f"frequency {frequency!r} is not supported")
        if reasons:
            excluded[name] = reasons
        else:
            eligible.append(name)
    return eligible, excluded


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TSFMAdapter(Protocol):
    """Uniform interface every TSFM adapter must implement.

    Beyond the required forecasting surface below, adapters may implement
    two optional multi-task verbs, discovered via ``getattr`` and declared
    through ``TSFMCapabilities.tasks``:

    - ``reconstruct(history, mask=None) -> list[float]`` — a masked
      reconstruction of the series (``mask``: 1 = observed, 0 = missing).
      Reconstruction error powers anomaly scoring; reconstruction of
      masked points is imputation.
    - ``embed(history) -> list[float]`` — a fixed-length representation
      of the series, for downstream heads.

    An adapter without a verb simply doesn't list the task; capability
    flags are set only for implemented, tested behavior."""

    #: Stable identifier used in evidence, artifacts, and capabilities.
    name: str

    #: Approximate parameter count (in millions) for logging / display.
    params_m: float

    #: Whether the adapter provides native quantile forecasts.
    supports_quantiles: bool

    #: Optional native joint trajectories. Capability alone never admits
    #: these paths to a governed decision; the model-neutral boundary also
    #: requires a job-specific out-of-sample admission.
    supports_sample_paths: bool

    def predict(
        self,
        history: list[float],
        horizon: int,
        season: int,
    ) -> list[float]:
        """Return point forecasts for ``horizon`` future steps.

        Args:
            history: Observed values, ordered oldest → newest.
            horizon: Number of future periods to forecast.
            season: Seasonal period for the data frequency (e.g. 24 for hourly).

        Returns:
            List of ``horizon`` point forecasts.

        Raises:
            TSFMUnavailable: If the model or its dependency is not installed.
            TSFMError: If inference fails for any reason.
        """
        ...

    def predict_quantiles(
        self,
        history: list[float],
        horizon: int,
        season: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[dict[str, float]] | None:
        """Return quantile forecasts, one dict per future step.

        If the adapter does not support native quantiles, return ``None``
        and Gnomon will fall back to residual-based intervals.

        Returns:
            List of ``horizon`` dicts mapping quantile → value, or ``None``.
        """
        ...

    def predict_samples(
        self, history: list[float], horizon: int, season: int,
        samples: int,
    ) -> list[list[float]] | None:
        """Return optional native paths after adapter-specific inference."""
        ...


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TSFMUnavailable(Exception):
    """Raised when a TSFM adapter's optional dependency is not installed."""
    pass


class TSFMError(Exception):
    """Raised when a TSFM adapter fails during inference."""
    pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Each entry: name → factory that returns a TSFMAdapter (or raises TSFMUnavailable).
# Factories are called lazily on first use; the registry itself never imports
# optional dependencies at module load time.
_REGISTRY: dict[str, Callable[[], TSFMAdapter]] = {}


def register_tsfm(name: str, factory: Callable[[], TSFMAdapter]) -> None:
    """Register a TSFM adapter factory."""
    _REGISTRY[name] = factory


def available_tsfms() -> list[str]:
    """Return names of all registered TSFM adapters (not necessarily installed)."""
    return sorted(_REGISTRY.keys())


def get_tsfm(name: str) -> TSFMAdapter:
    """Instantiate a registered TSFM adapter.

    Raises:
        TSFMUnavailable: If the adapter's optional dependency is not installed.
        KeyError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown TSFM adapter: {name}. Registered: {available_tsfms()}")
    return _REGISTRY[name]()


#: The optional import each built-in adapter needs. An adapter registered
#: from outside this module is absent here, which is not the same thing as
#: "its dependency is missing" -- see ``dependency_missing``.
_ADAPTER_DEPENDENCIES: dict[str, str] = {
    "chronos_bolt_mini": "chronos",
    "chronos_bolt_small": "chronos",
    "toto2_4m": "toto2",
    "toto2_22m": "toto2",
    "flowstate": "tsfm_public",
    "ttm": "tsfm_public",
    "moirai2_small": "uni2ts",
    "moment_small": "momentfm",
}


def dependency_missing(name: str) -> bool:
    """``True`` only when this adapter is *known* to be unusable here.

    ``check_tsfm`` answers False both for "dependency missing" and for "not
    one of ours", so it cannot filter a candidate pool: it would silently drop
    every externally registered adapter. This answers the narrower question
    and defaults to False for anything it does not know about.
    """
    if name in _LOCAL_CHECKPOINT_ENV or name in _ADAPTER_DEPENDENCIES:
        return not check_tsfm(name)
    return False


def check_tsfm(name: str) -> bool:
    """Return ``True`` if the adapter's dependency is importable."""
    if name not in _REGISTRY:
        return False
    # Check if torch is available first (required by all adapters)
    try:
        __import__("torch")
    except ImportError:
        return False
    # Check the adapter's specific dependency
    if name in _LOCAL_CHECKPOINT_ENV:
        # No optional package to probe: a local adapter is "installed" when a
        # checkpoint is configured and readable.
        try:
            local_checkpoint_dir(name)
            return True
        except TSFMUnavailable:
            return False
    dep = _ADAPTER_DEPENDENCIES.get(name)
    if dep is None:
        return False
    try:
        __import__(dep)
        return True
    except ImportError:
        return False


def installed_tsfms() -> list[str]:
    """Return names of TSFM adapters whose dependencies are importable."""
    return [name for name in sorted(_REGISTRY) if check_tsfm(name)]


# ---------------------------------------------------------------------------
# Helpers shared by adapters
# ---------------------------------------------------------------------------

def _try_import(module: str) -> Any:
    """Import an optional dependency or raise TSFMUnavailable."""
    try:
        return __import__(module)
    except ImportError as exc:
        raise TSFMUnavailable(
            f"Optional dependency '{module}' is not installed. "
            f"Install it with: pip install {module.replace('-', '_')}"
        ) from exc


def _import_torch():
    """Import torch, a hard dependency for all TSFM adapters."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise TSFMUnavailable(
            "PyTorch is required for TSFM inference but is not installed. "
            "Install it with: pip install torch"
        ) from exc


# ---------------------------------------------------------------------------
# Adapter: Chronos-Bolt (Amazon)
# ---------------------------------------------------------------------------

class ChronosBoltAdapter:
    """Adapter for Amazon Chronos-Bolt models (mini / small).

    Chronos-Bolt is a T5-based encoder that produces quantile forecasts
    via direct regression (not sampling). Fast, runs on CPU.
    """

    name: str = "chronos_bolt"
    params_m: float
    supports_quantiles = True

    _MODEL_IDS = {
        "chronos_bolt_mini": "amazon/chronos-bolt-mini",
        "chronos_bolt_small": "amazon/chronos-bolt-small",
    }

    def __init__(self, variant: str = "chronos_bolt_mini"):
        if variant not in self._MODEL_IDS:
            raise TSFMUnavailable(f"Unknown Chronos-Bolt adapter: {variant}")
        self._variant = variant
        # Registry identity is part of selection, evidence, weight pinning,
        # and forecast identity.  A family-level name made mini and small
        # silently collide after instantiation.
        self.name = variant
        self.params_m = tsfm_parameter_count(variant)
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        torch = _import_torch()
        _try_import("chronos")
        model_id = self._MODEL_IDS.get(self._variant, self._MODEL_IDS["chronos_bolt_mini"])
        try:
            from chronos import BaseChronosPipeline
            self._pipeline = BaseChronosPipeline.from_pretrained(
                model_id,
                revision=pinned_revision(model_id),
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Failed to load Chronos-Bolt: {exc}") from exc

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        self._ensure_loaded()
        torch = _import_torch()
        try:
            context = torch.tensor(history, dtype=torch.float32)
            forecast = self._pipeline.predict(
                context=context,
                prediction_length=horizon,
            )
            # Chronos-Bolt returns quantile forecasts: shape [num_quantiles, prediction_length]
            # The median (0.5) is used as the point forecast.
            arr = forecast.numpy()
            if arr.ndim == 3:
                # [num_series, num_quantiles, horizon] — take first series
                arr = arr[0]
            # Find the median quantile index (usually index 1 for [0.1, 0.5, 0.9])
            median_idx = arr.shape[0] // 2
            return arr[median_idx].tolist()
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Chronos-Bolt prediction failed: {exc}") from exc

    def predict_quantiles(
        self,
        history: list[float],
        horizon: int,
        season: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[dict[str, float]]:
        self._ensure_loaded()
        torch = _import_torch()
        try:
            context = torch.tensor(history, dtype=torch.float32)
            forecast = self._pipeline.predict(
                context=context,
                prediction_length=horizon,
                quantile_levels=list(quantiles),
            )
            arr = forecast.numpy()
            if arr.ndim == 3:
                arr = arr[0]
            # arr shape: [num_quantiles, horizon]
            results = []
            for step in range(arr.shape[1]):
                row = {}
                for i, q in enumerate(quantiles):
                    row[str(q)] = float(arr[i, step])
                results.append(row)
            return results
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Chronos-Bolt quantile prediction failed: {exc}") from exc


def _register_chronos():
    register_tsfm("chronos_bolt_mini", lambda: ChronosBoltAdapter("chronos_bolt_mini"))
    register_tsfm("chronos_bolt_small", lambda: ChronosBoltAdapter("chronos_bolt_small"))


# ---------------------------------------------------------------------------
# Adapter: Toto-2.0 (Datadog)
# ---------------------------------------------------------------------------

class Toto2Adapter:
    """Adapter for Datadog Toto-2.0 checkpoints.

    Toto 2.0 is a decoder-only patched transformer with alternating
    time/variate attention and a quantile output head. SOTA on
    observability benchmarks (BOOM) and competitive on GIFT-Eval.
    """

    supports_quantiles = True
    min_history = 32
    _QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    _PATCH_SIZE = 32

    def __init__(self, name: str = "toto2_22m"):
        variants = {
            "toto2_4m": "Datadog/Toto-2.0-4m",
            "toto2_22m": "Datadog/Toto-2.0-22m",
        }
        if name not in variants:
            raise TSFMUnavailable(f"Unknown Toto 2.0 adapter: {name}")
        self.name = name
        self._MODEL_ID = variants[name]
        self.params_m = tsfm_parameter_count(name)
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        torch = _import_torch()
        _try_import("toto2")
        try:
            from toto2 import Toto2Model
            self._model = Toto2Model.from_pretrained(
                self._MODEL_ID, revision=pinned_revision(self._MODEL_ID),
            )
            device = torch.device("cpu")
            self._model = self._model.to(device).eval()
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Failed to load Toto-2.0: {exc}") from exc

    def _forecast_quantiles(self, history: list[float], horizon: int) -> Any:
        self._ensure_loaded()
        torch = _import_torch()
        try:
            device = next(self._model.parameters()).device
            # Toto's public forecast path reduces context into fixed-size
            # patches and currently requires the context axis to divide
            # exactly.  Real histories almost never satisfy that accidentally.
            # Left padding represents unobserved pre-history and preserves the
            # newest observation as the forecast anchor.
            padding = (-len(history)) % self._PATCH_SIZE
            padded = ([0.0] * padding) + list(history)
            observed = ([False] * padding) + ([True] * len(history))
            target = torch.tensor(padded, dtype=torch.float32, device=device)
            # Shape: (batch, n_variates, time_steps)
            target = target.unsqueeze(0).unsqueeze(0)
            target_mask = torch.tensor(
                observed, dtype=torch.bool, device=device,
            ).unsqueeze(0).unsqueeze(0)
            series_ids = torch.zeros(1, 1, dtype=torch.long, device=device)

            quantiles = self._model.forecast(
                {
                    "target": target,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=horizon,
                has_missing_values=bool(padding),
            )
            # Shape: (9, batch, n_variates, horizon)
            return quantiles
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Toto-2.0 forecast failed: {exc}") from exc

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        q = self._forecast_quantiles(history, horizon)
        arr = q.detach().cpu().numpy()
        # Preserve the horizon axis when horizon == 1. ``squeeze`` alone
        # collapses (9, 1, 1, 1) to (9,) and turns the median into a scalar.
        arr = arr.reshape(len(self._QUANTILE_LEVELS), -1)
        # Median is at index 4 (0.5 in [0.1..0.9])
        median_idx = 4
        return arr[median_idx].tolist()

    def predict_quantiles(
        self,
        history: list[float],
        horizon: int,
        season: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[dict[str, float]]:
        q = self._forecast_quantiles(history, horizon)
        arr = q.detach().cpu().numpy().reshape(
            len(self._QUANTILE_LEVELS), -1)
        # Map requested quantiles to the closest available levels
        results = []
        for step in range(arr.shape[1]):
            row = {}
            for rq in quantiles:
                # Find closest index in _QUANTILE_LEVELS
                idx = min(
                    range(len(self._QUANTILE_LEVELS)),
                    key=lambda i: abs(self._QUANTILE_LEVELS[i] - rq),
                )
                row[str(rq)] = float(arr[idx, step])
            results.append(row)
        return results


def _register_toto():
    register_tsfm("toto2_4m", lambda: Toto2Adapter("toto2_4m"))
    register_tsfm("toto2_22m", lambda: Toto2Adapter("toto2_22m"))


# ---------------------------------------------------------------------------
# Cascade (local checkpoint)
# ---------------------------------------------------------------------------

class CascadeAdapter:
    """A locally trained cascade-model checkpoint, loaded from disk.

    Unlike every other adapter here, this one wraps weights that were trained
    in-house rather than published. That changes two things and nothing else:
    the checkpoint directory comes from ``$GNOMON_CASCADE_CHECKPOINT``, and the
    revision is a content digest rather than a Hub commit. It competes on the
    same rolling-origin folds against the same mandatory baselines, and being
    ours buys it no exemption -- if it cannot beat the strongest baseline by
    the configured margin, Gnomon keeps the baseline. That is the point of
    routing a house model through the candidate pool instead of calling it
    directly: the evaluation is the thing that makes its output admissible.

    The checkpoint ships a self-contained ``forecast_wrapper.py`` and its own
    ``model.py``, so the adapter never imports the training package; a
    checkpoint stays loadable after the trainer moves on.
    """

    name = "cascade"
    backend = "local"
    supports_quantiles = True
    min_history = 64

    def __init__(self, name: str = "cascade"):
        self.name = name
        # Deliberately does not require a checkpoint: adapters are constructed
        # for metadata alone (parameter count, quantile support) in paths that
        # never run inference. An unconfigured checkpoint keeps this adapter
        # out of candidate pools through ``dependency_missing``, and raises
        # from ``_ensure_loaded`` if something asks it to forecast anyway.
        self.params_m = tsfm_parameter_count(name)
        self._pins = resolved_weights(name)
        self.revision = ",".join(
            f"{key}@{value}" for key, value in sorted(self._pins.items())
        ) or None
        self._wrapper = None
        self._levels: list[float] = []

    def _ensure_loaded(self):
        if self._wrapper is not None:
            return
        import importlib.util

        _import_torch()
        path = local_checkpoint_dir(self.name)
        if local_pin(self.name) != self._pins:
            raise TSFMError(
                f"{self.name} checkpoint changed after adapter construction; "
                "create a fresh adapter so its revision matches the loaded files"
            )
        try:
            spec = importlib.util.spec_from_file_location(
                f"gnomon_cascade_{path.name}", path / "forecast_wrapper.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._wrapper = module.Wrapper(str(path), "cpu")
            self._levels = [float(v) for v in self._wrapper.quantile_levels]
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Failed to load cascade checkpoint at {path}: {exc}") from exc

    def _quantile_array(self, history: list[float], horizon: int):
        self._ensure_loaded()
        try:
            out = self._wrapper.forecast_quantiles(list(history), horizon)
        except Exception as exc:
            raise TSFMError(f"Cascade forecast failed: {exc}") from exc
        # The wrapper returns a torch tensor on some checkpoints and a numpy
        # array on others; both are accepted rather than pinned to one, so a
        # re-exported checkpoint does not break the adapter.
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        # (1, horizon, num_quantiles) -> (horizon, num_quantiles)
        return out.reshape(horizon, len(self._levels))

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        arr = self._quantile_array(history, horizon)
        median = min(range(len(self._levels)), key=lambda i: abs(self._levels[i] - 0.5))
        return [float(value) for value in arr[:, median]]

    def predict_quantiles(
        self,
        history: list[float],
        horizon: int,
        season: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[dict[str, float]]:
        arr = self._quantile_array(history, horizon)
        rows = []
        for step in range(arr.shape[0]):
            # The head emits a fixed level grid, so a requested level is
            # answered by its nearest trained neighbour rather than by
            # interpolating between two levels the model never fit.
            rows.append({
                str(level): float(arr[step, min(
                    range(len(self._levels)),
                    key=lambda i: abs(self._levels[i] - level))])
                for level in quantiles
            })
        return rows


def _register_cascade():
    register_tsfm("cascade", lambda: CascadeAdapter("cascade"))


# ---------------------------------------------------------------------------
# Adapter: FlowState (IBM Granite)
# ---------------------------------------------------------------------------

class FlowStateAdapter:
    """Adapter for IBM Granite FlowState R1.1 (18.5M params).

    FlowState is a time-scale adjustable SSM-based foundation model.
    #1 on GIFT-Eval MASE among zero-shot models with public weights.
    Uses a scale factor to adapt to the data's sampling rate.
    """

    name: str = "flowstate"
    params_m: float = tsfm_parameter_count(name)
    supports_quantiles = True

    _MODEL_ID = "ibm-granite/granite-timeseries-flowstate-r1"
    #: The upstream branch the pinned commit was taken from. The load uses
    #: the commit, not the branch: a branch can move under a fixed name.
    _REVISION_BRANCH = "r1.1"

    _SCALE_FACTORS = {
        "h": 1.0,
        "D": 3.43,  # daily with weekly cycle
        "W": 0.46,
        "MS": 2.0,
    }

    def __init__(self, frequency: str = "h"):
        self._frequency = frequency
        self._predictor = None

    def _ensure_loaded(self):
        if self._predictor is not None:
            return
        torch = _import_torch()
        _try_import("tsfm_public")
        try:
            from tsfm_public import FlowStateForPrediction
            self._predictor = FlowStateForPrediction.from_pretrained(
                self._MODEL_ID,
                revision=pinned_revision(self._MODEL_ID),
            )
            device = torch.device("cpu")
            self._predictor = self._predictor.to(device)
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Failed to load FlowState: {exc}") from exc

    def _scale_factor(self) -> float:
        return self._SCALE_FACTORS.get(self._frequency, 1.0)

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        self._ensure_loaded()
        torch = _import_torch()
        try:
            device = next(self._predictor.parameters()).device
            ts = torch.tensor(history, dtype=torch.float32, device=device)
            # Shape: (context, batch, n_ch)
            ts = ts.unsqueeze(1).unsqueeze(2)
            result = self._predictor(
                ts,
                scale_factor=self._scale_factor(),
                prediction_length=horizon,
                batch_first=False,
            )
            # prediction_outputs: (batch, quantiles, forecast_length, n_ch)
            arr = result.prediction_outputs.detach().cpu().numpy()
            # Squeeze batch and channel: (quantiles, horizon)
            arr = arr.squeeze()
            # Median is typically at index 4 (0.5 quantile)
            median_idx = arr.shape[0] // 2
            return arr[median_idx].tolist()
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"FlowState prediction failed: {exc}") from exc

    def predict_quantiles(
        self,
        history: list[float],
        horizon: int,
        season: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[dict[str, float]]:
        self._ensure_loaded()
        torch = _import_torch()
        try:
            device = next(self._predictor.parameters()).device
            ts = torch.tensor(history, dtype=torch.float32, device=device)
            ts = ts.unsqueeze(1).unsqueeze(2)
            result = self._predictor(
                ts,
                scale_factor=self._scale_factor(),
                prediction_length=horizon,
                batch_first=False,
            )
            arr = result.prediction_outputs.detach().cpu().numpy().squeeze()
            # FlowState outputs 9 quantiles: [0.1..0.9]
            fs_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            results = []
            for step in range(arr.shape[1]):
                row = {}
                for rq in quantiles:
                    idx = min(range(len(fs_levels)), key=lambda i: abs(fs_levels[i] - rq))
                    row[str(rq)] = float(arr[idx, step])
                results.append(row)
            return results
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"FlowState quantile prediction failed: {exc}") from exc


def _register_flowstate():
    register_tsfm("flowstate", FlowStateAdapter)


# ---------------------------------------------------------------------------
# Adapter: TinyTimeMixer (IBM Granite)
# ---------------------------------------------------------------------------

class TinyTimeMixerAdapter:
    """Adapter for IBM Granite TinyTimeMixer R2 (1-3M params).

    TTM is the smallest TSFM available — under 3M parameters. Supports
    multivariate forecasting but only provides point forecasts (no native
    quantiles). Good for edge / CPU-only deployments.
    """

    name: str = "ttm"
    params_m: float = tsfm_parameter_count(name)
    supports_quantiles = False

    _MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        _import_torch()
        _try_import("tsfm_public")
        try:
            from tsfm_public import TinyTimeMixerForPrediction
            self._model = TinyTimeMixerForPrediction.from_pretrained(
                self._MODEL_ID, revision=pinned_revision(self._MODEL_ID),
            )
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Failed to load TTM: {exc}") from exc

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        self._ensure_loaded()
        torch = _import_torch()
        try:
            device = next(self._model.parameters()).device
            # TTM expects a specific input shape and context length
            # Pad or truncate context to the model's expected length
            ctx_len = min(len(history), 512)
            ctx = history[-ctx_len:]
            ts = torch.tensor(ctx, dtype=torch.float32, device=device)
            ts = ts.unsqueeze(0).unsqueeze(0)  # (batch, n_variates, time)

            # Use the model's forecast method
            output = self._model(ts, prediction_length=horizon)
            arr = output.prediction_outputs.detach().cpu().numpy().squeeze()
            if arr.ndim > 1:
                arr = arr[-1]  # take last channel if multivariate
            return arr[:horizon].tolist()
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"TTM prediction failed: {exc}") from exc

    def predict_quantiles(
        self,
        history: list[float],
        horizon: int,
        season: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[dict[str, float]] | None:
        return None  # TTM only supports point forecasting


def _register_ttm():
    register_tsfm("ttm", TinyTimeMixerAdapter)


# ---------------------------------------------------------------------------
# Adapter: Moirai-2.0 (Salesforce)
# ---------------------------------------------------------------------------

class Moirai2Adapter:
    """Adapter for Salesforce Moirai-2.0-R-Small (~14M params).

    Moirai 2.0 is a universal forecasting transformer with native
    quantile output. Strong general-purpose performance across domains.
    """

    name: str = "moirai2_small"
    params_m: float = tsfm_parameter_count(name)
    supports_quantiles = True

    _MODEL_ID = "Salesforce/moirai-2.0-R-small"

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        torch = _import_torch()
        _try_import("uni2ts")
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            module = Moirai2Module.from_pretrained(
                self._MODEL_ID, revision=pinned_revision(self._MODEL_ID),
            )
            self._model = Moirai2Forecast(
                module=module,
                prediction_length=1,  # set per-call
                context_length=min(512, 1000),
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            device = torch.device("cpu")
            self._model = self._model.to(device)
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Failed to load Moirai-2.0: {exc}") from exc

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        self._ensure_loaded()
        _import_torch()
        try:
            import pandas as pd
            from gluonts.dataset.pandas import PandasDataset

            # Build a minimal DataFrame
            idx = pd.date_range(
                start="2000-01-01", periods=len(history), freq="h"
            )
            df = pd.DataFrame({"target": history}, index=idx)
            ds = PandasDataset(dict(df))

            # Moirai stores this setting in Lightning hyperparameters; a plain
            # attribute assignment does not update the predictor or network.
            with self._model.hparams_context(prediction_length=horizon):
                predictor = self._model.create_predictor(batch_size=1)
                # ``history`` already contains only observations available at
                # the forecast origin. Splitting it at ``-horizon`` would
                # silently discard the newest observed horizon before inference.
                forecast = next(iter(predictor.predict(ds)))
            # Mean as point forecast
            return forecast.mean.tolist()
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Moirai-2.0 prediction failed: {exc}") from exc

    def predict_quantiles(
        self,
        history: list[float],
        horizon: int,
        season: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[dict[str, float]]:
        self._ensure_loaded()
        _import_torch()
        try:
            import pandas as pd
            from gluonts.dataset.pandas import PandasDataset

            idx = pd.date_range(
                start="2000-01-01", periods=len(history), freq="h"
            )
            df = pd.DataFrame({"target": history}, index=idx)
            ds = PandasDataset(dict(df))

            with self._model.hparams_context(prediction_length=horizon):
                predictor = self._model.create_predictor(batch_size=1)
                forecast = next(iter(predictor.predict(ds)))
            results = []
            for step in range(horizon):
                row = {}
                for q in quantiles:
                    row[str(q)] = float(forecast.quantile(q)[step])
                results.append(row)
            return results
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Moirai-2.0 quantile prediction failed: {exc}") from exc


def _register_moirai():
    register_tsfm("moirai2_small", Moirai2Adapter)


# ---------------------------------------------------------------------------
# Adapter: MOMENT-1 (CMU AutonLab)
# ---------------------------------------------------------------------------

class MomentAdapter:
    """Adapter for CMU MOMENT-1-Small (~38M params).

    MOMENT is a family of open-source foundation models for general-purpose
    time-series analysis. Supports forecasting, anomaly detection, and
    representation learning. Native quantile support via sampling.
    """

    name: str = "moment_small"
    params_m: float = tsfm_parameter_count(name)
    supports_quantiles = False

    _MODEL_ID = "AutonLab/MOMENT-1-small"

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        _import_torch()
        _try_import("momentfm")
        try:
            from momentfm import MOMENTPipeline
            self._model = MOMENTPipeline.from_pretrained(
                self._MODEL_ID,
                revision=pinned_revision(self._MODEL_ID),
                model_kwargs={
                    "task_name": "forecasting",
                    "forecast_horizon": 1,
                },
            )
            self._model.init()
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Failed to load MOMENT: {exc}") from exc

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        self._ensure_loaded()
        torch = _import_torch()
        try:
            device = next(self._model.model.parameters()).device
            # MOMENT expects context length of 512
            ctx_len = min(len(history), 512)
            ctx = history[-ctx_len:]
            # Pad to 512 if needed
            if len(ctx) < 512:
                pad = [0.0] * (512 - len(ctx))
                ctx = pad + ctx
            ts = torch.tensor(ctx, dtype=torch.float32, device=device)
            ts = ts.unsqueeze(0).unsqueeze(0)  # (batch, n_variates, time)

            # MOMENT forecasts in chunks; set horizon
            self._model.model.forecast_horizon = horizon
            output = self._model.model(ts)
            # Output shape: (batch, horizon)
            arr = output.reconstruction.detach().cpu().numpy()
            if arr.ndim > 1:
                arr = arr.squeeze()
            return arr[:horizon].tolist()
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"MOMENT prediction failed: {exc}") from exc

    def predict_quantiles(
        self,
        history: list[float],
        horizon: int,
        season: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> list[dict[str, float]] | None:
        # MOMENT can provide quantiles via sampling, but the public API
        # for forecasting is reconstruction-based. Return None to fall back
        # to Gnomon's residual-based intervals.
        return None

    def _reconstruction_pipeline(self):
        _import_torch()
        _try_import("momentfm")
        try:
            from momentfm import MOMENTPipeline
            model = MOMENTPipeline.from_pretrained(
                self._MODEL_ID, revision=pinned_revision(self._MODEL_ID),
                model_kwargs={"task_name": "reconstruction"},
            )
            model.init()
            return model
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Failed to load MOMENT reconstruction head: {exc}") from exc

    def _windowed(self, history: list[float]):
        torch = _import_torch()
        ctx_len = min(len(history), 512)
        ctx = history[-ctx_len:]
        padding = 512 - len(ctx)
        padded = [0.0] * padding + ctx
        ts = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_mask = torch.tensor(
            [0.0] * padding + [1.0] * len(ctx), dtype=torch.float32,
        ).unsqueeze(0)
        return ts, input_mask, padding, ctx_len

    def reconstruct(
        self, history: list[float], mask: list[int] | None = None,
    ) -> list[float]:
        """Masked reconstruction of the trailing (≤512-point) window.

        ``mask`` (1 = observed, 0 = missing) marks points the model must
        not see: reconstruction there is imputation; elsewhere the gap
        between value and reconstruction is the anomaly signal."""
        torch = _import_torch()
        model = self._reconstruction_pipeline()
        try:
            ts, input_mask, padding, ctx_len = self._windowed(history)
            if mask is not None:
                tail_mask = mask[-ctx_len:]
                for offset, observed in enumerate(tail_mask):
                    if not observed:
                        input_mask[0, padding + offset] = 0.0
                        ts[0, 0, padding + offset] = 0.0
            with torch.no_grad():
                output = model.model(x_enc=ts, input_mask=input_mask)
            values = output.reconstruction.detach().cpu().numpy().squeeze()
            return values[padding:].tolist()
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"MOMENT reconstruction failed: {exc}") from exc

    def embed(self, history: list[float]) -> list[float]:
        """Fixed-length embedding of the trailing (≤512-point) window."""
        torch = _import_torch()
        _try_import("momentfm")
        try:
            from momentfm import MOMENTPipeline
            model = MOMENTPipeline.from_pretrained(
                self._MODEL_ID, revision=pinned_revision(self._MODEL_ID),
                model_kwargs={"task_name": "embedding"},
            )
            model.init()
            ts, input_mask, _, _ = self._windowed(history)
            with torch.no_grad():
                output = model.model(x_enc=ts, input_mask=input_mask)
            embedding = output.embeddings.detach().cpu().numpy().squeeze()
            if embedding.ndim > 1:
                embedding = embedding.mean(axis=0)
            return embedding.tolist()
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"MOMENT embedding failed: {exc}") from exc


def _register_moment():
    register_tsfm("moment_small", MomentAdapter)


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------

def _register_all():
    """Register all known adapters. Called at module import."""
    _register_chronos()
    _register_toto()
    _register_cascade()
    _register_flowstate()
    _register_ttm()
    _register_moirai()
    _register_moment()


_register_all()


# ---------------------------------------------------------------------------
# Public convenience: iterate candidates for evaluation
# ---------------------------------------------------------------------------

def tsfm_candidates(
    requested: list[str] | None = None,
    frequency: str = "h",
) -> list[TSFMAdapter]:
    """Return instantiated TSFM adapters for evaluation.

    Only adapters whose dependencies are actually installed are returned.
    If ``requested`` is provided, only those names are tried; otherwise all
    registered adapters are tried.

    Args:
        requested: Optional list of adapter names to try.
        frequency: Data frequency, passed to adapters that need it (e.g. FlowState).

    Returns:
        List of ready-to-use TSFMAdapter instances.
    """
    names = requested or available_tsfms()
    candidates: list[TSFMAdapter] = []
    for name in names:
        try:
            adapter = get_tsfm(name)
            if isinstance(adapter, FlowStateAdapter):
                adapter._frequency = frequency
            candidates.append(adapter)
            logger.info("TSFM adapter '%s' loaded successfully", name)
        except TSFMUnavailable:
            logger.debug("TSFM adapter '%s' not available (dependency missing)", name)
        except Exception:
            logger.debug("TSFM adapter '%s' failed to initialize", name, exc_info=True)
    return candidates
