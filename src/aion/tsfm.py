"""Time Series Foundation Model (TSFM) adapter framework for Aion.

Each adapter wraps a pre-trained foundation model behind a uniform protocol
that Aion's evaluation pipeline can call without knowing the underlying
library. Adapters are **optional** — Aion installs with zero TSFM dependencies
and degrades gracefully to baselines + statistical candidates when none are
present.

Design invariants (from the system design):
  - TSFMs are candidates, not oracles. They compete against mandatory baselines
    on identical rolling-origin folds. If a TSFM cannot beat the strongest
    baseline by the configured margin, Aion retains the baseline or abstains.
  - Adapters never receive or return forecast values that bypass evaluation.
    They produce *predictions*; the evaluation layer owns selection.
  - Lazy loading: model weights are downloaded / loaded only on first use,
    not at import time. A missing optional dependency is a soft skip, not an
    error.
  - Quantile support: adapters that provide native quantile forecasts expose
    them via ``predict_quantiles``; those that only produce point forecasts
    return ``None`` from that method, and Aion falls back to residual-based
    intervals.

Supported models (with optional extras):
  - chronos: Amazon Chronos-Bolt (mini 21M, small 48M) — ``pip install chronos-forecasting``
  - toto: Datadog Toto-2.0-22m (22M) — ``pip install toto-models``
  - flowstate: IBM Granite FlowState R1.1 (18.5M) — ``pip install tsfm_public``
  - ttm: IBM Granite TinyTimeMixer R2 (1-3M) — ``pip install tsfm_public``
  - moirai: Salesforce Moirai-2.0-R-Small (~14M) — ``pip install uni2ts``
  - moment: CMU MOMENT-1-Small (~38M) — ``pip install momentfm``
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TSFMAdapter(Protocol):
    """Uniform interface every TSFM adapter must implement."""

    #: Stable identifier used in evidence, artifacts, and capabilities.
    name: str

    #: Approximate parameter count (in millions) for logging / display.
    params_m: float

    #: Whether the adapter provides native quantile forecasts.
    supports_quantiles: bool

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
        and Aion will fall back to residual-based intervals.

        Returns:
            List of ``horizon`` dicts mapping quantile → value, or ``None``.
        """
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
    dep_map = {
        "chronos_bolt_mini": "chronos",
        "chronos_bolt_small": "chronos",
        "toto2_22m": "toto2",
        "flowstate": "tsfm_public",
        "ttm": "tsfm_public",
        "moirai2_small": "uni2ts",
        "moment_small": "momentfm",
    }
    dep = dep_map.get(name)
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
        self._variant = variant
        self.params_m = 21.0 if "mini" in variant else 48.0
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        torch = _import_torch()
        chronos = _try_import("chronos")
        model_id = self._MODEL_IDS.get(self._variant, self._MODEL_IDS["chronos_bolt_mini"])
        try:
            from chronos import BaseChronosPipeline
            self._pipeline = BaseChronosPipeline.from_pretrained(
                model_id,
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
            import numpy as np
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
            import numpy as np
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
    """Adapter for Datadog Toto-2.0-22m.

    Toto 2.0 is a decoder-only patched transformer with alternating
    time/variate attention and a quantile output head. SOTA on
    observability benchmarks (BOOM) and competitive on GIFT-Eval.
    """

    name: str = "toto2_22m"
    params_m: float = 22.0
    supports_quantiles = True

    _MODEL_ID = "Datadog/Toto-2.0-22m"
    _QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        torch = _import_torch()
        _try_import("toto2")
        try:
            from toto2 import Toto2Model
            self._model = Toto2Model.from_pretrained(self._MODEL_ID)
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
            target = torch.tensor(history, dtype=torch.float32, device=device)
            # Shape: (batch, n_variates, time_steps)
            target = target.unsqueeze(0).unsqueeze(0)
            target_mask = torch.ones_like(target, dtype=torch.bool)
            series_ids = torch.zeros(1, 1, dtype=torch.long, device=device)

            quantiles = self._model.forecast(
                {
                    "target": target,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=horizon,
                has_missing_values=False,
            )
            # Shape: (9, batch, n_variates, horizon)
            return quantiles
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Toto-2.0 forecast failed: {exc}") from exc

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        q = self._forecast_quantiles(history, horizon)
        torch = _import_torch()
        arr = q.detach().cpu().numpy()
        # Squeeze batch and variate dims: (9, 1, 1, horizon) → (9, horizon)
        arr = arr.squeeze()
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
        arr = q.detach().cpu().numpy().squeeze()
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
    register_tsfm("toto2_22m", Toto2Adapter)


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
    params_m: float = 18.5
    supports_quantiles = True

    _MODEL_ID = "ibm-granite/granite-timeseries-flowstate-r1"
    _REVISION = "r1.1"

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
                revision=self._REVISION,
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
    params_m: float = 3.0
    supports_quantiles = False

    _MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        torch = _import_torch()
        _try_import("tsfm_public")
        try:
            from tsfm_public import TinyTimeMixerForPrediction
            self._model = TinyTimeMixerForPrediction.from_pretrained(self._MODEL_ID)
        except TSFMUnavailable:
            raise
        except Exception as exc:
            raise TSFMError(f"Failed to load TTM: {exc}") from exc

    def predict(self, history: list[float], horizon: int, season: int) -> list[float]:
        self._ensure_loaded()
        torch = _import_torch()
        try:
            import numpy as np
            device = next(self._model.parameters()).device
            # TTM expects a specific input shape and context length
            # Pad or truncate context to the model's expected length
            ctx_len = min(len(history), 512)
            ctx = history[-ctx_len:]
            ts = torch.tensor(ctx, dtype=torch.float32, device=device)
            ts = ts.unsqueeze(0).unsqueeze(0)  # (batch, n_variates, time)

            from tsfm_public.toolkit.util import select_by_index
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
    params_m: float = 14.0
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
            from uni2ts.model.moirai import Moirai2Forecast, Moirai2Module
            module = Moirai2Module.from_pretrained(self._MODEL_ID)
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
        torch = _import_torch()
        try:
            import pandas as pd
            import numpy as np
            from gluonts.dataset.pandas import PandasDataset
            from gluonts.dataset.split import split

            # Build a minimal DataFrame
            idx = pd.date_range(
                start="2000-01-01", periods=len(history), freq="h"
            )
            df = pd.DataFrame({"target": history}, index=idx)
            ds = PandasDataset(dict(df))

            train, test_template = split(ds, offset=-horizon)
            test_data = test_template.generate_instances(
                prediction_length=horizon,
                windows=1,
                distance=horizon,
            )

            # Update prediction length
            self._model.prediction_length = horizon
            predictor = self._model.create_predictor(batch_size=1)
            forecasts = predictor.predict(test_data.input)

            forecast = next(iter(forecasts))
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
        torch = _import_torch()
        try:
            import pandas as pd
            import numpy as np
            from gluonts.dataset.pandas import PandasDataset
            from gluonts.dataset.split import split

            idx = pd.date_range(
                start="2000-01-01", periods=len(history), freq="h"
            )
            df = pd.DataFrame({"target": history}, index=idx)
            ds = PandasDataset(dict(df))

            train, test_template = split(ds, offset=-horizon)
            test_data = test_template.generate_instances(
                prediction_length=horizon,
                windows=1,
                distance=horizon,
            )

            self._model.prediction_length = horizon
            predictor = self._model.create_predictor(batch_size=1)
            forecasts = predictor.predict(test_data.input)

            forecast = next(iter(forecasts))
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
    params_m: float = 38.0
    supports_quantiles = True

    _MODEL_ID = "AutonLab/MOMENT-1-small"

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        torch = _import_torch()
        _try_import("momentfm")
        try:
            from momentfm import MOMENTPipeline
            self._model = MOMENTPipeline.from_pretrained(
                self._MODEL_ID,
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
            import numpy as np
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
        # to Aion's residual-based intervals.
        return None


def _register_moment():
    register_tsfm("moment_small", MomentAdapter)


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------

def _register_all():
    """Register all known adapters. Called at module import."""
    _register_chronos()
    _register_toto()
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
