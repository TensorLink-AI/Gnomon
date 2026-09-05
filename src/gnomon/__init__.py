"""Time-series execution tools with shared Python, CLI and MCP contracts.

Inference providers, optional immutable evidence, and opt-in temporal calculations.
Advanced evaluated macros remain lazily available for explicit legacy workflows.
"""

from importlib import import_module

from .forecast_adapter import AdapterCapabilities, ForecastRequest, ForecastResult
from .inference import ForecastExecution, InferenceEngine
from .ephemeris import EphemerisProvider
from .ledger import TemporalLedger
from .session import GnomonSession

# Direct inference does not import the evaluation/publication/context stack.
# Preserve the existing Python front door without requiring another package.
_LAZY_EXPORTS = {
    "temporal_operation": "temporal_ops",
    "EvaluationBudget": "backtesting", "evaluate_reference": "backtesting",
    "covariate_guide": "covariates", "load_covariates": "covariates",
    "validate_covariate_file": "covariates", "decide": "macros",
    "detect_anomalies": "macros", "investigate_change": "macros", "monitor": "macros",
    "capabilities": "runtime", "forecast": "runtime", "inspect_dataset": "runtime",
    "TemporalStore": "temporal_store", "TrackingStore": "tracking",
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_LAZY_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "AdapterCapabilities", "ForecastRequest", "ForecastResult",
    "ForecastExecution", "InferenceEngine",
    "EphemerisProvider",
    "TemporalLedger",
    "GnomonSession",
    "temporal_operation",
    "EvaluationBudget", "evaluate_reference",
    # The five verbs.
    "forecast",
    "investigate_change",
    "decide",
    "monitor",
    "detect_anomalies",
    # Inspection and capability disclosure.
    "inspect_dataset",
    "capabilities",
    # Point-in-time covariates.
    "covariate_guide",
    "load_covariates",
    "validate_covariate_file",
    # Persistent state: the bitemporal store and the tracking registry.
    "TemporalStore",
    "TrackingStore",
    "__version__",
]
from .versioning import RUNTIME_VERSION as __version__  # noqa: E402
