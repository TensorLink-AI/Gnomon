"""Provider-neutral time-series execution and optional temporal evidence."""

from .forecast_adapter import AdapterCapabilities, ForecastRequest, ForecastResult
from .inference import ForecastExecution, InferenceEngine
from .ephemeris import EphemerisProvider
from .ledger import TemporalLedger
from .session import GnomonSession
from .backtesting import EvaluationBudget, evaluate_reference
from .temporal_store import TemporalStore
from .product_contract import __version__
from .final_selection import forecast_completion, forecast_request_fingerprint, resolve_final_selection

__all__ = [
    "AdapterCapabilities", "ForecastRequest", "ForecastResult",
    "ForecastExecution", "InferenceEngine", "EphemerisProvider",
    "TemporalLedger", "GnomonSession", "EvaluationBudget", "evaluate_reference",
    "TemporalStore", "temporal_operation", "__version__",
    "forecast_completion", "forecast_request_fingerprint", "resolve_final_selection",
]


def __getattr__(name):
    if name == "temporal_operation":
        from .temporal_ops import temporal_operation
        return temporal_operation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .decision_memory import put_lesson as put_lesson
