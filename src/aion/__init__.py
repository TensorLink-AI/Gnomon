"""Aion forecasting runtime.

The Python API is one of three front doors, beside the CLI and the MCP
server, and it reaches the same product: all five verbs, the bitemporal
store, and the covariate helpers. Each macro returns ``(payload, path)`` —
the payload for reading, the path to the immutable artifact directory that
owns the numbers.
"""

from .covariates import covariate_guide, load_covariates, validate_covariate_file
from .macros import decide, detect_anomalies, investigate_change, monitor
from .runtime import capabilities, forecast, inspect_dataset
from .temporal_store import TemporalStore
from .tracking import TrackingStore

__all__ = [
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
]
__version__ = "0.4.0"
