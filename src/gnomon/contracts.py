"""Shared data schema and structured failures. Forecast types live in forecast_adapter."""

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class DataSchema:
    time_column: str
    target_column: str
    series_column: str | None
    frequency: str
    timezone: str | None
    missing_policy: str = "reject"
    duplicate_policy: str = "reject"


# Callers can supply instance-specific repairs; do not suggest retired workflows.
REPAIR_OPTIONS = {
    "INVALID_TIMESTAMP": [{"action": "correct_timestamp", "description": "Correct the source timestamp to ISO 8601, e.g. 2026-01-02 or 2026-01-02T00:00:00Z. Declare the actual source timezone with --timezone when needed. Aggressive dropping is limited to 5% of all input rows; it cannot infer a missing timestamp."}],
    "INVALID_TARGET": [{"action": "correct_target", "description": "Supply a finite numeric observation in the target column, or select the correct --target-column."}],
    "EMPTY_DATASET": [{"action": "supply_observations", "description": "Supply input containing observed timestamp/value rows; repair modes cannot create source observations."}],
    "TIMESTAMP_ALIGNMENT_CONFLICT": [{"action": "correct_conflicting_timestamps", "description": "Correct conflicting timestamps or measurements at the source. Safe repair cannot merge distinct observations. For exact conflicting duplicate rows only, aggressive chooses the last row subject to the 30% original-observation budget."}],
    "ERROR_DETAIL_LIMIT": [{"action": "simplify_request", "description": "Consult the tool schema and reduce invalid or oversized arguments; error details were not retained."}],
    "NON_FINITE_TARGET": [{"action": "supply_arguments", "description": "Supply finite observations; infinity and NaN are not measured values."}],
    "UNSUPPORTED_INPUT": [{"action": "convert_input", "description": "Supply CSV/TSV, JSON/JSONL, Parquet or Excel data with explicit column mappings."}],
    "INVALID_ARGUMENTS": [{"action": "supply_arguments", "description": "Use the tool's declared schema and provide valid arguments."}],
    "INTERNAL_ERROR": [{"action": "report_bug", "description": "Report the failing operation without credentials."}],
    "INPUT_NOT_FOUND": [{"action": "supply_input", "description": "Supply an existing input path."}],
    "EXECUTION_FAILED": [{"action": "check_provider_environment", "description": "Verify the provider configuration and run Gnomon in the same Python environment as the provider and its dependencies."}],
}

class GnomonError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None,
                 repair_options: list[dict[str, Any]] | None = None,
                 retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        #: Instance-specific repairs, when the useful advice depends on the
        #: particular failure rather than only on the code. Falls back to
        #: the code's entry in REPAIR_OPTIONS.
        self.repair_options = repair_options
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        repairs = (self.repair_options if self.repair_options is not None
                   else REPAIR_OPTIONS.get(self.code, []))
        return {
            "schema_version": "0.1",
            "status": "error",
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
                "repair_options": repairs,
            },
            "rejection": {
                "code": self.code, "reason": self.message,
                "missing": self.details,
                "admissibility_path": repairs[:1],
                "terminal": True,
            },
        }
