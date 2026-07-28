from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Support = Literal["supported", "weakly_supported", "unsupported"]


@dataclass(frozen=True)
class DataSchema:
    time_column: str
    target_column: str
    series_column: str | None
    frequency: str
    timezone: str | None
    missing_policy: str = "reject"
    duplicate_policy: str = "reject"


@dataclass(frozen=True)
class ForecastTask:
    input_path: str
    schema: DataSchema
    horizon: int
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    minimum_baseline_improvement: float = 0.02


@dataclass
class Evidence:
    evidence_id: str
    kind: str
    series: str
    payload: dict[str, Any]


@dataclass
class SeriesResult:
    series: str
    support: Support
    selected_model: str | None
    strongest_baseline: str | None
    selection_scores: dict[str, float | None]
    test_scores: dict[str, float | None]
    baseline_improvement: float | None
    interval_coverage: float | None
    warnings: list[str]
    forecast: list[dict[str, Any]]


@dataclass
class ForecastArtifact:
    schema_version: str
    forecast_id: str
    created_at: str
    status: Literal["complete", "partial"]
    task: ForecastTask
    source_fingerprint: str
    results: list[SeriesResult]
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeadwaterError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "status": "error",
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": False,
                "details": self.details,
            },
        }
