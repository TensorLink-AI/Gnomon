from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Support = Literal["supported", "weakly_supported", "degraded", "supported_ensemble", "unsupported"]

# The harness-wide vocabulary. ``Support`` above is the frozen v0.2 enum;
# new code speaks these.
SupportStatus = Literal[
    "supported", "conditionally_supported", "inconclusive", "unsupported", "invalid"
]
ClaimClass = Literal[
    "descriptive", "predictive", "associational", "causal", "counterfactual", "decision"
]


@dataclass(frozen=True)
class SupportReason:
    code: str
    message: str


@dataclass
class SupportAssessment:
    """The honest verdict on one requested output.

    ``inconclusive`` (not enough evidence either way) is not ``unsupported``
    (evidence against), and neither is ``invalid`` (the question was
    malformed) — and none of them is an operator failure."""

    status: SupportStatus
    reasons: list[SupportReason] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    sensitivity: dict[str, Any] = field(default_factory=dict)
    recovery_actions: list[SupportReason] = field(default_factory=list)
    legacy_support: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataSourceRef:
    """Reference to a temporal data source: a local file or ``store:<dataset>``."""

    ref: str
    time_column: str
    target_column: str
    series_column: str | None = None
    frequency: str | None = None


@dataclass(frozen=True)
class ForecastSpec:
    """Requested-output spec for the forecasting verb."""

    horizon: int
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    threshold: float | None = None
    minimum_baseline_improvement: float = 0.02


@dataclass(frozen=True)
class DecisionPolicy:
    """What the caller can act on, and — optionally — what outcomes cost.

    Utilities are optional by contract: without them a decision output must
    degrade to a feasible-action comparison with exceedance probabilities
    (``conditionally_supported: missing utility inputs``), never a silent
    guess and never a hard failure."""

    actions: tuple[str, ...] = ()
    utilities: tuple[tuple[str, float], ...] | None = None
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionBudget:
    max_wall_seconds: float | None = None
    max_steps: int | None = None


@dataclass(frozen=True)
class TemporalTask:
    """The general task contract: an objective compiled into validated,
    snapshot-bound execution. Forecasting is one ``task_type`` among several."""

    objective: str
    task_type: Literal["forecast", "investigate_change", "decide", "monitor"]
    sources: tuple[DataSourceRef, ...]
    outputs: tuple[Any, ...]
    as_of: str | None = None
    decision_policy: DecisionPolicy | None = None
    budget: ExecutionBudget | None = None
    permissions: tuple[str, ...] = ("read_local",)

    def task_id(self) -> str:
        from .ids import content_id
        return content_id("task", asdict(self))


def forecast_task(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    horizon: int,
    series_column: str | None = None,
    frequency: str | None = None,
    threshold: float | None = None,
    minimum_baseline_improvement: float = 0.02,
    as_of: str | None = None,
    objective: str | None = None,
) -> TemporalTask:
    """Thin constructor: a ForecastTask is a TemporalTask with a ForecastSpec."""
    return TemporalTask(
        objective=objective or f"Forecast {target_column} {horizon} periods ahead",
        task_type="forecast",
        sources=(DataSourceRef(input_path, time_column, target_column, series_column, frequency),),
        outputs=(ForecastSpec(horizon, threshold=threshold,
                              minimum_baseline_improvement=minimum_baseline_improvement),),
        as_of=as_of,
    )


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
    as_of: str | None = None


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
    context: dict[str, Any] | None = None
    covariates: dict[str, Any] | None = None
    threshold: dict[str, Any] | None = None
    support_assessment: dict[str, Any] | None = None


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


class AionError(Exception):
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
