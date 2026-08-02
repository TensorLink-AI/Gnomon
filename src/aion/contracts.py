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
    #: Correct-but-surprising facts about how this result was produced.
    #: A disclosure never changes ``status`` — that is what separates it
    #: from a reason — but a reader who does not see it will misread the
    #: numbers. Typed rather than free text so an agent can branch on it.
    disclosures: list[SupportReason] = field(default_factory=list)

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
    # Informational disclosures (e.g. an uninstalled-but-eligible TSFM tier);
    # unlike warnings these never downgrade support.
    notes: list[str] = field(default_factory=list)
    # Answers to "what if this event happens", each with its own
    # `conditional_on_event` support and assumptions. Strictly additive: every
    # field above keeps its unconditional value, so a reader that ignores this
    # key sees what it has always seen.
    conditional_forecasts: list[dict[str, Any]] = field(default_factory=list)


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
        payload = asdict(self)
        for result in payload.get("results", []):
            # Additive keys appear only when they carry something. A run that
            # produced no conditional forecast serialises exactly as it did
            # before the feature existed, which is what makes the v0.2 surface
            # freeze verifiable byte-for-byte rather than by inspection.
            if not result.get("conditional_forecasts"):
                result.pop("conditional_forecasts", None)
        return payload


# Machine-readable repair options per error code: what a host model can do
# next without human help. This layer is what lets hosts self-correct — it
# appreciates as models improve, because better models act on it better.
REPAIR_OPTIONS: dict[str, list[dict[str, str]]] = {
    "INVALID_HORIZON": [
        {"action": "set_horizon", "description": "Pass a horizon of at least 1 period."},
    ],
    "MISSING_COLUMNS": [
        {"action": "inspect_dataset", "description": "Call aion_inspect to see the available columns, then correct the mapping."},
    ],
    "INVALID_TIMESTAMP": [
        {"action": "fix_timestamps", "description": "Convert the timestamp column to ISO-8601; the offending row is in details."},
        {"action": "enable_repair", "description": "Pass repair=aggressive to drop unparseable rows (capped and disclosed)."},
    ],
    "INVALID_TARGET": [
        {"action": "fix_target", "description": "Make the target column numeric; the offending row is in details."},
        {"action": "enable_repair", "description": "Pass repair=aggressive to drop rows without a numeric reading (capped and disclosed)."},
    ],
    "DUPLICATE_TIMESTAMPS": [
        {"action": "deduplicate", "description": "Remove duplicate timestamps, or ingest revisions into the store with distinct known-at times."},
        {"action": "enable_repair", "description": "Identical duplicate rows collapse under the default repair=safe; conflicting values need repair=aggressive (last row wins, disclosed)."},
    ],
    "IRREGULAR_TIME_GRID": [
        {"action": "fill_or_resample", "description": "Fill the missing period named in details, or resample to a coarser regular frequency."},
        {"action": "enable_repair", "description": "Pass repair=aggressive to interpolate interior gaps and snap jittered timestamps — capped, and every fix becomes a warning."},
    ],
    "MIXED_TIMEZONES": [
        {"action": "align_timezones", "description": "Make every timestamp consistently timezone-aware or consistently naive."},
        {"action": "enable_repair", "description": "Pass repair=aggressive to assume naive timestamps are UTC (disclosed as an assumption)."},
    ],
    "AMBIGUOUS_DATE_ORDER": [
        {"action": "fix_timestamps", "description": "Use ISO-8601 dates, or include at least one date whose day exceeds 12 so the order is provable."},
        {"action": "enable_repair", "description": "Pass repair=aggressive to default to month-first (disclosed as an assumption)."},
    ],
    "EXCESSIVE_REPAIR": [
        {"action": "fix_source_data", "description": "Too much of this series would be invented by repair; fix the export at the source. Counts are in details."},
    ],
    "INVALID_REPAIR_LEVEL": [
        {"action": "set_repair_level", "description": "Use one of: off, safe, aggressive."},
    ],
    "INVALID_ENCODING": [
        {"action": "fix_encoding", "description": "Re-export the file as UTF-8, or use the default repair level, which assumes Windows-1252 with disclosure."},
    ],
    "UNSUPPORTED_INPUT": [
        {"action": "convert_input", "description": "Convert to a supported format: .csv, .tsv, .json, .jsonl (each optionally .gz), .parquet, or .xlsx (excel extra)."},
    ],
    "MISSING_OPTIONAL_DEPENDENCY": [
        {"action": "install_extra", "description": "Install the package named in details.install."},
    ],
    "FREQUENCY_MISMATCH": [
        {"action": "set_frequency", "description": "Pass the inferred frequency from details, or omit frequency to infer."},
    ],
    "EMPTY_SNAPSHOT": [
        {"action": "adjust_as_of", "description": "Choose an as_of at or after the first known observation."},
    ],
    "DATASET_NOT_FOUND": [
        {"action": "list_datasets", "description": "Run `aion store list`; available dataset names are in details."},
        {"action": "ingest", "description": "Ingest the data first with `aion ingest`."},
    ],
    "MULTIPLE_SERIES_UNSUPPORTED": [
        {"action": "set_series_name", "description": "Pass series_name; the available series are in details."},
    ],
    "MISSING_COVARIATE_MAPPING": [
        {"action": "set_covariate_mapping", "description": "Pass covariate_mapping as name:type:future_known entries."},
    ],
    "TEMPORAL_LEAKAGE": [
        {"action": "remove_post_cutoff_data", "description": "Drop rows whose known_time lies after the task as_of; the offending known_time is in details."},
    ],
    "CLAIM_VERIFICATION_FAILED": [
        {"action": "review_violations", "description": "Each violation in details names the claim and the deterministic rule it broke."},
    ],
    "UNSUPPORTED_SCHEMA_VERSION": [
        {"action": "upgrade_runtime", "description": "This artifact was written by a newer Aion; upgrade to read it."},
    ],
    "SERIES_NOT_FOUND": [
        {"action": "list_series", "description": "Call aion_inspect to list series names."},
    ],
    "INVALID_COSTS": [
        {"action": "fix_costs", "description": "alert_cost must be >= 0 and miss_cost > 0."},
    ],
    "SNAPSHOT_TIMEZONE_MISMATCH": [
        {"action": "align_timezones", "description": "Use an as_of with the same timezone-awareness as the data."},
    ],
    "ARTIFACT_NOT_FOUND": [
        {"action": "check_path", "description": "Pass the artifact directory returned by the macro (it contains artifact.json)."},
    ],
}


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
                "repair_options": REPAIR_OPTIONS.get(self.code, []),
            },
        }
