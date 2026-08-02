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


#: The band measured interval coverage must fall inside for a
#: probability-bearing claim to carry probability weight, at a nominal 80%
#: interval.
#:
#: The verifier's calibration gate used to be satisfied by the mere
#: *existence* of a `rolling_evaluation` record, so a run whose intervals
#: covered 57.1% of a nominal 80% still emitted verified `predictive`
#: claims. Coverage that far below nominal means the interval is not what
#: it says it is, and a claim resting on it is not calibrated in any useful
#: sense.
#:
#: The band is wide because the measurement is small — one test fold of
#: `horizon` points — and a narrow band would reject honest runs for sample
#: noise. It catches calibration that is plainly broken, which is the job.
MIN_VERIFIABLE_COVERAGE = 0.5
MAX_VERIFIABLE_COVERAGE = 1.0


def interval_calibration_is_verifiable(coverage: float | None) -> bool:
    """Whether measured coverage can carry a probability-bearing claim.

    Unmeasured coverage (``None``) is not disqualifying: the run already
    warns that no test fold remained, and treating "unknown" as "bad" would
    refuse every two-fold evaluation. Measured-and-outside-the-band is.
    """
    if coverage is None:
        return True
    return MIN_VERIFIABLE_COVERAGE <= float(coverage) <= MAX_VERIFIABLE_COVERAGE


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

    # --- Schema and frequency -------------------------------------------
    "AMBIGUOUS_FREQUENCY": [
        {"action": "set_frequency", "description": "Pass frequency explicitly; the supported codes are in details.supported."},
        {"action": "inspect_dataset", "description": "Call aion_inspect to see the observed step between timestamps."},
        {"action": "restamp_to_month_start", "description": "Month-end dates (Jan 31, Feb 28, …) are not a supported grid. Restamp each observation to the first of its month and pass frequency=MS."},
    ],
    "UNSUPPORTED_FREQUENCY": [
        {"action": "set_frequency", "description": "Use one of the supported codes listed in details.supported."},
    ],
    "MIXED_SERIES_FREQUENCIES": [
        {"action": "split_input", "description": "Forecast each frequency separately; details.per_series names each series' inferred step."},
        {"action": "set_frequency", "description": "Pass frequency to state which grid every series is on, if the inference is wrong."},
    ],
    "AMBIGUOUS_SCHEMA": [
        {"action": "supply_arguments", "description": "Name the column explicitly; the qualifying candidates are in details.candidates."},
        {"action": "inspect_dataset", "description": "Call aion_inspect to see every column and its inferred type."},
    ],
    "EMPTY_DATASET": [
        {"action": "check_input", "description": "The file parsed but contained no observations. Confirm it has a header row and at least one data row."},
        {"action": "check_delimiter", "description": "A mis-detected delimiter produces one column and no usable rows; re-export as comma-separated UTF-8."},
    ],

    # --- Inputs that do not exist ---------------------------------------
    "INPUT_NOT_FOUND": [
        {"action": "check_path", "description": "The path does not exist. Pass a path relative to the working directory, or an absolute one."},
        {"action": "list_datasets", "description": "For stored data use store:<dataset>; run `aion store list` to see the names."},
    ],
    "COVARIATES_NOT_FOUND": [
        {"action": "check_path", "description": "The covariates path does not exist; details names the resolved path."},
    ],
    "CONTEXT_FILE_NOT_FOUND": [
        {"action": "check_path", "description": "Produce the file with `aion context validate` first; it writes the typed events this flag expects."},
    ],
    "DOCUMENT_NOT_FOUND": [
        {"action": "check_path", "description": "Every --file passed to `aion context prompt` must exist and be readable."},
    ],
    "RESPONSE_NOT_FOUND": [
        {"action": "check_path", "description": "Pass the file containing the model's reply to the context prompt."},
    ],
    "ARGUMENT_FILE_NOT_FOUND": [
        {"action": "check_path", "description": "An @-prefixed argument reads a file; details.path is the path that was tried."},
    ],

    # --- Malformed arguments ---------------------------------------------
    "INVALID_ARGUMENTS": [
        {"action": "supply_arguments", "description": "Supply the arguments named in details.missing_arguments."},
        {"action": "show_usage", "description": "Run the command with --help for the full argument list."},
    ],
    "INVALID_JSON_ARGUMENT": [
        {"action": "fix_json", "description": "details.argument names which argument failed and details.parse_error says where."},
        {"action": "use_file", "description": "Pass @path/to/file.json instead of inline JSON to avoid shell quoting problems."},
    ],
    "INVALID_ACTIONS": [
        {"action": "fix_actions", "description": "actions is a list of objects, each with a 'name' and optionally 'feasible' and 'residual_risk'. details.example shows the shape."},
    ],
    "INVALID_MINIMUM_IMPROVEMENT": [
        {"action": "set_minimum_improvement", "description": "Use a value >= 0. Zero means the candidate must merely not be worse than the strongest baseline; the default 0.02 asks for 2% better."},
    ],
    "INVALID_PLAN": [
        {"action": "review_violations", "description": "Each entry in details.violations names the step and the rule it broke."},
        {"action": "repair_plan", "description": "Call the planner's repair pass, which fixes bounded structural faults automatically."},
    ],
    "UNKNOWN_TASK_TYPE": [
        {"action": "list_task_types", "description": "Use one of the five verbs: forecast, investigate_change, decide, monitor, detect_anomalies."},
    ],
    "UNKNOWN_TOOL": [
        {"action": "list_tools", "description": "Call tools/list for the available tool names; the planner tools appear only with AION_EXPERIMENTAL_PLANNER=1."},
    ],
    "OPERATOR_UNAVAILABLE": [
        {"action": "use_macro", "description": "This operator has no direct runner. Use the macro that owns it, or call it through a plan step."},
    ],

    # --- Covariates -------------------------------------------------------
    "INVALID_COVARIATE_MAPPING": [
        {"action": "set_covariate_mapping", "description": "Each entry is name:type:availability, comma-separated — for example `campaign:binary:future_known`."},
        {"action": "covariate_guide", "description": "Run `aion covariates guide` for the required format and the cutoffs your file must cover."},
    ],
    "INVALID_COVARIATE_TYPE": [
        {"action": "set_covariate_mapping", "description": "Supported types are continuous and binary."},
    ],
    "UNSUPPORTED_COVARIATE_AVAILABILITY": [
        {"action": "set_covariate_mapping", "description": "Availability must be future_known or past_observed. It is mandatory so a contemporaneous reading is never mistaken for one knowable at a past origin."},
    ],
    "UNSUPPORTED_COVARIATES": [
        {"action": "convert_input", "description": "Covariates currently require CSV input."},
    ],
    "MISSING_COVARIATE_COLUMNS": [
        {"action": "fix_columns", "description": "details.required lists what the mapping asked for and details.available what the file has."},
    ],
    "EMPTY_COVARIATES": [
        {"action": "check_input", "description": "The covariates file parsed but held no rows."},
    ],
    "INVALID_COVARIATE_TIMESTAMP": [
        {"action": "fix_timestamps", "description": "Both the timestamp and known_at columns must be ISO-8601; details names the offending row."},
    ],
    "INVALID_COVARIATE_VALUE": [
        {"action": "fix_target", "description": "Covariate values must be numeric; details names the offending row."},
    ],
    "INVALID_BINARY_COVARIATE": [
        {"action": "fix_target", "description": "A binary covariate takes 0 or 1 only; details names the offending row."},
    ],
    "MIXED_COVARIATE_TIMEZONES": [
        {"action": "align_timezones", "description": "Make timestamp and known_at consistently aware or consistently naive across the whole file."},
    ],
    "DUPLICATE_COVARIATE_VINTAGE": [
        {"action": "deduplicate", "description": "One value per (series, timestamp, known_at). Two rows sharing all three are ambiguous, not a revision — a revision has a later known_at."},
    ],
    "MISSING_HISTORICAL_VINTAGES": [
        {"action": "supply_earlier_known_at", "description": "The covariate's values were published too late to have been knowable at the backtest cutoffs. Each fold needs a row whose known_at precedes that fold's cutoff; details lists the cutoffs that came up empty."},
        {"action": "covariate_guide", "description": "Run `aion covariates guide` to see the exact cutoffs this dataset and horizon require."},
        {"action": "drop_covariate", "description": "Forecast without it. A covariate that was not knowable in the past cannot be backtested, and Aion will not pretend otherwise."},
    ],
    "MISSING_FORECAST_VALUES": [
        {"action": "supply_future_values", "description": "A future_known covariate needs a value for every horizon period, published by the forecast origin."},
    ],

    # --- Context events ---------------------------------------------------
    "INVALID_CONTEXT_FILE": [
        {"action": "regenerate", "description": "Produce the file with `aion context validate`, which writes the schema this flag reads."},
    ],
    "INVALID_CONTEXT_EVENT": [
        {"action": "fix_event", "description": "details names the field that failed. Timestamps need an explicit offset, or must match the dataset's naivety."},
    ],
    "INVALID_RESPONSE": [
        {"action": "regenerate", "description": "The model's reply did not parse as the requested JSON. Re-run `aion context prompt` and pass the reply verbatim."},
    ],

    # --- Internal ---------------------------------------------------------
    "TRACKING_ERROR": [
        {"action": "review_message", "description": "The registry rejected the operation; the message states why."},
    ],
    "INTERNAL_ERROR": [
        {"action": "report_bug", "description": "This is a defect in Aion, not in the input. details names the surface and the exception type; please report it with the command that produced it."},
    ],
    "QUANTILE_CROSSING": [
        {"action": "report_bug", "description": "Projected quantiles crossed, which the projection is meant to make impossible. This is a defect in Aion; details carries the offending row."},
    ],
    "RESIDUAL_PROVENANCE_MISMATCH": [
        {"action": "report_bug", "description": "A stage replaced the point forecast without replacing its residuals, so the interval would have described a different model. This is a defect in Aion; details names both models."},
    ],
}


class AionError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None,
                 repair_options: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        #: Instance-specific repairs, when the useful advice depends on the
        #: particular failure rather than only on the code. Falls back to
        #: the code's entry in REPAIR_OPTIONS.
        self.repair_options = repair_options

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "status": "error",
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": False,
                "details": self.details,
                "repair_options": (
                    self.repair_options
                    if self.repair_options is not None
                    else REPAIR_OPTIONS.get(self.code, [])
                ),
            },
        }
