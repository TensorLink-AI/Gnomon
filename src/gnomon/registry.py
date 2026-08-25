"""The operator and macro registry: one source of truth.

Every operator declares its input/output schema, minimum-data requirement,
determinism, leakage policy, the claim classes its output can support, and
its abstention criteria. Every macro declares its agent-facing schema. The
MCP server, the CLI, and (later) the plan validator all read this registry
rather than hand-maintained copies.

An operator without honest abstention criteria does not ship — which is why
the causal family is not in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import anomaly, operators
from .temporal_store import join_as_of


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    summary: str
    minimum_data: str
    abstention: str
    claim_classes: tuple[str, ...]
    deterministic: bool = True
    seeded: bool = False
    leakage_policy: str = "snapshot_only"
    runner: Any = None


OPERATORS: dict[str, OperatorSpec] = {
    spec.name: spec for spec in [
        OperatorSpec(
            "select_window", "Select a closed time window from a series.",
            "1 observation", "Never abstains; empty windows return empty output.",
            ("descriptive",), runner=operators.select_window,
        ),
        OperatorSpec(
            "resample_aggregate", "Aggregate a series onto a coarser grid.",
            "factor observations", "Never abstains; a truncated tail is reported.",
            ("descriptive",), runner=operators.resample_aggregate,
        ),
        OperatorSpec(
            "join_as_of", "Point-in-time join of a variable onto a grid using vintages.",
            "1 observation", "Grid points with no vintage known at the cutoff yield null.",
            ("descriptive",), runner=join_as_of,
        ),
        OperatorSpec(
            "difference", "Lagged differencing of a series.",
            "lag + 1 observations", "Raises on series shorter than the lag.",
            ("descriptive",), runner=operators.difference,
        ),
        OperatorSpec(
            "changepoint_detection", "Mean-shift changepoints by binary segmentation.",
            f"{operators.MIN_CHANGEPOINT_HISTORY} observations",
            "Inconclusive below minimum history; 'no change detected' is a supported result.",
            ("descriptive",), runner=operators.changepoint_detection,
        ),
        OperatorSpec(
            "regime_detection", "Classify changes as regime shift vs transient.",
            f"{operators.MIN_CHANGEPOINT_HISTORY} observations, {operators.MIN_PERSIST} post-change",
            "Inconclusive when the post-change window is too short to distinguish.",
            ("descriptive",), runner=operators.regime_detection,
        ),
        OperatorSpec(
            "anomaly_score", "Robust (median/MAD) anomaly scores, seasonally adjusted.",
            "8 observations", "Inconclusive below 8 observations.",
            ("descriptive",), runner=operators.anomaly_score,
        ),
        OperatorSpec(
            "detect_anomalies",
            "Graded anomaly detection: candidate detectors compete on a "
            "synthetic-injection grader (or supplied labels); the winner "
            "labels the series with every candidate's grade disclosed.",
            "16 observations",
            "Inconclusive below 16 observations; conditionally supported when "
            "the best grader F1 falls below the confidence floor.",
            ("descriptive",), seeded=True, runner=anomaly.detect_anomalies,
        ),
        OperatorSpec(
            "seasonality_analysis", "Detected period, strength, and per-phase profile.",
            "2 cycles", "Inconclusive below two cycles of the candidate period.",
            ("descriptive",), runner=operators.seasonality_analysis,
        ),
        OperatorSpec(
            "cross_correlation", "Lead/lag correlation on first differences.",
            f"{operators.MIN_CROSS_CORRELATION} paired observations",
            "Inconclusive below minimum pairs; below-significance results are unsupported.",
            ("associational",), runner=operators.cross_correlation,
        ),
        OperatorSpec(
            "event_study", "Pre/post window contrast around events.",
            "window observations each side",
            "Inconclusive when either window falls outside the observed range.",
            ("associational",), runner=operators.event_study,
        ),
        OperatorSpec(
            "forecast", "The evaluated forecast (baselines, candidates, abstention).",
            "horizon + 2 observations (degraded); more for separated folds",
            "Unsupported/inconclusive when the evaluation protocol cannot complete.",
            ("predictive",), runner=None,  # bound to runtime.forecast to avoid an import cycle
        ),
        OperatorSpec(
            "simulate_scenario", "Deterministic shift/scale scenarios over a forecast.",
            "1 forecast row",
            "Always conditionally supported: scenario outputs hold only under stated adjustments.",
            ("predictive", "descriptive"), runner=operators.simulate_scenario,
        ),
        OperatorSpec(
            "evaluate_threshold_risk", "Exceedance probabilities from backtest residuals.",
            "1 forecast row + calibration residuals",
            "Inconclusive without calibration residuals.",
            ("predictive",), runner=operators.evaluate_threshold_risk,
        ),
        OperatorSpec(
            "evaluate_actions", "Feasibility, constraints, and expected utility over actions.",
            "1 action, scenario probabilities summing to 1",
            "Conditionally supported without utilities (feasible-action comparison only); "
            "unsupported when no action is feasible; invalid when probabilities are malformed.",
            ("decision",), runner=operators.evaluate_actions,
        ),
    ]
}


@dataclass(frozen=True)
class MacroSpec:
    name: str
    tool_name: str
    summary: str
    operators_used: tuple[str, ...]
    input_schema: dict[str, Any] = field(default_factory=dict)


_COMMON_INPUT: dict[str, Any] = {
    "input": {"type": "string", "description": (
        "Path to a local CSV/Parquet file, or store:<dataset> for a dataset "
        "ingested into the bitemporal store. Callers without a filesystem "
        "pass `observations` inline instead."
    )},
    "observations": {"type": "array", "items": {"type": "object"}, "description": (
        "The observations supplied inline: row objects keyed by your "
        "column names. Mutually exclusive with `input`."
    )},
    "time_column": {"type": "string", "description": (
        "Timestamp column. Omit to infer when exactly one column "
        "qualifies (disclosed as an assumption); ambiguity fails loudly. "
        "Required for store:<dataset> inputs."
    )},
    "target_column": {"type": "string", "description": (
        "Numeric column to operate on. Omit to infer when exactly one "
        "non-time column qualifies (disclosed as an assumption); "
        "ambiguity fails loudly. Required for store:<dataset> inputs."
    )},
    "series_column": {"type": "string", "description": "Optional column identifying independent series."},
    "frequency": {
        "type": "string", "pattern": "^([1-9][0-9]*)?(s|min|h)$|^(D|W|MS)$",
        "description": (
            "Observation frequency: s/min/h/D/W/MS, or any whole-second "
            "sub-daily step as <N>s, <N>min, or <N>h (e.g. 7min, 90s, 2h). "
            "Omit to infer."
        ),
    },
    "regrid": {
        "type": "string", "enum": ["business_daily", "month_start"],
        "description": (
            "Declared calendar regrid, applied before validation: "
            "business_daily forward-fills weekends/holidays onto the "
            "continuous daily grid (implies frequency=D); month_start "
            "restamps monthly stamps to the first of each month (implies "
            "frequency=MS). Disclosed as warnings; not charged against "
            "the repair ceiling."
        ),
    },
    "as_of": {"type": "string", "description": (
        "Optional ISO instant: use only data known at or before this moment "
        "(historical replay)."
    )},
    "output_dir": {"type": "string", "description": "Directory for the immutable artifact (default ./gnomon-output)."},
}


MACROS: dict[str, MacroSpec] = {
    spec.name: spec for spec in [
        MacroSpec(
            "investigate_change", "gnomon_investigate_change",
            (
                "What changed? Detect changepoints, locate onsets, classify "
                "regime shift vs transient anomaly, retrieve concurrent events, "
                "and rank associational explanations with residual uncertainty. "
                "Never returns a cause."
            ),
            ("changepoint_detection", "regime_detection", "anomaly_score",
             "event_study", "cross_correlation"),
            {
                "type": "object",
                "properties": {
                    **_COMMON_INPUT,
                    "context_events_file": {"type": "string", "description": (
                        "Optional validated context-events JSON (output of "
                        "`gnomon context validate`) to rank as concurrent events."
                    )},
                    "context_events": {"type": "array", "description": (
                        "The same events supplied inline, for callers "
                        "without a filesystem; see gnomon_forecast's "
                        "context_events for the item shape."
                    ), "items": {"type": "object"}},
                    "suspected_cause": {"type": "string", "description": (
                        "Optional free text: what you suspect changed and "
                        "why (e.g. 'pricing change rolled out May 12'). "
                        "Recorded verbatim in the payload and artifact so "
                        "the hypothesis travels with the finding; it does "
                        "not influence detection or explanation ranking. "
                        "A suspicion with a date belongs in context_events "
                        "too, where it is ranked as a concurrent event."
                    )},
                },
                "required": [],
            },
        ),
        MacroSpec(
            "detect_anomalies", "gnomon_detect_anomalies",
            (
                "What is abnormal? Candidate detectors (robust z-score, "
                "rolling-median residual, forecast-interval exceedance) "
                "compete on a synthetic anomaly-injection grader — or on "
                "supplied labels — and the winner flags anomalies with every "
                "candidate's precision/recall/F1 disclosed. Deterministic: "
                "injection placement is seeded from the series content."
            ),
            ("detect_anomalies",),
            {
                "type": "object",
                "properties": {
                    **_COMMON_INPUT,
                    "threshold": {"type": "number", "description": (
                        "Detection threshold on standardised scores "
                        "(default 3.5)."
                    )},
                    "labels": {"type": "array", "items": {"type": "string"}, "description": (
                        "Optional ISO timestamps of known anomalies; when "
                        "given, detector selection uses label F1 instead of "
                        "the synthetic grader."
                    )},
                },
                "required": [],
            },
        ),
        MacroSpec(
            "forecast", "gnomon_forecast",
            "What happens next? The evaluated forecast with abstention.",
            ("forecast", "evaluate_threshold_risk"),
            {},  # frozen v0.2 schema lives in toolspec.py; preserved verbatim
        ),
        MacroSpec(
            "decide", "gnomon_decide",
            (
                "Compares actions; does not take one. Nothing is actuated. "
                "What should we do? Exceedance scenarios from an evaluated "
                "forecast, feasibility and constraint checks over candidate "
                "actions, and an expected-utility choice — or, without "
                "utilities, the feasible-action comparison with probabilities "
                "(conditionally_supported: missing utility inputs). "
                "A choice is offered only when aligned residual trajectories "
                "support a dependence-aware probability of any breach; peak "
                "marginals and independence composition are references only."
            ),
            ("forecast", "evaluate_threshold_risk", "evaluate_actions"),
            {
                "type": "object",
                "properties": {
                    **_COMMON_INPUT,
                    "horizon": {"type": "integer", "description": "Future periods to forecast."},
                    "threshold": {"type": "number", "description": "Decision threshold on the target."},
                    "actions": {"type": "array", "items": {"type": "object"}, "description": (
                        "Candidate actions: [{name, feasible?, residual_risk?}]."
                    )},
                    "utilities": {"type": "object", "description": (
                        "Optional payoff per action per scenario: "
                        "{action: {exceed: x, no_exceed: y}}. Omit for the degraded comparison."
                    )},
                    "max_acceptable_risk": {"type": "number", "description": (
                        "Optional constraint on each action's residual_risk."
                    )},
                    "series_name": {"type": "string", "description": "Series to decide for (required when multiple)."},
                    "project": {"type": "string", "description": (
                        "Optional tracking project: records a DecisionArtifact "
                        "for realised-outcome scoring (regret, calibration)."
                    )},
                    "questions": {"type": "array", "items": {"type": "object"},
                                  "description": "Optional typed temporal questions answered from the same immutable forecast."},
                },
                "required": ["horizon", "threshold", "actions"],
            },
        ),
        MacroSpec(
            "monitor", "gnomon_monitor",
            (
                "Defines an alert policy; does not run one. One evaluation "
                "against one forecast — nothing watches the series afterwards "
                "and nothing is actuated; re-run as observations arrive. "
                "When should we intervene? Trigger definition, sequential "
                "exceedance risk per horizon step, and an alert-cost-aware "
                "alert rule (cost-optimal with alert/miss costs; otherwise a "
                "flagged default)."
            ),
            ("forecast", "evaluate_threshold_risk"),
            {
                "type": "object",
                "properties": {
                    **_COMMON_INPUT,
                    "horizon": {"type": "integer", "description": "Future periods to monitor."},
                    "threshold": {"type": "number", "description": "Trigger threshold on the target."},
                    "alert_cost": {"type": "number", "description": "Cost of an unnecessary alert."},
                    "miss_cost": {"type": "number", "description": "Cost of a missed exceedance."},
                    "action_cost": {"type": "number", "description": (
                        "Cost paid whenever a single-shot mitigating action is "
                        "taken. Mutually exclusive with alert_cost; requires miss_cost."
                    )},
                    "mitigation_effectiveness": {"type": "number", "minimum": 0,
                        "maximum": 1, "description": (
                            "Fraction of miss loss removed by the action; default 1."
                        )},
                    "project": {"type": "string", "description": "Optional tracking project to register the underlying forecast in."},
                    "questions": {"type": "array", "items": {"type": "object"},
                                  "description": "Optional typed temporal questions answered from the same immutable forecast."},
                },
                "required": ["horizon", "threshold"],
            },
        ),
    ]
}


def registry_capabilities() -> dict[str, Any]:
    """Machine-readable registry view, merged into `gnomon capabilities`."""
    return {
        "operators": {
            name: {
                "summary": spec.summary,
                "minimum_data": spec.minimum_data,
                "abstention": spec.abstention,
                "claim_classes": list(spec.claim_classes),
                "deterministic": spec.deterministic,
                "leakage_policy": spec.leakage_policy,
            }
            for name, spec in sorted(OPERATORS.items())
        },
        "macros": {
            spec.tool_name: {
                "summary": spec.summary,
                "operators_used": list(spec.operators_used),
            }
            for spec in MACROS.values()
        },
    }
