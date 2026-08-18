"""The decision model: DecisionArtifact and realised-outcome scoring.

Replaces the v0.2 ``DecisionRecord`` (action / expected_outcome /
actual_outcome / bare ``correct``) with a structure that can say what the
old one could not: which actions were on the table, what constraints they
faced, what was known and assumed, what rule selected one — and, on
resolution, realised utility, regret against the best feasible action in
hindsight, and whether the choice was ex-ante optimal. A costly precaution
can be rational when the adverse event never occurs; this schema can say
so, which is exactly why bare ``correct`` is retired.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "0.1"


@dataclass
class ActionOption:
    name: str
    feasible: bool = True
    constraint_results: dict[str, Any] = field(default_factory=dict)
    expected_utility: float | None = None
    downside_risk: float | None = None


@dataclass
class DecisionOutcome:
    realised_scenario: str | None = None
    realised_utilities: dict[str, float] | None = None
    realised_utility: float | None = None
    best_feasible_realised_utility: float | None = None
    best_feasible_action_in_hindsight: str | None = None
    regret: float | None = None
    constraint_violations: list[str] = field(default_factory=list)
    ex_ante_optimal: bool | None = None
    risk_calibration: dict[str, Any] | None = None
    note: str | None = None


@dataclass
class DecisionArtifact:
    decision_id: str
    project: str
    forecast_id: str
    options: list[ActionOption] = field(default_factory=list)
    selected_action: str | None = None
    decision_rule: str | None = None
    scenario_probabilities: dict[str, float] | None = None
    utilities: dict[str, dict[str, float]] | None = None
    assumptions: list[str] = field(default_factory=list)
    sensitivity: dict[str, Any] = field(default_factory=dict)
    approval: dict[str, Any] | None = None
    created_at: str = ""
    resolved_at: str | None = None
    outcome: DecisionOutcome | None = None
    degraded: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecisionArtifact":
        options = [ActionOption(**option) for option in payload.get("options", [])]
        outcome = payload.get("outcome")
        return cls(**{
            **{key: value for key, value in payload.items()
               if key not in ("options", "outcome")},
            "options": options,
            "outcome": DecisionOutcome(**outcome) if outcome else None,
        })

    @classmethod
    def from_legacy(cls, record: Any) -> "DecisionArtifact":
        """Degrade a v0.2 DecisionRecord into this schema. Nothing is
        invented: unknown utilities stay unknown, and the old bare
        ``correct`` survives only as a note."""
        outcome = None
        if record.resolved_at is not None:
            outcome = DecisionOutcome(
                note=(
                    f"Migrated v0.2 resolution: actual_outcome="
                    f"{record.actual_outcome!r}, correct={record.correct!r}. "
                    "Bare correctness is not utility; no regret is computable."
                ),
            )
        return cls(
            decision_id=record.decision_id,
            project=record.project,
            forecast_id=record.forecast_id,
            options=[ActionOption(name=record.action)],
            selected_action=record.action,
            decision_rule=None,
            assumptions=[
                "Migrated from a v0.2 DecisionRecord; the option set, "
                "constraints, and utilities at decision time were not recorded.",
                f"Expected outcome as stated: {record.expected_outcome!r}.",
            ],
            created_at=record.created_at,
            resolved_at=record.resolved_at,
            outcome=outcome,
            degraded=True,
        )


def score_outcome(
    artifact: DecisionArtifact,
    *,
    realised_scenario: str | None = None,
    realised_utilities: dict[str, float] | None = None,
    constraint_violations: list[str] | None = None,
    note: str | None = None,
) -> DecisionOutcome:
    """Score a resolved decision. Realised utilities come either directly
    from the caller or from the declared utilities evaluated at the
    realised scenario; regret is measured against the best *feasible*
    action in hindsight, never against infeasible ones."""
    utilities = realised_utilities
    if utilities is None and artifact.utilities and realised_scenario is not None:
        utilities = {
            action: payoffs[realised_scenario]
            for action, payoffs in artifact.utilities.items()
            if realised_scenario in payoffs
        }
    outcome = DecisionOutcome(
        realised_scenario=realised_scenario,
        realised_utilities=utilities,
        constraint_violations=list(constraint_violations or []),
        note=note,
    )
    feasible = {option.name for option in artifact.options if option.feasible}
    if utilities:
        feasible_utilities = {name: value for name, value in utilities.items() if name in feasible}
        if feasible_utilities:
            best_action = max(sorted(feasible_utilities), key=lambda name: feasible_utilities[name])
            outcome.best_feasible_action_in_hindsight = best_action
            outcome.best_feasible_realised_utility = feasible_utilities[best_action]
        if artifact.selected_action is not None and artifact.selected_action in utilities:
            outcome.realised_utility = utilities[artifact.selected_action]
            if outcome.best_feasible_realised_utility is not None:
                outcome.regret = round(
                    outcome.best_feasible_realised_utility - outcome.realised_utility, 10,
                )
    if artifact.selected_action is not None:
        expected = {
            option.name: option.expected_utility
            for option in artifact.options
            if option.feasible and option.expected_utility is not None
        }
        if expected:
            best_expected = max(expected.values())
            selected_expected = expected.get(artifact.selected_action)
            outcome.ex_ante_optimal = (
                selected_expected is not None and selected_expected >= best_expected
            )
    if artifact.scenario_probabilities and realised_scenario is not None:
        outcome.risk_calibration = {
            "predicted_probabilities": artifact.scenario_probabilities,
            "realised_scenario": realised_scenario,
            "predicted_probability_of_realised": artifact.scenario_probabilities.get(realised_scenario),
        }
    return outcome


def robust_scenario_decision(
    *, decision_id: str, project: str, forecast_id: str,
    actions: list[dict[str, Any]],
    utilities: dict[str, dict[str, float]],
    scenario_ids: list[str], created_at: str,
) -> DecisionArtifact:
    """Choose the feasible action with best worst-case stated utility.

    Context scenarios do not receive invented probabilities. Maximin is
    therefore the honest default: rank by worst utility across the immutable
    primary and every credible conditioned scenario, then by mean utility and
    name for deterministic ties.
    """
    required = tuple(dict.fromkeys(["primary", *scenario_ids]))
    options: list[ActionOption] = []
    evaluations: dict[str, dict[str, float]] = {}
    for raw in actions:
        name = str(raw["name"])
        feasible = bool(raw.get("feasible", True))
        payoffs = utilities.get(name) or {}
        missing = [scenario for scenario in required if scenario not in payoffs]
        if missing and feasible:
            raise ValueError(
                f"action {name!r} lacks utilities for scenarios: {', '.join(missing)}"
            )
        values = [float(payoffs[scenario]) for scenario in required] if not missing else []
        worst = min(values) if values else None
        average = sum(values) / len(values) if values else None
        evaluations[name] = {"worst": worst, "mean": average} if values else {}
        options.append(ActionOption(
            name=name, feasible=feasible,
            constraint_results=dict(raw.get("constraint_results") or {}),
            expected_utility=None, downside_risk=worst,
        ))
    feasible_names = [option.name for option in options if option.feasible]
    selected = None
    if feasible_names:
        selected = max(
            sorted(feasible_names),
            key=lambda name: (evaluations[name]["worst"],
                              evaluations[name]["mean"]),
        )
    return DecisionArtifact(
        decision_id=decision_id, project=project, forecast_id=forecast_id,
        options=options, selected_action=selected,
        decision_rule="maximin across immutable primary and credible context scenarios",
        scenario_probabilities=None, utilities=utilities,
        assumptions=[
            "No probabilities were assigned to context scenarios.",
            "Utilities are caller supplied; Gnomon does not infer preferences.",
            "The primary scenario and every named conditioned scenario receive equal worst-case consideration.",
        ],
        sensitivity={"scenario_ids": list(required),
                     "action_evaluations": evaluations},
        created_at=created_at, degraded=False,
    )
