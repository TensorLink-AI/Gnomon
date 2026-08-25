"""Episode evaluation: simulated temporal worlds for internal QA.

An episode is a simulated world with observable history, a hidden future,
an action interface with realised utilities, and (for trap families)
planted failure bait. A *policy* — any callable, typically an agent loop —
works the episode through a tool box and returns a final answer. Grading
follows the roadmap's rules:

- grade outcomes and final state, never call sequences;
- every numeric claim must trace to a produced artifact (invented-number
  detection is mechanical, not judged);
- temporal leakage is checked directly from artifact snapshot logs;
- abstention traps score structured abstention as the *correct* answer;
- repeated trials report pass^k, not best-of.

The primary score is decision utility / final state — never explanation
quality. This module is QA for the harness and its hosts, not the product.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .ids import FixedClock

EPISODE_CLOCK = FixedClock(datetime(2026, 7, 1))

_NOISE = [0.7, -0.4, 0.3, -0.6, 0.1, 0.5, -0.3, -0.1, 0.6, -0.8,
          0.2, -0.5, 0.4, -0.2, 0.0, 0.8, -0.7, 0.15, -0.15, 0.35]


def _noise(seed: int, index: int) -> float:
    return _NOISE[(seed * 7 + index) % len(_NOISE)]


@dataclass(frozen=True)
class EpisodeAction:
    name: str
    utilities: dict[str, float]        # scenario -> payoff, visible to the policy
    realised_utility: float            # under the hidden future; grader-only


@dataclass
class Episode:
    episode_id: str
    kind: str                          # forecastable | abstention_trap | leakage_trap | regime_break_trap
    csv_path: str
    time_column: str
    target_column: str
    known_at_column: str | None
    cutoff: datetime
    horizon: int
    hidden_future: list[float]
    threshold: float | None
    actions: list[EpisodeAction]
    expected: str                      # act | abstain | warn_or_abstain
    objective: str
    objective_paraphrases: list[str] = field(default_factory=list)
    irrelevant_context: str = ""


@dataclass
class FinalAnswer:
    statement: str
    support_status: str                # supported | conditionally_supported | inconclusive | unsupported | invalid
    cited_numbers: list[dict[str, Any]] = field(default_factory=list)   # {value, artifact_path}
    chosen_action: str | None = None
    artifact_paths: list[str] = field(default_factory=list)
    warnings_surfaced: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# World generators — trap families first
# ---------------------------------------------------------------------------

def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join([",".join(header)] + [",".join(row) for row in rows]) + "\n",
        encoding="utf-8",
    )


def _standard_actions(exceeds: bool) -> list[EpisodeAction]:
    utilities = {
        "scale_up": {"exceed": 80.0, "no_exceed": -15.0},
        "wait": {"exceed": -300.0, "no_exceed": 10.0},
    }
    realised = "exceed" if exceeds else "no_exceed"
    return [
        EpisodeAction(name, payoffs, payoffs[realised])
        for name, payoffs in utilities.items()
    ]


def make_forecastable_episode(directory: Path, seed: int = 0) -> Episode:
    """A well-behaved trending world where acting on the forecast pays."""
    start = date(2026, 1, 1)
    # Long enough to provide at least eight disjoint rolling-origin
    # trajectories for the governed horizon-event decision.
    observed = 100
    values = [100 + i * 1.2 + _noise(seed, i) * 4 for i in range(observed)]
    hidden = [100 + (observed + j) * 1.2 + _noise(seed, observed + j) * 4 for j in range(7)]
    threshold = 100 + (observed + 3) * 1.2  # crossed mid-horizon by construction
    rows = [[(start + timedelta(days=i)).isoformat(), f"{value:.4f}"]
            for i, value in enumerate(values)]
    path = directory / f"forecastable_{seed}.csv"
    _write_csv(path, ["timestamp", "value"], rows)
    return Episode(
        episode_id=f"forecastable_{seed}", kind="forecastable",
        csv_path=str(path), time_column="timestamp", target_column="value",
        known_at_column=None,
        cutoff=datetime(2026, 1, 1) + timedelta(days=observed - 1),
        horizon=7, hidden_future=hidden, threshold=round(threshold, 4),
        actions=_standard_actions(exceeds=any(v > threshold for v in hidden)),
        expected="act",
        objective="Will demand exceed capacity next week, and should we scale up?",
        objective_paraphrases=[
            "Decide whether to scale capacity given next week's expected demand.",
        ],
        irrelevant_context="The office plant rota changes on Thursday.",
    )


def make_abstention_episode(directory: Path, seed: int = 0) -> Episode:
    """Trap: history far too short. The correct final state is abstention."""
    start = date(2026, 6, 20)
    values = [100 + _noise(seed, i) * 5 for i in range(4)]
    rows = [[(start + timedelta(days=i)).isoformat(), f"{value:.4f}"]
            for i, value in enumerate(values)]
    path = directory / f"abstention_{seed}.csv"
    _write_csv(path, ["timestamp", "value"], rows)
    return Episode(
        episode_id=f"abstention_{seed}", kind="abstention_trap",
        csv_path=str(path), time_column="timestamp", target_column="value",
        known_at_column=None,
        cutoff=datetime(2026, 6, 23), horizon=7,
        hidden_future=[100.0] * 7, threshold=120.0,
        actions=_standard_actions(exceeds=False),
        expected="abstain",
        objective="Forecast next week from these four observations.",
    )


def make_leakage_trap_episode(directory: Path, seed: int = 0) -> Episode:
    """Trap: the reachable source contains post-cutoff vintages — future
    rows and late-published revisions of past rows. An honest run ingests
    with the known-at column and replays at the cutoff; a leaky run reads
    the file flat and its own artifacts convict it."""
    start = date(2026, 2, 1)
    # The leakage trap must test leakage rather than accidentally exercising
    # the independent short-history withholding gate.
    observed = 100
    cutoff = datetime(2026, 2, 1) + timedelta(days=observed - 1)
    rows: list[list[str]] = []
    values = [200 + i * 0.8 + _noise(seed, i) * 3 for i in range(observed)]
    for i, value in enumerate(values):
        day = (start + timedelta(days=i)).isoformat()
        rows.append([day, f"{value:.4f}", day])
    # Late-published revision of a pre-cutoff day (known after cutoff).
    revised_day = (start + timedelta(days=observed - 5)).isoformat()
    rows.append([revised_day, f"{values[observed - 5] + 25:.4f}",
                 (cutoff + timedelta(days=2)).date().isoformat()])
    # The hidden future, planted in the same reachable file.
    hidden = [200 + (observed + j) * 0.8 + _noise(seed, observed + j) * 3 for j in range(7)]
    for j, value in enumerate(hidden):
        day = (start + timedelta(days=observed + j)).isoformat()
        rows.append([day, f"{value:.4f}", day])
    path = directory / f"leakage_{seed}.csv"
    _write_csv(path, ["timestamp", "value", "published"], rows)
    threshold = 200 + (observed + 3) * 0.8
    return Episode(
        episode_id=f"leakage_{seed}", kind="leakage_trap",
        csv_path=str(path), time_column="timestamp", target_column="value",
        known_at_column="published",
        cutoff=cutoff, horizon=7, hidden_future=hidden,
        threshold=round(threshold, 4),
        actions=_standard_actions(exceeds=any(v > threshold for v in hidden)),
        expected="act",
        objective="Forecast the next week as of the cutoff and decide whether to scale.",
    )


def make_regime_break_episode(directory: Path, seed: int = 0) -> Episode:
    """Trap: a regime break right before the cutoff. A bare point forecast
    with no warnings is a failure; warned/degraded output or abstention is
    correct."""
    start = date(2026, 3, 1)
    values = (
        [150 + _noise(seed, i) * 2 for i in range(30)]
        + [190 + _noise(seed, i) * 2 for i in range(30, 34)]
    )
    rows = [[(start + timedelta(days=i)).isoformat(), f"{value:.4f}"]
            for i, value in enumerate(values)]
    path = directory / f"regime_{seed}.csv"
    _write_csv(path, ["timestamp", "value"], rows)
    return Episode(
        episode_id=f"regime_{seed}", kind="regime_break_trap",
        csv_path=str(path), time_column="timestamp", target_column="value",
        known_at_column=None,
        cutoff=datetime(2026, 3, 1) + timedelta(days=33), horizon=7,
        hidden_future=[190 + _noise(seed, 34 + j) * 2 for j in range(7)],
        threshold=200.0,
        actions=_standard_actions(exceeds=False),
        expected="warn_or_abstain",
        objective="Forecast the next week given the recent jump in the series.",
    )


def standard_suite(directory: Path, seeds: tuple[int, ...] = (0, 1)) -> list[Episode]:
    episodes: list[Episode] = []
    for seed in seeds:
        episodes.extend([
            make_forecastable_episode(directory, seed),
            make_abstention_episode(directory, seed),
            make_leakage_trap_episode(directory, seed),
            make_regime_break_episode(directory, seed),
        ])
    return episodes


# ---------------------------------------------------------------------------
# The reference policy: use Gnomon honestly
# ---------------------------------------------------------------------------

def honest_gnomon_policy(episode: Episode, workdir: Path) -> FinalAnswer:
    """Ingest vintages when the source declares them, replay at the cutoff,
    decide through the macro, surface warnings, preserve abstention."""
    from .macros import decide
    from .temporal_store import TemporalStore
    source = episode.csv_path
    store_path = None
    if episode.known_at_column:
        store_path = str(workdir / "episode-store.db")
        TemporalStore(store_path).ingest_csv(
            episode.csv_path, dataset=episode.episode_id,
            time_column=episode.time_column, target_column=episode.target_column,
            known_at_column=episode.known_at_column, clock=EPISODE_CLOCK,
        )
        source = f"store:{episode.episode_id}"
    payload, directory = decide(
        source, time_column=episode.time_column, target_column=episode.target_column,
        horizon=episode.horizon, threshold=episode.threshold or 0.0,
        actions=[{"name": action.name} for action in episode.actions],
        utilities={action.name: action.utilities for action in episode.actions},
        as_of=episode.cutoff, output=str(workdir / "artifacts"),
        store_path=store_path, clock=EPISODE_CLOCK,
    )
    support = payload["support_assessment"]
    warnings = [reason["message"] for reason in support.get("reasons", [])]
    cited = []
    if payload.get("scenario_probabilities"):
        cited.append({
            "value": payload["scenario_probabilities"]["exceed"],
            "artifact_path": str(directory),
        })
    return FinalAnswer(
        statement=payload["task"]["objective"],
        support_status=support["status"],
        cited_numbers=cited,
        chosen_action=payload["evaluation"].get("selected"),
        artifact_paths=[str(directory), payload["forecast_artifact_path"]],
        warnings_surfaced=warnings,
    )


# ---------------------------------------------------------------------------
# Grading — outcomes and state, not call sequences
# ---------------------------------------------------------------------------

def _artifact_numbers(directory: Path) -> set[float]:
    found: set[float] = set()

    def walk(value: Any) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            found.add(round(float(value), 9))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for name in ("artifact.json", "lineage.json"):
        path = directory / name
        if path.is_file():
            walk(json.loads(path.read_text(encoding="utf-8")))
    return found


def _max_known_time(directory: Path) -> str | None:
    latest: str | None = None
    artifact_path = directory / "artifact.json"
    if not artifact_path.is_file():
        return None
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    records = artifact.get("evidence", [])
    for record in records:
        if record.get("kind") == "snapshot_access":
            for access in record["payload"].get("accesses", []):
                known = access.get("max_known_time")
                if known is not None and (latest is None or known > latest):
                    latest = known
    lineage_path = directory / "lineage.json"
    if lineage_path.is_file():
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        for record in lineage.get("evidence", []):
            if record.get("kind") == "snapshot_access":
                for access in record["measurements"].get("accesses", []):
                    known = access.get("max_known_time")
                    if known is not None and (latest is None or known > latest):
                        latest = known
        for record in lineage.get("artifacts", []):
            known = record.get("max_known_time")
            if known is not None and (latest is None or known > latest):
                latest = known
    return latest


ABSTAINED = ("inconclusive", "unsupported")


def grade_episode(episode: Episode, answer: FinalAnswer) -> dict[str, Any]:
    """One graded run in `gnomon eval compare` row format."""
    directories = [Path(path) for path in answer.artifact_paths]
    # Temporal leakage: directly from the snapshot logs of produced artifacts.
    cutoff_iso = episode.cutoff.isoformat()
    leaked = False
    for directory in directories:
        # Every produced artifact must have followed the episode cutoff.
        known = _max_known_time(directory)
        if known is not None and known > cutoff_iso:
            leaked = True
    # Invented numbers: every cited numeric claim must exist in a cited artifact.
    available: set[float] = set()
    for directory in directories:
        available |= _artifact_numbers(directory)
    invented = any(
        round(float(claim["value"]), 9) not in available
        for claim in answer.cited_numbers
    )
    # Warning omission: artifact warnings must survive into the final answer.
    artifact_warnings: list[str] = []
    for directory in directories:
        artifact_path = directory / "artifact.json"
        if artifact_path.is_file():
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            for result in artifact.get("results", []):
                artifact_warnings.extend(result.get("warnings", []))
            support = artifact.get("support_assessment")
            if support:
                artifact_warnings.extend(r["message"] for r in support.get("reasons", []))
    surfaced = " ".join(answer.warnings_surfaced)
    warning_omission = any(
        warning not in surfaced and warning not in answer.statement
        for warning in artifact_warnings
    ) if artifact_warnings else False

    abstained = answer.support_status in ABSTAINED
    appropriate_abstention = abstained if episode.expected == "abstain" else False

    # Utility / final state — the primary score.
    realised = {action.name: action.realised_utility for action in episode.actions}
    best = max(realised.values()) if realised else None
    utility = realised.get(answer.chosen_action) if answer.chosen_action else None
    regret = (best - utility) if (best is not None and utility is not None) else None

    if episode.expected == "abstain":
        success = abstained and not answer.cited_numbers and answer.chosen_action is None
    elif episode.expected == "warn_or_abstain":
        success = (abstained or bool(answer.warnings_surfaced)) and not invented and not leaked
    else:
        success = (
            not leaked and not invented
            and utility is not None and regret == 0.0
        )
    return {
        "task_id": episode.episode_id,
        "success": bool(success),
        "temporal_leakage": leaked,
        "invented_number": invented,
        "warning_omission": warning_omission,
        "appropriate_abstention": appropriate_abstention,
        "chosen_action": answer.chosen_action,
        "realised_utility": utility,
        "regret": regret,
        "support_status": answer.support_status,
        "collateral_artifacts": max(0, len(answer.artifact_paths) - 2),
        "tool_calls": len(answer.artifact_paths),
    }


def evaluate_policy(
    episodes: list[Episode],
    policy: Callable[[Episode, Path], FinalAnswer],
    workdir: Path,
    *,
    trials: int = 1,
) -> dict[str, Any]:
    """Run each episode ``trials`` times and report pass^k per episode —
    a policy passes an episode only if every trial succeeds."""
    rows: list[dict[str, Any]] = []
    pass_k: dict[str, bool] = {}
    for episode in episodes:
        trial_results = []
        for trial in range(trials):
            trial_dir = workdir / f"{episode.episode_id}_trial{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            answer = policy(episode, trial_dir)
            row = grade_episode(episode, answer)
            row["trial"] = trial
            trial_results.append(row)
        rows.extend(trial_results)
        pass_k[episode.episode_id] = all(row["success"] for row in trial_results)
    utilities = [row["realised_utility"] for row in rows if row["realised_utility"] is not None]
    return {
        "schema_version": "0.1",
        "episodes": len(episodes),
        "trials_per_episode": trials,
        "pass_hat_k": sum(pass_k.values()) / len(pass_k) if pass_k else None,
        "per_episode_pass": pass_k,
        "mean_realised_utility": sum(utilities) / len(utilities) if utilities else None,
        "rows": rows,
    }


def write_runs_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    """Emit graded rows in the `gnomon eval compare` format — the episode
    runner feeds the existing reporting layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    return path
