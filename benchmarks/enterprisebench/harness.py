"""EnterpriseBench: one harness, N domain packs, governed temporal decisions.

Each domain pack is a mechanistic simulator plus a cost model plus a
context schema for a real enterprise forecasting-and-decision job (cloud
budget breaches, cash-floor draws, order-up-to quantities, schedule
positions, provision thresholds, staffing calls). The suite answers, per
domain and per model: does Gnomon's governed layer reduce business loss
versus the model alone, versus the engine alone, and versus policies that
ignore the data — under leakage-proof, point-in-time-correct evaluation.

The differentiating layer is the bitemporal contract, enforced
mechanically here rather than promised in prose:

- every context item carries ``{item_id, kind, value, known_at,
  effective_from, effective_to, revises}``, and revision chains are
  first-class — simulators emit early noisy versions and later
  corrections;
- a case at cutoff T exposes only items with ``known_at <= T``, in the
  version known at T; the as-of resolver lives here, is tested once, and
  is used by every pack and every arm;
- a leakage lint over the serialized prompts fails the run before any
  API spend if a post-cutoff observation, post-cutoff revision value, or
  future-event outcome appears in any arm's prompt;
- trap cases (~15% per pack, disclosed) revise a fact so the correction
  flips the correct decision — trap accuracy is scored separately per
  arm, so an information-boundary violation becomes a measured quantity,
  not just a lint.

Arms are matched (temperature 0, prompts differing only by the treatment
block, symmetry verified pre-flight): ``model`` alone; the deterministic
``engine`` (real production ``forecast(threshold=...)`` plus the governed
breach-policy ladder mapped to the domain's decision, at zero API cost);
``model_facts_oracle`` (model plus engine outputs computed from
structured context); and ``governed_candidate`` (the model's forecast
admitted as a candidate inside the engine's contract: backtested on
inner folds against seasonal-naive under the same as-of snapshots,
published as primary only if it wins, engine fallback otherwise).

The primary metric is decision cost and regret in the domain's stated
units. Domains have different units; the rollup refuses to average them.

A new domain must be addable without touching this file — packs register
themselves via :func:`register` when ``benchmarks.enterprisebench.domains``
imports them, and a registry test enforces that this module names no
domain.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.common.openrouter import extract_json_objects  # noqa: E402

GENERATOR_VERSION = "0.1"
#: Arms that require a model call. ``engine`` is deterministic and free.
MODEL_ARMS = ("model", "model_facts_oracle", "governed_candidate")
ARMS = ("model", "engine") + MODEL_ARMS[1:]
#: Inner backtest folds for the governed-candidate arm. Each fold asks
#: the model to forecast from an earlier as-of snapshot of the same
#: series; the harness scores those folds against seasonal-naive from
#: the identical cutoffs before the model's forecast may publish.
CANDIDATE_FOLDS = 2
SYSTEM = """You make one operational decision for an enterprise metric.
Infer only from the supplied data and evidence. Answer with one JSON
object and nothing else."""


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------
# Bitemporal context contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextItem:
    """One dated context fact in one version.

    ``known_at``, ``effective_from`` and ``effective_to`` are step
    indices on the case's time grid (history step 0 is the oldest shown
    value; the cutoff index equals ``len(case.values)``). ``revises``
    names the item version this one supersedes; revision chains are
    resolved by :func:`as_of`.
    """

    item_id: str
    kind: str
    value: float
    known_at: int
    effective_from: int
    effective_to: int
    revises: str | None = None
    text_only: bool = False
    trap: bool = False
    #: Extra structured detail a pack wants rendered or used mechanically
    #: (stored as sorted pairs so cases stay hashable and deterministic).
    aux: tuple[tuple[str, Any], ...] = ()

    def aux_dict(self) -> dict[str, Any]:
        return dict(self.aux)


def as_of(items: tuple[ContextItem, ...] | list[ContextItem],
          cutoff: int) -> list[ContextItem]:
    """The version of every context fact known at ``cutoff``.

    Only items with ``known_at <= cutoff`` exist at all; among those, a
    version superseded by a visible revision is replaced by that
    revision (chains resolve transitively because every link marks its
    predecessor superseded). Facts whose every version is post-cutoff
    are absent — they have not happened yet from the case's viewpoint.
    """
    visible = [item for item in items if item.known_at <= cutoff]
    superseded = {item.revises for item in visible if item.revises}
    return sorted((item for item in visible
                   if item.item_id not in superseded),
                  key=lambda item: (item.known_at, item.item_id))


def hidden_versions(items: tuple[ContextItem, ...] | list[ContextItem],
                    cutoff: int) -> list[ContextItem]:
    """Item versions that must never reach a prompt: known post-cutoff."""
    return sorted((item for item in items if item.known_at > cutoff),
                  key=lambda item: (item.known_at, item.item_id))


# ---------------------------------------------------------------------------
# Domain pack protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """Named costs, break-even, and the domain's pricing of a decision."""

    #: Named costs in the domain's stated units (documented per pack).
    names: dict[str, float]
    #: The probability (binary domains) or fractile (quantity domains)
    #: at which acting and not acting price the same.
    break_even: float
    #: ``score(decision, case) -> {"cost": float, "regret": float}``.
    score: Callable[[dict[str, Any], "Case"], dict[str, float]]
    #: The recorded no-action default a malformed answer degrades to.
    no_action: Callable[["Case"], dict[str, Any]]
    #: The hindsight-optimal decision for the realized future.
    optimal: Callable[["Case"], dict[str, Any]]


@dataclass(frozen=True)
class Case:
    """One decision situation at one cutoff, with realized future truth."""

    case_id: str
    domain: str
    frequency: str
    values: tuple[float, ...]
    future: tuple[float, ...]
    horizon: int
    items: tuple[ContextItem, ...]
    #: The engine threshold derived from as-of structured facts, already
    #: resolved by the pack at generation (packs re-derive it from
    #: claims for the compiled arm).
    threshold: float | None
    trap: bool
    #: On trap cases: the optimal decision under the as-of-correct
    #: (revised) facts, and under the stale superseded version. The two
    #: differ by construction; matching the stale one is the trap.
    trap_optimal: dict[str, Any] | None
    stale_optimal: dict[str, Any] | None
    series_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cutoff(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class DomainPack:
    """Everything one enterprise decision job must declare.

    The harness owns arms, pairing, admission calls, stats, resume,
    lints, and references; a pack only declares content and mechanics.
    """

    name: str
    version: str
    #: "binary" (act/hold style) or "quantity" (a number is the decision).
    decision_kind: str
    #: ``simulate(seed, count) -> (cases, provenance)`` — mechanistic
    #: generator with known causal ground truth; deterministic in its
    #: inputs; futures generated by the same mechanism as histories.
    simulate: Callable[[int, int], tuple[list[Case], dict[str, Any]]]
    cost_model: CostModel
    #: The JSON contract the model must return: {"instruction": str,
    #: "fields": {...}} — instruction is appended verbatim to prompts.
    decision_schema: dict[str, Any]
    #: Typed fact kinds this domain emits: kind -> {"unit", "bounds"
    #: (plausibility for admission), "max_span" (effect prior on
    #: effective-window length, steps)}.
    context_kinds: dict[str, dict[str, Any]]
    #: The domain question text (shared across arms).
    question: Callable[[Case], str]
    #: Structured facts -> engine inputs (at minimum {"threshold": x};
    #: optionally {"series": [...], "basis": str} when the engine runs on
    #: a documented transform of the shown series).
    engine_inputs: Callable[[Case, list[ContextItem]], dict[str, Any]]
    #: Engine packet + inputs -> the domain decision (no LLM).
    engine_decision: Callable[
        [Case, dict[str, Any], dict[str, Any]], dict[str, Any]]
    #: A point forecast path -> the domain decision (candidate arm and
    #: the seasonal-naive / last-value references).
    decision_from_forecast: Callable[
        [Case, list[float], dict[str, Any]], dict[str, Any]]
    #: Constant policies that ignore the data, named in domain terms.
    constant_policies: Callable[[Case], dict[str, dict[str, Any]]]
    #: Validate one parsed JSON payload; return the decision dict or
    #: None when the payload does not meet the contract.
    parse_decision: Callable[[dict[str, Any], Case], dict[str, Any] | None]
    #: Reduce a decision to a comparable scalar (binary: 0/1) for trap
    #: agreement scoring.
    decision_scalar: Callable[[dict[str, Any]], float]
    #: Simulator parameters, hashed into the dataset identity.
    config: dict[str, Any] = field(default_factory=dict)
    season_length: int = 7


_REGISTRY: dict[str, DomainPack] = {}


def register(pack: DomainPack) -> DomainPack:
    if pack.name in _REGISTRY:
        raise ValueError(f"domain pack {pack.name!r} already registered")
    _REGISTRY[pack.name] = pack
    return pack


def registry() -> dict[str, DomainPack]:
    """All registered packs. ``benchmarks.enterprisebench.domains``
    imports every module in its package, so importing it populates
    this without the harness naming any domain."""
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Shared helpers packs build on (binary breach-style domains)
# ---------------------------------------------------------------------------

def binary_cost_model(act_cost: float, miss_cost: float,
                      act_name: str, miss_name: str) -> CostModel:
    """Act now (fixed cost, fully mitigates) vs a missed adverse event."""

    def score(decision: dict[str, Any], case: Case) -> dict[str, float]:
        acted = decision.get("action") == "act"
        cost = act_cost if acted else (
            miss_cost if case.meta["truth_event"] else 0.0)
        optimal = act_cost if case.meta["truth_event"] else 0.0
        return {"cost": cost, "regret": cost - optimal}

    def no_action(case: Case) -> dict[str, Any]:
        return {"action": "monitor", "event_expected": False,
                "first_event_step": None}

    def optimal(case: Case) -> dict[str, Any]:
        return {"action": "act" if case.meta["truth_event"] else "monitor",
                "event_expected": bool(case.meta["truth_event"]),
                "first_event_step": case.meta.get("truth_first_step")}

    return CostModel(
        names={act_name: act_cost, miss_name: miss_cost},
        break_even=act_cost / miss_cost,
        score=score, no_action=no_action, optimal=optimal)


def binary_decision_schema(horizon: int) -> dict[str, Any]:
    return {
        "instruction": (
            'Return {"event_expected": true|false, "first_event_step": '
            f'<1-{horizon} or null>, "action": "act"|"monitor"}}.'),
        "fields": {"event_expected": "bool", "first_event_step": "int|null",
                   "action": '"act"|"monitor"'},
    }


def parse_binary_decision(payload: dict[str, Any],
                          case: Case) -> dict[str, Any] | None:
    event = payload.get("event_expected")
    action = str(payload.get("action", "")).strip().lower()
    if not isinstance(event, bool) or action not in {"act", "monitor"}:
        return None
    raw_step = payload.get("first_event_step")
    # ``json.loads`` accepts NaN/Infinity, and ``True`` is an ``int``:
    # a malformed step must degrade to None, never crash a paid run.
    step = None
    if (isinstance(raw_step, (int, float)) and not isinstance(raw_step, bool)
            and math.isfinite(raw_step) and 1 <= int(raw_step) <= case.horizon):
        step = int(raw_step)
    return {"event_expected": event, "first_event_step": step,
            "action": action}


def governed_engine_decision(case: Case, packet: dict[str, Any],
                             act_cost: float,
                             miss_cost: float) -> dict[str, Any]:
    """Map the production governed breach ladder to an act/monitor call.

    Withholding (no probability could be formed) is monitor-by-omission
    in this one-shot benchmark and stays separately flagged, so the
    engine cannot improve apparent precision by withholding."""
    analysis = packet.get("threshold_analysis") or {}
    event = analysis.get("horizon_event") or {}
    decision = packet.get("governed_decision") or {}
    recommendation = decision.get("recommended_action")
    probability = event.get("probability_any_breach")
    expected = (float(probability) >= 0.5
                if probability is not None else False)
    step = event.get("first_breach_step_median_conditional") if expected \
        else None
    return {"action": recommendation or "monitor",
            "event_expected": expected, "first_event_step": step,
            "withheld": recommendation is None}


def first_crossing(path: list[float], threshold: float,
                   above: bool = True) -> int | None:
    for step, value in enumerate(path, 1):
        if (value > threshold) if above else (value < threshold):
            return step
    return None


def crossing_decision(case: Case, path: list[float], threshold: float,
                      above: bool = True) -> dict[str, Any]:
    step = first_crossing(path, threshold, above)
    return {"action": "act" if step is not None else "monitor",
            "event_expected": step is not None, "first_event_step": step}


# ---------------------------------------------------------------------------
# Forecast references and MASE
# ---------------------------------------------------------------------------

def seasonal_naive_path(history: list[float] | tuple[float, ...],
                        season: int, horizon: int) -> list[float]:
    if len(history) < season:
        last = history[-1] if history else 0.0
        return [float(last)] * horizon
    block = list(history[-season:])
    return [float(block[step % season]) for step in range(horizon)]


def mase(forecast: list[float], actual: list[float],
         history: list[float] | tuple[float, ...], season: int) -> float | None:
    """Mean absolute scaled error; affine-invariant because numerator and
    denominator transform identically under ``x -> a*x + b``."""
    if len(history) <= season or not actual:
        return None
    denominator = statistics.mean(
        abs(history[index] - history[index - season])
        for index in range(season, len(history)))
    if denominator <= 0:
        return None
    numerator = statistics.mean(
        abs(float(left) - float(right))
        for left, right in zip(forecast, actual))
    return numerator / denominator


def pinball(quantile_path: list[float], actual: list[float],
            tau: float) -> float | None:
    if not quantile_path or len(quantile_path) != len(actual):
        return None
    losses = []
    for predicted, realized in zip(quantile_path, actual):
        diff = float(realized) - float(predicted)
        losses.append(max(tau * diff, (tau - 1.0) * diff))
    return statistics.mean(losses)


def verify_mase_affine_invariance(pack: DomainPack, case: Case,
                                  tolerance: float = 0.01) -> None:
    """Pre-flight: the seasonal-naive reference must score identically
    across any positive affine transform of the same case."""
    season = pack.season_length
    history = list(case.values)
    actual = list(case.future)
    raw = mase(seasonal_naive_path(history, season, case.horizon),
               actual, history, season)
    a, b = 1.7, 230.0
    transformed = mase(
        seasonal_naive_path([a * v + b for v in history], season,
                            case.horizon),
        [a * v + b for v in actual], [a * v + b for v in history], season)
    if raw is None or transformed is None:
        raise ValueError(
            f"{pack.name}: MASE undefined on case {case.case_id}; the "
            "seasonal-naive reference cannot anchor this domain")
    if abs(raw - transformed) > tolerance:
        raise ValueError(
            f"{pack.name}: MASE not affine-invariant on {case.case_id} "
            f"({raw} vs {transformed})")


# ---------------------------------------------------------------------------
# Engine packets (real production output, cached per threshold)
# ---------------------------------------------------------------------------

def _grid_timestamps(frequency: str, count: int) -> list[str]:
    """Synthetic anchor timestamps on the domain's true cadence. Dates
    are arbitrary (values are anonymized anyway); the *step* is real,
    because Gnomon's season detection is frequency-aware."""
    from datetime import datetime, timedelta

    if frequency == "MS":
        return [f"{2000 + index // 12}-{1 + index % 12:02d}-01"
                for index in range(count)]
    steps = {"h": timedelta(hours=1), "W": timedelta(weeks=1),
             "D": timedelta(days=1)}
    step = steps.get(frequency, timedelta(days=1))
    start = datetime(2020, 1, 1)
    if step >= timedelta(days=1):
        return [(start + step * index).date().isoformat()
                for index in range(count)]
    return [(start + step * index).isoformat(sep=" ")
            for index in range(count)]


def grid_date(case: Case, step_index: int) -> str:
    """The grid date for a step index (history and future share a grid)."""
    return _grid_timestamps(case.frequency,
                            case.cutoff + case.horizon)[min(
                                step_index, case.cutoff + case.horizon - 1)]


def compute_engine_packet(case: Case, inputs: dict[str, Any],
                          cost_names: tuple[float, float],
                          cache: dict[tuple[str, float], dict[str, Any]],
                          ) -> dict[str, Any]:
    """Gnomon's real product output for this exact call, bounded for a
    prompt. Computed from the visible history (or the pack's documented
    transform of it) alone; cached per (case, threshold) so the compiled
    arm's re-derived threshold does not silently double compute."""
    import shutil

    from gnomon import forecast as gnomon_forecast
    from gnomon.contracts import GnomonError
    from gnomon.support import forecast_headline

    series = [float(v) for v in inputs.get("series") or case.values]
    threshold = float(inputs["threshold"])
    key = (case.case_id, round(threshold, 6))
    if key in cache:
        return cache[key]
    run_dir = Path(tempfile.mkdtemp(prefix="enterprisebench-"))
    try:
        csv_path = run_dir / "history.csv"
        stamps = _grid_timestamps(case.frequency, len(series))
        csv_path.write_text(
            "\n".join(["timestamp,value"] + [
                f"{stamp},{value!r}"
                for stamp, value in zip(stamps, series)]) + "\n",
            encoding="utf-8")
        try:
            artifact, _ = gnomon_forecast(
                str(csv_path), time_column="timestamp",
                target_column="value", horizon=case.horizon,
                frequency=case.frequency, threshold=threshold,
                output=str(run_dir / "out"))
        except GnomonError as error:
            packet = {"status": "abstained", "code": error.code,
                      "message": str(error.message)[:300]}
            cache[key] = packet
            return packet
        result = artifact.results[0]
        rows = result.forecast or []
        packet: dict[str, Any] = {
            "authority": "computed_gnomon_forecast_with_threshold_analysis",
            "support": result.support,
            "selected_model": result.selected_model,
            "headline": forecast_headline(
                result.support, result.support_assessment, rows),
            "basis": inputs.get("basis", "shown_series"),
            "forecast": [
                {"step": step,
                 "q50": round(float(row.get("q50", row["point"])), 4),
                 "q10": round(float(row["q10"]), 4),
                 "q90": round(float(row["q90"]), 4)}
                for step, row in enumerate(rows, 1)],
            "warnings": [str(item)[:200]
                         for item in (result.warnings or [])[:2]],
        }
        analysis = result.threshold or {}
        if analysis:
            horizon_event = analysis.get("horizon_event") or {}
            stamp_to_step = {str(row.get("timestamp")): step
                             for step, row in enumerate(rows, 1)}
            packet["threshold_analysis"] = {
                "threshold": threshold,
                "probability_above_per_step": [
                    round(float(value), 4)
                    for value in (analysis.get("probability_above") or [])],
                "first_step_point_above": stamp_to_step.get(
                    str(analysis.get("first_timestamp_point_above"))),
                "basis": analysis.get("basis"),
                "horizon_event": horizon_event or None,
            }
            if horizon_event:
                from gnomon.breach import (
                    BreachDecisionPolicy, apply_breach_policy,
                )
                packet["governed_decision"] = apply_breach_policy(
                    horizon_event,
                    BreachDecisionPolicy(cost_names[0], cost_names[1]))
        else:
            packet["threshold_analysis"] = {
                "threshold": threshold,
                "unavailable": ("exceedance probabilities require "
                                "calibrated residuals, which these rows "
                                "do not have")}
        cache[key] = packet
        return packet
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def render_context_structured(case: Case) -> str:
    """The as-of view of the context record, typed and dated."""
    resolved = as_of(case.items, case.cutoff)
    payload = [
        {"item_id": item.item_id, "kind": item.kind,
         "value": round(item.value, 4),
         "known_at": grid_date(case, item.known_at),
         "effective_from": grid_date(case, item.effective_from),
         "effective_to": grid_date(case, item.effective_to),
         **({"revises": item.revises} if item.revises else {}),
         **({"detail": item.aux_dict()} if item.aux else {})}
        for item in resolved
    ]
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def cost_block(pack: DomainPack) -> str:
    names = ", ".join(f"{name}={value}"
                      for name, value in sorted(pack.cost_model.names.items()))
    return (f"Costs in this domain's units: {names}. "
            "An unanswerable or malformed reply is scored as the "
            "no-action default.")


def base_prompt(case: Case, pack: DomainPack) -> str:
    """The shared question block: identical across every model arm."""
    return (
        f"Domain: {pack.name}. Series {case.series_id} at frequency "
        f"{case.frequency}, values oldest first (the next {case.horizon} "
        "observations are not shown):\n"
        f"{json.dumps(list(case.values), separators=(',', ':'))}\n"
        f"Dated context record (as of {grid_date(case, case.cutoff)}, "
        "each item in the version known at that date):\n"
        f"{render_context_structured(case)}\n"
        f"{cost_block(pack)}\n"
        f"{pack.question(case)}"
    )


def candidate_fold_cutoffs(case: Case) -> list[int]:
    return [case.cutoff - fold * case.horizon
            for fold in range(CANDIDATE_FOLDS, 0, -1)]


def prompt_for(case: Case, pack: DomainPack, arm: str,
               packet: dict[str, Any] | None) -> str:
    body = base_prompt(case, pack)
    if arm == "model_facts_oracle":
        body += ("\nComputed Gnomon evidence (deterministic, from the same "
                 "shown history and the structured context record):\n"
                 + json.dumps(packet, separators=(",", ":"), sort_keys=True))
    if arm == "governed_candidate":
        folds = candidate_fold_cutoffs(case)
        return body + (
            "\nYou are a forecast candidate under admission review. First "
            "backtest yourself: for each inner cutoff in "
            f"{folds} produce the {case.horizon}-step forecast you would "
            "have issued using only the values before that cutoff (they "
            "are scored against the later shown values, so copying them "
            "is detectable as over-promise). Then forecast the "
            f"{case.horizon} unshown future values.\n"
            'Return {"inner_forecasts": [' +
            ", ".join(f"[{case.horizon} numbers from cutoff {c}]"
                      for c in folds) +
            f'], "forecast": [{case.horizon} numbers]}}.')
    return body + "\n" + pack.decision_schema["instruction"]


def verify_arm_symmetry(cases: list[Case], pack: DomainPack,
                        prompts: dict[tuple[str, str], str]) -> None:
    for case in cases:
        shared = base_prompt(case, pack)
        for arm in MODEL_ARMS:
            text = prompts[(case.case_id, arm)]
            if not text.startswith(shared):
                raise ValueError(
                    f"arm {arm} altered the shared question for "
                    f"{case.case_id}")


# ---------------------------------------------------------------------------
# Leakage lint (pre-spend, per arm)
# ---------------------------------------------------------------------------

def _number_markers(value: float) -> list[str]:
    """Distinctive renderings of a hidden numeric value. Short strings
    ("25") would false-positive on unrelated numbers, so only
    sufficiently distinctive renderings are scanned for."""
    markers = []
    for text in (repr(round(float(value), 4)), f"{float(value):.2f}"):
        if len(text.replace("-", "").replace(".", "")) >= 4:
            markers.append(text)
    return markers


def leakage_lint(cases: list[Case], pack: DomainPack,
                 prompts: dict[tuple[str, str], str]) -> None:
    """Fail before any API spend if held-out information appears in any
    arm's serialized prompt: future observations (numeric marker,
    history-excised, >= 8 values) or the value of any item version known
    only after the cutoff (post-cutoff observations and post-cutoff
    revisions alike)."""
    for case in cases:
        future_marker = None
        if len(case.future) >= 8:
            future_marker = json.dumps(
                [round(float(v), 4) for v in case.future[:8]],
                separators=(",", ":"))[1:-1]
        history_blob = json.dumps(list(case.values), separators=(",", ":"))
        hidden = hidden_versions(case.items, case.cutoff)
        for arm in MODEL_ARMS:
            text = prompts[(case.case_id, arm)]
            scanned = text.replace(history_blob, "", 1)
            if future_marker and future_marker in scanned:
                raise ValueError(
                    f"{pack.name}: held-out future leaked into arm {arm} "
                    f"for {case.case_id}")
            for item in hidden:
                for marker in _number_markers(item.value):
                    if marker in scanned:
                        raise ValueError(
                            f"{pack.name}: post-cutoff item "
                            f"{item.item_id} value leaked into arm {arm} "
                            f"for {case.case_id}")


# ---------------------------------------------------------------------------
# Parsing model answers
# ---------------------------------------------------------------------------

def parse_decision_answer(text: str, pack: DomainPack,
                          case: Case) -> dict[str, Any]:
    """Balanced-span candidates, not a greedy regex: a correct JSON
    answer wrapped in prose (or preceded by an echoed evidence packet,
    which only the oracle arm's prompt even contains) must not be scored
    as an invalid answer the model never gave."""
    for payload in extract_json_objects(text):
        inner = payload.get("decision")
        if isinstance(inner, dict):
            payload = inner
        decision = pack.parse_decision(payload, case)
        if decision is not None:
            return {"valid": True, "decision": decision}
    return {"valid": False, "decision": pack.cost_model.no_action(case)}


def _finite_path(raw: Any, horizon: int) -> list[float] | None:
    if not isinstance(raw, list) or len(raw) != horizon:
        return None
    path = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            return None
        path.append(float(value))
    return path


def parse_candidate_answer(text: str, case: Case) -> dict[str, Any]:
    for payload in extract_json_objects(text):
        forecast = _finite_path(payload.get("forecast"), case.horizon)
        if forecast is None:
            continue
        inner_raw = payload.get("inner_forecasts")
        inner: list[list[float]] | None = []
        if isinstance(inner_raw, list) and len(inner_raw) == CANDIDATE_FOLDS:
            for fold in inner_raw:
                path = _finite_path(fold, case.horizon)
                if path is None:
                    inner = None
                    break
                inner.append(path)
        else:
            inner = None
        return {"valid": True, "forecast": forecast,
                "inner_forecasts": inner}
    return {"valid": False, "forecast": None, "inner_forecasts": None}


# ---------------------------------------------------------------------------
# Candidate admission (backtest against seasonal-naive, same snapshots)
# ---------------------------------------------------------------------------

def candidate_admission(case: Case, pack: DomainPack,
                        inner: list[list[float]] | None,
                        ) -> dict[str, Any]:
    """Backtest the model's inner-fold forecasts against seasonal-naive
    from the identical as-of cutoffs. The model is published as primary
    only when it wins; the row records the admission and the backtest
    promise so post-admission out-of-sample error can expose
    over-promising (a model that copied the visible answers into its
    'backtest' will promise far better than it later delivers)."""
    season = pack.season_length
    fold_cutoffs = candidate_fold_cutoffs(case)
    naive_scores = []
    model_scores = [] if inner is not None else None
    for index, fold_cutoff in enumerate(fold_cutoffs):
        history = list(case.values[:fold_cutoff])
        actual = list(case.values[fold_cutoff:fold_cutoff + case.horizon])
        naive = seasonal_naive_path(history, season, case.horizon)
        naive_mase = mase(naive, actual, history, season)
        if naive_mase is not None:
            naive_scores.append(naive_mase)
            if model_scores is not None:
                model_mase = mase(inner[index], actual, history, season)
                if model_mase is None:
                    model_scores = None
                else:
                    model_scores.append(model_mase)
    admitted = bool(model_scores and naive_scores
                    and statistics.mean(model_scores)
                    < statistics.mean(naive_scores))
    return {
        "admitted": admitted,
        "backtest_model_mase": (statistics.mean(model_scores)
                                if model_scores else None),
        "backtest_naive_mase": (statistics.mean(naive_scores)
                                if naive_scores else None),
        "backtest_folds": len(naive_scores),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def trap_agreement(decision: dict[str, Any], case: Case,
                   pack: DomainPack) -> bool | None:
    """Did the arm act on the as-of-correct (revised) fact rather than
    the stale superseded version? Scored as which optimal the decision
    scalar sits closer to; ties go to the stale side, so agreement is
    strict."""
    if not case.trap or case.trap_optimal is None \
            or case.stale_optimal is None:
        return None
    value = pack.decision_scalar(decision)
    to_correct = abs(value - pack.decision_scalar(case.trap_optimal))
    to_stale = abs(value - pack.decision_scalar(case.stale_optimal))
    return to_correct < to_stale


def score_decision_row(decision: dict[str, Any], valid: bool, case: Case,
                       pack: DomainPack) -> dict[str, Any]:
    outcome = pack.cost_model.score(decision, case)
    row: dict[str, Any] = {
        "valid": valid, "decision": decision,
        "cost": round(outcome["cost"], 6),
        "regret": round(outcome["regret"], 6),
        "action_optimal": outcome["regret"] == 0.0,
        "trap": case.trap,
        "trap_correct": trap_agreement(decision, case, pack),
    }
    truth_step = case.meta.get("truth_first_step")
    answered_step = decision.get("first_event_step")
    row["timing_error"] = (abs(answered_step - truth_step)
                           if truth_step is not None
                           and answered_step is not None else None)
    return row


def _arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Decision economics are scored over every row — an unparseable
    # answer is priced as the no-action default the operator lived
    # through. Call-quality metrics are scored over valid answers only.
    if not rows:
        return {"rows": 0}
    valid = [row for row in rows if row["valid"]]
    trap_rows = [row for row in rows if row.get("trap_correct") is not None]
    timing = [row["timing_error"] for row in valid
              if row.get("timing_error") is not None]
    metrics = {
        "rows": len(rows),
        "mean_cost": statistics.mean(row["cost"] for row in rows),
        "mean_regret": statistics.mean(row["regret"] for row in rows),
        "action_optimal_rate": statistics.mean(
            row["action_optimal"] for row in rows),
        "invalid_rate": statistics.mean(not row["valid"] for row in rows),
        "call_metrics_scored": len(valid),
        "trap_cases": len(trap_rows),
        "trap_accuracy": (statistics.mean(
            row["trap_correct"] for row in trap_rows)
            if trap_rows else None),
        # Timing is over each arm's self-selected answered subset; the
        # answer rate is reported next to the error for that reason.
        "timing_mae": statistics.mean(timing) if timing else None,
        "timing_answer_rate": (len(timing) / len(valid)) if valid else None,
    }
    return metrics


def exact_sign_p(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k)
               for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def paired_cost_comparison(treatment: list[dict[str, Any]],
                           reference: list[dict[str, Any]],
                           ) -> dict[str, Any]:
    """Exact sign test on per-case decision cost, pair counts disclosed."""
    by_case = {row["case_id"]: row for row in reference}
    treatment_cheaper = reference_cheaper = pairs = 0
    for row in treatment:
        other = by_case.get(row["case_id"])
        if other is None:
            continue
        pairs += 1
        if row["cost"] < other["cost"]:
            treatment_cheaper += 1
        elif other["cost"] < row["cost"]:
            reference_cheaper += 1
    return {
        "paired_cases": pairs,
        "treatment_cheaper": treatment_cheaper,
        "reference_cheaper": reference_cheaper,
        "exact_sign_p": exact_sign_p(treatment_cheaper, reference_cheaper),
        "mean_cost_delta": (
            statistics.mean(row["cost"] - by_case[row["case_id"]]["cost"]
                            for row in treatment
                            if row["case_id"] in by_case)
            if pairs else None),
    }


# ---------------------------------------------------------------------------
# The run itself (one domain)
# ---------------------------------------------------------------------------

def dataset_identity(pack: DomainPack, seed: int, count: int) -> str:
    config_sha = hashlib.sha256(json.dumps(
        pack.config, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()
    return (f"enterprisebench-{GENERATOR_VERSION}:{pack.name}:"
            f"pack={pack.version}:seed={seed}:cases={count}:"
            f"config={config_sha[:12]}")


def _load_resume_rows(rows_path: Path, identity: str, model_name: str,
                      valid_ids: set[str],
                      ) -> dict[tuple[str, str], dict[str, Any]]:
    """Resume rejects mismatched dataset identity, mismatched model,
    unknown case ids, and crash-truncated lines — disclosed, because
    silently pooling rows scored against different cases is the one
    thing a paired benchmark must never do."""
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if not rows_path.exists():
        return completed
    stale = malformed = 0
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if (row.get("case_id") in valid_ids and row.get("arm") in ARMS
                and row.get("dataset") == identity
                and row.get("model") == model_name):
            completed[(row["case_id"], row["arm"])] = row
        else:
            stale += 1
    if stale or malformed:
        print(f"resume: ignored {stale} stale and {malformed} malformed "
              f"rows", flush=True)
    return completed


def _candidate_row(case: Case, pack: DomainPack, answer: dict[str, Any],
                   engine_dec: dict[str, Any], inputs: dict[str, Any],
                   ) -> dict[str, Any]:
    season = pack.season_length
    admission = candidate_admission(case, pack,
                                    answer.get("inner_forecasts"))
    if answer["valid"] and admission["admitted"]:
        decision = pack.decision_from_forecast(
            case, answer["forecast"], inputs)
        published = "model_candidate"
    else:
        decision = dict(engine_dec)
        published = "engine_fallback"
    row = score_decision_row(decision, answer["valid"], case, pack)
    oos = (mase(answer["forecast"], list(case.future), list(case.values),
                season) if answer["valid"] else None)
    row.update({
        "candidate_admitted": admission["admitted"],
        "published": published,
        "backtest_model_mase": admission["backtest_model_mase"],
        "backtest_naive_mase": admission["backtest_naive_mase"],
        "oos_model_mase": oos,
        # Over-promise: positive means the model delivered worse
        # out-of-sample than its inner backtest promised.
        "over_promise": (oos - admission["backtest_model_mase"]
                         if oos is not None
                         and admission["backtest_model_mase"] is not None
                         else None),
    })
    return row


def run_domain(pack: DomainPack, args: Any, client: Any) -> dict[str, Any]:
    """One domain, all arms, full operational bar: pre-spend lints,
    resume with identity checks, failure tolerance, references at zero
    API cost, paired verdicts."""
    cases, provenance = pack.simulate(args.seed, args.cases)
    for case in cases[:3]:
        verify_mase_affine_invariance(pack, case)

    engine_cache: dict[tuple[str, float], dict[str, Any]] = {}
    cost_pair = _binary_cost_pair(pack)
    packets: dict[str, dict[str, Any]] = {}
    engine_inputs_by_case: dict[str, dict[str, Any]] = {}
    engine_decisions: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        inputs = pack.engine_inputs(case, as_of(case.items, case.cutoff))
        engine_inputs_by_case[case.case_id] = inputs
        packet = compute_engine_packet(case, inputs, cost_pair, engine_cache)
        packets[case.case_id] = packet
        engine_decisions[case.case_id] = pack.engine_decision(
            case, packet, inputs)
        if (index + 1) % 25 == 0:
            print(f"{pack.name}: engine runs {index + 1}/{len(cases)}",
                  flush=True)

    prompts = {(case.case_id, arm): prompt_for(
        case, pack, arm, packets[case.case_id])
        for case in cases for arm in MODEL_ARMS}
    verify_arm_symmetry(cases, pack, prompts)
    leakage_lint(cases, pack, prompts)

    references_rows = _reference_rows(cases, pack, engine_decisions,
                                      engine_inputs_by_case)
    references = {name: _arm_metrics(rows)
                  for name, rows in references_rows.items()}
    references["hindsight_optimal"] = {
        "mean_cost": statistics.mean(
            pack.cost_model.score(pack.cost_model.optimal(case), case)["cost"]
            for case in cases),
        "mean_regret": 0.0,
    }
    engine_rows = references_rows["engine"]
    references["engine"]["withholding_rate"] = statistics.mean(
        bool(row["decision"].get("withheld")) for row in engine_rows)
    references["engine"]["forecast_quality"] = _engine_forecast_quality(
        cases, packets)

    identity = dataset_identity(pack, args.seed, args.cases)
    model_name = getattr(args, "model", None)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    valid_ids = {case.case_id for case in cases}
    completed = (_load_resume_rows(rows_path, identity, model_name,
                                   valid_ids)
                 if getattr(args, "resume", False) else {})
    lock = threading.Lock()

    def one(case: Case, arm: str) -> dict[str, Any]:
        text = client.completions([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompts[(case.case_id, arm)]},
        ], n=1)[0]
        base = {"case_id": case.case_id, "arm": arm, "dataset": identity,
                "model": model_name, "domain": pack.name,
                "series_id": case.series_id}
        if arm == "governed_candidate":
            answer = parse_candidate_answer(text, case)
            return {**base, **_candidate_row(
                case, pack, answer, engine_decisions[case.case_id],
                engine_inputs_by_case[case.case_id])}
        parsed = parse_decision_answer(text, pack, case)
        return {**base, **score_decision_row(
            parsed["decision"], parsed["valid"], case, pack)}

    jobs = [(case, arm) for case in cases for arm in MODEL_ARMS
            if (case.case_id, arm) not in completed]
    failures: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {pool.submit(one, *job): job for job in jobs}
        for future in as_completed(pending):
            case, arm = pending[future]
            try:
                row = future.result()
            except Exception as error:
                # One dead endpoint call must not discard the paid work
                # already on disk: record, finish the rest, fail loudly
                # at the end. Never score an API failure as an answer.
                failures.append((case.case_id, arm, repr(error)[:300]))
                continue
            with lock:
                completed[(row["case_id"], row["arm"])] = row
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(
            f"{pack.name}: {len(failures)}/{len(jobs)} model calls "
            f"failed; completed rows are saved in {rows_path} — rerun "
            f"with --resume to finish. First failure: {failures[0]}")

    summary = _domain_summary(pack, args, cases, provenance, identity,
                              model_name, completed, references,
                              references_rows)
    if hasattr(client, "usage_summary"):
        summary["usage"] = client.usage_summary
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return summary


def _binary_cost_pair(pack: DomainPack) -> tuple[float, float]:
    """(act_cost, miss_cost) for the governed breach-policy projection.

    Quantity domains price per unit; the ladder still needs a pairwise
    policy, so packs expose the pair through the cost-model names with
    the convention that the first sorted name is the act-side cost."""
    names = pack.cost_model.names
    if "act_cost" in names and "miss_cost" in names:
        return names["act_cost"], names["miss_cost"]
    values = [names[key] for key in sorted(names)]
    return min(values), max(values)


def _reference_rows(cases: list[Case], pack: DomainPack,
                    engine_decisions: dict[str, dict[str, Any]],
                    engine_inputs_by_case: dict[str, dict[str, Any]],
                    ) -> dict[str, list[dict[str, Any]]]:
    """References at zero API cost: the governed engine, seasonal naive,
    last value, the pack's constant policies, all priced per case."""
    rows: dict[str, list[dict[str, Any]]] = {"engine": [],
                                             "seasonal_naive": [],
                                             "last_value": []}
    for case in cases:
        inputs = engine_inputs_by_case[case.case_id]
        rows["engine"].append(
            {"case_id": case.case_id,
             **score_decision_row(engine_decisions[case.case_id], True,
                                  case, pack)})
        naive = seasonal_naive_path(case.values, pack.season_length,
                                    case.horizon)
        rows["seasonal_naive"].append(
            {"case_id": case.case_id,
             **score_decision_row(pack.decision_from_forecast(
                 case, naive, inputs), True, case, pack)})
        last = [float(case.values[-1])] * case.horizon
        rows["last_value"].append(
            {"case_id": case.case_id,
             **score_decision_row(pack.decision_from_forecast(
                 case, last, inputs), True, case, pack)})
        for name, decision in pack.constant_policies(case).items():
            rows.setdefault(name, []).append(
                {"case_id": case.case_id,
                 **score_decision_row(decision, True, case, pack)})
    return rows


def _engine_forecast_quality(cases: list[Case],
                             packets: dict[str, dict[str, Any]],
                             ) -> dict[str, Any]:
    mases, pinballs = [], {"q10": [], "q50": [], "q90": []}
    for case in cases:
        packet = packets[case.case_id]
        rows = packet.get("forecast") or []
        if len(rows) != case.horizon or packet.get("basis") != "shown_series":
            continue
        actual = list(case.future)
        score = mase([row["q50"] for row in rows], actual,
                     list(case.values), 7)
        if score is not None:
            mases.append(score)
        for quantile, tau in (("q10", .1), ("q50", .5), ("q90", .9)):
            loss = pinball([row[quantile] for row in rows], actual, tau)
            if loss is not None:
                pinballs[quantile].append(loss)
    return {
        "mase": statistics.mean(mases) if mases else None,
        "pinball": {q: (statistics.mean(v) if v else None)
                    for q, v in pinballs.items()},
        "scored_cases": len(mases),
        "note": ("engine forecast quality is computed on cases whose "
                 "engine ran on the shown series itself; documented "
                 "transforms (e.g. negated balances) are excluded"),
    }


def _domain_summary(pack: DomainPack, args: Any, cases: list[Case],
                    provenance: dict[str, Any], identity: str,
                    model_name: str | None,
                    completed: dict[tuple[str, str], dict[str, Any]],
                    references: dict[str, Any],
                    references_rows: dict[str, list[dict[str, Any]]],
                    ) -> dict[str, Any]:
    arm_rows = {arm: [row for row in completed.values()
                      if row["arm"] == arm] for arm in MODEL_ARMS}
    metrics = {arm: _arm_metrics(rows) for arm, rows in arm_rows.items()}
    metrics["engine"] = references["engine"]
    candidate_rows = arm_rows["governed_candidate"]
    if candidate_rows:
        oos = [row["oos_model_mase"] for row in candidate_rows
               if row.get("oos_model_mase") is not None]
        promises = [row["over_promise"] for row in candidate_rows
                    if row.get("over_promise") is not None]
        metrics["governed_candidate"].update({
            "admission_rate": statistics.mean(
                row["candidate_admitted"] for row in candidate_rows),
            "oos_model_mase": statistics.mean(oos) if oos else None,
            "mean_over_promise": (statistics.mean(promises)
                                  if promises else None),
        })

    engine_rows = references_rows["engine"]
    constant_names = [name for name in references_rows
                      if name not in ("engine", "seasonal_naive",
                                      "last_value")]
    best_constant = min(
        constant_names,
        key=lambda name: references[name]["mean_regret"])
    treatment_arm = _treatment_arm(arm_rows)
    treatment_rows = arm_rows[treatment_arm]
    verdicts = {
        "treatment_arm": treatment_arm,
        "vs_model_alone": paired_cost_comparison(
            treatment_rows, arm_rows["model"]),
        "vs_engine_alone": paired_cost_comparison(
            treatment_rows, engine_rows),
        "vs_best_constant_policy": {
            "policy": best_constant,
            **paired_cost_comparison(
                treatment_rows, references_rows[best_constant])},
        "candidate_admission_value": {
            "candidate_regret": metrics["governed_candidate"].get(
                "mean_regret"),
            "engine_regret": references["engine"]["mean_regret"],
            "admission_rate": metrics["governed_candidate"].get(
                "admission_rate"),
            "mean_over_promise": metrics["governed_candidate"].get(
                "mean_over_promise"),
            **paired_cost_comparison(candidate_rows, engine_rows),
        },
        "trap_integrity": _trap_integrity(arm_rows, references),
        "reading": {
            "useful": ("useful requires the treatment arm cheaper than "
                       "the model alone AND the engine alone AND the "
                       "best constant policy; anything less means one "
                       "component carried the others or nobody beat a "
                       "policy that ignores the data"),
            "candidate_admission_value": (
                "the governed candidate lane is working when its regret "
                "is no worse than the engine's, admission is selective, "
                "and admitted forecasts do not over-promise (deliver "
                "materially worse out-of-sample than their backtest)"),
            "trap_integrity": (
                "trap accuracy is the measured information boundary: an "
                "arm resolving revisions as of the cutoff scores high; "
                "suspiciously high accuracy on the hidden-reversal "
                "subset would indicate leakage of post-cutoff versions"),
        },
    }
    return {
        "schema_version": GENERATOR_VERSION,
        "domain": pack.name,
        "seed": args.seed, "cases": args.cases,
        "model": model_name, "temperature": 0,
        "cost_model": {"names": pack.cost_model.names,
                       "break_even": pack.cost_model.break_even,
                       "units": pack.config.get("units", "domain_units")},
        "provenance": {
            "evaluated_commit": _git_sha(),
            "harness_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "dataset_identity": identity,
            "simulator": provenance,
        },
        "references": references,
        "metrics": metrics,
        "verdicts": verdicts,
        "design": {
            "matched": True,
            "arms_differ_by_treatment_block_only": True,
            "truth_is_realized_simulated_future": True,
            "held_out_future_absent_from_prompts_verified": True,
            "as_of_resolution": "harness_owned_single_resolver",
            "engine_packet_is_production_output": True,
            "costs_stated_in_prompt": True,
            "independence": (
                "each case simulates an independent series (per-series "
                "case counts in provenance); labels can still co-move "
                "through shared regime parameters within a domain — a "
                "caveat, not an independence claim"),
        },
    }


def _treatment_arm(arm_rows: dict[str, list[dict[str, Any]]]) -> str:
    """The arm the domain verdicts judge. Once the compiled text arm
    exists it is the treatment (it is the loop a client actually runs);
    until then the oracle-facts arm stands in, labelled as such."""
    if arm_rows.get("model_facts_compiled"):
        return "model_facts_compiled"
    return "model_facts_oracle"


def _trap_integrity(arm_rows: dict[str, list[dict[str, Any]]],
                    references: dict[str, Any]) -> dict[str, Any]:
    per_arm = {}
    for arm, rows in arm_rows.items():
        trap_rows = [row for row in rows
                     if row.get("trap_correct") is not None]
        if not trap_rows:
            per_arm[arm] = None
            continue
        per_arm[arm] = {
            "trap_cases": len(trap_rows),
            "trap_accuracy": statistics.mean(
                row["trap_correct"] for row in trap_rows),
        }
    per_arm["engine"] = {
        "trap_cases": references["engine"].get("trap_cases"),
        "trap_accuracy": references["engine"].get("trap_accuracy"),
    }
    return per_arm
