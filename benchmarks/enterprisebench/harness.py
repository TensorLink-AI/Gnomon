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
import random
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

GENERATOR_VERSION = "0.3"
#: Arms that require a model call. ``engine`` is deterministic and free.
#: ``model_facts_compiled`` is the loop a client actually runs: text
#: context in, ONE call out returning ``{claims, decision}`` — the
#: model's numerification of the text plus its decision; the harness
#: feeds the claims through the production context-admission gate and
#: recomputes the governed pipeline on admitted claims only.
MODEL_ARMS = ("model", "model_facts_oracle", "model_facts_compiled",
              "governed_candidate")
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
    #: Explicit (act_cost, miss_cost) for the governed breach-policy
    #: projection. Binary packs must set it (``binary_cost_model``
    #: does); quantity packs leave None — their engine mapping is
    #: quantile-based and never reaches the ladder.
    governed_pair: tuple[float, float] | None = None


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
    #: Case count the runner uses when --cases is not given. Binary
    #: domains tie heavily on identical actions, so their sign tests
    #: need more pairs than quantity domains (pilot power check in
    #: EVALUATION-READINESS.md).
    recommended_cases: int = 120
    #: Optional per-decision secondary metrics (e.g. hierarchical
    #: coherence error), averaged per arm under ``metrics.extras``.
    extra_metrics: Callable[[dict[str, Any], "Case"],
                            dict[str, float]] | None = None


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

def tail_shock(rng: Any, sigma: float, df: int = 5) -> float:
    """Observation noise with the fat tails real operational data has:
    a Student-t draw with ``df`` degrees of freedom scaled by ``sigma``.
    Gaussian noise makes every excursion mean something; real spend,
    cash, load, and contact series throw occasional wild points that a
    robust decision layer must not chase. Deterministic in the caller's
    seeded rng."""
    import math as _math

    z = rng.gauss(0.0, 1.0)
    chi2 = sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(df))
    return sigma * z / _math.sqrt(max(chi2 / df, 1e-12))


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
        score=score, no_action=no_action, optimal=optimal,
        governed_pair=(act_cost, miss_cost))


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
    steps = {"5min": timedelta(minutes=5), "h": timedelta(hours=1),
             "W": timedelta(weeks=1), "D": timedelta(days=1)}
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

    series = [float(v) for v in inputs.get("series") or case.values]
    threshold = (float(inputs["threshold"])
                 if inputs.get("threshold") is not None else None)
    key = (case.case_id,
           round(threshold, 6) if threshold is not None else None)
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
        # Numbers and tiers only — no headline, no interpretation
        # packet, no canonical semantic conclusions: the ablation
        # findings showed semantic conclusions in a treatment packet
        # turn a decision benchmark into a transcription benchmark.
        packet: dict[str, Any] = {
            "authority": "computed_gnomon_forecast_with_threshold_analysis",
            "support": result.support,
            "selected_model": result.selected_model,
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
        analysis = (result.threshold or {}) if threshold is not None else {}
        if threshold is None:
            # Quantity domains price a position, not a breach: the
            # packet carries the quantile paths and no threshold block.
            pass
        elif analysis:
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

def prev_values_for(case: Case) -> dict[str, float]:
    """For every as-of resolved revision item, the value of the version
    it superseded — revision renderings mention both figures."""
    by_id = {item.item_id: item for item in case.items}
    return {item.item_id: by_id[item.revises].value
            for item in as_of(case.items, case.cutoff)
            if item.revises and item.revises in by_id}


def text_context(case: Case) -> tuple[str, dict[str, float]]:
    """The native context form: every as-of resolved item rendered as a
    dated memo. Returns the block and the per-item shown values (exact
    extraction ground truth)."""
    from benchmarks.enterprisebench import textgen

    resolved = as_of(case.items, case.cutoff)
    return textgen.render_context_block(case, resolved,
                                        prev_values_for(case))


def cost_block(pack: DomainPack) -> str:
    names = ", ".join(f"{name}={value}"
                      for name, value in sorted(pack.cost_model.names.items()))
    return (f"Costs in this domain's units: {names}. "
            "An unanswerable or malformed reply is scored as the "
            "no-action default.")


def base_prompt(case: Case, pack: DomainPack, context_block: str) -> str:
    """The shared question block: identical across every model arm.
    Context arrives as text, the way it does in a real company."""
    return (
        f"Domain: {pack.name}. Series {case.series_id} at frequency "
        f"{case.frequency}, values oldest first (the next {case.horizon} "
        "observations are not shown):\n"
        f"{json.dumps(list(case.values), separators=(',', ':'))}\n"
        f"Context memos as of {grid_date(case, case.cutoff)}, each dated "
        "when it became known. A memo may revise an earlier figure; use "
        "the version in force as of the cutoff:\n"
        f"{context_block}\n"
        f"{cost_block(pack)}\n"
        f"{pack.question(case)}"
    )


def claims_instruction(pack: DomainPack, case: Case) -> str:
    kinds = ", ".join(sorted(pack.context_kinds))
    example_date = grid_date(case, case.cutoff)
    return (
        "First numerify the context memos into typed claims — one claim "
        "per fact currently in force as of the cutoff (kinds: "
        f"{kinds}); dates use the same format as {example_date}. Then "
        "decide. Return one JSON object: "
        '{"claims": [{"kind": "...", "value": <number>, '
        '"effective_from": "<date>", "effective_to": "<date>"}, ...], '
        '"decision": ' + pack.decision_schema["instruction"].removeprefix(
            "Return ").rstrip(".") + "}.")


def candidate_fold_cutoffs(case: Case) -> list[int]:
    return [case.cutoff - fold * case.horizon
            for fold in range(CANDIDATE_FOLDS, 0, -1)]


def prompt_for(case: Case, pack: DomainPack, arm: str,
               packet: dict[str, Any] | None,
               context_block: str) -> str:
    body = base_prompt(case, pack, context_block)
    if arm == "model_facts_compiled":
        return body + "\n" + claims_instruction(pack, case)
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
                        prompts: dict[tuple[str, str], str],
                        context_blocks: dict[str, str]) -> None:
    for case in cases:
        shared = base_prompt(case, pack, context_blocks[case.case_id])
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


def _contains_number(scanned: str, marker: str) -> bool:
    """Marker occurrences must stand alone as a number: a marker that is
    merely a digit-substring of a longer legitimate value (adjacent
    months of a smooth series share leading digits) is not a leak."""
    start = scanned.find(marker)
    while start != -1:
        before = scanned[start - 1] if start > 0 else " "
        after_index = start + len(marker)
        after = scanned[after_index] if after_index < len(scanned) else " "
        if before not in "0123456789." and after not in "0123456789.":
            return True
        start = scanned.find(marker, start + 1)
    return False


def leakage_lint(cases: list[Case], pack: DomainPack,
                 prompts: dict[tuple[str, str], str]) -> None:
    """Fail before any API spend if held-out information appears in any
    arm's serialized prompt: future observations (numeric marker,
    history-excised, >= 8 values), the value of any item version known
    only after the cutoff (post-cutoff observations and post-cutoff
    revisions alike), or the distinctive generated text reference of a
    hidden version. Also verifies the text layer rendered every as-of
    resolved item — a fact silently dropped from the memos would turn
    extraction scoring into fiction."""
    from benchmarks.enterprisebench.textgen import ref_code

    for case in cases:
        future_blobs = [list(case.future)]
        # Packs with auxiliary held-out series (e.g. per-leaf futures in
        # a hierarchical domain) disclose them under meta.extra_futures
        # so the lint covers them identically.
        for extra in (case.meta.get("extra_futures") or {}).values():
            future_blobs.append(list(extra))
        # min(8, horizon) values: a short-horizon domain (monthly packs)
        # must not silently lose its future-leak check — three or four
        # rounded floats joined are still a distinctive marker.
        future_markers = [
            json.dumps([round(float(v), 4) for v in blob[:8]],
                       separators=(",", ":"))[1:-1]
            for blob in future_blobs if len(blob) >= 3]
        history_blob = json.dumps(list(case.values), separators=(",", ":"))
        hidden = hidden_versions(case.items, case.cutoff)
        resolved = as_of(case.items, case.cutoff)
        for arm in MODEL_ARMS:
            text = prompts[(case.case_id, arm)]
            scanned = text.replace(history_blob, "", 1)
            if any(marker in scanned for marker in future_markers):
                raise ValueError(
                    f"{pack.name}: held-out future leaked into arm {arm} "
                    f"for {case.case_id}")
            for item in hidden:
                if ref_code(case.case_id, item.item_id) in scanned:
                    raise ValueError(
                        f"{pack.name}: post-cutoff item {item.item_id} "
                        f"text reference leaked into arm {arm} for "
                        f"{case.case_id}")
                for marker in _number_markers(item.value):
                    if _contains_number(scanned, marker):
                        raise ValueError(
                            f"{pack.name}: post-cutoff item "
                            f"{item.item_id} value leaked into arm {arm} "
                            f"for {case.case_id}")
            for item in resolved:
                if ref_code(case.case_id, item.item_id) not in text:
                    raise ValueError(
                        f"{pack.name}: as-of item {item.item_id} missing "
                        f"from the text context of arm {arm} for "
                        f"{case.case_id}")


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


def parse_compiled_answer(text: str, pack: DomainPack,
                          case: Case) -> dict[str, Any]:
    """One call, two independently degradable parts: a decision with
    valid claims but a malformed decision block (and vice versa) keeps
    the valid part and records the malformed one separately — never
    crashes, never silently patches."""
    claims: list[Any] | None = None
    decision: dict[str, Any] | None = None
    for payload in extract_json_objects(text):
        if claims is None and isinstance(payload.get("claims"), list):
            claims = payload["claims"]
        raw_decision = payload.get("decision")
        if decision is None and isinstance(raw_decision, dict):
            decision = pack.parse_decision(raw_decision, case)
        if claims is not None and decision is not None:
            break
    return {
        "claims": claims if claims is not None else [],
        "claims_valid": claims is not None,
        "decision": (decision if decision is not None
                     else pack.cost_model.no_action(case)),
        "decision_valid": decision is not None,
    }


# ---------------------------------------------------------------------------
# The production context-admission gate (compiled arm)
# ---------------------------------------------------------------------------

def _step_from_date(case: Case, raw: Any) -> int | None:
    """Map a claimed date back onto the case grid; unparseable dates are
    a schema rejection, dates beyond the grid clamp to its edges."""
    from datetime import datetime

    grid = _grid_timestamps(case.frequency,
                            case.cutoff + case.horizon)
    text = str(raw).strip()
    if text in grid:
        return grid.index(text)
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    # Models often append a timezone the naive grid does not carry;
    # comparing aware to naive raises, so normalize to naive wall time.
    if moment.tzinfo is not None:
        moment = moment.replace(tzinfo=None)
    parsed_grid = [datetime.fromisoformat(stamp) for stamp in grid]
    if moment <= parsed_grid[0]:
        return 0
    for index, stamp in enumerate(parsed_grid):
        if stamp >= moment:
            return index
    return len(parsed_grid) - 1


def admit_claims(raw_claims: list[Any], case: Case,
                 pack: DomainPack) -> dict[str, Any]:
    """Schema validation, the production ``ContextEvent`` contract,
    plausibility bounds, and effect priors — the ContextBench compiled
    route applied to the model's numerification of the text.

    The gate never sees ground truth: it rejects on structure and
    plausibility alone. How often it happens to reject hallucinated
    claims is measured downstream against the simulator's exact record.
    """
    from gnomon.context import (
        UNVERIFIED_EXTERNAL_CREATOR,
        ContextEvent,
        validate_context_event,
    )

    admitted: list[ContextItem] = []
    rejected: list[dict[str, Any]] = []
    parsed: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_claims):
        if not isinstance(raw, dict):
            rejected.append({"claim": raw, "reason": "schema_not_object"})
            continue
        kind = str(raw.get("kind", ""))
        value = raw.get("value")
        start = _step_from_date(case, raw.get("effective_from"))
        end = _step_from_date(case, raw.get("effective_to"))
        if end is None:
            end = start
        if (kind not in pack.context_kinds or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value) or start is None):
            rejected.append({"claim": raw, "reason": "schema"})
            continue
        claim = {"kind": kind, "value": float(value),
                 "effective_from": min(start, end),
                 "effective_to": max(start, end)}
        parsed.append(claim)
        event = ContextEvent(
            event_id=f"{case.case_id}-claim-{index}",
            event_type=kind, entity_scope=(case.series_id,),
            effective_start=grid_date(case, claim["effective_from"])[:10]
            + "T00:00:00+00:00",
            effective_end=grid_date(case, claim["effective_to"])[:10]
            + "T23:59:59+00:00",
            known_at=grid_date(case, case.cutoff)[:10] + "T00:00:00+00:00",
            created_by=UNVERIFIED_EXTERNAL_CREATOR)
        problems = validate_context_event(event)
        if problems:
            rejected.append({"claim": raw, "reason": "contract",
                             "problems": problems[:3]})
            continue
        spec = pack.context_kinds[kind]
        low, high = spec["bounds"]
        if not low <= claim["value"] <= high:
            rejected.append({"claim": raw, "reason": "implausible_value"})
            continue
        if claim["effective_to"] - claim["effective_from"] \
                > spec["max_span"]:
            rejected.append({"claim": raw, "reason": "effect_span_prior"})
            continue
        admitted.append(ContextItem(
            f"claim-{index}", kind, claim["value"], case.cutoff,
            claim["effective_from"], claim["effective_to"]))
    return {"admitted": admitted, "rejected": rejected, "parsed": parsed}


def score_extraction(raw_claims: list[Any], gate: dict[str, Any],
                     case: Case, pack: DomainPack,
                     shown_values: dict[str, float]) -> dict[str, Any]:
    """Extraction fidelity against the simulator's exact record: the
    ground truth for each rendered memo is the number the text actually
    displayed (``shown_values``) and the item's true window. Matching is
    greedy within kind by window proximity then value agreement."""
    resolved = as_of(case.items, case.cutoff)
    truth = [{"item": item, "shown": shown_values[item.item_id]}
             for item in resolved if item.item_id in shown_values]
    parsed = list(gate["parsed"])
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unmatched_claims = list(parsed)
    for entry in truth:
        item = entry["item"]
        candidates = [claim for claim in unmatched_claims
                      if claim["kind"] == item.kind
                      and abs(claim["effective_from"] - item.effective_from)
                      <= max(5, case.horizon // 2)]
        if not candidates:
            entry["matched"] = None
            continue
        best = min(candidates, key=lambda claim: (
            abs(claim["effective_from"] - item.effective_from),
            abs(claim["value"] - entry["shown"])))
        unmatched_claims.remove(best)
        entry["matched"] = best
        matched.append((entry, best))
    value_errors = [abs(claim["value"] - entry["shown"])
                    / max(abs(entry["shown"]), 1.0)
                    for entry, claim in matched]
    window_errors = [
        (abs(claim["effective_from"] - entry["item"].effective_from)
         + abs(claim["effective_to"] - entry["item"].effective_to)) / 2.0
        for entry, claim in matched]
    hallucinated = len(unmatched_claims)
    admitted_values = {(item.kind, item.value, item.effective_from)
                       for item in gate["admitted"]}
    hallucinated_admitted = sum(
        1 for claim in unmatched_claims
        if (claim["kind"], claim["value"], claim["effective_from"])
        in admitted_values)
    revision_flags = []
    by_id = {item.item_id: item for item in case.items}
    for entry, claim in matched:
        item = entry["item"]
        if not item.trap or item.revises is None:
            continue
        prev = by_id.get(item.revises)
        if prev is None:
            continue
        if abs(round(prev.value, 4) - entry["shown"]) > 1e-9:
            # A value revision: the claim must carry the corrected
            # figure, not the superseded one the text also mentions.
            revision_flags.append(
                abs(claim["value"] - entry["shown"])
                < abs(claim["value"] - round(prev.value, 4)))
        else:
            # A schedule revision (same figure, moved window): the
            # claim must carry the corrected effective window.
            revision_flags.append(
                abs(claim["effective_from"] - item.effective_from)
                < abs(claim["effective_from"] - prev.effective_from))
    return {
        "truth_items": len(truth),
        "claims_parsed": len(parsed),
        "claims_raw": len(raw_claims),
        "matched": len(matched),
        "missed_items": len(truth) - len(matched),
        "value_relative_error": (statistics.mean(value_errors)
                                 if value_errors else None),
        "window_error_steps": (statistics.mean(window_errors)
                               if window_errors else None),
        "hallucinated_claims": hallucinated,
        "hallucinated_admitted": hallucinated_admitted,
        "gate_rejected": len(gate["rejected"]),
        "revision_correct": (statistics.mean(revision_flags)
                             if revision_flags else None),
    }


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

def trap_hidden_reversal(case: Case) -> bool:
    """True when a trap case carries a post-cutoff revision of the trap
    chain whose value sits closer to the superseded (stale) version than
    to the as-of-correct one — a hidden reversal. No legal arm can see
    it, so an arm whose trap accuracy drops specifically on this subset
    is exhibiting the signature of an information-boundary violation:
    leakage becomes a measured quantity, not just a lint."""
    if not case.trap:
        return False
    by_id = {item.item_id: item for item in case.items}
    trap_heads = [item for item in case.items
                  if item.trap and item.known_at <= case.cutoff]
    for hidden in hidden_versions(case.items, case.cutoff):
        if hidden.revises is None:
            continue
        for head in trap_heads:
            if hidden.revises != head.item_id or head.revises is None:
                continue
            stale = by_id.get(head.revises)
            if stale is None:
                continue
            value_to_stale = abs(hidden.value - stale.value)
            value_to_head = abs(hidden.value - head.value)
            if value_to_stale < value_to_head:
                return True
            if value_to_stale == value_to_head and \
                    abs(hidden.effective_from - stale.effective_from) \
                    < abs(hidden.effective_from - head.effective_from):
                # Schedule traps revise the window, not the figure.
                return True
    return False


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
        "trap_hidden_reversal": trap_hidden_reversal(case),
        "truth_event": bool(case.meta.get("truth_event")),
        "outcome_cell": case.meta.get("outcome_cell"),
    }
    truth_step = case.meta.get("truth_first_step")
    answered_step = decision.get("first_event_step")
    row["timing_error"] = (abs(answered_step - truth_step)
                           if truth_step is not None
                           and answered_step is not None else None)
    row["timing_answerable"] = ("first_event_step" in decision
                                and truth_step is not None)
    if pack.extra_metrics is not None:
        row["extras"] = pack.extra_metrics(decision, case)
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
        # answer rate (over valid answers to cases where a first-step
        # answer was possible) is reported next to the error because an
        # arm can dodge the MAE by never naming a step.
        "timing_mae": statistics.mean(timing) if timing else None,
        "timing_answer_rate": (
            len(timing) / len([row for row in valid
                               if row.get("timing_answerable")])
            if any(row.get("timing_answerable") for row in valid)
            else None),
    }
    extra_keys = sorted({key for row in rows
                         for key in (row.get("extras") or {})})
    if extra_keys:
        metrics["extras"] = {
            key: statistics.mean(row["extras"][key] for row in rows
                                 if key in (row.get("extras") or {}))
            for key in extra_keys}
    cells = sorted({row.get("outcome_cell") for row in rows
                    if row.get("outcome_cell")})
    if cells:
        metrics["regret_by_outcome_cell"] = {
            cell: statistics.mean(row["regret"] for row in rows
                                  if row.get("outcome_cell") == cell)
            for cell in cells}
    return metrics


def exact_sign_p(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k)
               for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _paired_bootstrap_ci(deltas: list[float],
                         replicates: int = 2000) -> dict[str, Any] | None:
    """Seeded percentile bootstrap of the mean per-case cost delta — the
    effect size interval the sign test alone does not provide. Seeded
    from the deltas themselves, so identical rows always reproduce the
    identical interval (no wall clock, no global random state)."""
    if not deltas:
        return None
    seed = hashlib.sha256(json.dumps(
        [round(value, 9) for value in deltas]).encode()).hexdigest()[:16]
    rng = random.Random(f"eb-paired-bootstrap:{seed}")
    n = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(replicates))
    return {
        "low_95": round(means[int(0.025 * replicates)], 6),
        "high_95": round(means[int(0.975 * replicates) - 1], 6),
        "replicates": replicates,
        "method": "seeded_paired_percentile_bootstrap",
    }


def paired_cost_comparison(treatment: list[dict[str, Any]],
                           reference: list[dict[str, Any]],
                           ) -> dict[str, Any]:
    """Exact sign test on per-case decision cost (ties dropped, pair
    counts disclosed) plus a seeded bootstrap interval on the mean
    delta. Negative delta means the treatment is cheaper."""
    by_case = {row["case_id"]: row for row in reference}
    treatment_cheaper = reference_cheaper = 0
    deltas: list[float] = []
    # Sorted by case id: rows arrive in thread-completion order, and the
    # bootstrap seed is derived from the delta sequence — an unsorted
    # sequence would make identical runs publish different intervals.
    for row in sorted(treatment, key=lambda entry: entry["case_id"]):
        other = by_case.get(row["case_id"])
        if other is None:
            continue
        deltas.append(row["cost"] - other["cost"])
        if row["cost"] < other["cost"]:
            treatment_cheaper += 1
        elif other["cost"] < row["cost"]:
            reference_cheaper += 1
    return {
        "paired_cases": len(deltas),
        "treatment_cheaper": treatment_cheaper,
        "reference_cheaper": reference_cheaper,
        "ties": len(deltas) - treatment_cheaper - reference_cheaper,
        "exact_sign_p": exact_sign_p(treatment_cheaper, reference_cheaper),
        "mean_cost_delta": (statistics.mean(deltas) if deltas else None),
        "mean_cost_delta_ci": _paired_bootstrap_ci(deltas),
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


def _compiled_row(case: Case, pack: DomainPack, answer: dict[str, Any],
                  cost_pair: tuple[float, float],
                  engine_cache: dict[tuple[str, float], dict[str, Any]],
                  shown_values: dict[str, float]) -> dict[str, Any]:
    """The full agent loop a client actually runs, scored twice: the
    model's own decision, and the governed decision recomputed on the
    claims that survive the production admission gate. The same
    extractions fed in raw (gate bypassed) are also priced, so the
    gate's worth is a number in the domain's own units."""
    gate = admit_claims(answer["claims"], case, pack)
    admitted_inputs = pack.engine_inputs(case, gate["admitted"])
    governed = pack.engine_decision(
        case, compute_engine_packet(case, admitted_inputs, cost_pair,
                                    engine_cache), admitted_inputs)
    raw_items = [ContextItem(f"raw-{index}", claim["kind"], claim["value"],
                             case.cutoff, claim["effective_from"],
                             claim["effective_to"])
                 for index, claim in enumerate(gate["parsed"])]
    raw_inputs = pack.engine_inputs(case, raw_items)
    raw_governed = pack.engine_decision(
        case, compute_engine_packet(case, raw_inputs, cost_pair,
                                    engine_cache), raw_inputs)
    row = score_decision_row(governed, answer["claims_valid"], case, pack)
    model_own = score_decision_row(answer["decision"],
                                   answer["decision_valid"], case, pack)
    raw_outcome = pack.cost_model.score(raw_governed, case)
    row.update({
        "claims_valid": answer["claims_valid"],
        "decision_valid": answer["decision_valid"],
        "model_own_cost": model_own["cost"],
        "model_own_regret": model_own["regret"],
        "model_own_action_optimal": model_own["action_optimal"],
        "model_own_trap_correct": model_own["trap_correct"],
        "raw_governed_cost": round(raw_outcome["cost"], 6),
        "raw_governed_regret": round(raw_outcome["regret"], 6),
        "admitted_claims": len(gate["admitted"]),
        "rejected_claims": len(gate["rejected"]),
        "rejection_reasons": sorted({entry["reason"]
                                     for entry in gate["rejected"]}),
        "extraction": score_extraction(answer["claims"], gate, case, pack,
                                       shown_values),
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
    text_only_items = 0
    for index, case in enumerate(cases):
        resolved = as_of(case.items, case.cutoff)
        # The engine consumes structured context only; items marked
        # text-only exist solely in the memos (disclosed) and can be
        # recovered for the governed path only through extraction.
        structured = [item for item in resolved if not item.text_only]
        text_only_items += len(resolved) - len(structured)
        inputs = pack.engine_inputs(case, structured)
        engine_inputs_by_case[case.case_id] = inputs
        packet = compute_engine_packet(case, inputs, cost_pair, engine_cache)
        packets[case.case_id] = packet
        engine_decisions[case.case_id] = pack.engine_decision(
            case, packet, inputs)
        if (index + 1) % 25 == 0:
            print(f"{pack.name}: engine runs {index + 1}/{len(cases)}",
                  flush=True)

    context_blocks: dict[str, str] = {}
    shown_values_by_case: dict[str, dict[str, float]] = {}
    for case in cases:
        block, shown = text_context(case)
        context_blocks[case.case_id] = block
        shown_values_by_case[case.case_id] = shown
    prompts = {(case.case_id, arm): prompt_for(
        case, pack, arm, packets[case.case_id],
        context_blocks[case.case_id])
        for case in cases for arm in MODEL_ARMS}
    verify_arm_symmetry(cases, pack, prompts, context_blocks)
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
        cases, packets, pack.season_length)

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
        if arm == "model_facts_compiled":
            answer = parse_compiled_answer(text, pack, case)
            return {**base, **_compiled_row(
                case, pack, answer, cost_pair, engine_cache,
                shown_values_by_case[case.case_id])}
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

    provenance = {**provenance,
                  "text_only_items_across_cases": text_only_items}
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

    Explicit roles only — inferring which cost is the act side from
    magnitudes would silently invert the ladder's break-even the day a
    domain prices mitigation above the miss. A binary pack without the
    declaration is a configuration error; quantity packs never reach the
    ladder (their engine runs without a threshold), so the placeholder
    is inert by construction."""
    pair = pack.cost_model.governed_pair
    if pair is not None:
        return pair
    if pack.decision_kind == "binary":
        raise ValueError(
            f"{pack.name}: a binary pack must declare "
            "cost_model.governed_pair (use binary_cost_model)")
    return (1.0, 2.0)


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
                             season: int) -> dict[str, Any]:
    mases, pinballs = [], {"q10": [], "q50": [], "q90": []}
    for case in cases:
        packet = packets[case.case_id]
        rows = packet.get("forecast") or []
        if len(rows) != case.horizon or packet.get("basis") != "shown_series":
            continue
        actual = list(case.future)
        score = mase([row["q50"] for row in rows], actual,
                     list(case.values), season)
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

    compiled_rows = arm_rows.get("model_facts_compiled") or []
    if compiled_rows:
        metrics["model_facts_compiled"].update(
            _compiled_metrics(compiled_rows))

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
        "compiled_vs_oracle_gap": (
            {"gap_mean_cost":
                metrics["model_facts_compiled"]["mean_cost"]
                - metrics["model_facts_oracle"]["mean_cost"],
             **paired_cost_comparison(compiled_rows,
                                      arm_rows["model_facts_oracle"])}
            if compiled_rows and arm_rows["model_facts_oracle"] else None),
        "admission_value": (_admission_value(compiled_rows)
                            if compiled_rows else None),
        "text_pipeline_integrity": (
            _text_pipeline_integrity(metrics, compiled_rows)
            if compiled_rows else None),
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
        "trap_integrity": _trap_integrity(arm_rows, engine_rows),
        "reading": {
            "compiled_vs_oracle_gap": (
                "the cost of imperfect extraction: the governed decision "
                "on the model's own extraction versus the model given "
                "oracle engine facts — remaining headroom for better "
                "extraction"),
            "admission_value": (
                "what the production admission gate is worth in this "
                "domain's units: gated governed cost versus the same "
                "extractions fed in raw"),
            "text_pipeline_integrity": (
                "the compiled arm may only be claimed viable when "
                "extraction fidelity, hallucination rejection, and the "
                "compiled-vs-oracle gap are published together; a good "
                "cost number with unpublished extraction fidelity is "
                "not a result"),
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
                "arm resolving revisions as of the cutoff scores high, "
                "and a drop concentrated on the hidden-reversal subset "
                "(which no legal arm can see) is the signature of "
                "post-cutoff leakage"),
        },
    }
    verdicts["useful"] = _useful_verdict(verdicts)
    event_rate = (statistics.mean(
        bool(case.meta.get("truth_event")) for case in cases)
        if pack.decision_kind == "binary" else None)
    return {
        "schema_version": GENERATOR_VERSION,
        "domain": pack.name,
        "seed": args.seed, "cases": args.cases,
        # Seeds 9xxxxxxx are frozen for validation; everything else is
        # development, and its numbers are diagnostic by construction.
        # Stamped mechanically so a diagnostic run cannot be quietly
        # presented as validation.
        "scope": ("validation"
                  if 90_000_000 <= args.seed < 100_000_000
                  else "diagnostic"),
        "model": model_name, "temperature": 0,
        "cost_model": {"names": pack.cost_model.names,
                       "break_even": pack.cost_model.break_even,
                       "units": pack.config.get("units", "domain_units"),
                       # Base rates are held near the break-even so
                       # constant policies cannot masquerade as skill;
                       # the achieved rate is disclosed, not assumed.
                       "achieved_event_rate": event_rate,
                       "event_rate_minus_break_even": (
                           round(event_rate - pack.cost_model.break_even,
                                 6)
                           if event_rate is not None else None)},
        "statistics": {
            "primary_endpoint": "per_case_decision_cost",
            "paired_test": ("exact two-sided sign test; ties dropped "
                            "and disclosed next to the pair counts"),
            "effect_interval": ("seeded paired percentile bootstrap, "
                                "95%, on the mean per-case cost delta"),
            "multiple_comparisons": (
                "p-values are per-comparison; no family-wise "
                "correction is applied. The useful verdict is a "
                "conjunction of three point estimates, not a pooled "
                "test — read it with the per-comparison intervals."),
            "caveat": ("cases are independent simulated series, but "
                       "labels can co-move through shared parameter "
                       "ranges; see provenance.independence"),
        },
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
            "extraction_ground_truth": (
                "the number the rendered text actually shows (vague "
                "renderings round to two significant figures, so their "
                "shown number is the target); decision truth always "
                "uses the precise structured value"),
            "context_noise": (
                "one to three seeded pure-noise distractor memos per "
                "case, interleaved by date; a claim extracted from one "
                "matches no simulator fact and scores as a "
                "hallucination"),
            "data_freshness": (
                "the corpus is a pure function of (seed, simulator "
                "config); any unused seed regenerates entirely new "
                "series, futures, facts, and memos — there is no "
                "static dataset to memorize or overfit"),
            "independence": (
                "each case simulates an independent series (per-series "
                "case counts in provenance); labels can still co-move "
                "through shared regime parameters within a domain — a "
                "caveat, not an independence claim"),
        },
    }


def _useful_verdict(verdicts: dict[str, Any]) -> dict[str, Any]:
    """The suite's headline question, computed rather than narrated:
    is the treatment arm cheaper than the model alone AND the engine
    alone AND the best constant policy? Point-estimate conjunction, with
    each component's sign test and bootstrap interval carried along so
    a reader can judge the evidence, not just the direction."""
    components = {}
    for key in ("vs_model_alone", "vs_engine_alone",
                "vs_best_constant_policy"):
        comparison = verdicts[key]
        delta = comparison.get("mean_cost_delta")
        ci = comparison.get("mean_cost_delta_ci")
        components[key] = {
            "mean_cost_delta": delta,
            "treatment_cheaper_on_average": (delta is not None
                                             and delta < 0),
            # Primary evidence: the bootstrap interval, because ties
            # blunt the sign test in binary domains — a point-estimate
            # win whose interval straddles zero is direction, not proof.
            "ci_excludes_zero": bool(ci and (ci["high_95"] < 0
                                             or ci["low_95"] > 0)),
            "exact_sign_p": comparison.get("exact_sign_p"),
            "mean_cost_delta_ci": ci,
        }
    return {
        "all_three_cheaper": all(
            entry["treatment_cheaper_on_average"]
            for entry in components.values()),
        "all_three_ci_excluding_zero": all(
            entry["ci_excludes_zero"] for entry in components.values()),
        "components": components,
        "note": ("a conjunction of paired point estimates in this "
                 "domain's own units; diagnostic until run on a frozen "
                 "validation seed"),
    }


def _treatment_arm(arm_rows: dict[str, list[dict[str, Any]]]) -> str:
    """The arm the domain verdicts judge. Once the compiled text arm
    exists it is the treatment (it is the loop a client actually runs);
    until then the oracle-facts arm stands in, labelled as such."""
    if arm_rows.get("model_facts_compiled"):
        return "model_facts_compiled"
    return "model_facts_oracle"


def _mean_of(rows: list[dict[str, Any]], key: str,
             nested: str | None = None) -> float | None:
    values = []
    for row in rows:
        source = row.get(nested) or {} if nested else row
        value = source.get(key)
        if value is not None:
            values.append(value)
    return statistics.mean(values) if values else None


def _compiled_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extraction scored in its own right (exact ground truth from the
    generated text) alongside the model's own decision — the compiled
    arm's headline cost is the governed decision on admitted claims."""
    hallucinated = sum(row["extraction"]["hallucinated_claims"]
                       for row in rows)
    hallucinated_admitted = sum(
        row["extraction"]["hallucinated_admitted"] for row in rows)
    return {
        "model_own": {
            "mean_cost": _mean_of(rows, "model_own_cost"),
            "mean_regret": _mean_of(rows, "model_own_regret"),
            "action_optimal_rate": statistics.mean(
                row["model_own_action_optimal"] for row in rows),
            "decision_invalid_rate": statistics.mean(
                not row["decision_valid"] for row in rows),
            # The model's own revision handling, separate from the
            # governed path built on its claims.
            "trap_accuracy": _mean_of(rows, "model_own_trap_correct"),
        },
        "extraction": {
            "value_relative_error": _mean_of(rows, "value_relative_error",
                                             "extraction"),
            "window_error_steps": _mean_of(rows, "window_error_steps",
                                           "extraction"),
            "missed_rate": (
                sum(row["extraction"]["missed_items"] for row in rows)
                / max(1, sum(row["extraction"]["truth_items"]
                             for row in rows))),
            "hallucinated_claims": hallucinated,
            "hallucination_admission_rate": (
                hallucinated_admitted / hallucinated
                if hallucinated else None),
            "revision_correct_rate": _mean_of(rows, "revision_correct",
                                              "extraction"),
            "claims_invalid_rate": statistics.mean(
                not row["claims_valid"] for row in rows),
        },
    }


def _admission_value(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_reference = [{"case_id": row["case_id"],
                      "cost": row["raw_governed_cost"]} for row in rows]
    return {
        "gated_mean_cost": _mean_of(rows, "cost"),
        "raw_mean_cost": _mean_of(rows, "raw_governed_cost"),
        **paired_cost_comparison(rows, raw_reference),
    }


def _text_pipeline_integrity(metrics: dict[str, Any],
                             rows: list[dict[str, Any]]) -> dict[str, Any]:
    extraction = metrics["model_facts_compiled"].get("extraction") or {}
    required = ("value_relative_error", "missed_rate",
                "hallucinated_claims", "hallucination_admission_rate",
                "revision_correct_rate")
    return {
        "published_together": all(key in extraction for key in required),
        "extraction_fidelity": {key: extraction.get(key)
                                for key in required},
        "note": ("a compiled-arm cost claim without these fields "
                 "published alongside it is not a result"),
    }


def _trap_split(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    trap_rows = [row for row in rows
                 if row.get("trap_correct") is not None]
    if not trap_rows:
        return None
    with_hidden = [row for row in trap_rows
                   if row.get("trap_hidden_reversal")]
    without = [row for row in trap_rows
               if not row.get("trap_hidden_reversal")]
    entry: dict[str, Any] = {
        "trap_cases": len(trap_rows),
        "trap_accuracy": statistics.mean(
            row["trap_correct"] for row in trap_rows),
        # The measured leakage signature: no legal arm can see the
        # hidden reversal, so its accuracy on this subset should match
        # the rest. A drop concentrated here means the arm is following
        # post-cutoff versions.
        "hidden_reversal_cases": len(with_hidden),
        "trap_accuracy_hidden_reversal": (
            statistics.mean(row["trap_correct"] for row in with_hidden)
            if with_hidden else None),
        "trap_accuracy_no_hidden_reversal": (
            statistics.mean(row["trap_correct"] for row in without)
            if without else None),
    }
    return entry


def _trap_integrity(arm_rows: dict[str, list[dict[str, Any]]],
                    engine_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_arm: dict[str, Any] = {
        arm: _trap_split(rows) for arm, rows in arm_rows.items()}
    per_arm["engine"] = _trap_split(engine_rows)
    return per_arm
