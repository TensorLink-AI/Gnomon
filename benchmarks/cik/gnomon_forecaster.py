"""Treatment condition for CiK: Gnomon owns the numbers.

Two modes, both scored by the untouched official RCRPS metric:

``pure``
    No LLM at all. The task's numeric history is handed to
    ``gnomon.forecast`` and the textual context is deliberately ignored.
    This measures the harness floor — what a backtested deterministic
    selection achieves with zero context.

``agent``
    The "LLM proposes; Gnomon validates and computes" contract. An
    OpenRouter model reads the task's textual context (background,
    constraints, scenario) plus non-numeric metadata about the history
    window, and proposes *typed context events only* — never numbers.
    The events are handed to Gnomon, whose deterministic admission gate
    (identical-fold ablation, known-at gating, stability checks) decides
    whether they may influence the forecast at all.

Adapter decisions, disclosed:

- CiK provides three quantities per horizon step once Gnomon has run:
  q10/q50/q90. RCRPS is estimated from samples, so samples are drawn
  deterministically from the piecewise-linear inverse CDF through those
  quantiles, with flat tails clamped at q10/q90 (conservative: no
  invented tail spread). See :func:`samples_from_quantile_rows`.
- CiK context text is benchmark-supplied ground truth available at
  forecast time, so proposed events carry a verifiable source of type
  ``dataset`` referencing the task's context, with ``known_at`` set to
  the start of the history window. Gnomon's gate still decides admission.
- CiK task indexes are timezone-naive; they are written as UTC because
  Gnomon's context path requires timezone-aware timestamps.
- When Gnomon abstains, the adapter raises :class:`GnomonAbstained` rather
  than fabricating samples; the run is recorded as an abstention, and
  scores simply do not exist for it. The most dangerous forecast is the
  confident one that shouldn't exist.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.openrouter import OpenRouterClient, extract_json_array  # noqa: E402

QUANTILE_LEVELS = (0.1, 0.5, 0.9)
MAX_EVENTS = 8

EVENT_PROPOSAL_INSTRUCTIONS = """\
You assist a deterministic forecasting engine. Read the task context and
propose the calendar events it describes or clearly implies. You must NOT
predict numbers; the engine computes all values itself and will validate
every event you propose against held-out data before using it.

Return ONLY a JSON array (possibly empty) of at most {max_events} events:
[
  {{
    "event_type": "<short snake_case label, e.g. maintenance_window>",
    "effective_start": "<ISO-8601 timestamp with timezone, e.g. 2024-05-01T00:00:00+00:00>",
    "effective_end": "<ISO-8601 timestamp with timezone>",
    "confidence": <0.0-1.0>,
    "rationale": "<one sentence quoting or paraphrasing the context>"
  }}
]

Rules:
- Only events stated or directly implied by the context. Do not invent.
- Timestamps must lie within [{window_start}, {window_end}] and carry an
  explicit timezone offset.
- An instantaneous event may use the same start and end.
- If the context describes no time-locatable event, return [].
"""


class GnomonAbstained(RuntimeError):
    """Gnomon refused to produce a forecast; there are no numbers to score."""

    def __init__(self, reasons: list[str]):
        super().__init__("GNOMON_ABSTAINED: " + "; ".join(reasons))
        self.reasons = reasons


def samples_from_quantile_rows(
    rows: list[dict[str, Any]], n_samples: int
) -> list[list[float]]:
    """Deterministic samples from per-step (q10, q50, q90) quantiles.

    Uses stratified probabilities ``(i + 0.5) / n_samples`` mapped through
    the piecewise-linear inverse CDF defined by the three quantiles.
    Probabilities beyond the outer quantiles clamp to q10/q90, which
    understates tail spread rather than inventing it.

    Returns a list of ``n_samples`` trajectories, each of length
    ``len(rows)``.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    quantiles: list[tuple[float, float, float]] = []
    for row in rows:
        q50 = float(row.get("q50", row["point"]))
        q10 = float(row.get("q10", q50))
        q90 = float(row.get("q90", q50))
        low, mid, high = sorted((q10, q50, q90))
        quantiles.append((low, mid, high))

    def inverse_cdf(p: float, low: float, mid: float, high: float) -> float:
        if p <= QUANTILE_LEVELS[0]:
            return low
        if p >= QUANTILE_LEVELS[2]:
            return high
        if p <= QUANTILE_LEVELS[1]:
            fraction = (p - QUANTILE_LEVELS[0]) / (QUANTILE_LEVELS[1] - QUANTILE_LEVELS[0])
            return low + fraction * (mid - low)
        fraction = (p - QUANTILE_LEVELS[1]) / (QUANTILE_LEVELS[2] - QUANTILE_LEVELS[1])
        return mid + fraction * (high - mid)

    samples = []
    for index in range(n_samples):
        probability = (index + 0.5) / n_samples
        samples.append(
            [inverse_cdf(probability, low, mid, high) for low, mid, high in quantiles]
        )
    return samples


def build_context_text(task_instance: Any) -> str:
    parts = []
    if getattr(task_instance, "background", None):
        parts.append(f"Background: {task_instance.background}")
    if getattr(task_instance, "constraints", None):
        parts.append(f"Constraints: {task_instance.constraints}")
    if getattr(task_instance, "scenario", None):
        parts.append(f"Scenario: {task_instance.scenario}")
    return "\n".join(parts)


def events_from_proposals(
    proposals: list[Any],
    *,
    task_name: str,
    known_at: str,
    window_start: str,
    window_end: str,
) -> tuple[list[Any], list[str]]:
    """Convert LLM proposals into validated ``ContextEvent`` objects.

    Malformed proposals are dropped with a note instead of failing the
    run: a bad proposal is the LLM's failure, and the treatment's answer
    to that is falling back to the history-only forecast.
    """
    from datetime import datetime

    from gnomon.context import ContextEvent, ContextSource, validate_context_event

    window_start_dt = datetime.fromisoformat(window_start)
    window_end_dt = datetime.fromisoformat(window_end)
    events: list[Any] = []
    notes: list[str] = []
    for number, raw in enumerate(proposals[:MAX_EVENTS], start=1):
        if not isinstance(raw, dict):
            notes.append(f"proposal {number} is not an object")
            continue
        try:
            confidence = float(raw.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        event = ContextEvent(
            event_id=f"cik-{task_name}-evt{number}",
            event_type=str(raw.get("event_type", "")).strip() or "context_event",
            entity_scope=("*",),
            effective_start=str(raw.get("effective_start", "")),
            effective_end=str(raw.get("effective_end", "")),
            known_at=known_at,
            confidence=min(max(confidence, 0.0), 1.0),
            attributes={"rationale": str(raw.get("rationale", ""))[:500]},
            source=ContextSource("dataset", f"cik:{task_name}#context"),
            created_by="llm",
        )
        problems = validate_context_event(event)
        if problems:
            notes.append(f"proposal {number} rejected: {'; '.join(problems)}")
            continue
        try:
            start = datetime.fromisoformat(event.effective_start)
            end = datetime.fromisoformat(event.effective_end)
        except ValueError:
            notes.append(f"proposal {number} rejected: unparseable timestamps")
            continue
        if end < window_start_dt or start > window_end_dt:
            notes.append(f"proposal {number} rejected: outside the task window")
            continue
        events.append(event)
    return events, notes


class GnomonForecaster:
    """A CiK ``Baseline``-compatible callable backed by ``gnomon.forecast``.

    Implements the official baseline interface (``__call__(task_instance,
    n_samples)`` returning ``(samples, extra_info)`` with samples shaped
    ``[n_samples, horizon, 1]`` and a ``cache_name`` property) without
    importing the heavy benchmark package at module load.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        mode: str = "pure",
        openrouter_model: str | None = None,
        *,
        temperature: float = 1.0,
        work_dir: str | None = None,
    ) -> None:
        if mode not in ("pure", "agent"):
            raise ValueError("mode must be 'pure' or 'agent'")
        if mode == "agent" and not openrouter_model:
            raise ValueError("mode='agent' requires an OpenRouter model id")
        self.mode = mode
        self.openrouter_model = openrouter_model
        self.client = (
            OpenRouterClient(openrouter_model, temperature=temperature)
            if mode == "agent"
            else None
        )
        self.work_dir = work_dir

    # -- CiK Baseline interface -------------------------------------------
    @property
    def cache_name(self) -> str:
        model = (self.openrouter_model or "none").replace("/", "-")
        return f"GnomonForecaster_mode={self.mode}_model={model}"

    def __str__(self) -> str:
        return self.cache_name

    def __call__(self, task_instance: Any, n_samples: int):
        import numpy as np

        started = time.time()
        history = self._history_frame(task_instance)
        horizon = len(task_instance.future_time)
        run_dir = Path(
            tempfile.mkdtemp(prefix="cik-gnomon-", dir=self.work_dir)
        )
        csv_path = run_dir / "history.csv"
        timestamps = self._write_history_csv(history, csv_path)

        events: list[Any] = []
        proposal_notes: list[str] = []
        if self.mode == "agent":
            events, proposal_notes = self._propose_events(
                task_instance, timestamps, horizon
            )

        from gnomon import forecast as gnomon_forecast
        from gnomon.contracts import GnomonError

        try:
            artifact, _ = gnomon_forecast(
                str(csv_path),
                time_column="timestamp",
                target_column="value",
                horizon=horizon,
                output=str(run_dir / "gnomon-output"),
                context_events=events or None,
            )
        except GnomonError as error:
            raise GnomonAbstained([f"{error.code}: {error.message}"]) from error

        result = artifact.results[0]
        if result.support == "unsupported" or not result.forecast:
            reasons = [str(w) for w in result.warnings] or ["unsupported series"]
            raise GnomonAbstained(reasons)

        samples = np.asarray(
            samples_from_quantile_rows(result.forecast, n_samples), dtype=float
        )[:, :, None]

        extra_info = {
            "adapter": self.cache_name,
            "support": result.support,
            "selected_model": result.selected_model,
            "strongest_baseline": result.strongest_baseline,
            "warnings": list(result.warnings),
            "context": result.context,
            "proposed_events": [event.event_id for event in events],
            "proposal_notes": proposal_notes,
            "total_time": time.time() - started,
        }
        if self.client is not None:
            extra_info["llm_usage"] = self.client.usage_summary
        return samples, extra_info

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _history_frame(task_instance: Any):
        import pandas as pd

        frame = task_instance.past_time
        index = frame.index
        if isinstance(index, pd.PeriodIndex):
            index = index.to_timestamp()
        index = pd.DatetimeIndex(index)
        if index.tz is None:
            index = index.tz_localize("UTC")
        values = frame[frame.columns[-1]].astype(float).values
        return pd.DataFrame({"value": values}, index=index)

    @staticmethod
    def _write_history_csv(history, csv_path: Path) -> list[str]:
        timestamps = [ts.isoformat() for ts in history.index]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "value"])
            for timestamp, value in zip(timestamps, history["value"].values):
                writer.writerow([timestamp, repr(float(value))])
        return timestamps

    def _propose_events(
        self, task_instance: Any, timestamps: list[str], horizon: int
    ) -> tuple[list[Any], list[str]]:
        import pandas as pd

        context = build_context_text(task_instance)
        if not context.strip():
            return [], ["task has no textual context"]

        future_index = task_instance.future_time.index
        if isinstance(future_index, pd.PeriodIndex):
            future_index = future_index.to_timestamp()
        future_index = pd.DatetimeIndex(future_index)
        if future_index.tz is None:
            future_index = future_index.tz_localize("UTC")
        window_start = timestamps[0]
        window_end = future_index[-1].isoformat()

        instructions = EVENT_PROPOSAL_INSTRUCTIONS.format(
            max_events=MAX_EVENTS,
            window_start=window_start,
            window_end=window_end,
        )
        prompt = (
            f"{instructions}\n"
            f"History window: {timestamps[0]} to {timestamps[-1]} "
            f"({len(timestamps)} observations).\n"
            f"Forecast window: {future_index[0].isoformat()} to {window_end} "
            f"({horizon} steps).\n\n"
            f"Task context:\n{context}\n"
        )
        try:
            completion = self.client.completions(
                [{"role": "user", "content": prompt}], n=1
            )[0]
            proposals = extract_json_array(completion)
        except ValueError:
            return [], ["no JSON array in event-proposal output"]
        except Exception as error:  # network failures fall back to history-only
            return [], [f"event proposal failed: {error}"]
        return events_from_proposals(
            proposals,
            task_name=task_instance.name,
            known_at=window_start,
            window_start=window_start,
            window_end=window_end,
        )
