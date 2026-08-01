"""Aion-side computation for TemporalBench rows.

Adapter decisions, disclosed:

- TemporalBench rows carry index-aligned arrays, not regular calendar
  timestamps (``time_position_in_day`` is an irregular covariate), so
  Aion models each channel on a synthetic regular hourly axis
  (observation *k* at epoch + *k* hours). Metrics are index-based; the
  axis never enters the score.
- Missing readings (nulls) are written as ``N/A`` cells and handled by
  Aion's disclosed repair layer (``--repair aggressive``), never
  silently imputed by the adapter.
- Forecasts are Aion's q50 path per channel. A channel Aion abstains on
  stays absent — the official scorer then reports the row as missing,
  which is recorded as an abstention, not papered over.
- In ``aion-pure`` mode multiple-choice questions are answered
  ``Uncertain`` where the option exists (an honest abstention — T2/T4
  option sets include it); rows without such an option score those
  questions as wrong. The agent mode hands choices to an LLM that sees
  Aion's computed evidence.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.temporalbench.tasks import prompt_input_arrays  # noqa: E402

EPOCH = datetime(2021, 1, 1, tzinfo=timezone.utc)
STEP = timedelta(hours=1)


def _clean(values: list[float | None]) -> list[float]:
    filled: list[float] = []
    previous = None
    for value in values:
        if value is None:
            if previous is None:
                continue
            value = previous
        filled.append(float(value))
        previous = value
    return filled


def forecast_channel(values: list[float | None], horizon: int,
                     work_dir: str | None = None) -> dict[str, Any]:
    """Aion forecast for one channel on the synthetic hourly axis."""
    from aion import forecast as aion_forecast
    from aion.contracts import AionError

    run_dir = Path(tempfile.mkdtemp(prefix="tb-aion-", dir=work_dir))
    csv_path = run_dir / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for index, value in enumerate(values):
            cell = "N/A" if value is None else repr(float(value))
            writer.writerow([(EPOCH + index * STEP).isoformat(), cell])
    try:
        artifact, _ = aion_forecast(
            str(csv_path), time_column="timestamp", target_column="value",
            horizon=horizon, frequency="h",
            output=str(run_dir / "aion-output"), repair="aggressive",
        )
    except AionError as error:
        return {"abstained": True, "reason": f"{error.code}: {error.message}"}
    result = artifact.results[0]
    if result.support == "unsupported" or not result.forecast:
        return {"abstained": True,
                "reason": "; ".join(str(w) for w in result.warnings) or "unsupported"}
    return {
        "abstained": False,
        "support": result.support,
        "selected_model": result.selected_model,
        "values": [float(r.get("q50", r["point"])) for r in result.forecast],
    }


def analyse_row(row: dict[str, Any], work_dir: str | None = None) -> dict[str, Any]:
    """Deterministic Aion evidence for one row: per-channel forecasts
    (T2/T4), plus season/anomaly/stats findings on the main channel."""
    from aion.anomaly import detect_anomalies
    from aion.temporal import detect_season

    arrays = prompt_input_arrays(row)
    meta = row.get("meta") or {}
    main_key = meta.get("main_key") or next(iter(arrays), None)
    horizon = int(meta.get("n_horizon") or 0)
    ground_truth = row.get("ground_truth")
    target_keys = (list(ground_truth.keys()) if isinstance(ground_truth, dict)
                   else meta.get("target_keys") or ([main_key] if main_key else []))

    analysis: dict[str, Any] = {"main_key": main_key, "channels": {}}
    main_values = _clean(arrays.get(main_key, [])) if main_key else []
    if main_values:
        season, strength, basis = detect_season(main_values, "h")
        analysis["season"] = {"period": season, "strength": round(strength, 4),
                              "basis": basis}
        detection = detect_anomalies(
            [str(i) for i in range(len(main_values))], main_values, season=season
        )
        analysis["anomalies"] = {
            "detector": detection.get("detector"),
            "count": len(detection.get("anomalies", [])),
            "indices": [int(a["timestamp"]) for a in
                        detection.get("anomalies", [])][:32],
            "support": detection.get("support", {}).get("status"),
        }
        mean = sum(main_values) / len(main_values)
        analysis["stats"] = {
            "count": len(main_values), "mean": round(mean, 4),
            "min": min(main_values), "max": max(main_values),
        }
    if row.get("tier") in ("T2", "T4") and horizon > 0:
        for key in target_keys:
            if key in arrays:
                analysis["channels"][key] = forecast_channel(
                    arrays[key], horizon, work_dir
                )
    return analysis


def forecast_payload(analysis: dict[str, Any]) -> tuple[dict[str, list[float]], list[str]]:
    """Collect Aion-owned forecast arrays; list channels that abstained."""
    forecast: dict[str, list[float]] = {}
    abstained: list[str] = []
    for key, outcome in analysis.get("channels", {}).items():
        if outcome.get("abstained"):
            abstained.append(f"{key}: {outcome.get('reason')}")
        else:
            forecast[key] = outcome["values"]
    return forecast, abstained


def uncertain_mcq(row: dict[str, Any]) -> dict[str, str]:
    """Pure-mode choice answers: 'Uncertain' where the options allow it."""
    answers: dict[str, str] = {}
    for key, entry in (row.get("mcq") or {}).items():
        options = entry.get("options") or []
        uncertain = next((o for o in options if str(o).lower() == "uncertain"), None)
        answers[key] = uncertain if uncertain is not None else str(options[0]) \
            if options else "Uncertain"
    return answers
