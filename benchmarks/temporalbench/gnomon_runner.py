"""Gnomon-side computation for TemporalBench rows.

Adapter decisions, disclosed:

- TemporalBench rows carry index-aligned arrays, not regular calendar
  timestamps (``time_position_in_day`` is an irregular covariate), so
  Gnomon models each channel on a synthetic regular hourly axis
  (observation *k* at epoch + *k* hours). Metrics are index-based; the
  axis never enters the score.
- Missing readings (nulls) are omitted, and the readings that exist are
  laid out consecutively on that axis. TemporalBench's clinical
  channels are recorded irregularly, so a dense hourly grid would be
  >30% holes — asking Gnomon to invent most of a series (its repair cap
  rightly refuses at 30%, which abstained on 48 of 50 rows) instead of
  forecasting the readings actually taken. The missingness was the
  adapter's own construction, not the data's. Targets are the next *H*
  entries either way, so the alignment of the forecast is unchanged.
- Forecasts are Gnomon's q50 path per channel. A channel Gnomon abstains on
  stays absent — the official scorer then reports the row as missing,
  which is recorded as an abstention, not papered over.
- In ``gnomon-pure`` mode multiple-choice questions are answered
  ``Uncertain`` where the option exists (an honest abstention — T2/T4
  option sets include it); rows without such an option score those
  questions as wrong. The agent mode hands choices to an LLM that sees
  Gnomon's computed evidence.
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


def _observed(values: list[float | None]) -> list[float]:
    """The readings that exist, in order. Nulls are absent, not filled:
    forward-filling would flatten real variation and hand the anomaly
    and season detectors runs of values nobody recorded."""
    return [float(value) for value in values if value is not None]


def forecast_channel(values: list[float | None], horizon: int,
                     work_dir: str | None = None) -> dict[str, Any]:
    """Gnomon forecast for one channel on the synthetic hourly axis."""
    from gnomon import forecast as gnomon_forecast
    from gnomon.contracts import GnomonError

    run_dir = Path(tempfile.mkdtemp(prefix="tb-gnomon-", dir=work_dir))
    csv_path = run_dir / "history.csv"
    observed = _observed(values)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for position, value in enumerate(observed):
            writer.writerow([(EPOCH + position * STEP).isoformat(), repr(value)])
    try:
        artifact, _ = gnomon_forecast(
            str(csv_path), time_column="timestamp", target_column="value",
            horizon=horizon, frequency="h",
            output=str(run_dir / "gnomon-output"),
        )
    except GnomonError as error:
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


def forecast_channels(
    channels: dict[str, list[float | None]], horizon: int,
    work_dir: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Batched Gnomon forecasts for several channels in ONE invocation.

    One wide CSV, one shared load pass, per-channel evaluation run
    concurrently by ``forecast_multi`` — each channel's numbers are
    identical to a ``forecast_channel`` call (the parity is pinned by
    Gnomon's own tests), so the benchmark metrics are unchanged; only the
    wall clock is. Channels keep their own consecutive synthetic axes:
    a shorter channel's trailing cells are blank, which the safe repair
    level drops as absent observations (disclosed, non-assumptive), so
    the values each channel models are exactly the single-call ones. An
    abstaining channel reports its own reason and never blocks the rest.
    """
    from gnomon.contracts import GnomonError
    from gnomon.runtime import forecast_multi

    observed = {key: _observed(values) for key, values in channels.items()}
    keys = list(observed)
    length = max((len(values) for values in observed.values()), default=0)
    if len(keys) < 2 or length == 0 or "timestamp" in keys:
        return {key: forecast_channel(channels[key], horizon, work_dir)
                for key in channels}
    run_dir = Path(tempfile.mkdtemp(prefix="tb-gnomon-", dir=work_dir))
    csv_path = run_dir / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp"] + keys)
        for position in range(length):
            writer.writerow(
                [(EPOCH + position * STEP).isoformat()]
                + [repr(observed[key][position]) if position < len(observed[key])
                   else "" for key in keys]
            )
    try:
        artifact, _ = forecast_multi(
            str(csv_path), time_column="timestamp", target_columns=keys,
            horizon=horizon, frequency="h",
            output=str(run_dir / "gnomon-output"),
        )
    except GnomonError as error:
        reason = f"{error.code}: {error.message}"
        return {key: {"abstained": True, "reason": reason} for key in keys}
    outcomes: dict[str, dict[str, Any]] = {}
    for result in artifact.results:
        if result.support == "unsupported" or not result.forecast:
            outcomes[result.series] = {
                "abstained": True,
                "reason": "; ".join(str(w) for w in result.warnings) or "unsupported",
            }
        else:
            outcomes[result.series] = {
                "abstained": False,
                "support": result.support,
                "selected_model": result.selected_model,
                "values": [float(r.get("q50", r["point"])) for r in result.forecast],
            }
    return outcomes


def analyse_row(row: dict[str, Any], work_dir: str | None = None) -> dict[str, Any]:
    """Deterministic Gnomon evidence for one row: per-channel forecasts
    (T2/T4), plus season/anomaly/stats findings on the main channel."""
    from gnomon.anomaly import detect_anomalies
    from gnomon.temporal import detect_season

    arrays = prompt_input_arrays(row)
    meta = row.get("meta") or {}
    main_key = meta.get("main_key") or next(iter(arrays), None)
    horizon = int(meta.get("n_horizon") or 0)
    ground_truth = row.get("ground_truth")
    target_keys = (list(ground_truth.keys()) if isinstance(ground_truth, dict)
                   else meta.get("target_keys") or ([main_key] if main_key else []))

    analysis: dict[str, Any] = {"main_key": main_key, "channels": {}}
    main_values = _observed(arrays.get(main_key, [])) if main_key else []
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
        # One batched invocation for every channel instead of one run per
        # channel: same per-channel numbers, one shared load, concurrent
        # evaluation. forecast_channels falls back to the single-channel
        # path when only one channel needs forecasting.
        wanted = {key: arrays[key] for key in target_keys if key in arrays}
        if wanted:
            analysis["channels"].update(
                forecast_channels(wanted, horizon, work_dir)
            )
    return analysis


def forecast_payload(analysis: dict[str, Any]) -> tuple[dict[str, list[float]], list[str]]:
    """Collect Gnomon-owned forecast arrays; list channels that abstained."""
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
