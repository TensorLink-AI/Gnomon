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
  which is recorded as an abstention, not papered over. Under the
  official all-channels rule one absent channel voids the whole record
  (measured on the MIMIC split: abstentions on the sparse
  ``temperature_c`` channel voided 38 otherwise-complete records and
  left one comparable record), so cross-arm comparison goes through
  ``score_per_channel.py`` — see that module and the README.
- ``best_effort`` (opt-in, default off, ``--best-effort`` on the
  runner) passes Gnomon's own best-effort flag through: a channel that
  would abstain publishes the engine's disclosed naive fallback rows
  instead, labeled ``support: "best_effort"`` and carrying Gnomon's NO
  RELIABLE FORECAST warning. Those rows are not supported forecasts,
  so every consumer of the per-channel outcomes keeps the support
  label, and the runner reports the support-label mix beside any score
  that includes them. Off by default because publishing unsupported
  numbers unlabeled-by-default would be exactly the laundering the
  abstention contract forbids.
- In ``gnomon-pure`` mode multiple-choice questions are answered
  ``Uncertain`` where the option exists (an honest abstention — T2/T4
  option sets include it); questions without such an option are answered
  with the :data:`MCQ_ABSTAIN` sentinel, which exact-matches no real
  option and therefore deterministically scores wrong — never a lucky
  hit on a real option. Those questions are recorded as abstentions in
  the row's bookkeeping. The agent mode hands choices to an LLM that
  sees Gnomon's computed evidence.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
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


def forecast_target_map(row: dict[str, Any], arrays: dict[str, list[Any]]) -> dict[str, str]:
    """Map official forecast keys to their historical input series.

    TemporalBench uses two representations.  Panel rows keep the same names
    in history and ground truth (for example MIMIC's ``heart_rate``), while
    single-target rows deliberately call the truth ``future_main`` or
    ``future_sales``.  Those aliases are output identities, not columns in
    the historical input.  Keeping the mapping explicit prevents an empty
    forecast from being mistaken for model abstention on the latter rows.
    """
    meta = row.get("meta") or {}
    ground_truth = row.get("ground_truth")
    truth_keys = list(ground_truth) if isinstance(ground_truth, dict) else []
    if not truth_keys:
        main = meta.get("main_key") or meta.get("target")
        return {str(main): str(main)} if main in arrays else {}
    if len(truth_keys) > 1:
        return {str(key): str(key) for key in truth_keys if key in arrays}

    output_key = str(truth_keys[0])
    candidates = [
        meta.get("main_key"), meta.get("target"), meta.get("target_col"),
        "sales_censored" if output_key == "future_sales" else None,
        output_key.removeprefix("future_"),
    ]
    input_key = next((str(key) for key in candidates if key in arrays), None)
    if input_key is None:
        # Last-resort structural rule for third-party additions: select the
        # sole non-time history series, but refuse ambiguity instead of
        # guessing among covariates.
        non_time = [str(key) for key in arrays
                    if str(key).lower() not in {"timestamp", "timestamps", "time"}]
        if len(non_time) == 1:
            input_key = non_time[0]
    return {output_key: input_key} if input_key is not None else {}


def forecast_channel(values: list[float | None], horizon: int,
                     work_dir: str | None = None,
                     best_effort: bool = False,
                     model_evidence_registry: str | None = None,
                     candidates: list[str] | None = None) -> dict[str, Any]:
    """Gnomon forecast for one channel on the synthetic hourly axis.

    With ``best_effort`` the engine's disclosed fallback lane is enabled:
    a channel that would abstain returns rows labeled ``support:
    "best_effort"`` instead — not a supported forecast, and the label
    must travel with the values everywhere they are scored.
    """
    from gnomon import forecast as gnomon_forecast
    from gnomon.contracts import GnomonError

    config = None
    if model_evidence_registry:
        from gnomon.config import GnomonConfig

        config = GnomonConfig()
        config.models.admission_policy = "evidence_weighted"
        config.models.evidence_registry_path = str(model_evidence_registry)

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
            # The benchmark's two arms are the strict engine (abstention
            # possible, degraded results still published — the pre-
            # graduated default) and the disclosed fallback lane. The
            # engine's own default floor is now best_effort, so the strict
            # arm pins the old behaviour explicitly.
            minimum_support=("best_effort" if best_effort
                             else "conditionally_supported"),
            config=config,
            candidates=candidates,
        )
    except GnomonError as error:
        return {"abstained": True, "reason": f"{error.code}: {error.message}"}
    result = artifact.results[0]
    if result.support == "unsupported" or not result.forecast:
        return {"abstained": True,
                "reason": "; ".join(str(w) for w in result.warnings) or "unsupported"}
    outcome = {
        "abstained": False,
        "support": result.support,
        "selected_model": result.selected_model,
        "values": [float(r.get("q50", r["point"])) for r in result.forecast],
        "warnings": list(result.warnings),
    }
    if result.admission is not None:
        outcome["admission"] = result.admission
    if result.model_assisted is not None:
        outcome["model_assisted"] = result.model_assisted
    return outcome


def forecast_channels(
    channels: dict[str, list[float | None]], horizon: int,
    work_dir: str | None = None, best_effort: bool = False,
    named_tsfm: str | None = None,
    model_evidence_registry: str | None = None,
    max_workers: int | None = None,
    candidates: list[str] | None = None,
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

    if named_tsfm and model_evidence_registry:
        raise ValueError(
            "named_tsfm and model_evidence_registry are separate experimental "
            "arms and cannot be combined")
    if named_tsfm and candidates:
        raise ValueError(
            "named_tsfm bypasses selection and cannot use a candidate pool")
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    observed = {key: _observed(values) for key, values in channels.items()}
    if named_tsfm:
        # Model-supply experiment: bypass local model selection explicitly,
        # while labelling every output as an unevaluated named-model prior.
        # TemporalBench's ~50-point histories cannot support its 29-step
        # horizon plus separated selection/calibration/test windows, so this
        # arm answers the distinct question "what does the pinned TSFM itself
        # add?" It must never be reported as Gnomon's governed default.
        from gnomon.tsfm_sandbox import SubprocessAdapter

        def predict(item: tuple[str, list[float]]) -> tuple[str, dict[str, Any]]:
            key, history = item
            try:
                adapter = SubprocessAdapter(named_tsfm, frequency="h")
                values = adapter.predict(history, horizon, 24)
                return key, {
                    "abstained": False,
                    "support": "experimental_named_model",
                    "selected_model": named_tsfm,
                    "candidate_identity": {
                        "kind": "tsfm", "name": named_tsfm,
                        "backend": adapter.backend,
                        "revision": adapter.revision,
                        "selection_policy": "caller_named_unvalidated_prior",
                    },
                    "values": [float(value) for value in values],
                }
            except Exception as error:
                return key, {"abstained": True,
                             "reason": f"{type(error).__name__}: {error}"}

        workers = (min(3, max(1, len(observed))) if max_workers is None
                   else min(max_workers, max(1, len(observed))))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return dict(executor.map(predict, observed.items()))
    keys = list(observed)
    length = max((len(values) for values in observed.values()), default=0)
    if len(keys) < 2 or length == 0 or "timestamp" in keys:
        return {key: forecast_channel(channels[key], horizon, work_dir,
                                      best_effort=best_effort,
                                      model_evidence_registry=(
                                          model_evidence_registry),
                                      candidates=candidates)
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
    config = None
    if model_evidence_registry:
        from gnomon.config import GnomonConfig

        config = GnomonConfig()
        config.models.admission_policy = "evidence_weighted"
        config.models.evidence_registry_path = str(model_evidence_registry)
    try:
        artifact, _ = forecast_multi(
            str(csv_path), time_column="timestamp", target_columns=keys,
            horizon=horizon, frequency="h",
            output=str(run_dir / "gnomon-output"),
            minimum_support=("best_effort" if best_effort
                             else "conditionally_supported"),
            config=config,
            max_workers=max_workers,
            candidates=candidates,
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
            outcome = {
                "abstained": False,
                "support": result.support,
                "selected_model": result.selected_model,
                "values": [float(r.get("q50", r["point"])) for r in result.forecast],
                "warnings": list(result.warnings),
            }
            if result.admission is not None:
                outcome["admission"] = result.admission
            if result.model_assisted is not None:
                outcome["model_assisted"] = result.model_assisted
            outcomes[result.series] = outcome
    return outcomes


def analyse_row(row: dict[str, Any], work_dir: str | None = None,
                best_effort: bool = False,
                named_tsfm: str | None = None,
                model_evidence_registry: str | None = None,
                max_workers: int | None = None,
                candidates: list[str] | None = None) -> dict[str, Any]:
    """Deterministic Gnomon evidence for one row: per-channel forecasts
    (T2/T4), plus season/anomaly/stats findings on the main channel."""
    from gnomon.anomaly import detect_anomalies
    from gnomon.temporal import detect_season

    arrays = prompt_input_arrays(row)
    meta = row.get("meta") or {}
    main_key = meta.get("main_key") or next(iter(arrays), None)
    horizon = int(meta.get("n_horizon") or 0)
    target_map = forecast_target_map(row, arrays)

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
        wanted = {output_key: arrays[input_key]
                  for output_key, input_key in target_map.items()}
        if wanted:
            analysis["channels"].update(
                forecast_channels(wanted, horizon, work_dir,
                                  best_effort=best_effort,
                                  named_tsfm=named_tsfm,
                                  model_evidence_registry=(
                                      model_evidence_registry),
                                  max_workers=max_workers,
                                  candidates=candidates)
            )
    return analysis


def forecast_payload(
    analysis: dict[str, Any],
) -> tuple[dict[str, list[float]], list[str], dict[str, str]]:
    """Collect Gnomon-owned forecast arrays; list channels that abstained.

    The third element maps each forecast channel to its support label
    (``supported`` / ``interval_only`` / ``best_effort`` / ...), so the
    label always travels with the values: a ``best_effort`` channel is a
    disclosed fallback carrying NO RELIABLE FORECAST, not a supported
    forecast, and any score that includes it must say so.
    """
    forecast: dict[str, list[float]] = {}
    abstained: list[str] = []
    support: dict[str, str] = {}
    for key, outcome in analysis.get("channels", {}).items():
        if outcome.get("abstained"):
            abstained.append(f"{key}: {outcome.get('reason')}")
        else:
            forecast[key] = outcome["values"]
            support[key] = str(outcome.get("support"))
    return forecast, abstained, support


#: Sentinel answered when an MCQ has no ``Uncertain`` option. It never
#: exact-matches a real option string, so the question deterministically
#: scores wrong — answering ``options[0]`` instead would coincidentally
#: match the label about 1/n of the time, quietly inflating accuracy.
MCQ_ABSTAIN = "ABSTAIN"


def _abstain_sentinel(options: list[Any]) -> str:
    """A sentinel guaranteed not to match any option.

    Scoring compares strip+lower normalised strings, so a hypothetical
    option that normalises to ``abstain`` would make the plain sentinel
    score CORRECT — the guarantee is enforced here per question, not
    assumed of the dataset.
    """
    normalised = {str(option).strip().lower() for option in options}
    sentinel = MCQ_ABSTAIN
    while sentinel.strip().lower() in normalised:
        sentinel += "-"
    return sentinel


def uncertain_mcq(row: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Pure-mode choice answers: 'Uncertain' where the options allow it.

    Questions whose option set has no ``Uncertain`` entry are answered
    with the :data:`MCQ_ABSTAIN` sentinel, so they score wrong
    deterministically instead of guessing. The second return value lists
    those question keys for the row's abstention bookkeeping.
    """
    answers: dict[str, str] = {}
    abstained: list[str] = []
    for key, entry in (row.get("mcq") or {}).items():
        options = entry.get("options") or []
        uncertain = next((o for o in options if str(o).lower() == "uncertain"), None)
        if uncertain is not None:
            answers[key] = uncertain
        else:
            answers[key] = _abstain_sentinel(options)
            abstained.append(f"mcq/{key}: no Uncertain option")
    return answers, abstained
