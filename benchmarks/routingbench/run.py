"""Run the frozen P5 paired-outcome routing streams serially and resumably."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal
import subprocess
from typing import Any

from gnomon.adapter_promotion import AdapterOutcomeLedger


STREAM_VERSION = "p5-shadow-routing-1"
CANDIDATE = "challenger"
REVISION = "sha256:route-v1"
CHAMPION = "last_value"
SUBDAILY = {"frequency_class": "subdaily"}
DAILY = {"frequency_class": "daily_weekly"}

STREAMS = {
    "stable_gain": [.70] * 20,
    "gain_then_drift": [.70] * 12 + [1.40] * 8,
    "mixed_control": [.90 if index % 2 == 0 else 1.10
                      for index in range(20)],
}


class StreamTimeout(RuntimeError):
    pass


def _timeout(_signum: int, _frame: Any) -> None:
    raise StreamTimeout("stream exceeded 30 seconds")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _instant(index: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)


def _route(ledger: AdapterOutcomeLedger, project: str,
           regime: dict[str, str], as_of: datetime,
           revision: str | None = REVISION) -> dict[str, Any]:
    return ledger.route(
        project=project, candidate=CANDIDATE, revision=revision,
        champion=CHAMPION, regime=regime,
        as_of=as_of.isoformat(),
    ).to_dict()


def run_stream(ledger: AdapterOutcomeLedger, name: str,
               errors: list[float]) -> dict[str, Any]:
    decisions = []
    routed_error = 0.0
    for index, candidate_error in enumerate(errors):
        known_at = _instant(index)
        decision = _route(
            ledger, name, SUBDAILY, known_at - timedelta(seconds=1))
        error = (candidate_error if decision["recommendation"] == CANDIDATE
                 else 1.0)
        routed_error += error
        decisions.append({
            "index": index,
            "as_of": decision["as_of"],
            "paired_outcomes": decision["paired_outcomes"],
            "recommendation": decision["recommendation"],
            "recommended_pool": decision["recommended_pool"],
            "reasons": decision["reasons"],
            "receipt_id": decision["receipt_id"],
            "routed_error": error,
            "candidate_error": candidate_error,
            "champion_error": 1.0,
            "authority": {
                key: decision[key] for key in (
                    "automatic_promotion", "automation_eligible",
                    "job_local_admission_required", "routing_authority",
                    "rollback_condition")
            },
        })
        ledger.record(
            project=name, outcome_id=f"{name}-{index}",
            candidate=CANDIDATE, revision=REVISION, baseline=CHAMPION,
            candidate_error=candidate_error, baseline_error=1.0,
            known_at=known_at.isoformat(), regime=SUBDAILY,
        )
    champion_error = float(len(errors))
    return {
        "schema_version": 1,
        "stream": name,
        "complete": len(decisions) == len(errors),
        "outcomes": len(errors),
        "routed_error": routed_error,
        "always_champion_error": champion_error,
        "relative_improvement_vs_champion": (
            (champion_error - routed_error) / champion_error),
        "challenger_routes": sum(
            row["recommendation"] == CANDIDATE for row in decisions),
        "decisions": decisions,
    }


def run_regime_and_replay_probes(ledger: AdapterOutcomeLedger) -> dict[str, Any]:
    project = "regime_isolation"
    for index in range(8):
        for regime, error in ((SUBDAILY, .7), (DAILY, 1.4)):
            ledger.record(
                project=project,
                outcome_id=f"{regime['frequency_class']}-{index}",
                candidate=CANDIDATE, revision=REVISION, baseline=CHAMPION,
                candidate_error=error, baseline_error=1.0,
                known_at=_instant(index).isoformat(), regime=regime,
            )
    cutoff = _instant(8)
    subdaily = _route(ledger, project, SUBDAILY, cutoff)
    daily = _route(ledger, project, DAILY, cutoff)
    before = _route(ledger, project, SUBDAILY, cutoff)
    ledger.record(
        project=project, outcome_id="future", candidate=CANDIDATE,
        revision=REVISION, baseline=CHAMPION, candidate_error=0.0,
        baseline_error=1.0,
        known_at="2026-02-01T10:00:00+10:00", regime=SUBDAILY)
    after = _route(ledger, project, SUBDAILY, cutoff)
    unpinned_project = "unpinned"
    for index in range(8):
        ledger.record(
            project=unpinned_project, outcome_id=str(index),
            candidate=CANDIDATE, revision=None, baseline=CHAMPION,
            candidate_error=.1, baseline_error=1.0,
            known_at=_instant(index).isoformat(), regime=SUBDAILY)
    unpinned = _route(
        ledger, unpinned_project, SUBDAILY, cutoff, revision=None)
    deterministic = _route(ledger, project, SUBDAILY, cutoff)
    return {
        "schema_version": 1,
        "stream": "regime_and_replay_probes",
        "complete": True,
        "subdaily": subdaily,
        "daily_weekly": daily,
        "future_before": before,
        "future_after": after,
        "future_replay_equal": before == after,
        "unpinned": unpinned,
        "deterministic_replay_equal": after == deterministic,
    }


def summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stable = rows["stable_gain"]
    drift = rows["gain_then_drift"]
    mixed = rows["mixed_control"]
    probes = rows["regime_and_replay_probes"]
    cold_start = all(
        all(decision["recommendation"] == CHAMPION
            and "insufficient_paired_outcomes" in decision["reasons"]
            for decision in row["decisions"][:8])
        for row in (stable, drift, mixed)
    )
    rollback_indices = [
        decision["index"] for decision in drift["decisions"][12:]
        if decision["recommendation"] == CHAMPION
    ]
    rollback_delay = ((rollback_indices[0] - 12)
                      if rollback_indices else None)
    authority_rows = [decision for name in STREAMS
                      for decision in rows[name]["decisions"]]
    authority_safe = all(
        decision["authority"]["automatic_promotion"] is False
        and decision["authority"]["automation_eligible"] is False
        and decision["authority"]["job_local_admission_required"] is True
        and decision["authority"]["routing_authority"] == "candidate_pool_only"
        and CHAMPION in decision["authority"]["rollback_condition"]
        for decision in authority_rows)
    gates = {
        "completion": all(row["complete"] for row in rows.values()),
        "cold_start": cold_start,
        "stable_skill": stable["relative_improvement_vs_champion"] >= .15,
        "drift_rollback_delay": rollback_delay is not None
                                and rollback_delay <= 2,
        "drift_nonpositive_regret": (
            drift["routed_error"] <= drift["always_champion_error"]),
        "mixed_control_safety": mixed["challenger_routes"] == 0
                                and mixed["routed_error"]
                                == mixed["always_champion_error"],
        "regime_isolation": (
            probes["subdaily"]["recommendation"] == CANDIDATE
            and probes["daily_weekly"]["recommendation"] == CHAMPION
            and probes["subdaily"]["paired_outcomes"] == 8
            and probes["daily_weekly"]["paired_outcomes"] == 8),
        "point_in_time_replay": probes["future_replay_equal"],
        "unpinned_safety": probes["unpinned"]["recommendation"] == CHAMPION,
        "determinism": probes["deterministic_replay_equal"],
        "authority": authority_safe,
    }
    return {
        "schema_version": 1,
        "stream_version": STREAM_VERSION,
        "denominators": {
            "streams": 4,
            "prospective_decisions": len(authority_rows),
            "completed_streams": sum(row["complete"] for row in rows.values()),
        },
        "metrics": {
            "stable_relative_improvement":
                stable["relative_improvement_vs_champion"],
            "drift_regret_vs_champion":
                drift["routed_error"] - drift["always_champion_error"],
            "rollback_delay_harmful_outcomes": rollback_delay,
            "mixed_challenger_routes": mixed["challenger_routes"],
        },
        "gates": gates,
        "decision_ready": all(gates.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = args.output / "streams"
    ledger = AdapterOutcomeLedger(args.output / "routing.sqlite")
    signal.signal(signal.SIGALRM, _timeout)
    runners = {
        **{name: (lambda name=name, errors=errors:
                  run_stream(ledger, name, errors))
           for name, errors in STREAMS.items()},
        "regime_and_replay_probes": lambda: run_regime_and_replay_probes(ledger),
    }
    for name, runner in runners.items():
        target = cases / f"{name}.json"
        if target.exists():
            continue
        error = None
        for attempt in range(args.retries + 1):
            try:
                signal.alarm(args.timeout)
                row = runner()
                signal.alarm(0)
                row["attempt"] = attempt + 1
                _atomic_json(target, row)
                error = None
                break
            except Exception as exc:
                signal.alarm(0)
                error = f"{type(exc).__name__}: {exc}"[:1000]
        if error is not None:
            _atomic_json(target, {
                "schema_version": 1, "stream": name, "complete": False,
                "attempt": args.retries + 1, "error": error,
            })
    rows = {path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(cases.glob("*.json"))}
    summary = summarize(rows)
    identity = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stream_version": STREAM_VERSION,
        "jobs": 1, "retries": args.retries,
        "timeout_seconds": args.timeout,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=False).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True,
            check=False).stdout.splitlines(),
    }
    _atomic_json(args.output / "run_identity.json", identity)
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["decision_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
