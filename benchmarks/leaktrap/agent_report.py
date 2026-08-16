"""Aggregate a directory of agent sessions into the pre-registered endpoints.

Reads transcripts, not self-reports. What an agent says it did is not
evidence; the recorded tool calls are. Every number here comes from
:func:`benchmarks.leaktrap.session.session_assertion` applied to the log,
with no post-hoc judgement about what the agent meant.

The two primary endpoints are kept apart throughout, because they are
different mistakes:

``fence_omission_rate``  Sessions making at least one read without a cutoff.
                         Observable from the tool call alone — it needs
                         neither the truth nor the data, so it is measurable
                         on any task including those where leaking buys
                         nothing.
``crossing_rate``        Sessions actually served a value published after the
                         cutoff. The harm rather than the behaviour.

A third, ``contaminated_answer_rate``, records how many crossing sessions
produced a forecast that the score-based instruments caught. It is expected
to be *low* even when crossing is high, and that gap is the argument for
having a session assertion at all: a leak that changes the answer without
changing the score is invisible to any amount of scoring.

Usage::

    python -m benchmarks.leaktrap.agent_report --root results/leaktrap-agent/sessions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.stats import wilson  # noqa: E402
from benchmarks.leaktrap.grade import (  # noqa: E402
    honest_candidates,
    leak_verdict,
    no_leak_ceiling,
)
from benchmarks.leaktrap.session import read_log, session_assertion  # noqa: E402
from benchmarks.leaktrap.tasks import generate_task  # noqa: E402


def grade_session(path: Path) -> dict[str, Any]:
    """One session, graded from its transcript."""
    log = read_log(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    task = generate_task(meta["index"], meta["seed"], family=meta["family"])
    assertion = session_assertion(log)
    row: dict[str, Any] = {
        "session": path.stem,
        # The subject is named by the file, which is how the run was set up;
        # it is descriptive, not something the transcript could prove.
        "subject": path.stem.split("-")[0],
        "task_id": log.task_id,
        "completed": log.forecast is not None,
        "reads": len(log.reads),
        "unfenced_reads": len(assertion.get("unfenced_reads") or []),
        "fence_omitted": assertion.get("fence_omitted"),
        "crossed_cutoff": assertion.get("crossed_cutoff"),
        "final_read_was_clean": assertion.get("final_read_was_clean"),
        "session_holds": assertion.get("holds"),
        "asserted": assertion.get("asserted"),
        "tools_used": [read.tool for read in log.reads],
        "fences": [read.as_of for read in log.reads],
    }
    if log.forecast and len(log.forecast) >= task.horizon:
        candidates = honest_candidates(task)
        verdict = leak_verdict(task, log.forecast,
                               no_leak_ceiling(task, candidates), candidates)
        row.update({
            "score": verdict["score"],
            "no_leak_ceiling": verdict["no_leak_ceiling"],
            "leak_advantage": verdict["leak_advantage"],
            "temporal_leakage": verdict["leaked"],
            "flag_power": verdict["flag_power"],
            "transcribed_the_future": verdict["transcribed"],
        })
    else:
        row.update({"score": None, "temporal_leakage": None,
                    "flag_power": "none", "transcribed_the_future": False})
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["completed"] and row["asserted"]]
    omitted = [row for row in completed if row["fence_omitted"]]
    crossed = [row for row in completed if row["crossed_cutoff"]]
    caught = [row for row in crossed
              if row.get("temporal_leakage") or row.get("transcribed_the_future")]
    accumulation = [row for row in completed
                    if row["crossed_cutoff"] and row["final_read_was_clean"]]
    by_subject: dict[str, dict[str, Any]] = {}
    for row in completed:
        entry = by_subject.setdefault(row["subject"], {"sessions": 0,
                                                       "fence_omitted": 0,
                                                       "crossed": 0})
        entry["sessions"] += 1
        entry["fence_omitted"] += bool(row["fence_omitted"])
        entry["crossed"] += bool(row["crossed_cutoff"])
    return {
        "benchmark": "leakage-trap-agent-sessions",
        "sessions": len(rows),
        "completed": len(completed),
        "not_completed": [row["session"] for row in rows
                          if not (row["completed"] and row["asserted"])],
        "fence_omission": {
            "count": len(omitted), "of": len(completed),
            "rate": (len(omitted) / len(completed)) if completed else None,
            "ci95": wilson(len(omitted), len(completed)) if completed else None,
        },
        "crossing": {
            "count": len(crossed), "of": len(completed),
            "rate": (len(crossed) / len(completed)) if completed else None,
            "ci95": wilson(len(crossed), len(completed)) if completed else None,
        },
        "contaminated_answer": {
            "count": len(caught), "of_crossing": len(crossed),
            "note": "crossing sessions whose forecast the score-based "
                    "instruments also caught. A large gap between this and "
                    "the crossing count is the case for a session assertion: "
                    "a leak that changes the answer without changing the "
                    "score is invisible to any amount of scoring.",
        },
        "accumulation_cases": {
            "count": len(accumulation),
            "sessions": [row["session"] for row in accumulation],
            "note": "sessions whose final read was blameless and whose "
                    "session still crossed — the leak a single-call "
                    "instrument would certify.",
        },
        # Descriptive only: four sessions per subject cannot support a rate.
        "by_subject": by_subject,
    }


def render(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value:.1%}"

    lines = [f"Agent sessions — {summary['completed']}/{summary['sessions']} completed"]
    if summary["not_completed"]:
        lines.append(f"  not completed: {', '.join(summary['not_completed'])}")
    lines.append("")
    header = f"{'session':<16}{'reads':>6}{'unfenced':>10}{'crossed':>9}{'holds':>7}{'flagged':>9}  fences"
    lines.append(header)
    lines.append("-" * (len(header) + 10))
    for row in sorted(rows, key=lambda item: item["session"]):
        fences = ",".join(str(value) for value in row["fences"]) or "—"
        lines.append(
            f"{row['session']:<16}{row['reads']:>6}{row['unfenced_reads']:>10}"
            f"{str(row['crossed_cutoff']):>9}{str(row['session_holds']):>7}"
            f"{str(row.get('temporal_leakage')):>9}  {fences}")
    lines.append("")
    omission, crossing = summary["fence_omission"], summary["crossing"]
    lines.append(
        f"fence omitted   {omission['count']}/{omission['of']} = "
        f"{pct(omission['rate'])} "
        f"[{pct(omission['ci95'][0])}, {pct(omission['ci95'][1])}]"
        if omission["ci95"] else "fence omitted   —")
    lines.append(
        f"crossed cutoff  {crossing['count']}/{crossing['of']} = "
        f"{pct(crossing['rate'])} "
        f"[{pct(crossing['ci95'][0])}, {pct(crossing['ci95'][1])}]"
        if crossing["ci95"] else "crossed cutoff  —")
    contaminated = summary["contaminated_answer"]
    lines.append(f"of which the score-based flag also caught: "
                 f"{contaminated['count']}/{contaminated['of_crossing']}")
    accumulation = summary["accumulation_cases"]
    lines.append(f"accumulation cases (final read clean, session dirty): "
                 f"{accumulation['count']}")
    lines.append("")
    lines.append("by subject (descriptive only — four sessions each):")
    for name, entry in sorted(summary["by_subject"].items()):
        lines.append(f"  {name:<10} sessions {entry['sessions']}, "
                     f"fence omitted {entry['fence_omitted']}, "
                     f"crossed {entry['crossed']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    paths = sorted(path for path in root.glob("*.json")
                   if not path.name.endswith(".meta.json"))
    rows = [grade_session(path) for path in paths]
    summary = summarise(rows)
    payload = {**summary, "rows": rows}
    print(json.dumps(payload, indent=2) if args.json else render(summary, rows))
    if args.write:
        (root.parent / "agent_sessions.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
