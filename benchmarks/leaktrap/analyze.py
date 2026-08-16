"""Read a whole leakage-trap family and report what its instruments support.

The numbers this prints used to be assembled by hand: read three summaries,
pick the interesting counts, compute a McNemar p-value in a throwaway script,
write the table into a document. Every step of that is a place for a
favourable reading to enter, and one of them produced a paired test against
an arm whose flag column was a structural constant — a p-value over a
quantity that could not have varied. So the reading is code now, and it
refuses the comparisons it cannot support:

- **One instrument across the whole family.** Every row is regraded under
  the current ceiling basis before anything is compared, so an arm recorded
  under an older basis is not silently mixed with a newer one. Rows that
  carry their forecast are regraded completely; older rows that carry only a
  score get their leak advantage recomputed and are labelled, because a
  score is enough for the advantage but not enough to tell whether the flag
  had any power.
- **Refusal over a printed number.** Manifests that disagree on the task
  set, an empty matched subset, or a paired leak test against an arm the
  flag cannot reach: each is refused with its reason instead of rendered.
- **Rates with intervals, and abstentions bracketed.** A leakage rate over
  35 answered tasks out of 40 is three numbers, not one: the rate, its
  interval, and what it would be under each way of counting the five.

Usage::

    python -m benchmarks.leaktrap.analyze --root results/leaktrap
    python -m benchmarks.leaktrap.analyze --root results/leaktrap --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.manifest import incompatibilities, read_manifest  # noqa: E402
from benchmarks.common.stats import mcnemar_exact, wilson  # noqa: E402
from benchmarks.leaktrap import baselines  # noqa: E402
from benchmarks.leaktrap.baselines import CEILING_BASIS  # noqa: E402
from benchmarks.leaktrap.grade import (  # noqa: E402
    LEAK_MARGIN,
    honest_candidates,
    no_leak_ceiling,
    transcription_verdict,
    wape,
)
from benchmarks.leaktrap.run_leaktrap import THRESHOLD_SWEEP  # noqa: E402
from benchmarks.leaktrap.tasks import generate_tasks  # noqa: E402

#: Arms whose whole purpose is to leak. They validate the trap; reading them
#: as failures of anything would be reading the thermometer as a fever.
ADVERSARIAL = ("oracle-leak", "naive-leak", "gnomon-leaky")


def parse_target(target: str) -> dict[str, int]:
    """``seed=7,horizon=14,history=120`` to a task-set specification.

    The task set is a pure function of these, which is what lets the
    analysis regenerate the tasks and recompute the ceilings rather than
    trusting whatever ceiling was recorded at the time.
    """
    fields: dict[str, int] = {}
    for part in str(target).split(","):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        try:
            fields[key.strip()] = int(value)
        except ValueError:
            continue
    return fields


def load_arm(run_dir: Path) -> dict[str, Any]:
    """One arm's manifest, summary and per-task rows."""
    rows: list[dict[str, Any]] = []
    path = run_dir / "gnomonbench.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    summary: dict[str, Any] = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    return {"name": run_dir.name, "dir": run_dir, "rows": rows,
            "summary": summary, "manifest": read_manifest(run_dir)}


def regrade(rows: list[dict[str, Any]], tasks: dict[str, Any]) -> list[dict[str, Any]]:
    """Every row scored again under the current instruments.

    A row that carries its forecast is regraded from the forecast: score,
    ceiling, advantage, transcription, and whether the flag had power
    against it. A row from before forecasts were recorded carries only a
    score; its advantage is still exactly recomputable — the advantage is a
    function of the score and the ceiling alone — but whether the flag could
    have fired is not, so it is marked ``unspecified`` rather than assumed
    either way.
    """
    graded: list[dict[str, Any]] = []
    cache: dict[str, tuple[dict[str, Any], dict[str, list[float]]]] = {}
    for row in rows:
        task = tasks.get(row.get("task_id"))
        if task is None:
            graded.append({**row, "regrade": "no_task"})
            continue
        if task.task_id not in cache:
            candidates = honest_candidates(task)
            cache[task.task_id] = (no_leak_ceiling(task, candidates), candidates)
        ceiling, candidates = cache[task.task_id]
        forecast = row.get("forecast")
        if forecast:
            score = wape(task.truth, forecast)
            reproduces = baselines.reproduced_by_basis(forecast, candidates)
            flag_power = "measured" if reproduces is None else "none"
            transcription = transcription_verdict(task, forecast, score)
            source = "forecast"
        else:
            score = row.get("score")
            reproduces = None
            flag_power = "unspecified"
            transcription = {
                "transcribed": bool(row.get("transcribed_the_future")),
                "near_transcription": bool(row.get("near_transcription")),
            }
            source = "score"
        advantage = None
        leaked = None
        if score is not None and ceiling["score"]:
            advantage = (ceiling["score"] - score) / ceiling["score"]
            leaked = advantage > LEAK_MARGIN
        if score is None:
            flag_power = "none"
        graded.append({
            **row,
            "score": score,
            "no_leak_ceiling": ceiling["score"],
            "ceiling_strategy": ceiling["strategy"],
            "ceiling_basis": CEILING_BASIS,
            "leak_advantage": advantage,
            "temporal_leakage": leaked,
            "flag_power": flag_power,
            "reproduces_basis_strategy": reproduces,
            **{"transcribed_the_future": transcription["transcribed"],
               "near_transcription": transcription["near_transcription"]},
            "regrade": source,
        })
    return graded


def describe(arm: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One arm, as the instruments actually support describing it."""
    scored = [row for row in rows if row.get("score") is not None]
    reachable = [row for row in scored if row["flag_power"] != "none"]
    unspecified = [row for row in reachable if row["flag_power"] == "unspecified"]
    leaked = [row for row in reachable if row.get("temporal_leakage")]
    unanswered = len(rows) - len(scored)
    advantages = [row["leak_advantage"] for row in scored
                  if row["leak_advantage"] is not None]
    asserted = [row for row in rows
                if (row.get("structural_claim") or {}).get("asserted")]
    holds = [row for row in asserted
             if (row.get("structural_claim") or {}).get("holds")]
    copied = [row for row in rows if row.get("transcribed_the_future")]
    return {
        "arm": arm["name"],
        "condition": arm["manifest"].get("condition") or arm["summary"].get("condition"),
        "model": arm["manifest"].get("model"),
        "prompt_variant": arm["manifest"].get("prompt_variant"),
        "tasks": len(rows),
        "answered": len(scored),
        "unanswered": unanswered,
        "unanswered_reasons": {
            reason: sum(1 for row in rows if row.get("abstention_reason") == reason)
            for reason in sorted({row.get("abstention_reason") for row in rows
                                  if row.get("abstention_reason")})
        },
        "mean_score": mean([row["score"] for row in scored]) if scored else None,
        "mean_no_leak_ceiling": (
            mean([row["no_leak_ceiling"] for row in scored
                  if row["no_leak_ceiling"] is not None]) if scored else None),
        "mean_leak_advantage": mean(advantages) if advantages else None,
        "median_leak_advantage": median(advantages) if advantages else None,
        "flag_reach": len(reachable),
        "flag_reach_unspecified": len(unspecified),
        "flagged": len(leaked),
        "leak_rate": (len(leaked) / len(reachable)) if reachable else None,
        "leak_rate_95ci": wilson(len(leaked), len(reachable)) if reachable else None,
        # What the rate would be if every abstention were honest, and if
        # every abstention had leaked. The truth is inside this bracket and
        # the benchmark cannot narrow it, so it prints both ends.
        "leak_rate_bounds": (
            [len(leaked) / (len(reachable) + unanswered),
             (len(leaked) + unanswered) / (len(reachable) + unanswered)]
            if (len(reachable) + unanswered) else None),
        "transcribed": len(copied),
        "near_transcribed": sum(1 for row in rows if row.get("near_transcription")),
        "structural_asserted": len(asserted),
        "structural_holds": len(holds),
        "structural_violated": len(asserted) - len(holds),
        "threshold_sweep": {
            f"{threshold:.2f}": sum(
                1 for row in reachable
                if row["leak_advantage"] is not None
                and row["leak_advantage"] > threshold)
            for threshold in THRESHOLD_SWEEP
        },
        "regraded_from_score_only": sum(1 for row in rows
                                        if row.get("regrade") == "score"),
    }


def paired_leak_test(left: dict[str, Any], left_rows: list[dict[str, Any]],
                     right: dict[str, Any], right_rows: list[dict[str, Any]],
                     ) -> dict[str, Any]:
    """McNemar on the leak flag, or the reason there is no test to run.

    The refusal that matters: an arm whose forecasts the ceiling's own basis
    reproduces cannot be flagged, so its flag column is a constant. Testing
    a random variable against a constant produces a p-value that means
    nothing, and printing one is how a structural fact gets published as
    evidence. Where this refuses, the structural assertion is the instrument
    — see the structural columns, which are not refused.
    """
    problems = incompatibilities(left["manifest"], right["manifest"])
    if problems:
        return {"refused": "; ".join(problems)}
    measured: dict[str, int] = {}
    unspecified: dict[str, int] = {}
    for name, rows in ((left["name"], left_rows), (right["name"], right_rows)):
        scored = [row for row in rows if row.get("score") is not None]
        measured[name] = sum(1 for row in scored
                             if row["flag_power"] == "measured")
        unspecified[name] = sum(1 for row in scored
                                if row["flag_power"] == "unspecified")
    powerless = [name for name, count in measured.items()
                 if count == 0 and not unspecified[name]]
    if powerless:
        return {"refused": (
            f"the leak flag has no power against {', '.join(powerless)}: every "
            f"forecast it produced is reproduced by the ceiling's own basis, "
            f"so its flag column is a structural constant and a paired test "
            f"against it would be testing arithmetic. Read the structural "
            f"assertion for that arm instead.")}
    unknown = [name for name, count in measured.items() if count == 0]
    if unknown:
        # The row that would settle it — the forecast — was not recorded when
        # these arms ran. Establishing that the flag *could* have fired is a
        # precondition of the test, so an unestablished precondition refuses
        # rather than being assumed in the direction that yields a p-value.
        return {"refused": (
            f"cannot establish that the leak flag had any power against "
            f"{', '.join(unknown)}: its rows predate recorded forecasts, so "
            f"whether the ceiling's own basis reproduces them is unknown. "
            f"Re-run the arm to record forecasts, or read the structural "
            f"assertion instead.")}

    def flags(rows: list[dict[str, Any]]) -> dict[str, bool]:
        return {row["task_id"]: bool(row.get("temporal_leakage"))
                for row in rows
                if row.get("score") is not None and row["flag_power"] != "none"}

    left_flags, right_flags = flags(left_rows), flags(right_rows)
    shared = set(left_flags) & set(right_flags)
    if not shared:
        return {"refused": "no task was answered by both arms"}
    result = mcnemar_exact(left_flags, right_flags)
    return {
        "baseline": left["name"], "treatment": right["name"],
        "matched_tasks": len(shared),
        "baseline_flagged": sum(1 for key in shared if left_flags[key]),
        "treatment_flagged": sum(1 for key in shared if right_flags[key]),
        **result,
    }


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"LeakTrap family — ceiling basis {report['ceiling_basis']}, "
                 f"leak margin {report['leak_margin']}")
    lines.append(f"task set: {report['target']}")
    lines.append("")
    header = (f"{'arm':<22}{'ans':>8}{'flag reach':>12}{'flagged':>10}"
              f"{'leak rate (95% CI)':>26}{'copies':>8}{'structural':>14}")
    lines.append(header)
    lines.append("-" * len(header))
    for arm in report["arms"]:
        interval = arm["leak_rate_95ci"]
        # A rate whose denominator is entirely rows of unknown reach is
        # marked, not printed as if it were established.
        mark = "?" if arm["flag_reach_unspecified"] == arm["flag_reach"] else ""
        rate = (f"{_percent(arm['leak_rate'])}{mark} "
                f"[{_percent(interval[0])}, {_percent(interval[1])}]"
                if interval else "no power")
        structural = (f"{arm['structural_holds']}/{arm['structural_asserted']}"
                      if arm["structural_asserted"] else "not asserted")
        lines.append(
            f"{arm['arm']:<22}{arm['answered']:>4}/{arm['tasks']:<3}"
            f"{arm['flag_reach']:>12}{arm['flagged']:>10}{rate:>26}"
            f"{arm['transcribed']:>8}{structural:>14}")
    lines.append("")
    for arm in report["arms"]:
        if arm["flag_reach_unspecified"]:
            lines.append(
                f"note: {arm['flag_reach_unspecified']} of {arm['arm']}'s rows "
                f"predate recorded forecasts, so whether the flag could reach "
                f"them is unknown; their advantage is exact, their reach is not.")
        if arm["unanswered"]:
            bounds = arm["leak_rate_bounds"]
            lines.append(
                f"note: {arm['arm']} left {arm['unanswered']} task(s) unanswered "
                f"({arm['unanswered_reasons'] or 'reason not recorded'}); the "
                f"leak rate lies between {_percent(bounds[0])} and "
                f"{_percent(bounds[1])} depending on how they are counted.")
    lines.append("")
    lines.append("Trap validity (leaking must measurably help, or nothing else means anything)")
    for arm in report["arms"]:
        if arm["condition"] in ADVERSARIAL:
            lines.append(
                f"  {arm['arm']:<20} median advantage "
                f"{_percent(arm['median_leak_advantage'])}, "
                f"mean {_percent(arm['mean_leak_advantage'])}, "
                f"flagged {arm['flagged']}/{arm['flag_reach']}")
    lines.append("")
    lines.append("Paired leak tests")
    for test in report["paired_tests"]:
        if "refused" in test:
            lines.append(f"  {test['pair']}: REFUSED — {test['refused']}")
        else:
            lines.append(
                f"  {test['baseline']} vs {test['treatment']}: "
                f"{test['baseline_flagged']} vs {test['treatment_flagged']} flagged "
                f"over {test['matched_tasks']} matched tasks, "
                f"{test['discordant']} discordant, p = {test['p_value']:.5f}")
    return "\n".join(lines)


def build_report(root: Path) -> dict[str, Any]:
    arms = [load_arm(path) for path in sorted(root.iterdir()) if path.is_dir()]
    arms = [arm for arm in arms if arm["rows"]]
    if not arms:
        raise SystemExit(f"no arms with per-task rows under {root}")
    targets = {arm["manifest"].get("target") for arm in arms
               if arm["manifest"].get("target")}
    if len(targets) > 1:
        raise SystemExit("arms describe different task sets and cannot be read "
                         f"together: {sorted(targets)}")
    target = next(iter(targets)) if targets else "seed=7,horizon=14,history=120"
    spec = parse_target(target)
    limit = max(len(arm["rows"]) for arm in arms)
    tasks = {task.task_id: task for task in generate_tasks(
        limit, spec.get("seed", 7), history=spec.get("history", 120),
        horizon=spec.get("horizon", 14))}

    graded = {arm["name"]: regrade(arm["rows"], tasks) for arm in arms}
    descriptions = [describe(arm, graded[arm["name"]]) for arm in arms]
    by_name = {arm["name"]: arm for arm in arms}

    tests: list[dict[str, Any]] = []
    honest = [arm for arm in arms
              if (arm["manifest"].get("condition")
                  or arm["summary"].get("condition")) not in ADVERSARIAL]
    for index, left in enumerate(honest):
        for right in honest[index + 1:]:
            outcome = paired_leak_test(left, graded[left["name"]],
                                       right, graded[right["name"]])
            tests.append({"pair": f"{left['name']} vs {right['name']}", **outcome})
    return {
        "benchmark": "leakage-trap",
        "ceiling_basis": CEILING_BASIS,
        "leak_margin": LEAK_MARGIN,
        "target": target,
        "arms": descriptions,
        "paired_tests": tests,
        "arms_read": sorted(by_name),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="directory holding one subdirectory per arm")
    parser.add_argument("--json", action="store_true",
                        help="print the report as JSON instead of a table")
    parser.add_argument("--write", action="store_true",
                        help="also write analysis.json beside the arms")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    report = build_report(root)
    print(json.dumps(report, indent=2) if args.json else render(report))
    if args.write:
        (root / "analysis.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
