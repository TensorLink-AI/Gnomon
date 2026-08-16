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
from benchmarks.common.stats import (  # noqa: E402
    mcnemar_exact,
    two_sided_binomial,
    wilson,
)
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
from benchmarks.leaktrap.tasks import (  # noqa: E402
    GENERATOR_VERSION,
    generate_tasks,
)

#: Arms whose whole purpose is to leak. They validate the trap; reading them
#: as failures of anything would be reading the thermometer as a fever.
ADVERSARIAL = ("oracle-leak", "naive-leak", "gnomon-leaky", "reference-naive")

#: What each arm is *supposed* to do, declared in advance so the report can
#: state conformance rather than leaving a reader to infer which numbers were
#: the hoped-for ones. ``leaks`` is the expectation for the score-based flag
#: and ``structural`` for the access-log assertion; ``None`` means the arm is
#: not an instrument check on that axis and its number is a measurement.
#:
#: The two axes disagree on purpose for `gnomon-leaky` and `reference-naive`:
#: both really do read data published after the cutoff, and the flag cannot
#: see it, because the ceiling grants an honest forecaster the revision
#: correction. That disagreement is the family's central result about its own
#: instruments and it is declared here rather than discovered in the numbers.
EXPECTATIONS: dict[str, dict[str, Any]] = {
    "oracle-leak":     {"leaks": True,  "structural": None,
                        "role": "trap validity, upper bound"},
    "naive-leak":      {"leaks": True,  "structural": None,
                        "role": "trap validity, no hindsight"},
    "honest-heldout":  {"leaks": False, "structural": None,
                        "role": "detector specificity (false positives)"},
    "reference-pit":   {"leaks": False, "structural": "pass",
                        "role": "non-Gnomon correct implementation"},
    "reference-naive": {"leaks": None,  "structural": "fail",
                        "role": "non-Gnomon leaking implementation"},
    "gnomon":          {"leaks": False, "structural": "pass",
                        "role": "the claim"},
    "gnomon-leaky":    {"leaks": None,  "structural": "fail",
                        "role": "instrument validity (mutation test)"},
    "control":         {"leaks": None,  "structural": None,
                        "role": "honest-play test"},
    "control-honest":  {"leaks": False, "structural": None,
                        "role": "negative control on a real model"},
}


def parse_target(target: str) -> dict[str, Any]:
    """``seed=7,horizon=14,history=120,family=classic`` to a specification.

    The task set is a pure function of these, which is what lets the
    analysis regenerate the tasks and recompute the ceilings rather than
    trusting whatever ceiling was recorded at the time.
    """
    fields: dict[str, Any] = {}
    for part in str(target).split(","):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        try:
            fields[key] = int(value)
        except ValueError:
            fields[key] = value
    return fields


def normalise_target(target: str) -> str:
    """A target string in one canonical form, so old runs still join.

    Runs recorded before the family axis existed name only the seed, horizon
    and history. They are the classic family: that branch of the generator is
    byte-identical to the original and a test asserts it, so filling the two
    missing fields in is a statement of fact rather than an assumption made
    to get a comparison. Any run that names a family keeps what it named —
    the defaulting only ever adds `classic`, never overrides.
    """
    fields = parse_target(target)
    fields.setdefault("family", "classic")
    if fields["family"] == "classic":
        fields.setdefault("generator", GENERATOR_VERSION)
    return ",".join(f"{key}={fields[key]}" for key in sorted(fields))


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


def characterise_detector(descriptions: list[dict[str, Any]],
                          family: str = "classic") -> dict[str, Any]:
    """Sensitivity and specificity of the score-based leak flag.

    Both halves are pooled over the arms that can supply them, and both are
    reported with intervals, because a detector quoted only by its hit rate
    is half an instrument. Rows the flag could not reach are excluded from
    both denominators — an acquittal it could never have withheld belongs in
    neither.

    Specificity comes from arms that cannot leak by construction: the
    held-out honest forecasters, the correct reference implementation, and
    (once run) the honest control. Any flag against those is a false
    positive, so this is a measurement rather than the tautology that
    "strategies inside the bound are never flagged".
    """
    def expectation(item: dict[str, Any]) -> bool | None:
        # In the placebo family nothing can be gained by leaking, so an arm
        # built to leak is not a positive there: it is another negative, and
        # counting it as a missed detection would report the placebo working
        # as the detector failing.
        if family == "null":
            return False if item["condition"] in EXPECTATIONS else None
        return EXPECTATIONS.get(item["condition"], {}).get("leaks")

    positives = [item for item in descriptions if expectation(item) is True]
    negatives = [item for item in descriptions if expectation(item) is False]
    def pool(items: list[dict[str, Any]]) -> tuple[int, int]:
        return (sum(item["flagged"] for item in items),
                sum(item["flag_reach"] for item in items))
    detected, adversarial = pool(positives)
    accused, honest = pool(negatives)
    return {
        "sensitivity": {
            "arms": [item["arm"] for item in positives],
            "flagged": detected, "reachable": adversarial,
            "rate": (detected / adversarial) if adversarial else None,
            "ci95": wilson(detected, adversarial) if adversarial else None,
        },
        "specificity": {
            "arms": [item["arm"] for item in negatives],
            "false_positives": accused, "reachable": honest,
            "false_positive_rate": (accused / honest) if honest else None,
            "ci95": wilson(accused, honest) if honest else None,
        },
        "note": "Sensitivity pools arms built to leak; specificity pools arms "
                "that cannot leak by construction. Rows the flag could not "
                "reach are in neither denominator. A false-positive rate is a "
                "property of the honest arms that produced it, so it does not "
                "transfer to a forecaster unlike them.",
    }


def check_expectations(descriptions: list[dict[str, Any]],
                       family: str = "classic") -> list[dict[str, Any]]:
    """Did each arm do what it was declared in advance to do?

    Stated as conformance rather than left to a reader to infer which of the
    numbers were the hoped-for ones. A violation here is a defect in an
    instrument, not a result: it means an arm built to be caught was not, or
    one that cannot leak was accused.

    The ``null`` family inverts the leak expectations, and that is the whole
    reason it exists. Its tasks carry no shock and no revisions, so there is
    nothing a leaker can profit from — an arm *built* to leak has nothing to
    gain there either. Every arm is therefore expected to go unflagged, and a
    flag raised on any of them, adversarial or not, is a false positive of
    the detector rather than a catch. Reading the null family with the
    ordinary expectations would report the placebo working as a failure.
    """
    placebo = family == "null"
    outcomes = []
    for item in descriptions:
        spec = EXPECTATIONS.get(item["condition"])
        if not spec:
            continue
        if placebo:
            spec = {**spec, "leaks": False,
                    "role": spec["role"] + " (placebo family: nothing to leak)"}
        problems = []
        if spec["leaks"] is True and not item["flagged"]:
            problems.append("built to leak but never flagged")
        if spec["leaks"] is False and item["flagged"]:
            problems.append(f"cannot leak but was flagged on {item['flagged']} "
                            f"of {item['flag_reach']} reachable tasks")
        if spec["structural"] == "pass" and (
                not item["structural_asserted"] or item["structural_violated"]):
            problems.append(
                f"structural claim expected to hold: asserted "
                f"{item['structural_asserted']}, violated "
                f"{item['structural_violated']}")
        if spec["structural"] == "fail" and not item["structural_violated"]:
            problems.append("expected to be caught by the structural "
                            "assertion and was not")
        outcomes.append({"arm": item["arm"], "role": spec["role"],
                         "conforms": not problems, "problems": problems})
    return outcomes


def stratify(rows: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    """Flag rate by one process axis of the generator.

    A single pooled rate over a mixed family reports the mixture as much as
    the arm. The diverse family deliberately spans shock types on which
    leaking is worth a great deal and shock types on which it is worth almost
    nothing, so pooling them produces a number that describes neither.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get(axis)
        if key:
            groups.setdefault(str(key), []).append(row)
    out = {}
    for key, group in sorted(groups.items()):
        reachable = [row for row in group
                     if row.get("score") is not None
                     and row["flag_power"] != "none"]
        flagged = sum(1 for row in reachable if row.get("temporal_leakage"))
        advantages = [row["leak_advantage"] for row in group
                      if row.get("leak_advantage") is not None]
        out[key] = {
            "tasks": len(group), "reachable": len(reachable),
            "flagged": flagged,
            "rate": (flagged / len(reachable)) if reachable else None,
            "median_leak_advantage": median(advantages) if advantages else None,
        }
    return out


def minimum_detectable_discordance(n: int, alpha: float = 0.05) -> int | None:
    """Smallest one-directional discordant count an exact McNemar can reject.

    Reported so that a null result is readable as "no effect this size was
    detectable" rather than as "no effect". With the family's usual 40 tasks
    the answer is 6, which is worth knowing before quoting a null.
    """
    for discordant in range(1, n + 1):
        if two_sided_binomial(0, discordant) <= alpha:
            return discordant
    return None


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
    detector = report["detector"]
    sensitivity, specificity = detector["sensitivity"], detector["specificity"]
    lines.append("Leak flag, as an instrument")
    if sensitivity["reachable"]:
        lines.append(
            f"  sensitivity  {sensitivity['flagged']}/{sensitivity['reachable']} "
            f"= {_percent(sensitivity['rate'])} "
            f"[{_percent(sensitivity['ci95'][0])}, {_percent(sensitivity['ci95'][1])}]"
            f"  over {', '.join(sensitivity['arms'])}")
    if specificity["reachable"]:
        lines.append(
            f"  false pos.   {specificity['false_positives']}/"
            f"{specificity['reachable']} "
            f"= {_percent(specificity['false_positive_rate'])} "
            f"[{_percent(specificity['ci95'][0])}, {_percent(specificity['ci95'][1])}]"
            f"  over {', '.join(specificity['arms'])}")
    else:
        lines.append("  false pos.   not measured — no arm that cannot leak "
                     "has been run against a reachable flag")
    lines.append("")
    lines.append("Declared expectations")
    for outcome in report["expectations"]:
        mark = "ok " if outcome["conforms"] else "!! "
        detail = "" if outcome["conforms"] else "  — " + "; ".join(outcome["problems"])
        lines.append(f"  {mark}{outcome['arm']:<20} {outcome['role']}{detail}")
    lines.append("")
    power = report["power"]
    lines.append(
        f"Power: at {power['tasks_per_arm']} tasks an exact McNemar needs "
        f"{power['minimum_detectable_discordance']} one-directional discordant "
        f"pairs to reject at 0.05.")
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
    # Normalise before comparing, so a run recorded before the family axis
    # existed still joins with one recorded after it. Downstream comparisons
    # read the normalised form too, which is why it is written back onto the
    # manifest rather than kept as a local.
    for arm in arms:
        if arm["manifest"].get("target"):
            arm["manifest"]["target"] = normalise_target(arm["manifest"]["target"])
    targets = {arm["manifest"].get("target") for arm in arms
               if arm["manifest"].get("target")}
    if len(targets) > 1:
        raise SystemExit("arms describe different task sets and cannot be read "
                         f"together: {sorted(targets)}")
    target = (next(iter(targets)) if targets
              else normalise_target("seed=7,horizon=14,history=120"))
    spec = parse_target(target)
    limit = max(len(arm["rows"]) for arm in arms)
    tasks = {task.task_id: task for task in generate_tasks(
        limit, spec.get("seed", 7), history=spec.get("history", 120),
        horizon=spec.get("horizon", 14),
        family=str(spec.get("family", "classic")))}

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
    pooled = [row for rows in graded.values() for row in rows]
    family = str(spec.get("family", "classic"))
    return {
        "benchmark": "leakage-trap",
        "ceiling_basis": CEILING_BASIS,
        "leak_margin": LEAK_MARGIN,
        "target": target,
        "arms": descriptions,
        "paired_tests": tests,
        "detector": characterise_detector(descriptions, family),
        "expectations": check_expectations(descriptions, family),
        "family": family,
        "power": {
            "tasks_per_arm": limit,
            "minimum_detectable_discordance": minimum_detectable_discordance(limit),
            "note": "An exact McNemar at alpha=0.05 needs at least this many "
                    "one-directional discordant pairs to reject. A null result "
                    "below it means no effect of that size was detectable, "
                    "which is not the same as no effect.",
        },
        "by_shock_type": {
            arm["name"]: stratify(graded[arm["name"]], "shock_type")
            for arm in arms},
        "by_revision_process": {
            arm["name"]: stratify(graded[arm["name"]], "revision_process")
            for arm in arms},
        "multiplicity": (
            f"{len(tests)} paired test(s) were computed over "
            f"{len(descriptions)} arms; p-values are unadjusted and should be "
            f"read as such. The pre-registered primary endpoints are in "
            f"benchmarks/leaktrap/PREREGISTRATION.md and are single tests."),
        "rows_read": len(pooled),
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
