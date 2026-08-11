"""Compare benchmark arms on the tasks they both answered.

Every comparison in this suite was, until now, assembled by hand: load
two result files, decide which rows correspond, pick a statistic, run a
test in a throwaway script. That is exactly where a favourable reading
creeps in — and where three genuinely mismatched comparisons slipped
through in one session.

This does it the same way every time:

- **Matched subset only.** Arms are joined on task id; means are reported
  over the intersection, so an arm cannot look better by having scored a
  different (easier) set of tasks. Unmatched tasks are counted and named
  in the output rather than dropped silently.
- **Paired tests.** Binary outcomes get an exact McNemar test, continuous
  metrics an exact sign test — both paired, because the arms answer the
  same tasks.
- **Refusal over guessing.** If the manifests disagree on benchmark or
  target, or the matched subset is empty, the comparison is refused with
  the reason. A number that should not have been computed is worse than
  no number.

Usage::

    python -m benchmarks.report --root results/glm52
    python -m benchmarks.report --root results/glm52 --compare tb-control tb-gnomon
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common.manifest import incompatibilities, read_manifest  # noqa: E402

#: Direction registry. A metric matches by substring against these token
#: lists; a name matching neither is NOT silently assumed — it is compared
#: as higher-better with an explicit `direction_recognised: false` flag on
#: the entry and a warning in the rendered line. The registry exists
#: because the old default ("anything unrecognised is higher-better")
#: silently inverted LeakTrap's `score` (a WAPE, lower-better): the sign
#: test reported treatment "wins" on the tasks it did worse on.
LOWER_IS_BETTER = ("mse", "mae", "rmse", "mape", "rcrps", "mase", "smape",
                   "wape", "crps", "pinball", "loss", "score")
HIGHER_IS_BETTER = ("f1", "accuracy", "precision", "recall", "auc",
                    "success", "coverage")


def metric_direction(name: str) -> tuple[bool, bool]:
    """``(lower_is_better, recognised)`` for a metric name.

    Lower-better tokens win when a name matches both lists ("f1_loss" is
    a loss); a name matching neither is treated as higher-better but
    flagged unrecognised so the caller discloses the assumption instead
    of printing an inverted comparison as fact.
    """
    lowered = name.lower()
    if any(token in lowered for token in LOWER_IS_BETTER):
        return True, True
    if any(token in lowered for token in HIGHER_IS_BETTER):
        return False, True
    return False, False


def is_voided(record: dict[str, Any]) -> bool:
    """Whether the harness, not the system under test, ended this row.

    Adapters mark a row the harness voided (a breached cap, a run that
    never submitted) with ``row_abstained``. Such a row did not answer
    the task wrongly — it did not answer it — so it must not enter a
    success comparison as a model failure.
    """
    return bool(record.get("row_abstained") or record.get("voided"))


# ---------------------------------------------------------------------------
# Loading: every adapter's per-task output, reduced to {task_id: outcome}
# ---------------------------------------------------------------------------

def normalise_task_id(task_id: str) -> str:
    """One task, one id, whichever layout wrote it.

    Our adapters name an MTBench sample `shard#0007`; the official script
    names the same sample `shard_0007.json`. Joining on the raw strings
    finds nothing in common and refuses a comparison that is in fact
    perfectly valid.
    """
    text = str(task_id)
    if text.endswith(".json"):
        text = text[: -len(".json")]
    return text.replace("#", "_")


def _load_gnomonbench(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Rows written by our adapters (`gnomonbench.jsonl`)."""
    path = run_dir / "gnomonbench.jsonl"
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = record.get("task_id")
        if task_id is None:
            continue
        rows[normalise_task_id(task_id)] = record
    return rows


def _load_mtbench_official(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-sample details written by MTBench's official script (control) or
    by our adapter (treatment). Task ids are normalised to a common form so
    the two layouts join: `train-...#0007` and `train-..._0007.json` are the
    same task."""
    details = run_dir / "output_details"
    if not details.is_dir():
        return {}
    rows = {}
    for path in sorted(details.iterdir()):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, IsADirectoryError):
            continue
        rows[normalise_task_id(path.name)] = record
    return rows


def load_run(run_dir: Path) -> dict[str, Any]:
    """One arm: its manifest, summary, and per-task outcomes."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    tasks = _load_gnomonbench(run_dir) or _load_mtbench_official(run_dir)
    return {"name": run_dir.name, "dir": run_dir, "manifest": read_manifest(run_dir),
            "summary": summary, "tasks": tasks}


# ---------------------------------------------------------------------------
# Outcome extraction
# ---------------------------------------------------------------------------

def derived_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Metrics computed from a record's own trajectory, when it carries one.

    An official script stores `predict` and `ground_truth` and leaves the
    aggregation to itself; our adapters store the scored metrics. Deriving
    the metrics here lets the two be compared per task without either side
    re-running, and — because both sides are computed by the same code on
    the same pairs — without inheriting either aggregation's filters.
    """
    predicted, truth = record.get("predict"), record.get("ground_truth")
    if not isinstance(predicted, list) or not isinstance(truth, list):
        return {}
    if len(predicted) != len(truth):
        # A wrong-length prediction is unscoreable, not scoreable-on-the-
        # overlap: zip would silently drop the tail and understate the
        # error — and the adapters' own scorers refuse the same condition,
        # so scoring it here would grade one arm by a rule the other
        # arm's rows never got.
        return {}
    pairs = [(float(t), float(p)) for t, p in zip(truth, predicted)
             if isinstance(t, (int, float)) and isinstance(p, (int, float))]
    if not pairs:
        return {}
    n = len(pairs)
    mse = sum((t - p) ** 2 for t, p in pairs) / n
    mae = sum(abs(t - p) for t, p in pairs) / n
    nonzero = [(t, p) for t, p in pairs if t != 0]
    metrics = {"mse": mse, "mae": mae, "rmse": mse ** 0.5}
    if nonzero:
        metrics["mape"] = 100.0 * sum(abs((t - p) / t) for t, p in nonzero) / len(nonzero)
    return metrics


def metric_value(record: dict[str, Any], metric: str) -> float | None:
    for key in (metric, f"mean_{metric}"):
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    extra = record.get("extra")
    if isinstance(extra, dict):
        value = extra.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return derived_metrics(record).get(metric)


#: Numeric per-task fields that are bookkeeping, not quality metrics. The
#: default sweep used to sign-test every number it found — task parameters
#: identical across arms (`shock`, `choice_total`), grading intermediates
#: (`no_leak_ceiling`, `leak_advantage`), and run accounting — burying the
#: real metric under lines of all-tie noise.
NON_COMPARABLE_FIELDS = frozenset({
    "latency_seconds", "cost_usd", "tool_calls", "run_tokens",
    # LeakTrap bookkeeping: the trap's own parameters and the grading
    # intermediates. `score` is the metric; `leak_advantage` is derived
    # from it against a per-task ceiling and has no cross-arm direction.
    "shock", "no_leak_ceiling", "leak_advantage",
    # TemporalBench choice bookkeeping: totals are task parameters, and
    # raw correct-counts are only meaningful over their totals.
    "choice_correct", "choice_total",
    # Task/run parameters that can appear as numbers.
    "seed", "horizon",
})


def available_metrics(tasks: dict[str, dict[str, Any]]) -> list[str]:
    found = set()
    for record in tasks.values():
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found.add(key)
        extra = record.get("extra")
        if isinstance(extra, dict):
            for key, value in extra.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    found.add(key)
    for record in tasks.values():
        found.update(derived_metrics(record))
    return sorted(found - NON_COMPARABLE_FIELDS)


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def _two_sided_binomial(successes: int, trials: int) -> float:
    """Exact two-sided p under p=0.5 (used by both paired tests)."""
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(0, min(successes, trials - successes) + 1))
    return min(1.0, 2 * tail / 2 ** trials)


def mcnemar(baseline: dict[str, bool], treatment: dict[str, bool]) -> dict[str, Any]:
    """Paired test on binary outcomes over the matched subset."""
    shared = sorted(set(baseline) & set(treatment))
    fixed = sum(1 for k in shared if not baseline[k] and treatment[k])
    broken = sum(1 for k in shared if baseline[k] and not treatment[k])
    discordant = fixed + broken
    return {"test": "mcnemar_exact", "n": len(shared), "treatment_fixed": fixed,
            "treatment_broke": broken,
            "p_value": _two_sided_binomial(min(fixed, broken), discordant)}


def sign_test(baseline: dict[str, float], treatment: dict[str, float],
              lower_is_better: bool) -> dict[str, Any]:
    """Paired test on a continuous metric: how often does treatment win?"""
    shared = sorted(set(baseline) & set(treatment))
    wins = ties = 0
    for key in shared:
        difference = treatment[key] - baseline[key]
        if difference == 0:
            ties += 1
        elif (difference < 0) == lower_is_better:
            wins += 1
    trials = len(shared) - ties
    return {"test": "sign_exact", "n": len(shared), "ties": ties,
            "treatment_wins": wins, "treatment_losses": trials - wins,
            "p_value": _two_sided_binomial(min(wins, trials - wins), trials)}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def penalized_mean(baseline_values: dict[str, float],
                   treatment_values: dict[str, float],
                   answered_by_baseline: set[str]) -> dict[str, Any] | None:
    """Treatment mean with abstentions imputed at the baseline's score.

    A scored-only mean rewards abstention: refuse the hard tasks and the
    average improves. The matched subset fixes the comparison but hides
    the cost of refusing. Here every task the baseline answered and the
    treatment did not is charged to the treatment at the baseline's own
    result — the outcome a caller falls back to when Gnomon declines.

    This is a *lower* bound on the cost of abstention: a real fallback
    (seasonal-naive on the same task) may do worse than the baseline did.
    """
    missing = sorted(answered_by_baseline - set(treatment_values))
    if not missing:
        return None
    imputed = dict(treatment_values)
    for task in missing:
        imputed[task] = baseline_values[task]
    common = sorted(set(baseline_values) & set(imputed))
    if not common:
        return None
    return {
        "abstentions_imputed": len(missing),
        "baseline_mean": round(statistics.mean(baseline_values[t] for t in common), 6),
        "treatment_mean": round(statistics.mean(imputed[t] for t in common), 6),
        "basis": "abstentions charged at the baseline's score on the same task",
    }


def compare(baseline: dict[str, Any], treatment: dict[str, Any],
            metric: str | None = None) -> dict[str, Any]:
    """Compare two arms on the tasks both answered, or refuse and say why."""
    problems = incompatibilities(baseline["manifest"], treatment["manifest"])
    missing_manifest = [run["name"] for run in (baseline, treatment)
                        if not run["manifest"]]
    shared = sorted(set(baseline["tasks"]) & set(treatment["tasks"]))
    if problems:
        return {"comparable": False, "reason": "; ".join(problems),
                "baseline": baseline["name"], "treatment": treatment["name"]}
    if not shared:
        return {"comparable": False,
                "reason": (f"no task ids in common "
                           f"({len(baseline['tasks'])} vs {len(treatment['tasks'])} tasks)"),
                "baseline": baseline["name"], "treatment": treatment["name"]}

    result: dict[str, Any] = {
        "comparable": True,
        "baseline": baseline["name"], "treatment": treatment["name"],
        "matched_tasks": len(shared),
        "baseline_only": len(set(baseline["tasks"]) - set(treatment["tasks"])),
        "treatment_only": len(set(treatment["tasks"]) - set(baseline["tasks"])),
    }
    if missing_manifest:
        result["warning"] = (
            "no manifest for " + ", ".join(missing_manifest)
            + ": comparability could not be verified, only assumed"
        )

    # Binary success, when the adapters record it. Rows the harness voided
    # (row_abstained: a breached cap, a run that never submitted) are
    # excluded pairwise and counted: scoring them as failures reported a
    # harness cap as a model failure — in either arm, on exactly the rows
    # where the harness lost the answer.
    if all("success" in run["tasks"][shared[0]] for run in (baseline, treatment)):
        voided_pairs = [k for k in shared
                        if is_voided(baseline["tasks"][k])
                        or is_voided(treatment["tasks"][k])]
        graded = [k for k in shared if k not in set(voided_pairs)]
        if voided_pairs:
            result["success_voided_excluded"] = {
                "pairs": len(voided_pairs),
                "baseline_voided": sum(
                    1 for k in voided_pairs if is_voided(baseline["tasks"][k])),
                "treatment_voided": sum(
                    1 for k in voided_pairs if is_voided(treatment["tasks"][k])),
                "basis": "rows the harness ended without an answer are not "
                         "wrong answers; they are excluded from the success "
                         "test and counted here",
            }
        if graded:
            base_success = {k: bool(baseline["tasks"][k].get("success"))
                            for k in graded}
            treat_success = {k: bool(treatment["tasks"][k].get("success"))
                             for k in graded}
            result["success_rate"] = {
                "baseline": round(sum(base_success.values()) / len(graded), 4),
                "treatment": round(sum(treat_success.values()) / len(graded), 4),
            }
            result["success_test"] = mcnemar(base_success, treat_success)
            # `success` does not mean one thing across adapters (TemporalBench
            # T2/T4 records completion, T1/T3 all-choices-correct). Where rows
            # declare their basis, a pooled rate over mixed bases is a blend,
            # so the per-basis split is reported beside it.
            bases: dict[str, list[str]] = {}
            for k in graded:
                basis = (baseline["tasks"][k].get("success_basis")
                         or treatment["tasks"][k].get("success_basis"))
                if basis:
                    bases.setdefault(str(basis), []).append(k)
            if len(bases) > 1:
                result["success_by_basis"] = {
                    basis: {
                        "n": len(keys),
                        "baseline": round(
                            sum(base_success[k] for k in keys) / len(keys), 4),
                        "treatment": round(
                            sum(treat_success[k] for k in keys) / len(keys), 4),
                    }
                    for basis, keys in sorted(bases.items())
                }

    # A continuous metric, when one is present in both arms.
    candidates = ([metric] if metric else
                  [m for m in available_metrics(baseline["tasks"])
                   if m in available_metrics(treatment["tasks"])])
    for name in candidates:
        base_values = {k: metric_value(baseline["tasks"][k], name) for k in shared}
        treat_values = {k: metric_value(treatment["tasks"][k], name) for k in shared}
        paired = {k for k in shared
                  if base_values.get(k) is not None and treat_values.get(k) is not None}
        base_values = {k: base_values[k] for k in paired}
        treat_values = {k: treat_values[k] for k in paired}
        lower, direction_recognised = metric_direction(name)
        penalized = penalized_mean(
            {k: v for k, v in
             ((k, metric_value(baseline["tasks"][k], name))
              for k in baseline["tasks"]) if v is not None},
            {k: v for k, v in
             ((k, metric_value(treatment["tasks"][k], name))
              for k in treatment["tasks"]) if v is not None},
            {k for k in baseline["tasks"]
             if metric_value(baseline["tasks"][k], name) is not None},
        )
        # A treatment that abstains on most tasks leaves too small a
        # matched subset to test — but that is precisely when the
        # penalized view is the only honest summary, so the entry is
        # emitted with whichever halves exist.
        if len(paired) < 2 and not penalized:
            continue
        entry: dict[str, Any] = {
            "scored_by_both": len(paired),
            "lower_is_better": lower,
            "direction_recognised": direction_recognised,
        }
        if not direction_recognised:
            entry["direction_warning"] = (
                f"metric name {name!r} matches no known direction token; "
                f"treated as higher-is-better. If that is wrong the wins and "
                f"means below are inverted — add the metric to "
                f"benchmarks.report.LOWER_IS_BETTER or HIGHER_IS_BETTER."
            )
        if penalized:
            entry["penalized"] = penalized
        if len(paired) >= 2:
            entry.update({
                "baseline_mean": round(statistics.mean(base_values.values()), 6),
                "treatment_mean": round(statistics.mean(treat_values.values()), 6),
                "baseline_median": round(statistics.median(base_values.values()), 6),
                "treatment_median": round(statistics.median(treat_values.values()), 6),
                "test": sign_test(base_values, treat_values, lower),
            })
        result.setdefault("metrics", {})[name] = entry
    return result


def cost_of(run: dict[str, Any]) -> dict[str, Any]:
    usage = (run.get("summary") or {}).get("llm_usage") or {}
    return {"cost_usd": usage.get("cost_usd"), "requests": usage.get("requests"),
            "truncation_escalations": usage.get("truncation_escalations")}


def format_comparison(result: dict[str, Any]) -> str:
    head = f"{result['baseline']}  vs  {result['treatment']}"
    if not result["comparable"]:
        return f"{head}\n  REFUSED: {result['reason']}"
    lines = [head, f"  matched tasks: {result['matched_tasks']}"
                   f"  (baseline-only {result['baseline_only']},"
                   f" treatment-only {result['treatment_only']})"]
    if result.get("warning"):
        lines.append(f"  warning: {result['warning']}")
    if result.get("success_voided_excluded"):
        voided = result["success_voided_excluded"]
        lines.append(
            f"  harness-voided rows excluded from the success test: "
            f"{voided['pairs']} pair(s) "
            f"(baseline {voided['baseline_voided']}, "
            f"treatment {voided['treatment_voided']})"
        )
    if "success_rate" in result:
        test = result["success_test"]
        lines.append(
            f"  success: {result['success_rate']['baseline']:.3f} ->"
            f" {result['success_rate']['treatment']:.3f}"
            f"  (fixed {test['treatment_fixed']}, broke {test['treatment_broke']},"
            f" p={test['p_value']:.4f})"
        )
        for basis, split in (result.get("success_by_basis") or {}).items():
            lines.append(
                f"    {basis} (n={split['n']}): "
                f"{split['baseline']:.3f} -> {split['treatment']:.3f}"
            )
        if result.get("success_by_basis"):
            lines.append(
                "    (success means different things per basis; the pooled "
                "rate above blends them)"
            )
    for name, entry in (result.get("metrics") or {}).items():
        direction = "lower better" if entry["lower_is_better"] else "higher better"
        if not entry.get("direction_recognised", True):
            direction = "direction UNRECOGNISED, assumed higher better"
        test = entry.get("test")
        if test is None:
            lines.append(f"  {name} ({direction}): too few tasks scored by both"
                         f" ({entry['scored_by_both']}) to test")
        else:
            lines.append(
                f"  {name} ({direction}, n={entry['scored_by_both']}):"
                f" mean {entry['baseline_mean']:.4g} -> {entry['treatment_mean']:.4g},"
                f" median {entry['baseline_median']:.4g} -> {entry['treatment_median']:.4g},"
                f" wins {test['treatment_wins']}/{test['treatment_wins'] + test['treatment_losses']},"
                f" p={test['p_value']:.4f}"
            )
        if entry.get("penalized"):
            pen = entry["penalized"]
            lines.append(
                f"    penalized ({pen['abstentions_imputed']} abstentions charged"
                f" at the baseline's score): {pen['baseline_mean']:.4g}"
                f" -> {pen['treatment_mean']:.4g}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", required=True,
                        help="directory of run directories")
    parser.add_argument("--compare", nargs=2, action="append", default=None,
                        metavar=("BASELINE", "TREATMENT"),
                        help="compare two run names; repeatable")
    parser.add_argument("--metric", default=None,
                        help="restrict continuous comparison to this metric")
    parser.add_argument("--json", action="store_true",
                        help="emit the comparison objects as JSON")
    args = parser.parse_args()

    root = Path(args.root)
    runs = {path.name: load_run(path) for path in sorted(root.iterdir())
            if path.is_dir()}
    if not runs:
        raise SystemExit(f"no run directories under {root}")

    pairs = args.compare
    if not pairs:
        # Default: pair every control-ish arm with every other arm of the
        # same benchmark, so a bare invocation still says something.
        pairs = []
        by_benchmark: dict[str, list[str]] = {}
        for name, run in runs.items():
            key = run["manifest"].get("benchmark") or name.split("-")[0]
            by_benchmark.setdefault(key, []).append(name)
        for names in by_benchmark.values():
            controls = [n for n in names
                        if any(token in n for token in ("control", "direct"))]
            others = [n for n in names if n not in controls]
            for control in controls:
                for other in others:
                    pairs.append([control, other])

    results = []
    for baseline_name, treatment_name in pairs:
        for name in (baseline_name, treatment_name):
            if name not in runs:
                raise SystemExit(f"no run directory named {name!r} under {root}")
        results.append(compare(runs[baseline_name], runs[treatment_name],
                               metric=args.metric))

    if args.json:
        print(json.dumps({"comparisons": results,
                          "cost": {n: cost_of(r) for n, r in runs.items()}},
                         indent=2))
        return 0

    for result in results:
        print(format_comparison(result))
        print()
    total = sum(cost_of(run)["cost_usd"] or 0.0 for run in runs.values())
    print(f"total LLM cost across {len(runs)} runs: ${total:.2f}")
    if not pairs:
        print("no comparable pairs found; name arms *-control / *-direct "
              "and their treatments consistently, or pass --compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
