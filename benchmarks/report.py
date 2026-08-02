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
    python -m benchmarks.report --root results/glm52 --compare tb-control tb-aion
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

#: Lower is better for these; everything else is treated as higher-better.
LOWER_IS_BETTER = ("mse", "mae", "rmse", "mape", "rcrps", "mase", "smape")


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


def _load_aionbench(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Rows written by our adapters (`aionbench.jsonl`)."""
    path = run_dir / "aionbench.jsonl"
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
    tasks = _load_aionbench(run_dir) or _load_mtbench_official(run_dir)
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
    return sorted(found - {"latency_seconds", "cost_usd", "tool_calls"})


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
    result — the outcome a caller falls back to when Aion declines.

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

    # Binary success, when the adapters record it.
    if all("success" in run["tasks"][shared[0]] for run in (baseline, treatment)):
        base_success = {k: bool(baseline["tasks"][k].get("success")) for k in shared}
        treat_success = {k: bool(treatment["tasks"][k].get("success")) for k in shared}
        result["success_rate"] = {
            "baseline": round(sum(base_success.values()) / len(shared), 4),
            "treatment": round(sum(treat_success.values()) / len(shared), 4),
        }
        result["success_test"] = mcnemar(base_success, treat_success)

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
        lower = any(token in name.lower() for token in LOWER_IS_BETTER)
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
        }
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
    if "success_rate" in result:
        test = result["success_test"]
        lines.append(
            f"  success: {result['success_rate']['baseline']:.3f} ->"
            f" {result['success_rate']['treatment']:.3f}"
            f"  (fixed {test['treatment_fixed']}, broke {test['treatment_broke']},"
            f" p={test['p_value']:.4f})"
        )
    for name, entry in (result.get("metrics") or {}).items():
        direction = "lower better" if entry["lower_is_better"] else "higher better"
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
