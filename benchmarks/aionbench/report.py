"""Turning graded rows into a defensible table.

The headline this package cares about is not a leaderboard position — it is
the *paired* difference between a model on its own and the same model with
Aion, on identical items. So the report always reports:

- accuracy with a bootstrap interval, not a bare point estimate;
- the paired uplift with an exact McNemar p-value, so a two-point difference
  on 150 items is visibly not a result;
- the failure taxonomy, so "worse accuracy" can be distinguished from "the
  model never produced a parseable answer";
- for treatment arms, how often a correct answer was actually grounded in a
  successful Aion call.

Transport failures are excluded from every denominator; they are counted and
reported separately so the exclusion is visible rather than convenient.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import CONTROL_CONDITIONS, FAILURE_KINDS, TREATMENT_CONDITIONS
from .metrics import accuracy, bootstrap_ci, mcnemar


@dataclass(frozen=True)
class ArmSummary:
    model: str
    condition: str
    graded: int
    excluded: int
    accuracy: float
    ci_low: float
    ci_high: float
    mean_score: float | None
    failures: dict[str, int]
    tool_calls: float
    tool_calls_ok: float
    grounded_rate: float
    grounded_correct_rate: float | None
    cost_usd: float
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self) -> dict[str, Any]:
        payload = {key: getattr(self, key) for key in self.__annotations__}
        payload["failures"] = dict(self.failures)
        return payload


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read ``rows.jsonl`` from a run directory or a file path."""
    target = Path(path)
    if target.is_dir():
        target = target / "rows.jsonl"
    if not target.is_file():
        raise FileNotFoundError(f"No graded rows at {target}")
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def summarise_arm(rows: Sequence[Mapping[str, Any]]) -> ArmSummary:
    """Collapse one model-condition arm into a summary."""
    graded = [row for row in rows if row.get("correct") is not None]
    excluded = len(rows) - len(graded)
    flags = [bool(row.get("correct")) for row in graded]

    failures: dict[str, int] = {kind: 0 for kind in FAILURE_KINDS}
    for row in rows:
        kind = row.get("failure")
        if kind in failures:
            failures[kind] += 1

    scores = [row["score"] for row in graded if isinstance(row.get("score"), (int, float))]
    grounded_rows = [row for row in rows if row.get("grounded")]
    grounded_correct = [row for row in graded if row.get("grounded") and row.get("correct")]
    correct_rows = [row for row in graded if row.get("correct")]

    low, high = bootstrap_ci(flags)
    return ArmSummary(
        model=str(rows[0].get("model", "")) if rows else "",
        condition=str(rows[0].get("condition", "")) if rows else "",
        graded=len(graded),
        excluded=excluded,
        accuracy=round(accuracy(flags), 6),
        ci_low=low,
        ci_high=high,
        mean_score=round(sum(scores) / len(scores), 6) if scores else None,
        failures=failures,
        tool_calls=_mean(rows, "tool_calls"),
        tool_calls_ok=_mean(rows, "tool_calls_ok"),
        grounded_rate=round(len(grounded_rows) / len(rows), 6) if rows else 0.0,
        grounded_correct_rate=(
            round(len(grounded_correct) / len(correct_rows), 6) if correct_rows else None
        ),
        cost_usd=round(sum(float(row.get("cost_usd") or 0.0) for row in rows), 6),
        latency_seconds=_mean(rows, "latency_seconds"),
        prompt_tokens=int(sum(int(row.get("prompt_tokens") or 0) for row in rows)),
        completion_tokens=int(sum(int(row.get("completion_tokens") or 0) for row in rows)),
    )


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Full report: per-arm summaries, per-category breakdown, paired uplift."""
    by_arm: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[(str(row.get("model")), str(row.get("condition")))].append(row)

    arms = {
        f"{model}::{condition}": summarise_arm(arm_rows).to_dict()
        for (model, condition), arm_rows in sorted(by_arm.items())
    }

    return {
        "schema_version": "0.1",
        "suite": rows[0].get("suite") if rows else None,
        "rows": len(rows),
        "models": sorted({str(row.get("model")) for row in rows}),
        "conditions": sorted({str(row.get("condition")) for row in rows}),
        "arms": arms,
        "categories": _by_category(rows),
        "uplift": uplift(rows),
        "totals": {
            "cost_usd": round(sum(float(row.get("cost_usd") or 0.0) for row in rows), 6),
            "transport_failures": sum(1 for row in rows if row.get("failure") == "transport"),
        },
    }


def uplift(
    rows: Sequence[Mapping[str, Any]],
    *,
    control: str | None = None,
    treatment: str | None = None,
) -> list[dict[str, Any]]:
    """Paired control-vs-treatment comparison, per model.

    Pairs are formed on (task_id, repeat) and only over items both arms
    graded. An item one arm failed to transport is dropped from *both*, so
    the comparison never rests on a different set of questions.
    """
    present = {str(row.get("condition")) for row in rows}
    control = control or next((item for item in CONTROL_CONDITIONS if item in present), None)
    treatment = treatment or next((item for item in TREATMENT_CONDITIONS if item in present), None)
    if not control or not treatment or control == treatment:
        return []

    results: list[dict[str, Any]] = []
    for model in sorted({str(row.get("model")) for row in rows}):
        control_map = _graded_map(rows, model, control)
        treatment_map = _graded_map(rows, model, treatment)
        shared = sorted(set(control_map) & set(treatment_map))
        if not shared:
            continue

        control_flags = [control_map[key] for key in shared]
        treatment_flags = [treatment_map[key] for key in shared]
        test = mcnemar(control_flags, treatment_flags)
        control_accuracy = accuracy(control_flags)
        treatment_accuracy = accuracy(treatment_flags)
        control_error = 1.0 - control_accuracy

        results.append({
            "model": model,
            "control": control,
            "treatment": treatment,
            "paired_items": len(shared),
            "control_accuracy": round(control_accuracy, 6),
            "treatment_accuracy": round(treatment_accuracy, 6),
            "absolute_uplift": round(treatment_accuracy - control_accuracy, 6),
            "relative_error_reduction": (
                round((treatment_accuracy - control_accuracy) / control_error, 6)
                if control_error > 0 else None
            ),
            "fixed_by_treatment": test.treatment_only,
            "broken_by_treatment": test.control_only,
            "p_value": test.p_value,
            "significant_at_05": test.p_value < 0.05,
        })
    return results


def _by_category(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row.get("category") or "uncategorised")][
            f"{row.get('model')}::{row.get('condition')}"
        ].append(row)

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for category, arms in sorted(grouped.items()):
        out[category] = {}
        for arm, arm_rows in sorted(arms.items()):
            graded = [row for row in arm_rows if row.get("correct") is not None]
            flags = [bool(row.get("correct")) for row in graded]
            out[category][arm] = {
                "graded": len(graded),
                "accuracy": round(accuracy(flags), 6),
            }
    return out


def _graded_map(rows: Sequence[Mapping[str, Any]], model: str, condition: str) -> dict[tuple[str, int], bool]:
    return {
        (str(row.get("task_id")), int(row.get("repeat") or 0)): bool(row.get("correct"))
        for row in rows
        if str(row.get("model")) == model
        and str(row.get("condition")) == condition
        and row.get("correct") is not None
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key) or 0.0) for row in rows) / len(rows), 4)


# -- rendering -------------------------------------------------------------

def render_markdown(summary: Mapping[str, Any], *, title: str = "AionBench") -> str:
    lines = [f"# {title} — {summary.get('suite', 'run')}", ""]
    lines.append(
        f"{summary['rows']} graded rows across {len(summary['models'])} model(s) "
        f"and {len(summary['conditions'])} condition(s). "
        f"Total spend ${summary['totals']['cost_usd']:.4f}."
    )
    transport = summary["totals"]["transport_failures"]
    if transport:
        lines.append(
            f"\n> {transport} row(s) failed in transport and are excluded from every "
            f"accuracy denominator."
        )

    lines += ["", "## Arms", ""]
    lines.append(
        "| Model | Condition | n | Accuracy | 95% CI | Structural | Refusal | "
        "Tool calls | Grounded | Cost |"
    )
    lines.append("| --- | --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |")
    for arm in summary["arms"].values():
        lines.append(
            f"| {arm['model']} | {arm['condition']} | {arm['graded']} | "
            f"{arm['accuracy']:.1%} | {arm['ci_low']:.1%}–{arm['ci_high']:.1%} | "
            f"{arm['failures']['structural']} | {arm['failures']['refusal']} | "
            f"{arm['tool_calls']:.1f} | {arm['grounded_rate']:.0%} | "
            f"${arm['cost_usd']:.4f} |"
        )

    if summary["uplift"]:
        lines += ["", "## Paired uplift (Aion vs control, identical items)", ""]
        lines.append("| Model | Control | Treatment | Pairs | Control | Treatment | Δ | Fixed | Broken | p |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for item in summary["uplift"]:
            marker = "" if item["significant_at_05"] else " (ns)"
            lines.append(
                f"| {item['model']} | {item['control']} | {item['treatment']} | "
                f"{item['paired_items']} | {item['control_accuracy']:.1%} | "
                f"{item['treatment_accuracy']:.1%} | {item['absolute_uplift']:+.1%} | "
                f"{item['fixed_by_treatment']} | {item['broken_by_treatment']} | "
                f"{item['p_value']:.4f}{marker} |"
            )
        lines.append("")
        lines.append(
            "*Fixed* counts items the treatment got right and the control got wrong; "
            "*broken* is the reverse. The p-value is an exact two-sided McNemar test "
            "over those discordant items only. `(ns)` marks differences this run "
            "cannot distinguish from noise."
        )

    if summary["categories"]:
        lines += ["", "## By category", ""]
        arms = sorted({arm for category in summary["categories"].values() for arm in category})
        lines.append("| Category | " + " | ".join(arms) + " |")
        lines.append("| --- |" + " ---: |" * len(arms))
        for category, values in summary["categories"].items():
            cells = []
            for arm in arms:
                entry = values.get(arm)
                cells.append(f"{entry['accuracy']:.1%} ({entry['graded']})" if entry else "—")
            lines.append(f"| {category} | " + " | ".join(cells) + " |")

    lines += [
        "", "---", "",
        "Accuracy excludes transport failures. Intervals are seeded percentile "
        "bootstraps over the graded items and describe sampling noise only — they "
        "say nothing about whether the corpus is representative of your data.",
    ]
    return "\n".join(lines) + "\n"


def export_eval_compare(
    rows: Sequence[Mapping[str, Any]],
    directory: str | Path,
    *,
    model: str,
    control: str = "direct",
    treatment: str = "aion",
) -> tuple[Path, Path]:
    """Write baseline/treatment JSONL for ``aion eval compare``.

    That command requires identical task-id sets, so only items both arms
    graded are exported, and the task id carries the repeat index to keep
    multi-trial runs distinct.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    control_map = _row_map(rows, model, control)
    treatment_map = _row_map(rows, model, treatment)
    shared = sorted(set(control_map) & set(treatment_map))

    paths = []
    for condition, mapping in ((control, control_map), (treatment, treatment_map)):
        path = target / f"{_safe(model)}-{condition}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for key in shared:
                row = mapping[key]
                handle.write(json.dumps({
                    "task_id": f"{key[0]}#{key[1]}",
                    "success": bool(row.get("correct")),
                    "invented_number": bool(
                        condition in TREATMENT_CONDITIONS
                        and row.get("correct") and not row.get("grounded")
                    ),
                    "temporal_leakage": False,
                    "warning_omission": False,
                    "appropriate_abstention": row.get("failure") == "refusal",
                    "tool_calls": int(row.get("tool_calls") or 0),
                    "latency_seconds": float(row.get("latency_seconds") or 0.0),
                    "cost_usd": float(row.get("cost_usd") or 0.0),
                }) + "\n")
        paths.append(path)
    return paths[0], paths[1]


def _row_map(
    rows: Sequence[Mapping[str, Any]], model: str, condition: str,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {
        (str(row.get("task_id")), int(row.get("repeat") or 0)): row
        for row in rows
        if str(row.get("model")) == model
        and str(row.get("condition")) == condition
        and row.get("correct") is not None
    }


def _safe(text: str) -> str:
    return "".join(char if char.isalnum() or char in "-._" else "_" for char in text)


def write_report(run_dir: str | Path, rows: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Summarise a run directory and write report.json / report.md into it."""
    directory = Path(run_dir)
    data = list(rows) if rows is not None else load_rows(directory)
    summary = summarise(data)
    (directory / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (directory / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary
