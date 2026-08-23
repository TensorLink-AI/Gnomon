"""Human-readable realised track records from the existing tracking store."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

from .tracking import TrackingStore


def _in_month(timestamp: str | None, month: str | None) -> bool:
    return bool(timestamp) and (month is None or str(timestamp).startswith(month + "-"))


def build_project_report(
    store: TrackingStore, project: str, *, month: str | None = None,
) -> dict[str, Any]:
    if month is not None:
        try:
            datetime.strptime(month, "%Y-%m")
        except ValueError as exc:
            raise ValueError("month must use YYYY-MM") from exc
    forecasts = [item for item in store.list_forecasts(project, limit=100000)
                 if _in_month(item.created_at, month)]
    scored = [item for item in forecasts if item.scored]
    abstained = [item for item in forecasts if item.support == "unsupported"]
    coverage = [item.coverage for item in scored if item.coverage is not None]
    mase = [item.mase for item in scored if item.mase is not None]
    decisions = [item for item in store.list_decision_artifacts(project)
                 if _in_month(item.created_at, month)]
    resolved_decisions = [item for item in decisions
                          if item.resolved_at is not None and item.outcome is not None]
    regrets = [item.outcome.regret for item in resolved_decisions
               if item.outcome.regret is not None]
    temporal = store.temporal_answer_receipts(project)
    temporal = [item for item in temporal if _in_month(item.get("created_at"), month)]
    abstained_answers = [item for item in temporal if item.get("support") in {
        "unsupported", "inconclusive", "abstained",
    }]
    resolved_abstentions = [item for item in abstained_answers if item.get("resolved_at")]
    return {
        "schema_version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "period": month or "all_time",
        "forecasts": {
            "registered": len(forecasts), "resolved": len(scored),
            "awaiting_actuals": len(forecasts) - len(scored),
            "unsupported": len(abstained),
            "mean_mase": mean(mase) if mase else None,
            "mean_interval_coverage": mean(coverage) if coverage else None,
            "coverage_denominator": len(coverage),
        },
        "decisions": {
            "registered": len(decisions), "resolved": len(resolved_decisions),
            "regret_scored": len(regrets),
            "mean_regret_vs_best_in_hindsight": mean(regrets) if regrets else None,
        },
        "abstentions": {
            "typed_answers": len(abstained_answers),
            "resolved": len(resolved_abstentions),
            "unresolved": len(abstained_answers) - len(resolved_abstentions),
            "note": "Resolved abstentions retain their realised outcome; this report does not relabel an abstention as right or wrong without a property-specific scoring rule.",
        },
        "models": [asdict(item) for item in store.leaderboard(project)],
        "limitations": [
            "Realised performance is observational evidence, not a causal model comparison.",
            "Metrics omit unresolved forecasts and always show their denominator.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    forecasts, decisions, abstentions = (
        report["forecasts"], report["decisions"], report["abstentions"])
    def metric(value: Any) -> str:
        return "not measured" if value is None else f"{float(value):.4g}"
    lines = [
        f"# Gnomon track record — {report['project']}", "",
        f"Period: {report['period']}", "", "## Forecasts", "",
        f"- Registered: {forecasts['registered']}",
        f"- Resolved: {forecasts['resolved']}",
        f"- Awaiting actuals: {forecasts['awaiting_actuals']}",
        f"- Unsupported: {forecasts['unsupported']}",
        f"- Mean MASE: {metric(forecasts['mean_mase'])}",
        f"- Mean interval coverage: {metric(forecasts['mean_interval_coverage'])} "
        f"(n={forecasts['coverage_denominator']})", "", "## Decisions", "",
        f"- Registered: {decisions['registered']}",
        f"- Resolved: {decisions['resolved']}",
        f"- Mean regret versus best in hindsight: "
        f"{metric(decisions['mean_regret_vs_best_in_hindsight'])} "
        f"(n={decisions['regret_scored']})", "", "## Abstentions", "",
        f"- Typed abstentions: {abstentions['typed_answers']}",
        f"- Resolved: {abstentions['resolved']}",
        f"- Unresolved: {abstentions['unresolved']}",
        f"- Note: {abstentions['note']}", "", "## Limitations", "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def render_html(report: dict[str, Any], markdown: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Gnomon track record</title>
<style>body{{font:15px/1.55 system-ui,sans-serif;max-width:850px;margin:2rem auto;padding:0 1rem;color:#172033}}pre{{white-space:pre-wrap;background:#f8fafc;border:1px solid #dbe3ed;padding:1.25rem;border-radius:10px}}</style>
</head><body><pre>{escape(markdown)}</pre></body></html>"""


def write_project_report(
    store: TrackingStore, project: str, output: str | Path,
    *, month: str | None = None,
) -> tuple[Path, Path]:
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report = build_project_report(store, project, month=month)
    markdown = render_markdown(report)
    markdown_path, html_path = destination / "report.md", destination / "report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(render_html(report, markdown), encoding="utf-8")
    return markdown_path, html_path
