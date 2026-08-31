"""Offline, deterministic HTML renderings of immutable Gnomon artifacts."""

from __future__ import annotations

from html import escape
from math import isfinite
from typing import Any


_COLOURS = {
    "supported": "#15803d",
    "conditionally_supported": "#a16207",
    "best_effort": "#b91c1c",
    "unsupported": "#64748b",
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _forecast_chart(result: dict[str, Any], history: list[dict[str, Any]] | None = None) -> str:
    rows = result.get("forecast") or []
    if not isinstance(rows, list):
        return ""
    plotted = [row for row in rows if isinstance(row, dict)]
    observed = [row for row in (history or []) if isinstance(row, dict)]
    values = [
        value for row in plotted
        for value in (_number(row.get("q10")), _number(row.get("q50")),
                      _number(row.get("q90"))) if value is not None
    ]
    threshold = _number((result.get("threshold") or {}).get("value"))
    if threshold is not None:
        values.append(threshold)
    values.extend(value for row in observed
                  if (value := _number(row.get("value"))) is not None)
    if not plotted or not values:
        return '<p class="empty">No numeric path was published.</p>'
    width, height, pad = 760, 260, 28
    low, high = min(values), max(values)
    if high <= low:
        high, low = high + .5, low - .5

    total = len(observed) + len(plotted)

    def x(index: int) -> float:
        return pad + index * (width - 2 * pad) / max(1, total - 1)

    def y(value: float) -> float:
        return pad + (high - value) * (height - 2 * pad) / (high - low)

    offset = len(observed)
    history_line = [(x(index), y(value)) for index, row in enumerate(observed)
                    if (value := _number(row.get("value"))) is not None]
    median = [(x(offset + index), y(value)) for index, row in enumerate(plotted)
              if (value := _number(row.get("q50", row.get("point")))) is not None]
    upper = [(x(offset + index), y(value)) for index, row in enumerate(plotted)
             if (value := _number(row.get("q90"))) is not None]
    lower = [(x(offset + index), y(value)) for index, row in reversed(list(enumerate(plotted)))
             if (value := _number(row.get("q10"))) is not None]
    polygon = " ".join(f"{left:.1f},{top:.1f}" for left, top in upper + lower)
    line = " ".join(f"{left:.1f},{top:.1f}" for left, top in median)
    threshold_svg = ""
    if threshold is not None:
        threshold_svg = (
            f'<line x1="{pad}" y1="{y(threshold):.1f}" x2="{width-pad}" '
            f'y2="{y(threshold):.1f}" class="threshold" />'
            f'<text x="{pad + 3}" y="{y(threshold) - 5:.1f}">threshold '
            f'{escape(format(threshold, ".5g"))}</text>'
        )
    points = []
    for index, row in enumerate(plotted):
        value = _number(row.get("q50", row.get("point")))
        if value is None:
            continue
        tier = str(row.get("tier") or result.get("support") or "unsupported")
        colour = _COLOURS.get(tier, "#334155")
        points.append(
            f'<circle cx="{x(offset + index):.1f}" cy="{y(value):.1f}" r="3" '
            f'fill="{colour}"><title>{escape(str(row.get("timestamp", index)))}: '
            f'{escape(format(value, ".6g"))} ({escape(tier)})</title></circle>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Forecast median and 10th to 90th percentile interval">'
        f'<polygon points="{polygon}" class="band" />'
        f'{threshold_svg}<polyline points="{" ".join(f"{left:.1f},{top:.1f}" for left, top in history_line)}" class="history" />'
        f'<polyline points="{line}" class="median" />'
        f'{"".join(points)}</svg>'
    )


def _support_panel(payload: dict[str, Any]) -> str:
    items: list[str] = []
    assessments = []
    if isinstance(payload.get("support_assessment"), dict):
        assessments.append(payload["support_assessment"])
    for result in payload.get("results") or []:
        if isinstance(result, dict) and isinstance(result.get("support_assessment"), dict):
            assessments.append(result["support_assessment"])
        if isinstance(result, dict):
            for warning in result.get("warnings") or []:
                items.append(f"<li><strong>Warning:</strong> {escape(str(warning))}</li>")
            for note in result.get("notes") or []:
                items.append(f"<li><strong>Note:</strong> {escape(str(note))}</li>")
    for trigger in payload.get("triggers") or []:
        if isinstance(trigger, dict) and isinstance(
            trigger.get("support_assessment"), dict,
        ):
            assessments.append(trigger["support_assessment"])
    for assessment in assessments:
        for group, label in (("reasons", "Reason"), ("disclosures", "Disclosure"),
                             ("recovery_actions", "Next step")):
            for entry in assessment.get(group) or []:
                if isinstance(entry, dict):
                    message = entry.get("message") or entry.get("description") or entry
                    code = entry.get("code") or entry.get("action") or ""
                else:
                    message, code = entry, ""
                items.append(
                    f"<li><strong>{label}{' · ' + escape(str(code)) if code else ''}:</strong> "
                    f"{escape(str(message))}</li>"
                )
    if not items:
        items.append("<li>No additional limitations or disclosures.</li>")
    return "<section><h2>Disclosures and next steps</h2><ul>" + "".join(items) + "</ul></section>"


def render_artifact_html(
    artifact_id: str, payload: dict[str, Any],
    *, history: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Render only fields persisted in ``artifact.json``.

    The report therefore cannot drift from the receipt, and can be regenerated
    without access to the source system.  Raw history is intentionally not
    reopened from a mutable input path; a future artifact schema may embed a
    governed history preview explicitly.
    """
    title = f"Gnomon artifact {artifact_id}"
    task = payload.get("task") or {}
    kind = (task.get("task_type") if isinstance(task, dict) else None) or "result"
    sections = []
    results = payload.get("results") or []
    for result in results:
        if not isinstance(result, dict):
            continue
        series = escape(str(result.get("series") or "series"))
        from .support import weakest_support_tier
        rows = result.get("forecast") or []
        support = escape(str(weakest_support_tier(
            [result.get("support"),
             (result.get("support_assessment") or {}).get("status"),
             *(row.get("tier") for row in rows if isinstance(row, dict))],
            published=bool(rows),
        ) or "unknown"))
        chart = _forecast_chart(result, (history or {}).get(str(result.get("series"))))
        details = [f"<dt>Support</dt><dd>{support}</dd>"]
        for key, label in (("selected_model", "Selected model"),
                           ("strongest_baseline", "Robust baseline")):
            if result.get(key) is not None:
                details.append(f"<dt>{label}</dt><dd>{escape(str(result[key]))}</dd>")
        sections.append(
            f"<section><h2>{series}</h2><dl>{''.join(details)}</dl>{chart}</section>"
        )
    for trigger in payload.get("triggers") or []:
        if isinstance(trigger, dict):
            assessment = trigger.get("support_assessment") or {}
            from .support import weakest_support_tier
            support = weakest_support_tier(
                [assessment.get("status"), assessment.get("legacy_support"),
                 (trigger.get("horizon_event") or {}).get("support")],
                published=bool(trigger.get("horizon_event")),
            ) or "unknown"
            sections.append(
                f"<section><h2>Trigger · {escape(str(trigger.get('series', 'series')))}</h2>"
                f"<dl><dt>Support</dt><dd>{escape(str(support))}</dd>"
                f"<dt>Armed</dt><dd>{escape(str(trigger.get('armed')))}</dd></dl></section>"
            )
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        sections.append(
            f"<section><h2>Decision</h2><p>Selected: "
            f"<strong>{escape(str(evaluation.get('selected', 'none')))}</strong></p></section>"
        )
    if not sections:
        sections.append("<section><p class=\"empty\">This artifact contains no plottable numeric path.</p></section>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
body{{font:15px/1.5 system-ui,sans-serif;color:#172033;max-width:920px;margin:2rem auto;padding:0 1rem;background:#f8fafc}}
header,section{{background:white;border:1px solid #dbe3ed;border-radius:10px;padding:1rem 1.25rem;margin:1rem 0}}
h1,h2{{line-height:1.2}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:.25rem 1rem}}dd{{margin:0}}
svg{{width:100%;height:auto;background:#fff}}.band{{fill:#dbeafe}}.history{{fill:none;stroke:#334155;stroke-width:1.5}}.median{{fill:none;stroke:#1d4ed8;stroke-width:2}}
.threshold{{stroke:#b91c1c;stroke-width:1.5;stroke-dasharray:6 4}}svg text{{font-size:11px;fill:#7f1d1d}}.empty{{color:#64748b}}
</style></head><body><header><h1>{escape(title)}</h1><p>Type: {escape(str(kind))} · Created: {escape(str(payload.get('created_at', 'unknown')))}</p>
<p>This offline view is rendered only from the immutable artifact. Open <code>artifact.json</code> for the complete receipt.</p></header>
{''.join(sections)}{_support_panel(payload)}</body></html>"""
