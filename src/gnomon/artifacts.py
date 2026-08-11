from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from .contracts import GnomonError, ForecastArtifact

logger = logging.getLogger(__name__)


def write_artifact(
    artifact: ForecastArtifact, output_parent: str,
    lineage: dict[str, Any] | None = None,
    output_config: Any = None,
) -> Path:
    """Write the immutable artifact directory.

    ``output_config`` is `gnomon.yaml`'s `output` section, whose six switches
    were documented and never read — a user who set `write_summary: false`
    still got a summary. `artifact.json` and `lineage.json` are not
    switchable: the artifact is the run's identity and the lineage is what
    the verifier checked, so a run without them is not a run.
    """
    write_forecast_csv = getattr(output_config, "write_forecast_csv", True)
    write_summary = getattr(output_config, "write_summary", True)
    write_evidence = getattr(output_config, "write_evidence", True)
    parent = Path(output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / artifact.forecast_id
    if final.is_dir():
        # Content-addressed IDs: an existing directory holds this exact run.
        # Artifacts are immutable, so the first write wins.
        return final
    temporary = parent / f".{artifact.forecast_id}.tmp"
    if temporary.is_dir():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        with (temporary / "artifact.json").open("w", encoding="utf-8") as handle:
            json.dump(artifact.to_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")
        if write_evidence:
            with (temporary / "evidence.jsonl").open("w", encoding="utf-8") as handle:
                for record in artifact.evidence:
                    handle.write(json.dumps(record.__dict__, allow_nan=False) + "\n")
        if lineage is not None:
            with (temporary / "lineage.json").open("w", encoding="utf-8") as handle:
                json.dump(lineage, handle, indent=2, allow_nan=False)
                handle.write("\n")
        # The first six columns are the frozen ones and keep their order and
        # meaning; any additional quantile levels the run emitted are appended
        # after them, so a reader that indexes the leading columns is
        # unaffected and one that reads by header name gains the rest.
        from .evaluation import QUANTILE_LEVELS, quantile_key

        core = ["series", "timestamp", "point", "q10", "q50", "q90"]
        present = {key for result in artifact.results
                   for row in result.forecast for key in row}
        extra = [quantile_key(level) for level in QUANTILE_LEVELS
                 if quantile_key(level) in present
                 and quantile_key(level) not in core]
        # `point` is the raw model output and every quantile is recentred on
        # the median residual, so the two are not the same number. The gap
        # ships as a column rather than being left for the reader to derive.
        if "point_bias_correction" in present:
            extra.append("point_bias_correction")
        # The unstrippable label: every published row names its tier, in
        # the CSV exactly as in the payload. Uniform forecasts repeat one
        # value; a horizon split changes it at the split point.
        if "tier" in present:
            extra.append("tier")
        if write_forecast_csv:
            with (temporary / "forecast.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=core + extra)
                writer.writeheader()
                for result in artifact.results:
                    for row in result.forecast:
                        writer.writerow({"series": result.series, **row})
        lines = [f"# Forecast {artifact.forecast_id}", ""]
        for result in artifact.results:
            lines.extend([
                f"## {result.series}", "",
                f"- Support: {result.support}",
                f"- Selected model: {result.selected_model or 'none'}",
                f"- Strongest baseline: {result.strongest_baseline or 'none'}",
            ])
            if result.interval_coverage is not None:
                # The count belongs beside the rate: one test fold of
                # `horizon` points makes "100%" nearly information-free.
                lines.append(
                    f"- Final-test 80% interval coverage: "
                    f"{result.interval_coverage:.1%} "
                    f"({len(result.forecast)} points, one fold)"
                )
            if result.selected_model == "last_value" and result.forecast:
                lines.append(
                    "- Note: no model beat the last-value baseline on backtest; "
                    "the point forecast is a flat line at the last observed value."
                )
            lines.extend(f"- Warning: {warning}" for warning in result.warnings)
            lines.extend(f"- Note: {note}" for note in result.notes)
            # Correct-but-surprising facts read here or nowhere: a human
            # reading summary.md is exactly the reader who would otherwise
            # take `point` for the median or a flat interval for a bug.
            for disclosure in (result.support_assessment or {}).get("disclosures", []):
                lines.append(f"- Disclosure ({disclosure['code']}): {disclosure['message']}")
            if result.threshold:
                lines.extend([
                    "", f"### Threshold {result.threshold['value']}", "",
                    f"- First timestamp with point above: {result.threshold['first_timestamp_point_above'] or 'never in horizon'}",
                    f"- First timestamp with q90 above: {result.threshold['first_timestamp_interval_above'] or 'never in horizon'}",
                    f"- Peak probability above: {max(result.threshold['probability_above']):.1%}",
                ])
            if result.covariates:
                lines.extend([
                    "", "### Covariates", "",
                    f"- Retained: {', '.join(result.covariates.get('retained', [])) or 'none'}",
                    f"- Rejected: {len(result.covariates.get('rejected', []))}",
                ])
            lines.append("")
        if write_summary:
            (temporary / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        os.replace(temporary, final)
    except Exception:
        # The partial directory is kept for diagnosis and never exposed as a
        # complete run — but nothing else would ever mention it, and the
        # next attempt at the same id silently rmtree's it. Say where it is.
        logger.warning(
            "Artifact write failed; the partial directory is retained at %s "
            "and will be removed by the next attempt at this id.", temporary,
        )
        raise
    return final


def write_json_artifact(
    artifact_id: str, payload: dict[str, Any], output_parent: str,
    lineage: dict[str, Any] | None = None,
) -> Path:
    """Immutable artifact directory for non-forecast macros: artifact.json,
    lineage.json, and summary.md, atomically placed, first write wins.

    Only forecasts used to get a human-readable rendering, so `investigate`,
    `detect`, `decide`, and `monitor` were JSON or nothing.
    """
    parent = Path(output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / artifact_id
    if final.is_dir():
        return final
    temporary = parent / f".{artifact_id}.tmp"
    if temporary.is_dir():
        shutil.rmtree(temporary)
    temporary.mkdir()
    with (temporary / "artifact.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if lineage is not None:
        with (temporary / "lineage.json").open("w", encoding="utf-8") as handle:
            json.dump(lineage, handle, indent=2, allow_nan=False)
            handle.write("\n")
    (temporary / "summary.md").write_text(
        _macro_summary(artifact_id, payload), encoding="utf-8",
    )
    os.replace(temporary, final)
    return final


def _macro_summary(artifact_id: str, payload: dict[str, Any]) -> str:
    """A human-readable rendering of any macro payload.

    Deliberately generic: it reads the shapes the macros share — a support
    assessment, a per-series `results` list — rather than switching on the
    verb, so a new macro gets a summary without a new branch here.
    """
    kind = str(payload.get("task", {}).get("task_type", "result"))
    lines = [f"# {kind.replace('_', ' ').title()} {artifact_id}", ""]
    created = payload.get("created_at")
    if created:
        lines.extend([f"- Created: {created}", ""])

    def render_support(assessment: dict[str, Any] | None, indent: str = "") -> None:
        if not assessment:
            return
        lines.append(f"{indent}- Support: {assessment.get('status', 'unknown')}")
        for reason in assessment.get("reasons", []):
            lines.append(f"{indent}- Reason ({reason['code']}): {reason['message']}")
        for disclosure in assessment.get("disclosures", []):
            lines.append(
                f"{indent}- Disclosure ({disclosure['code']}): {disclosure['message']}"
            )
        for action in assessment.get("recovery_actions", []):
            lines.append(f"{indent}- Next ({action['code']}): {action['message']}")

    render_support(payload.get("support_assessment"))

    for result in payload.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        lines.extend(["", f"## {result.get('series', 'result')}", ""])
        for key in ("classification", "onset", "detector", "selection_basis"):
            if result.get(key) is not None:
                lines.append(f"- {key.replace('_', ' ').capitalize()}: {result[key]}")
        for change in result.get("ranked_changes", []) or []:
            marker = " **(dominant)**" if change.get("dominant") else ""
            share = change.get("share_of_variation_explained")
            lines.append(
                f"- Change at {change['timestamp']}: "
                f"{change.get('before_mean'):.4g} → {change.get('after_mean'):.4g} "
                f"(shift {change.get('shift'):+.4g}"
                + (f", explains {share:.1%} of the variation" if share is not None else "")
                + f"){marker}"
            )
        for rank, explanation in enumerate(result.get("explanations", []) or [], 1):
            lines.append(f"- Explanation {rank}: {explanation}")
        if result.get("residual_uncertainty"):
            lines.append(f"- Residual uncertainty: {result['residual_uncertainty']}")
        render_support(result.get("support_assessment"))

    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("selected") is not None:
        lines.extend(["", "## Decision", "", f"- Selected: {evaluation['selected']}"])
        for item in evaluation.get("evaluations", []) or []:
            lines.append(
                f"- {item.get('action')}: feasible={item.get('feasible')}"
                + (f", expected utility {item['expected_utility']:.4g}"
                   if item.get("expected_utility") is not None else "")
            )

    for trigger in payload.get("triggers", []) or []:
        lines.extend(["", f"## Trigger — {trigger.get('series', 'series')}", ""])
        lines.append(f"- Armed: {trigger.get('armed')}")
        if trigger.get("first_alert_step") is not None:
            lines.append(f"- First alert at step: {trigger['first_alert_step']}")
        render_support(trigger.get("support_assessment"))

    lines.append("")
    return "\n".join(lines)


def read_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    """Load a stored artifact.json, enforcing the schema-versioning rule."""
    from .versioning import ensure_readable

    path = Path(artifact_dir).expanduser() / "artifact.json"
    if not path.is_file():
        raise GnomonError("ARTIFACT_NOT_FOUND", f"No artifact.json under {path.parent}")
    data = json.loads(path.read_text(encoding="utf-8"))
    ensure_readable(data.get("schema_version"))
    return data
