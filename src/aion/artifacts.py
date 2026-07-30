from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .contracts import AionError, ForecastArtifact


def write_artifact(
    artifact: ForecastArtifact, output_parent: str,
    lineage: dict[str, Any] | None = None,
) -> Path:
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
        with (temporary / "evidence.jsonl").open("w", encoding="utf-8") as handle:
            for record in artifact.evidence:
                handle.write(json.dumps(record.__dict__, allow_nan=False) + "\n")
        if lineage is not None:
            with (temporary / "lineage.json").open("w", encoding="utf-8") as handle:
                json.dump(lineage, handle, indent=2, allow_nan=False)
                handle.write("\n")
        with (temporary / "forecast.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["series", "timestamp", "point", "q10", "q50", "q90"])
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
                lines.append(f"- Final-test 80% interval coverage: {result.interval_coverage:.1%}")
            if result.selected_model == "last_value" and result.forecast:
                lines.append(
                    "- Note: no model beat the last-value baseline on backtest; "
                    "the point forecast is a flat line at the last observed value."
                )
            lines.extend(f"- Warning: {warning}" for warning in result.warnings)
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
        (temporary / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        os.replace(temporary, final)
    except Exception:
        # Preserve the temporary directory for diagnosis; never expose it as a complete run.
        raise
    return final


def write_json_artifact(
    artifact_id: str, payload: dict[str, Any], output_parent: str,
    lineage: dict[str, Any] | None = None,
) -> Path:
    """Immutable artifact directory for non-forecast macros: artifact.json
    plus lineage.json, atomically placed, first write wins."""
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
    os.replace(temporary, final)
    return final


def read_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    """Load a stored artifact.json, enforcing the schema-versioning rule."""
    from .versioning import ensure_readable

    path = Path(artifact_dir).expanduser() / "artifact.json"
    if not path.is_file():
        raise AionError("ARTIFACT_NOT_FOUND", f"No artifact.json under {path.parent}")
    data = json.loads(path.read_text(encoding="utf-8"))
    ensure_readable(data.get("schema_version"))
    return data
