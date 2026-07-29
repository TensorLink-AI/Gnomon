from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .contracts import ForecastArtifact


def write_artifact(artifact: ForecastArtifact, output_parent: str) -> Path:
    parent = Path(output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / artifact.forecast_id
    temporary = parent / f".{artifact.forecast_id}.tmp"
    temporary.mkdir()
    try:
        with (temporary / "artifact.json").open("w", encoding="utf-8") as handle:
            json.dump(artifact.to_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")
        with (temporary / "evidence.jsonl").open("w", encoding="utf-8") as handle:
            for record in artifact.evidence:
                handle.write(json.dumps(record.__dict__, allow_nan=False) + "\n")
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
            lines.extend(f"- Warning: {warning}" for warning in result.warnings)
            lines.append("")
        (temporary / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        os.replace(temporary, final)
    except Exception:
        # Preserve the temporary directory for diagnosis; never expose it as a complete run.
        raise
    return final

