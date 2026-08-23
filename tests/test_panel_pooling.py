from __future__ import annotations

from datetime import date, timedelta

from gnomon.panel_pooling import PANEL_POOLED_TREND, PanelTrendCandidate
from gnomon.runtime import forecast, forecast_multi


def _panel(length: int = 15) -> dict[str, list[float]]:
    return {
        f"metric_{index}": [100.0 * (index + 1) + step * (index + 1)
                            for step in range(length)]
        for index in range(5)
    }


def test_pooled_candidate_never_reads_donor_future() -> None:
    series = _panel()
    candidate = PanelTrendCandidate("metric_0", series)
    before = candidate(10, 3)
    for donor in candidate.donors:
        series[donor][10:] = [999999.0] * 5
    # Candidate copied its immutable panel and, more importantly, origin 10
    # only ever exposes prefixes ending at 10.
    assert candidate(10, 3) == before


def test_short_wide_panel_can_earn_distinct_pooled_admission(tmp_path) -> None:
    panel = _panel()
    source = tmp_path / "panel.csv"
    columns = list(panel)
    rows = ["timestamp," + ",".join(columns)]
    start = date(2026, 1, 1)
    for step in range(15):
        rows.append(
            f"{(start + timedelta(days=step)).isoformat()},"
            + ",".join(str(panel[name][step]) for name in columns)
        )
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    artifact, _ = forecast_multi(
        str(source), time_column="timestamp", target_columns=columns,
        horizon=5, frequency="D", output=str(tmp_path / "out"),
        max_workers=1,
    )
    assert {result.selected_model for result in artifact.results} == {
        PANEL_POOLED_TREND}
    admission = [item.payload for item in artifact.evidence
                 if item.kind == "model_admission"]
    assert len(admission) == len(columns)
    assert {item["state"] for item in admission} == {"pooled_validated"}
    assert all(item["evidence"]["independent_folds"] == 1
               for item in admission)


def test_equivalent_long_panel_uses_same_pooling_lane(tmp_path) -> None:
    panel = _panel()
    source = tmp_path / "panel-long.csv"
    rows = ["timestamp,series,value"]
    start = date(2026, 1, 1)
    for name, values in panel.items():
        rows.extend(
            f"{(start + timedelta(days=step)).isoformat()},{name},{value}"
            for step, value in enumerate(values)
        )
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        series_column="series", horizon=5, frequency="D",
        output=str(tmp_path / "out"),
    )
    assert {result.selected_model for result in artifact.results} == {
        PANEL_POOLED_TREND}
    assert {result.admission["state"] for result in artifact.results} == {
        "pooled_validated"}


def test_heterogeneous_panel_rejects_itself() -> None:
    series = _panel()
    for channel, direction in (("metric_2", -1), ("metric_3", -1),
                               ("metric_4", 1)):
        level = series[channel][0]
        series[channel] = [level + direction * (index + 1) * step
                           for step in range(len(series[channel]))
                           for index in [int(channel.rsplit("_", 1)[1])]]
    candidate = PanelTrendCandidate("metric_0", series)
    assert candidate.lightweight_evidence(5, 1, 0.02) is None
