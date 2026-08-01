"""Regenerate ``tsqa_smoke.jsonl``.

The smoke corpus exists so the whole harness — loading, prompting, the tool
loop, scoring, reporting — can be exercised offline with no key and no
download. It is a wiring check, not a benchmark: sixteen hand-specified items
with answers that follow from construction. Never quote accuracy on it.

    python benchmarks/aionbench/data/generate_smoke.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tsqa_smoke.jsonl"
LENGTH = 96
SEED = 20260801


def _round(values: list[float]) -> list[float]:
    return [round(value, 3) for value in values]


def build() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    noise = lambda scale: [rng.gauss(0, scale) for _ in range(LENGTH)]

    flat = [50.0 + item for item in noise(1.0)]
    rising = [50.0 + 0.6 * index + item for index, item in enumerate(noise(1.0))]
    falling = [110.0 - 0.5 * index + item for index, item in enumerate(noise(1.0))]
    seasonal = [50.0 + 12 * math.sin(2 * math.pi * index / 12) + item
                for index, item in enumerate(noise(0.8))]
    seasonal_trend = [40.0 + 0.4 * index + 10 * math.sin(2 * math.pi * index / 12) + item
                      for index, item in enumerate(noise(0.8))]

    spiked = list(flat)
    spiked[61] = 96.0

    shifted = list(flat)
    shifted = shifted[:48] + [value + 25.0 for value in shifted[48:]]

    quiet = [50.0 + item for item in noise(0.3)]
    loud = [50.0 + item for item in noise(6.0)]

    lead = [50.0 + 10 * math.sin(2 * math.pi * index / 16) + item
            for index, item in enumerate(noise(0.5))]
    lag = [50.0] * 4 + [value + rng.gauss(0, 0.5) for value in lead[:-4]]

    items: list[dict[str, object]] = [
        {
            "id": "trend-up",
            "question": "Which statement best describes the long-run behaviour of this series?",
            "options": ["It trends upward", "It trends downward", "It has no trend"],
            "answer": "A",
            "series": [{"name": "metric", "values": _round(rising)}],
            "category": "trend",
        },
        {
            "id": "trend-down",
            "question": "Which statement best describes the long-run behaviour of this series?",
            "options": ["It trends upward", "It trends downward", "It has no trend"],
            "answer": "B",
            "series": [{"name": "metric", "values": _round(falling)}],
            "category": "trend",
        },
        {
            "id": "trend-none",
            "question": "Which statement best describes the long-run behaviour of this series?",
            "options": ["It trends upward", "It trends downward", "It has no trend"],
            "answer": "C",
            "series": [{"name": "metric", "values": _round(flat)}],
            "category": "trend",
        },
        {
            "id": "season-present",
            "question": "Does this series show a repeating seasonal cycle, and if so at what period?",
            "options": ["No seasonality", "Seasonality with period 12", "Seasonality with period 5"],
            "answer": "B",
            "series": [{"name": "metric", "values": _round(seasonal)}],
            "category": "seasonality",
        },
        {
            "id": "season-absent",
            "question": "Does this series show a repeating seasonal cycle, and if so at what period?",
            "options": ["No seasonality", "Seasonality with period 12", "Seasonality with period 5"],
            "answer": "A",
            "series": [{"name": "metric", "values": _round(flat)}],
            "category": "seasonality",
        },
        {
            "id": "season-and-trend",
            "question": "Which components are present in this series?",
            "options": [
                "Trend only", "Seasonality only", "Both trend and seasonality",
                "Neither", "A single level shift",
            ],
            "answer": "C",
            "series": [{"name": "metric", "values": _round(seasonal_trend)}],
            "category": "decomposition",
        },
        {
            "id": "anomaly-spike",
            "question": "The series contains at most one anomalous observation. Where is it?",
            "options": [
                "Around index 20", "Around index 61", "Around index 88",
                "There is no anomaly", "Every observation is anomalous",
            ],
            "answer": "B",
            "series": [{"name": "metric", "values": _round(spiked)}],
            "category": "anomaly",
        },
        {
            "id": "anomaly-none",
            "question": "The series contains at most one anomalous observation. Where is it?",
            "options": [
                "Around index 20", "Around index 61", "Around index 88",
                "There is no anomaly", "Every observation is anomalous",
            ],
            "answer": "D",
            "series": [{"name": "metric", "values": _round(flat)}],
            "category": "anomaly",
        },
        {
            "id": "level-shift",
            "question": "What kind of change occurs in this series?",
            "options": [
                "A sustained level shift about halfway through",
                "A single transient spike",
                "A gradual linear drift with no break",
            ],
            "answer": "A",
            "series": [{"name": "metric", "values": _round(shifted)}],
            "category": "changepoint",
        },
        {
            "id": "transient",
            "question": "What kind of change occurs in this series?",
            "options": [
                "A sustained level shift about halfway through",
                "A single transient spike",
                "A gradual linear drift with no break",
            ],
            "answer": "B",
            "series": [{"name": "metric", "values": _round(spiked)}],
            "category": "changepoint",
        },
        {
            "id": "noise-compare",
            "question": "Which of the two series is noisier?",
            "options": ["series_a", "series_b", "They are equally noisy"],
            "answer": "B",
            "series": [
                {"name": "series_a", "values": _round(quiet)},
                {"name": "series_b", "values": _round(loud)},
            ],
            "category": "noise",
        },
        {
            "id": "noise-compare-2",
            "question": "Which of the two series is noisier?",
            "options": ["series_a", "series_b", "They are equally noisy"],
            "answer": "A",
            "series": [
                {"name": "series_a", "values": _round(loud)},
                {"name": "series_b", "values": _round(quiet)},
            ],
            "category": "noise",
        },
        {
            "id": "lead-lag",
            "question": "series_b was constructed from series_a. What is their relationship?",
            "options": [
                "series_b leads series_a by roughly 4 steps",
                "series_b lags series_a by roughly 4 steps",
                "The two series are unrelated",
            ],
            "answer": "B",
            "series": [
                {"name": "series_a", "values": _round(lead)},
                {"name": "series_b", "values": _round(lag)},
            ],
            "category": "causality",
        },
        {
            "id": "unrelated",
            "question": "What is the relationship between these two series?",
            "options": [
                "series_b leads series_a by roughly 4 steps",
                "series_b lags series_a by roughly 4 steps",
                "The two series are unrelated",
            ],
            "answer": "C",
            "series": [
                {"name": "series_a", "values": _round(quiet)},
                {"name": "series_b", "values": _round(loud)},
            ],
            "category": "causality",
        },
        {
            "id": "forecast-direction",
            "question": "If the pattern continues, the next 10 observations will be",
            "options": [
                "higher than the last 10 on average",
                "lower than the last 10 on average",
                "about the same as the last 10 on average",
            ],
            "answer": "A",
            "series": [{"name": "metric", "values": _round(rising)}],
            "category": "forecast",
        },
        {
            "id": "forecast-flat",
            "question": "If the pattern continues, the next 10 observations will be",
            "options": [
                "higher than the last 10 on average",
                "lower than the last 10 on average",
                "about the same as the last 10 on average",
            ],
            "answer": "C",
            "series": [{"name": "metric", "values": _round(flat)}],
            "category": "forecast",
        },
    ]
    return items


def main() -> None:
    rows = build()
    OUT.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} items to {OUT}")


if __name__ == "__main__":
    main()
