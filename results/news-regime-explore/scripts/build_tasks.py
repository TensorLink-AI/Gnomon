"""Build a 50-task surrogate for MTBench finance_long (30 daily bars in,
7 out) from real S&P 500 daily closes.

The official MTBench processed datasets live only on Hugging Face, which
this environment's egress policy denies (403 on CONNECT). This builder
substitutes real large-cap daily closes from the plotly/datasets mirror of
the Kaggle S&P 500 5-year dump (2013-02 .. 2018-02) — the same asset
class, bar convention (trading days), window shape (30 in / 7 out), and
price scale regime as MTBench finance_long. No news text is attached: the
experiments this feeds are all LLM-free.

Deterministic: fixed seed, fixed eligibility rule, fixed CSV input.

Usage: python build_tasks.py <all_stocks_5yr.csv> <output_dir>
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 20260803
N_TASKS = 50
N_IN, N_OUT = 30, 7
# Keep the price-scale regime comparable to the official run's aggregate
# numbers (control MSE ~6.5 at MAPE ~2.8% implies prices of order $50-150):
# a $900 stock at the same relative error would dominate every MSE mean.
PRICE_BAND = (30.0, 200.0)


def main(csv_path: str, out_dir: str) -> None:
    series: dict[str, list[float]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            close = row.get("close")
            if close:
                try:
                    series[row["Name"]].append(float(close))
                except ValueError:
                    continue

    tickers = sorted(
        name for name, values in series.items()
        if len(values) >= 400
        and PRICE_BAND[0] <= sorted(values)[len(values) // 2] <= PRICE_BAND[1]
    )
    rng = random.Random(SEED)
    picks = rng.sample(tickers, N_TASKS)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for index, ticker in enumerate(picks):
        values = series[ticker]
        start = rng.randrange(0, len(values) - (N_IN + N_OUT))
        window = values[start : start + N_IN + N_OUT]
        task = {
            "input_window": window[:N_IN],
            "output_window": window[N_IN:],
            "input_timestamps": list(range(start, start + N_IN)),
            "text": None,
            "ticker": ticker,
            "start_index": start,
        }
        (out / f"task_{index:03d}_{ticker}.json").write_text(
            json.dumps(task) + "\n", encoding="utf-8"
        )
    print(f"wrote {N_TASKS} tasks to {out} from {len(tickers)} eligible tickers")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
