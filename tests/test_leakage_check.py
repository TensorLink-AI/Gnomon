"""The trading skill's leakage checker passes a causal policy and catches look-ahead."""
import runpy
from pathlib import Path

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "skills/trade-with-gnomon/scripts/leakage_check.py"))
check_causal = CHECK["check_causal"]

ROWS = [{"time": i, "price": 100 + (i % 7) - (i % 3) + 0.1 * i} for i in range(200)]


def trailing_signal(rows):
    prices = [r["price"] for r in rows]
    return {rows[i]["time"]: (1 if prices[i] > sum(prices[i - 20:i]) / 20 else -1) for i in range(20, len(rows))}


def full_sample_zscore(rows):  # look-ahead: normalises with the whole sample's mean
    prices = [r["price"] for r in rows]
    mean = sum(prices) / len(prices)
    return {r["time"]: round(r["price"] - mean, 9) for r in rows}


def centred_window(rows):  # look-ahead: averages 5 future rows
    p = [r["price"] for r in rows]
    return {rows[i]["time"]: sum(p[i - 5:i + 6]) / len(p[i - 5:i + 6]) for i in range(5, len(rows))}


def test_causal_policy_passes():
    report = check_causal(trailing_signal, ROWS)
    assert report["pass"] and report["checked"] == 12


def test_look_ahead_is_caught():
    for leaky in (full_sample_zscore, centred_window):
        report = check_causal(leaky, ROWS)
        assert not report["pass"] and report["mismatches"]
