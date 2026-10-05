"""Causality check for any backtest policy: no decision may change when later data is removed.

    from leakage_check import check_causal
    report = check_causal(decide, rows)          # decide(rows) -> {time: decision}
    assert report["pass"], report["mismatches"]

`decide` takes the rows known so far (sorted by `time_key`) and returns the decision it
would have made at each time: a number, a sequence or mapping of numbers, or a label.
The check runs it once on everything, then again on the rows up to each of `n_cuts`
sampled times, and requires the decision at that time to be identical (within `tol`).
A mismatch means the policy used information from after its decision time: a centred
window, a full-sample normalisation, a forward-filled future value, a fit on all data.

Pure Python, no dependencies. It finds look-ahead in the decision rule; it does not check
fills, costs or the engine.
"""
from __future__ import annotations

import math
import random


def _same(a, b, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def check_causal(decide, rows, *, time_key="time", n_cuts=12, seed=0, tol=1e-9, skip_first=0):
    """Re-run `decide` on data truncated at sampled decision times; report any changed decision.

    skip_first: ignore the first N decision times (e.g. a warm-up you do not trade).
    """
    rows = sorted(rows, key=lambda r: r[time_key])
    full = decide(rows)
    times = sorted(full)[skip_first:]
    if not times:
        return {"pass": False, "checked": 0, "mismatches": [], "error": "decide returned no decisions"}
    rng = random.Random(seed)
    cuts = sorted(rng.sample(times, min(n_cuts, len(times))))
    mismatches = []
    for t in cuts:
        part = decide([r for r in rows if r[time_key] <= t])
        if t not in part or not _same(full[t], part[t], tol):
            mismatches.append({"time": t, "full": full[t], "truncated": part.get(t, "<missing>")})
    return {"pass": not mismatches, "checked": len(cuts), "mismatches": mismatches[:5]}
