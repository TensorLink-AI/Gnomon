"""Optional numpy acceleration of episodic memory for offline replay (`replay_router(accelerate=True)`).

Gnomon has no runtime dependencies; this module is imported only when acceleration is
requested and needs numpy. It keeps matured episodes in arrays as they arrive, so each
replayed origin costs vectorised work instead of a Python loop over every remembered
episode. It reproduces `episodic_memory.episodic_evidence` (scores, effective_n,
neighbours, and the shrinkage / confidence_z options) up to floating-point summation
order; tests/test_episodic_memory.py checks replay decisions match the pure path.
`dedupe_seconds`, `novelty_threshold` and `diagnostics` are not accelerated.
"""

from __future__ import annotations

from datetime import datetime
import math

import numpy as np

from .episodic_memory import feature_names
from .forecast_adapter import ForecastAdapterError

UNSUPPORTED = ("dedupe_seconds", "novelty_threshold", "diagnostics")


def check_supported(memory: dict) -> None:
    used = [k for k in UNSUPPORTED if memory.get(k)]
    if used:
        raise ForecastAdapterError(f"accelerate=True does not support memory {used}; replay with accelerate=False",
                                   details={"rejected_fields": [f"memory.{k}" for k in used]})


class MemoryIndex:
    """Matured episodes as arrays, appended once each."""

    def __init__(self, policy: dict):
        self.policy = policy
        self.memory = policy["memory"]
        check_supported(self.memory)
        self.providers = [policy["baseline"], *policy["candidates"]]
        self.names = feature_names(self.memory)
        self._n = 0
        self._F = np.empty((64, len(self.names)))
        self._L = np.empty((64, len(self.providers)))
        self._S = np.empty(64, dtype=np.int64)          # series code
        self._codes, self._names = {}, []
        self._O = np.empty(64, dtype=object)
        self._T = np.empty(64)

    def add(self, row: dict) -> None:
        if row.get("features") is None or not all(p in row["losses"] for p in self.providers):
            return
        if self._n == len(self._T):  # grow by doubling: amortised O(1) appends
            grow = lambda a: np.concatenate([a, np.empty((len(a),) + a.shape[1:], dtype=a.dtype)])
            self._F, self._L, self._S, self._O, self._T = map(grow, (self._F, self._L, self._S, self._O, self._T))
        f, i = row["features"], self._n
        self._F[i] = [np.nan if f.get(n) is None else float(f[n]) for n in self.names]
        self._L[i] = [float(row["losses"][p]) for p in self.providers]
        code = self._codes.setdefault(row["series_id"], len(self._names))
        if code == len(self._names):
            self._names.append(row["series_id"])
        self._S[i], self._O[i] = code, row["origin"]
        self._T[i] = datetime.fromisoformat(row["origin"]).timestamp()
        self._n += 1

    def _arrays(self):
        n = self._n
        return self._F[:n], self._L[:n], self._S[:n], self._O[:n], self._T[:n]

    def evidence(self, own_series: str, query: dict, origin_time: str, oldest: str) -> dict:
        empty = {"scores": {}, "effective_n": 0.0, "neighbours": []}
        if not self._n or query is None:
            return empty
        F, L, S, O, T = self._arrays()
        pool = self.policy.get("pool")
        allowed = {own_series, *(pool["series"] if pool else [])}
        allowed_code = np.array([n in allowed for n in self._names])
        # origins are normalised UTC ISO strings, so timestamp order is their string order
        keep = (T >= datetime.fromisoformat(oldest).timestamp()) & allowed_code[S]
        if not keep.any():
            return empty
        F, L, S, O, T = F[keep], L[keep], S[keep], O[keep], T[keep]
        own_code = self._codes.get(own_series, -1)
        name_rank = np.argsort(np.argsort(np.array(self._names, dtype=object)))  # tie-break by series name
        names, memory = self.names, self.memory
        q = np.array([np.nan if query.get(n) is None else float(query[n]) for n in names])
        nf = len(names)
        scale = np.full(nf, np.nan)
        for j in range(nf):
            xs = np.sort(F[:, j][np.isfinite(F[:, j])])
            if len(xs) == 0:
                continue
            med = xs[len(xs) // 2]
            mad = np.sort(np.abs(xs - med))[len(xs) // 2]
            scale[j] = 1.4826 * mad if mad > 0 else (float(np.std(xs)) or 1.0)
        have = np.isfinite(F) & np.isfinite(q)[None, :] & np.isfinite(scale)[None, :]
        shared = have.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(have, (F - q[None, :]) / np.where(np.isfinite(scale), scale, 1.0)[None, :], 0.0)
            d = np.sqrt((z ** 2).sum(axis=1) * nf / shared)
        ok = shared * 2 >= nf
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            return empty
        order = idx[np.lexsort((name_rank[S[idx]], T[idx], d[idx]))][: memory["k"]]
        dk = d[order]
        bandwidth = float(np.sort(dk)[len(dk) // 2]) or 1e-9
        sums = np.bincount(S, weights=L[:, 0], minlength=len(self._names))
        counts = np.bincount(S, minlength=len(self._names))
        series_scale = (sums / np.maximum(counts, 1))[S[order]]
        w = np.exp(-0.5 * (dk / bandwidth) ** 2)
        w = np.where(S[order] == own_code, w * memory["own_weight"], w)
        half_life = memory["recency_half_life_days"]
        if origin_time and half_life:
            now = datetime.fromisoformat(origin_time).timestamp()
            w = w * 0.5 ** (np.maximum((now - T[order]) / 86400, 0.0) / half_life)
        use = (series_scale > 0) & (w > 0)
        w, sscale, Lk, Sk, Ok, dk = w[use], series_scale[use], L[order][use], S[order][use], O[order][use], dk[use]
        if w.sum() <= 0:
            return empty
        weighted = (w[:, None] * Lk / sscale[:, None]).sum(axis=0)
        if weighted[0] <= 0:
            return empty
        effective_n = float(w.sum() ** 2 / (w ** 2).sum())
        scores = {p: float(weighted[i] / weighted[0]) for i, p in enumerate(self.providers)}
        neighbours = [{"series_id": self._names[Sk[i]], "origin": Ok[i], "distance": round(float(dk[i]), 4),
                       "weight": round(float(w[i]), 4),
                       "best_provider": self.providers[int(np.argmin(Lk[i]))]} for i in range(len(w))]
        neighbours.sort(key=lambda x: -x["weight"])
        out = {"scores": scores, "effective_n": effective_n, "neighbours": neighbours}
        shrink, zc = memory.get("shrinkage", 0), memory.get("confidence_z", 0)
        if not (shrink or zc):
            return out
        a = Lk / sscale[:, None]
        resid = ((w[:, None] * (a - np.array([scores[p] for p in self.providers])[None, :] * a[:, [0]])) ** 2).sum(axis=0)
        se = {p: float(math.sqrt(resid[i]) / weighted[0]) for i, p in enumerate(self.providers)}
        base = self.providers[0]
        if shrink:
            factor = effective_n / (effective_n + shrink)
            scores = {p: (s if p == base else 1.0 + (s - 1.0) * factor) for p, s in scores.items()}
            se = {p: v * factor for p, v in se.items()}
            out["scores"] = scores
        if zc:
            out["selection_scores"] = {p: (s if p == base else s + zc * se[p]) for p, s in scores.items()}
        return out
