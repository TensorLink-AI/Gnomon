"""FASE statistical memory primitives (arXiv:2609.32689, equations 1–2, appendix E).

Univariate, standard-library implementation. No learned policy or LLM selector.
Feature group names follow the executable prompt in appendix E. Additional Gnomon
features each form their own group; caller context forms a single extra group.
"""
from __future__ import annotations

import cmath
import math
from collections import defaultdict

GROUPS = {
    "quality_profile": ("missing_ratio", "zero_ratio"),
    "global_recent_dynamics": ("global_trend_strength", "recent_trend_strength",
                               "recent_level_shift_strength", "recent_scale_change_strength"),
    "change_regime_analysis": ("change_strength",),
    "nonlinear_complexity": ("linear_fit_error_ratio", "turning_behavior_ratio", "direction_entropy"),
    "dependence_screen": ("spectral_concentration",),
    "background": ("spectral_entropy", "stability"),
    "missingness_structure": ("longest_missing_block",),
    "time_varying_periodicity": ("recent_period_strength",),
    "covariate_relationship": ("strongest_covariate_relation", "covariate_relation_stability"),
    "intermittent_structure": ("recent_event_rate_change",),
}
FEATURES = tuple(f"fase:{n}" for group in GROUPS.values() for n in group)
_GROUP = {f"fase:{n}": g for g, ns in GROUPS.items() for n in ns}


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return math.sqrt(mean([(x - m) ** 2 for x in xs]))


def scales(vectors, names):
    out = {}
    for name in names:
        xs = [v[name] for v in vectors if v.get(name) is not None and math.isfinite(v[name])]
        if xs:
            out[name] = (mean(xs), std(xs))
    return out


def distance(a, b, names, scale):
    groups = defaultdict(list)
    for n in names:
        if a.get(n) is None or b.get(n) is None or n not in scale:
            continue
        if not math.isfinite(a[n]) or not math.isfinite(b[n]):
            continue
        delta, sigma = abs(a[n] - b[n]), scale[n][1]
        # Continuous limit as sigma -> 0; unlike a unit fallback, scale invariant.
        rho = delta / (sigma + delta) if sigma + delta > 0 else 0.0
        group = _GROUP.get(n, "context" if n.startswith("context:") else n)
        groups[group].append(rho)
    return mean([mean(v) for v in groups.values()]) if groups else None


def _slope(pairs):
    if len(pairs) < 2:
        return None
    tx, vy = mean([p[0] for p in pairs]), mean([p[1] for p in pairs])
    den = sum((t - tx) ** 2 for t, _ in pairs)
    return sum((t - tx) * (v - vy) for t, v in pairs) / den if den else None


def _corr(pairs):
    if len(pairs) < 2:
        return None
    x, y = zip(*pairs)
    sx, sy = std(x), std(y)
    if not sx or not sy:
        return None
    mx, my = mean(x), mean(y)
    return max(-1.0, min(1.0, mean([(a-mx)*(b-my) for a, b in pairs]) / sx / sy))


def _fft(xs, inverse=False):
    """Radix-2 FFT; internal callers always provide a power-of-two length."""
    a, n = list(xs), len(xs)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    size = 2
    while size <= n:
        root = cmath.exp((2j if inverse else -2j) * math.pi / size)
        for start in range(0, n, size):
            w = 1
            for k in range(size // 2):
                u, v = a[start+k], w * a[start+k+size//2]
                a[start+k], a[start+k+size//2] = u+v, u-v
                w *= root
        size *= 2
    return [v/n for v in a] if inverse else a


def _power(xs):
    """Exact-length DFT via Bluestein, without padding the statistical window."""
    n = len(xs)
    size = 1 << (2*n-1).bit_length()
    chirp = [cmath.exp(-1j * math.pi * (k*k % (2*n)) / n) for k in range(n)]
    a = [x*c for x, c in zip(xs, chirp)] + [0j]*(size-n)
    b = [0j]*size
    for k, c in enumerate(chirp):
        b[k] = c.conjugate()
        if k:
            b[-k] = c.conjugate()
    conv = _fft([x*y for x, y in zip(_fft(a), _fft(b))], inverse=True)
    return [abs(conv[k]*chirp[k])**2 for k in range(1, n//2+1)]


def _detrend(xs):
    # Quadratic least squares on coordinates scaled to [-1, 1].
    n = len(xs)
    ts = [2*i/(n-1)-1 for i in range(n)]
    matrix = [[sum(t**(i+j) for t in ts) for j in range(3)] +
              [sum(v*t**i for t, v in zip(ts, xs))] for i in range(3)]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda j: abs(matrix[j][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        den = matrix[i][i]
        if abs(den) < 1e-12:
            return None
        matrix[i] = [v/den for v in matrix[i]]
        for j in range(3):
            if j != i:
                factor = matrix[j][i]
                matrix[j] = [v-factor*w for v, w in zip(matrix[j], matrix[i])]
    coefs = [r[-1] for r in matrix]
    return [v-sum(c*t**i for i, c in enumerate(coefs)) for t, v in zip(ts, xs)]


def features(history, *, window, season=1, mask=None, covariates=(), covariate_names=(), mask_name=None):
    raw = list(history)[-window:]
    n = len(raw)
    out = {name: None for name in FEATURES}
    if not n:
        return out
    mask = list(mask)[-n:] if mask is not None else [1]*n
    values = [float(v) if v is not None and math.isfinite(float(v)) and m > 0 else None
              for v, m in zip(raw, mask, strict=True)]
    observed = [v for v in values if v is not None]
    def put(k, v):
        out['fase:'+k] = v if v is None or math.isfinite(v) else None
    put('missing_ratio', 1-len(observed)/n)
    if not observed:
        put('longest_missing_block', n)
        return out
    put('zero_ratio', sum(v == 0 for v in observed)/len(observed))
    run = longest = 0
    for v in values:
        run = run+1 if v is None else 0
        longest = max(longest, run)
    if longest:
        put('longest_missing_block', longest)
    sd = std(observed)
    r = min(n, max(4, n//4))
    recent = [v for v in values[-r:] if v is not None]
    prior = [v for v in values[:-r] if v is not None]
    if sd:
        slope = _slope([(i, v) for i, v in enumerate(values) if v is not None])
        put('global_trend_strength', slope*n/sd if slope is not None else None)
        slope = _slope([(i, v) for i, v in enumerate(values[-r:]) if v is not None])
        put('recent_trend_strength', slope*r/sd if slope is not None else None)
        if recent and prior:
            put('recent_level_shift_strength', abs(mean(recent)-mean(prior))/sd)
        prefix, counts = [0.0], [0]
        for v in values:
            prefix.append(prefix[-1]+(v if v is not None else 0))
            counts.append(counts[-1]+(v is not None))
        shifts = [abs(prefix[i]/counts[i] - (prefix[-1]-prefix[i])/(counts[-1]-counts[i]))/sd
                  for i in range(max(1, math.ceil(n*.2)), min(n, math.floor(n*.8)+1))
                  if counts[i] and counts[-1] > counts[i]]
        if shifts:
            put('change_strength', max(shifts))
    if recent and prior and std(recent)+std(prior):
        put('recent_scale_change_strength', abs(std(recent)-std(prior))/(std(recent)+std(prior)))
    pairs = [(a,b) for a,b in zip(values, values[1:]) if a is not None and b is not None]
    if pairs:
        p = sum(b>a for a,b in pairs)/len(pairs)
        put('direction_entropy', -sum(q*math.log2(q) for q in (p, 1-p) if q))
        persistence = sum((b-a)**2 for a,b in pairs)
        slope = _slope(pairs)
        if slope is not None and persistence:
            intercept = mean([b for _,b in pairs])-slope*mean([a for a,_ in pairs])
            put('linear_fit_error_ratio', sum((b-intercept-slope*a)**2 for a,b in pairs)/persistence)
    sign = lambda x: (x>0)-(x<0)
    turns = [sign(b-a) != sign(c-b) for a,b,c in zip(values,values[1:],values[2:])
             if a is not None and b is not None and c is not None]
    if turns:
        put('turning_behavior_ratio', mean(turns))
    if len(observed) == n and sd and n >= 3:
        center = mean(observed)
        centered = [(v-center)/sd for v in observed]
        power = _power(centered)
        total = sum(power)
        if total and len(power)>1:
            put('spectral_entropy', -sum((p/total)*math.log(p/total) for p in power if p)/math.log(len(power)))
        width = season if season>1 else 10
        blocks = [mean(centered[i:i+width]) for i in range(0,n-width+1,width)]
        if len(blocks)>1:
            put('stability', sum((v-mean(blocks))**2 for v in blocks)/(len(blocks)-1))
        residual = _detrend(centered)
        if residual is not None and sum(v*v for v in residual)>1e-20*n:
            power = _power(residual)
            put('spectral_concentration', sum(sorted(power)[-3:])/sum(power))
        if n>=36:
            residual = _detrend(centered[-max(12,n//3):])
            if residual is not None and sum(v*v for v in residual)>1e-20*n:
                power = _power(residual)
                put('recent_period_strength', max(power)/sum(power))
    correlations, stability = [], []
    for j, name in enumerate(covariate_names):
        if name == mask_name:
            continue
        aligned = [(i, v, row[j]) for i,(v,row) in enumerate(zip(values,list(covariates)[-n:],strict=True))
                   if v is not None and row[j] is not None and math.isfinite(row[j])]
        corr = _corr([(v,c) for _,v,c in aligned])
        if corr is not None:
            correlations.append(abs(corr))
        first = _corr([(v,c) for i,v,c in aligned if i<n//2])
        last = _corr([(v,c) for i,v,c in aligned if i>=n//2])
        if first is not None and last is not None:
            stability.append(max(0,1-abs(first-last)))
    if correlations:
        put('strongest_covariate_relation', max(correlations))
    if stability:
        put('covariate_relation_stability', mean(stability))
    if n>=8 and out['fase:zero_ratio']>=.45:
        recent = [v for v in values[-max(8,n//4):] if v is not None]
        if recent:
            put('recent_event_rate_change', abs(mean([v!=0 for v in recent])-mean([v!=0 for v in observed])))
    return out


def episode_key(row):
    return (row['series_id'], row['origin'])


class Pool:
    """Recent FIFO plus long-term retention, updated only on completed instances.

    Retrieval emits a receipt of equation-2 contributions, to be recorded with the
    decision and applied when that decision matures. Scores never use its outcome
    before completion. Ties retain the incumbent long-term entry. Unscored legacy
    entries start with retention zero; their historical contributions are not invented.
    """
    def __init__(self, recent_capacity=100, long_term_capacity=900):
        self.recent_capacity, self.long_term_capacity = recent_capacity, long_term_capacity
        self.recent, self.long_term, self.values = [], [], {}

    @property
    def rows(self):
        return self.long_term + self.recent

    def complete(self, row):
        key = episode_key(row)
        if key in self.values:
            return
        for item in row.get('retention_receipt', []):
            old = (item['series_id'], item['origin'])
            if old in self.values:
                total, count = self.values[old]
                self.values[old] = (total+item['contribution'], count+1)
        self.values[key] = (0.0, 0)
        self.recent.append(row)
        if len(self.recent) <= self.recent_capacity:
            return
        candidate = self.recent.pop(0)
        key = episode_key(candidate)
        if len(self.long_term) < self.long_term_capacity:
            self.long_term.append(candidate)
            return
        def score(entry):
            total, count = self.values[episode_key(entry)]
            return total/count if count else 0.0
        if self.long_term:
            worst = min(range(len(self.long_term)), key=lambda i: (score(self.long_term[i]), i))
            if score(candidate) > score(self.long_term[worst]):
                evicted = self.long_term[worst]
                self.long_term[worst] = candidate
                del self.values[episode_key(evicted)]
                return
        del self.values[key]

    def receipt(self, query, names, k, providers, *, scored=None):
        if scored is None:
            scale = scales([r['features'] for r in self.rows], names)
            scored = []
            for row in self.rows:
                d = distance(query, row['features'], names, scale)
                if d is not None:
                    scored.append((d, episode_key(row), row))
            scored.sort(key=lambda x: (x[0], x[1][1], x[1][0]))
        else:
            scored = [(d, episode_key(row), row) for d, row in scored]
        selected = {key for _, key, _ in scored[:k]}
        partitions = defaultdict(list)
        for d, key, row in scored:
            best = min(row['losses'][p] for p in providers)
            winners = tuple(p for p in sorted(providers) if row['losses'][p] == best)
            partitions[(row.get('served_provider'), winners)].append((d, key))
        contributions = {key: 0.0 for key in selected}
        for group in partitions.values():
            first, key = group[0]
            if key in selected:
                contributions[key] = group[1][0]-first if len(group)>1 else 1-first
        return [{'series_id': key[0], 'origin': key[1], 'contribution': value}
                for key, value in sorted(contributions.items())]


class ReplayIndex:
    """Incrementally reconstruct each task pool; identical to replaying its receipts."""
    def __init__(self, policy, accelerate=False):
        self.accelerate = accelerate
        self.policy, self.completed, self.pools = policy, [], {}

    def add(self, row):
        self.completed.append(row)

    def evidence(self, own_series, query, origin, oldest=None):
        from .episodic_memory import episodic_evidence
        memory = self.policy['memory']
        providers = [self.policy['baseline'], *self.policy['candidates']]
        allowed = frozenset([own_series, *(self.policy.get('pool') or {}).get('series', [])])
        pool, offset = self.pools.get(allowed, (Pool(memory['recent_capacity'], memory['long_term_capacity']), 0))
        for row in self.completed[offset:]:
            if row['series_id'] in allowed and row.get('features') is not None and all(p in row['losses'] for p in providers):
                pool.complete(row)
        self.pools[allowed] = pool, len(self.completed)
        return episodic_evidence(pool.rows, self.policy, own_series, query, origin, retained_pool=pool, accelerate=self.accelerate)
