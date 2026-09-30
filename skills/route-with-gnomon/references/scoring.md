# How memory scores

## Feature definitions

Computed from the request's `history` values (not differences) at routing time. *S* is
the last `short_window` values, *L* the last `long_window`; sd is the population standard
deviation. All but `length_cycles` and `covariate_share` need at least `long_window`
values, and are null when the stated denominator is zero.

| Feature | Definition | Null when |
|---|---|---|
| `volatility_ratio` | log(max(sd(S), 0.001·sd(L)) / sd(L)), clipped to ±5 | sd(L) = 0 |
| `trend` | mean of first differences within *S*, divided by sd(L), clipped to ±5 | sd(L) = 0 |
| `seasonality` | lag-`season` autocorrelation over *L*: Σ(xₜ−m)(xₜ₊ₛ−m) / (sd(L)²·(len(L)−s)) | season = 1, len(L) < 2·season, sd(L) = 0 |
| `zero_share` | share of zeros among observed values of *L* (mask = `mask_covariate` > 0) | no observed values |
| `missing_share` | share of *L* with mask ≤ 0; without a mask, 1 − len(L)/expected steps from the median timestamp gap | neither mask nor enough timestamps |
| `length_cycles` | log(min(n, 4·`long_window`) / `season`) | empty history |
| `level_shift` | (mean(S) − mean(L)) / sd(L), clipped to ±5 | sd(L) = 0 |
| `cv` | log(sd(L) / \|mean(L)\|), clipped to ±5 | sd(L) = 0 or mean(L) = 0 |
| `covariate_share` | share of nonzero values of `future_covariate` over the forecast horizon | covariate absent |

`length_cycles` is constant once every series has more than four long windows of
history, so it only helps when some series are young.

## Scoring in detail

1. Candidates: matured episodes from this series and the pool inside the lookback whose
   features were recorded under the same memory spec.
2. Standardise each feature by the median and 1.4826 × MAD over those episodes (standard
   deviation, then 1, when the MAD is zero), so no future data enters the scaling.
3. Distance: Euclidean over features both vectors have, scaled by (all features / shared
   features); episodes sharing fewer than half are skipped. Keep the `k` nearest.
4. Weight: exp(−½(d/h)²) with h = median neighbour distance, × `own_weight` for the
   forecast series, × 0.5^(age / `recency_half_life_days`) when set.
5. Score(p) = Σ w·loss(p)/scaleₛ ÷ Σ w·loss(baseline)/scaleₛ, where scaleₛ is series s's
   mean baseline loss over its visible episodes (the baseline scores 1.0).
6. effective_n = (Σw)² / Σw². Below `min_effective_n` the router falls back to `context`,
   then to all evidence. Otherwise the usual cost-adjusted utility, `limits` and
   `min_improvement` decide.
