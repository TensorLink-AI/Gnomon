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
| `autocorrelation` | lag-1 autocorrelation over *L* | sd(L) = 0 |
| `skewness` | mean((x−m)³)/sd(L)³ over *L*, clipped to ±5 | sd(L) = 0 |
| `vol_of_vol` | sd of the sds of consecutive `short_window` blocks of *L* ÷ their mean, clipped to 5 | fewer than 2 blocks, or mean block sd = 0 |
| `demand_interval` | log(observed values in *L* / nonzero values) | no nonzero observed value |
| `context:<name>` | latest value of past covariate `<name>` at the origin (`memory.context`) | covariate absent or not finite |

Presets: `profile = "returns"` or `"intermittent"` replaces `features` (see below).

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

Optional steps (each off unless set): `dedupe_seconds` skips a neighbour within that many
seconds of an already-chosen one from the same series (step 3); `shrinkage` replaces each
candidate score s by 1 + (s − 1)·n/(n + shrinkage), n = effective_n; `confidence_z` selects
on s + z·SE, SE = √Σ(w·(aᵢ − s·bᵢ))² / Σw·bᵢ with aᵢ, bᵢ the scaled candidate and baseline
losses; `novelty_threshold` abstains when the median neighbour distance exceeds that
multiple of the typical one for remembered episodes; `switch_penalty` (top level) adds a
penalty to every provider except the last one served.

## Optional settings

All off by default; only `context` changes the memory spec, so episodes recorded before
the other settings were added stay usable.

| Setting | What it does | When to use it |
|---|---|---|
| `profile` | Feature preset instead of `features`: `levels` (the default set), `returns` (volatility ratio, level shift, trend, autocorrelation, skewness, vol-of-vol), `intermittent` (zero share, demand interval, missing share, cv, level shift, volatility ratio) | Return-like series (changes, log returns, P&L), where level features such as `cv` are unstable; intermittent demand |
| `context` | Past-covariate names whose **latest value at the origin** joins the distance as `context:<name>` | The state that decides which model wins is outside the series: a market-wide volatility, a promotion flag, a peer-group aggregate, the weather |
| `dedupe_seconds` | At most one neighbour per series within this window | Multi-step horizons or dense origins, where adjacent episodes share outcomes and would overstate `effective_n`; set about the horizon |
| `shrinkage` | Candidate scores pulled toward 1.0 by effective_n / (effective_n + shrinkage) | Noisy losses or small pools; stops a few lucky neighbours from switching the model |
| `confidence_z` | Selection uses score + z × standard error (delta-method SE of the weighted ratio) | Switch only on evidence that is better *and* clear |
| `novelty_threshold` | When the median distance to the k neighbours exceeds this multiple of the typical one among remembered episodes, memory abstains (`memory_abstained`) and the router falls back to context, then all evidence | Structural breaks and new regimes: "never seen this" should not borrow a confident answer |
| `diagnostics` | Adds `memory_diagnostics`: unshrunk scores, standard errors, 90% intervals, neighbour loss-ratio quantiles (10/50/90%), novelty | Auditing and agent explanations |

Top-level `switch_penalty` (any evidence level): every provider except the one last served
for the series pays this in utility, so the router switches only when the gain clears
`min_improvement + switch_penalty`. Reported as `incumbent`.

## Checking memory against its twin

`gnomon.adaptive_router.memory_ablation(folds, policy, histories)` replays the policy with
and without memory and returns both scores, the best fixed provider, the paired loss
difference with a 90% bootstrap interval (whole origins resampled), switch rates and a
`verdict` (`memory_better`, `memory_worse`, `not_distinguishable`). `replay_router(...,
accelerate=True)` uses numpy for large replays (not with `dedupe_seconds`,
`novelty_threshold` or `diagnostics`).
