# Adaptive routing

A router is a provider name that serves one of *your* models per forecast and learns
which one from outcomes recorded in the ledger. You choose the model set (built-ins,
StatsForecast, Ephemeris, your own callables) and, optionally, what each call costs
in dollars or latency. Routing needs a configured ledger.

```toml
schema_version = 1
ledger_path = "ledger.db"

[providers.ephemeris]
kind = "ephemeris"
base_url_env = "EPHEMERIS_BASE_URL"
token_env = "EPHEMERIS_API_TOKEN"
discover = true

[routers."router/sales"]
candidates = ["statsforecast/ets", "ephemeris/chronos-2"]
baseline = "seasonal_naive"
metric = "mae"
recent_origins = 20        # score each provider on its latest 20 matched origins
min_origins = 5            # serve the baseline until this much evidence exists
min_improvement = 0.02     # a candidate must beat the baseline's utility by this much
lookback_seconds = 2592000 # evidence window ending at the forecast origin (30 days)
shadow_every = 1           # run every provider on every call so evidence keeps accruing
identity_policy = "prospective_unattested"  # needed for Ephemeris (no attested revision)

[routers."router/sales".costs]
"ephemeris/chronos-2" = { usd_per_call = 0.002, latency_seconds = 1.5 }

[routers."router/sales".cost_weights]
usd = 10.0             # $0.001 per call counts as 1% relative error
latency_seconds = 0.01 # one second counts as 1% relative error

[routers."router/sales".limits]
max_latency_seconds = 3.0

[routers."router/sales".pool]
series = ["store-2/sales", "store-3/sales"]  # learn across these too (same unit and horizon)
own_weight = 2.0                              # the forecast series' own origins count double

[routers."router/sales".context]
feature = "volatility_ratio"  # or "trend"
on = "differences"
short_window = 7
long_window = 90
thresholds = [0.8, 1.25]      # labels volatility_ratio:bin0 / bin1 / bin2
```

Forecast with the router's name wherever a provider is accepted. Requests need
`series_id`, `timestamps` and `future_timestamps` so outcomes can be matched later.
Append actuals as they arrive; routing improves as recorded forecasts mature.

```python
from gnomon import GnomonSession
with GnomonSession.from_config("gnomon.toml") as session:
    reply = session.forecast("router/sales", request)
    print(reply["routing"]["served_provider"], reply["routing"]["reason"])
```

```bash
gnomon infer --providers-config gnomon.toml --provider router/sales --request request.json
gnomon capabilities --providers-config gnomon.toml   # routers are listed under ledger.routers
```

Over MCP, start the server with the same config; agents call `gnomon_forecast` with
`"provider": "router/sales"` and find routers under `gnomon_capabilities` →
`ledger.routers`. Outcomes are appended with the ledger's `append_actual` operation
(`gnomon_ledger` over MCP, `gnomon ledger --arguments` on the CLI), which needs
`allow_outcome_writes = true` in the operator TOML. Agents cannot create or change
routers; they are operator configuration.

## Before you start

- **Make the lookback longer than the horizon.** An origin becomes evidence only when
  *all* its targets are known, so nothing inside a window shorter than the horizon can
  ever mature. The default `lookback_seconds` is 7 days: at a 7-day or longer horizon
  the router serves the baseline forever. Use at least horizon + enough origins to
  score (for example 56-day horizon, daily origins: 200 days).
- **Windows count observations, not days.** `short_window`/`long_window` in `context`
  and `memory` are numbers of history values: 112 is 16 weeks of daily data but under
  10 hours of 5-minute bars. Scale them to your frequency.
- **Two different features are called memory.** `[routers."x".memory]` is this page's
  episodic routing memory. The top-level `[memory]` table (`auto_recall`, `ledger_ref`)
  is read-only evidence recall of recorded lessons; see
  [memory bridge](memory-bridge.md). They are independent.
- **Replay before deploying.** Run the rule on your own history first
  ([walkthrough](#evaluate-a-router-on-your-own-data)); in our tests plain rolling
  selection did not beat choosing one good model, while memory with a pool did.

## How the choice is made

At each forecast the router asks `compare_history` for matched, prospectively
recorded evidence over `lookback_seconds` ending at the origin. Only outcomes available
by the request's point in time count: its `known_time_cutoff`, or else its origin
(`routing.evidence_as_of`). A live request loses nothing; repeating a historical
request gets the same evidence it had then, never outcomes that matured later. Each
provider's score is its mean per-origin metric over the latest `recent_origins` matched
origins.

```
utility(p) = score(p) / score(baseline)
           + cost_weights.usd * usd_per_call(p)
           + cost_weights.latency_seconds * latency_seconds(p)
```

Lower is better. The best candidate is served only if it meets `limits` and its
utility is at least `min_improvement` below the baseline's; otherwise the baseline is
served. With fewer than `min_origins` matched origins the baseline is served and the
reason is `insufficient_evidence`. If the ledger refuses the window as
`incompatible_evidence` (task shape, such as `season`, or a provider identity changed
inside it), the baseline is served with reason `evidence_incompatible`; if evidence
cannot be read at all, `evidence_unavailable`. Costs are declared, not measured; each routed call
records measured latency so declarations can be checked.

## Pooled evidence across series

With `pool`, the router also learns from the listed series, so a new series is served
from other series' experience instead of cold-starting on the baseline. Each series is
scored relative to its *own* baseline (a series with large values cannot dominate) and
series are combined weighted by matched origins, with the forecast series multiplied by
`own_weight`. A pool may list up to 128 series. Pool members must share the request's unit and horizon. Every series'
evidence obeys the same clock: only outcomes known before this forecast count. A pool
member whose evidence cannot be read is skipped and listed in `pool_failures`.

## Context-conditioned routing

With `context`, each forecast is labelled with a regime computed only from its own
request history: `volatility_ratio` = std(last `short_window`) / std(last `long_window`),
or `trend` = mean(last `short_window`) / std(last `long_window`), on raw values or first
differences, binned by ascending `thresholds`. The router records that label in its
decision. Later forecasts first use evidence from past origins with the same label (and
the same context spec); if that has fewer than `min_origins` matched origins, they use
all evidence. `routing.evidence_level` says which was used.

`volatility_ratio` is relative: in a long calm or long volatile stretch it returns towards
1. Choose `long_window` long enough to span regimes if you want absolute "high/low
volatility" labels. Past origins routed before `context` was configured, or under a
different spec, carry no label and only count at the `all` level.

## Episodic memory

With `memory`, the router describes every forecast by features of its own request and
records them in its routing decision. At a new forecast it retrieves the `k` most similar
matured past origins from the forecast series and its `pool` (all within
`lookback_seconds`), and scores providers by their similarity-weighted losses relative to
the baseline. The question it asks is "when a series looked like this before, which model
did best?" rather than "which model did best recently?".

```toml
[routers."router/sales".memory]
features = ["volatility_ratio", "trend", "seasonality", "zero_share", "missing_share",
            "length_cycles", "level_shift", "cv"]      # add "covariate_share" with future_covariate
short_window = 14
long_window = 112
season = 7
k = 64                  # neighbours retrieved
min_effective_n = 16    # else fall back to context, then all evidence
own_weight = 2.0        # the forecast series' own episodes count double
mask_covariate = "observed"        # optional past covariate: 1 = observed, 0 = filled gap
future_covariate = "onpromotion"   # optional known-future covariate for covariate_share
```

- **Features** come only from the request: volatility ratio, trend, weekly (or `season`)
  autocorrelation, zero share among observed values, missing share (from the mask, else
  from timestamp gaps), history length in cycles (capped at 4 long windows, so it is not a
  proxy for the date), level shift, coefficient of variation, and the share of nonzero
  values of a known future covariate. A feature that cannot be computed (too little
  history, a flat window, an absent covariate) is recorded as null and never imputed;
  distances use shared features only, and an episode sharing fewer than half is skipped.
- **Scoring**: features are standardised by median and MAD over the episodes visible at
  that moment; neighbours are weighted by a Gaussian kernel (bandwidth = median neighbour
  distance) and `own_weight`, optionally by `recency_half_life_days`. Each series' losses
  are divided by its mean baseline loss, so large-volume series cannot dominate.
- **Evidence level**: `routing.evidence_level` is `memory` when the effective sample size
  (sum of weights squared over sum of squared weights) reaches `min_effective_n`.
  Responses add `effective_n` and the top five `memory_neighbours` (series, origin,
  distance, weight, best provider), which an agent can inspect before accepting or
  overriding the choice.
- **Point in time**: features are recorded when the router forecasts and are never
  recomputed; episodes count only once every target is known; features recorded under a
  different memory spec (features, windows, season, covariates) are ignored.
- **Cost**: memory reads the whole lookback window for each pool member (past the ledger's
  1,000-execution cap it reads the window in parts rather than truncating it). A pool of
  many series therefore means many evidence reads per forecast; replay first.

`replay_router(..., covariates={series: {name: [(time, value), ...]}})` computes the same
features from timestamped `histories`, so replay and live routing select identically.

### Optional memory settings

All off by default; a policy that does not set them behaves exactly as before, and only
`context` changes the memory spec (so episodes recorded under the old spec stay usable).

| Setting | What it does | When to use it |
|---|---|---|
| `profile` | Feature preset instead of `features`: `levels` (the default set), `returns` (volatility ratio, level shift, trend, autocorrelation, skewness, vol-of-vol), `intermittent` (zero share, demand interval, missing share, cv, level shift, volatility ratio) | Return-like series (changes, log returns, P&L), where level features such as `cv` are unstable; intermittent demand |
| `context` | Past-covariate names whose **latest value at the origin** joins the distance as `context:<name>` | The state that decides which model wins is outside the series: a market-wide volatility, a promotion flag, a peer-group aggregate, the weather |
| `dedupe_seconds` | At most one neighbour per series within this window | Multi-step horizons or dense origins, where adjacent episodes share outcomes and would overstate `effective_n`; set about the horizon |
| `shrinkage` | Candidate scores pulled toward 1.0 by effective_n / (effective_n + shrinkage) | Noisy losses or small pools; stops a few lucky neighbours from switching the model |
| `confidence_z` | Selection uses score + z × standard error (delta-method SE of the weighted ratio) | Switch only on evidence that is better *and* clear |
| `novelty_threshold` | When the median distance to the k neighbours exceeds this multiple of the typical one among remembered episodes, memory abstains (`memory_abstained`) and the router falls back to context, then all evidence | Structural breaks and new regimes: "never seen this" should not borrow a confident answer |
| `diagnostics` | Adds `memory_diagnostics`: unshrunk scores, standard errors, 90% intervals, neighbour loss-ratio quantiles (10/50/90%), novelty | Auditing and agent explanations |

Top-level `switch_penalty` (any evidence level) adds hysteresis: every provider other than
the one this router last served for the series pays the penalty in utility, so the router
switches only when the gain clears `min_improvement + switch_penalty`. The incumbent comes
from the router's own recorded decisions (live) or the previous replayed origin (replay),
and is reported as `incumbent`. Use it where forecast churn has a cost (re-planned orders,
re-traded positions).

Extra features for `features` lists (computed over *L*): `autocorrelation` (lag 1),
`skewness` (clipped ±5), `vol_of_vol` (sd of the standard deviations of consecutive
`short_window` blocks divided by their mean, clipped to 5) and `demand_interval`
(log of observed values per nonzero value).

### Experimental FASE-style memory

The optional `distance: "fase"`, `profile: "fase"`, and `retention: "fase"` settings
implement memory mechanisms described in [FASE](https://arxiv.org/html/2609.32689v1),
with Gnomon's existing deterministic, Gaussian-weighted selector. This is not a full
FASE reproduction: there is no learned MLP ranker or LLM controller.

- Distance uses the bounded discrepancy `abs(a-b)/(std + abs(a-b))`, averaged within
  the feature groups named in appendix E, then across observed groups. Standard
  deviations use only active episodes. With zero standard deviation, equal values
  contribute zero and unequal values one. No shared observations means no match.
  Additional Gnomon features each form a group; caller context forms one extra group.
- The profile contains 18 names prefixed with `fase:`. It computes the univariate
  statistics from appendix E in pure Python, including exact-length spectra, quadratic
  detrending and time-aligned historical covariate correlations. `long_window` defaults
  to 15,360 and `k` to 10 for this profile; explicit settings override these defaults.
  Undefined measurements are `None`. The mask covariate is excluded from correlations.
  Operational conventions include every integer split in the middle 20–80%, population
  standard deviations, and a nonzero detrended spectral peak as the periodicity candidate.
- Retention requires `distance: "fase"` and no dedupe. It replaces the memory lookback
  with a recent FIFO (default 100) and a long-term pool (default 900). Retrieval records
  contributions from equation 2, partitioned by invoked provider and the complete set
  of tied best providers. Receipts are applied only after the corresponding outcome
  matures. Long-term admission replaces the lowest running-average contribution only
  for a strictly higher value; unscored entries start at zero and ties keep incumbents.
  Legacy episodes without receipts have no invented historical contribution.

Live routing reconstructs retention from recorded decisions and visible outcomes;
replay maintains equivalent incremental pools. This can require reading substantially
more ledger history than the default lookback. NumPy remains optional and accelerates
replay distances only; the feature implementation needs no numerical dependency.
Install `gnomon-forecast[replay]` (or `.[replay]` from a checkout) to use acceleration.

For controlled ablations, keep the existing window, neighbour count and own-series
weight fixed while adding one mechanism at a time. Use `k: 10` and `own_weight: 1` for
a closer match to the paper's retrieval settings, and report this change separately.
Bounded distances limit individual-feature influence; they do not guarantee that
additional context improves retrieval or forecast accuracy.

Offline experiments can explicitly call `validate_policy(raw_policy, replay=True)`
to admit up to 31 candidates plus a baseline. The default validation and live schema
retain their eight-provider limit, matching ledger comparison capacity. The expanded
crypto experiment in `experiments/crypto_broad/` uses this option to compare 20
statistical and machine-learning configurations across separate indicator/horizon
replays. Each task has its own memory pool; a future-window label is admitted only
when the entire window has matured.

### Does memory earn its place?

`memory_ablation(folds, policy, histories, warmup_origins=..., covariates=...)` replays the
policy with and without its memory (everything else identical) and returns both scores,
the best fixed provider, the paired mean loss difference with a 90% bootstrap interval
(resampling whole origins, since series sharing a clock are not independent), the share of
folds served differently, each arm's switch rate and a `verdict`: `memory_better`,
`memory_worse` or `not_distinguishable`. Deploy memory only on `memory_better`, and check
`memory_beats_best_fixed` too: a router that beats its no-memory twin but not the best
single model is still not worth its cost.

`replay_router(..., accelerate=True)` scores memory with numpy (an optional dependency,
imported only then): about 9× faster on 10 series × 600 origins, with the same decisions.
It supports everything except `dedupe_seconds`, `novelty_threshold` and `diagnostics`.
`decision_losses=True` adds each decision's losses to `return_decisions` output.

### Feature definitions

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

### Scoring in detail

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

### Choosing settings

- **Which evidence mode.** Start with `pool` + `memory` when you have several related
  series; `memory` alone needs a long own history. `context` (one feature, fixed bins) is
  the cheaper fallback. Plain rolling scores (neither) are mainly a baseline.
- **Features.** The defaults are a reasonable start. In the Favorita study the dynamics
  group (`volatility_ratio`, `trend`, `level_shift`) carried the gain; add
  `covariate_share` when a known-future driver such as promotions matters. Check your
  own ablations with replay rather than adding features by default.
- **`k` and `min_effective_n`.** `k` should be a small fraction of the episodes in the
  window (pool size × matured origins); `min_effective_n` guards against acting on a
  handful of near-duplicates. We used k = 64, min_effective_n = 16 with pools of 5-96
  series.
- **Pool members** must share the unit and horizon. Up to 128 series; each is one
  evidence read per forecast.

## Evidence, shadows and cost

Only forecasts recorded before their first target count, so replays and backfills
never train the router. Every provider must have forecast the same origin for it to
count, which is what shadow runs provide. `shadow_every = 1` runs all providers on
every call (full evidence, full cost and latency); `shadow_every = k` runs them on a
deterministic 1-in-k sample of origins; `0` stops learning. A failed shadow never fails
the served forecast; it is recorded.

## Providers without an attested revision

Strict comparison (`identity_policy = "attested"`, the default) requires every
provider to report a revision, and pretrained providers to attest a training cutoff.
The Ephemeris connector attests neither, so admitting it needs
`identity_policy = "prospective_unattested"`. That is safe from target leakage: every
admitted forecast was recorded before its first target. It cannot detect a remote
model changing silently inside the window; responses list such providers under
`unattested_providers`.

## What is recorded and returned

Each routed forecast returns the served provider's normal response plus `routing`:
the served provider, reason, the utility table, matched origins, shadow execution IDs,
measured latencies, declared dollars for the call and `routing_decision_id`. That
decision (`kind: adaptive_route/1`) is stored in the ledger with the evidence window,
provider revisions and a `router_revision` hashed from the policy and revisions.

| `routing` field | Meaning |
|---|---|
| `served_provider`, `reason`, `evidence_based` | what was served and why (`insufficient_evidence`, `baseline_error_zero`, `evidence_no_eligible_candidate` (all excluded by `limits`), `evidence_improvement_below_threshold`, `evidence_and_cost_favour_candidate`, `evidence_incompatible`, `evidence_unavailable`) |
| `evidence_level` | `memory`, `context` or `all`; null when evidence could not be used |
| `effective_n`, `memory_neighbours` | memory only: sample size and the top five neighbours (series, origin, distance, weight, best provider) |
| `context_label`, `series_used`, `pool_failures` | regime label, series whose evidence counted, pool members that could not be read |
| `table`, `matched_origins`, `evidence_window`, `evidence_as_of` | per-provider scores and utilities, evidence size, window and point-in-time cutoff |
| `shadow_execution_ids`, `shadow_failures`, `measured_latency_seconds`, `declared_usd_this_call` | cost and shadow accounting |

The stored decision's `inputs` also hold `context_label`/`context_spec_id` and
`memory_features`/`memory_spec_id`, which later forecasts use to match episodes. A low
`effective_n` or neighbours from unrelated series are reasons to treat a choice as weak
evidence.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Always `insufficient_evidence` | `lookback_seconds` shorter than the horizon plus scoring origins; actuals not appended; `shadow_every = 0` | lengthen the lookback; append actuals; shadow |
| `evidence_level` never `memory` | fewer matured episodes than `min_effective_n`; features null (history shorter than `long_window`, flat series); memory settings changed (older episodes carry another spec) | lower `min_effective_n` or add pool series; shorten `long_window`; wait for new episodes |
| `evidence_incompatible` | task shape (for example `season`) or a provider revision changed inside the window | keep requests consistent, or let the window move past the change |
| `evidence_unavailable` | the ledger could not be read or compared; see `evidence_error` | fix the reported error; the baseline is served meanwhile |
| Slow routed calls | large pools, long lookbacks, long request histories | fewer pool series, shorter lookback, trim history to what models use |


## Evidence

Measured with `replay_router` on Favorita retail data (daily item-store unit sales,
56-day horizon, 450 rolling origins, scored on the last 300), with six StatsForecast
models run through Gnomon evaluation studies. Scores are mean MAE relative to
seasonal naive (lower is better); intervals are bootstrap 95% over series.

| Arm (96 series) | Score | vs best single model picked once (AutoTheta) |
|---|--:|---|
| Best model per origin in hindsight (unattainable) | 0.747 | |
| Memory + pool | **0.794** | -0.017 [-0.023, -0.012], better on 82/96 |
| Memory, own series only | 0.805 | -0.007 [-0.013, +0.001] |
| Regime labels (`context`) | 0.811 | -0.000 |
| Picked once (AutoTheta) | 0.812 | |
| Pooled rolling scores | 0.812 | +0.000 |
| Memory with random features (control) | 0.812 | +0.001 |
| Plain rolling scores | 0.817 | +0.005 |

- Plain rolling selection does not beat choosing one good model; retrieving similar
  past situations does. The random-feature control matches pooled scoring, so the
  gain comes from the features, not the pipeline.
- Dropping the dynamics features (volatility ratio, trend, level shift) removes the
  whole gain (+0.022 [+0.016, +0.029]); seasonality, sparsity, cv and a promotion
  covariate made no measurable difference here. `length_cycles` is constant once
  every series exceeds its cap.
- Every arm ran all six models at every origin. Shadow sampling (1 in 4) matched full
  shadowing in a 12-series study but has not been tested with memory.
- Live routing through the ledger and replay chose the same model at 600/600 origins
  (memory + pool + promotions, 4 series x 150 origins).

These are replay results on one dataset with classical models; confirm on your own
series before relying on a router.

## Evaluate a router before relying on it

`gnomon.adaptive_router.replay_router(folds, policy, histories)` replays the same selection
rule over saved folds (one-step or multi-step) from one or more series on a shared clock,
using only folds whose targets have all passed, computing labels and memory features from
history known at each origin,
and reports the router's score beside each fixed provider's, overall and per series. Folds come from a
`gnomon_evaluate` study (`origin`, target time, actual and each provider's point). Times
must carry a timezone and are compared as instants. Scores are mean per-fold losses
(`aggregation: mean_<metric>_over_folds`), matching the live per-origin score for
one-step folds; this is not an aggregate RMSLE over all folds. As in live comparison,
a fold with a negative RMSLE prediction or actual is excluded (`excluded_folds`), not
clipped.
Replay is simulation evidence; judge a deployed router on prospectively recorded
forecasts. In our parity checks live routing through the ledger and replay chose the
same model at every origin.

### Evaluate a router on your own data

Build folds with a Gnomon evaluation study per series, then replay each policy. Replays
need many folds; the default evaluation budget allows 8, so raise it in the operator
TOML.

```python
import csv, math
from datetime import datetime, timedelta, timezone

# Two small daily series; replace with your own CSVs (timestamp,value).
start = datetime(2026, 1, 1, tzinfo=timezone.utc)
for name, phase in (("store-1", 0.0), ("store-2", 1.5)):
    with open(f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value"])
        for i in range(200):
            w.writerow([(start + timedelta(days=i)).isoformat(), 50 + 10 * math.sin(i / 7 + phase) + (i % 5)])

open("gnomon.toml", "w").write("schema_version = 1\n[evaluation_limits]\nmax_folds = 500\nmax_calls = 2000\n")

from gnomon import GnomonSession
from gnomon.adaptive_router import replay_router, validate_policy

models = ["seasonal_naive", "last_value", "historical_mean"]
folds, histories = [], {}
with GnomonSession.from_config("gnomon.toml") as session:
    for name in ("store-1", "store-2"):
        ref = session.call("gnomon_inspect", {"input": f"{name}.csv", "unit": "units",
                                              "purpose": "evaluate"}, compact=False)["data_ref"]
        study = session.call("gnomon_evaluate", {"data_ref": ref, "baseline": models[0],
                                                 "candidates": models[1:], "horizon": 7, "folds": 60,
                                                 "stride": 1, "season": 7,
                                                 "budget": {"max_folds": 60, "max_calls": 180}}, compact=False)
        for fold in study["folds"]:
            if fold["status"] != "complete":
                continue
            actuals = sorted(fold["actuals"], key=lambda a: a["valid_time"])
            folds.append({"series_id": name, "origin": fold["origin"],
                          "target_time": actuals[-1]["valid_time"],  # matures when the last target is known
                          "actual": [a["value"] for a in actuals],
                          "points": {p: list(fold["runs"][p]["point"]) for p in models}})
        rows = list(csv.DictReader(open(f"{name}.csv")))
        histories[name] = [(r["timestamp"], float(r["value"])) for r in rows]

base = {"candidates": models[1:], "baseline": models[0], "min_origins": 5, "recent_origins": 20,
        "lookback_seconds": 90 * 86400, "pool": {"series": ["store-1", "store-2"], "own_weight": 2.0}}
memory = {"features": ["volatility_ratio", "trend", "level_shift", "cv"], "short_window": 7,
          "long_window": 56, "k": 32, "min_effective_n": 8, "own_weight": 2.0}
for label, policy in (("pooled", base), ("memory", {**base, "memory": memory})):
    result = replay_router(folds, validate_policy(policy), histories)
    print(label, round(result["router_score"], 3), result["fixed_provider_scores"], result["evidence_levels"])
```

Compare `router_score` with the best fixed provider, not only the baseline, and judge on
origins after a warm-up. `return_decisions=True` adds each origin's choice, evidence level
and label for held-out scoring; `covariates=` supplies the mask and future covariate for
memory features.

## Cost of routing

With default window retention, each routed call reads at most the evidence window
per series (its own plus pool members), independent of how long the ledger has run: the ledger indexes forecasts by
series, horizon and origin, and routing decisions by kind and series. Ledgers created
before these indexes gain them the next time they are opened for writing. Per-call
work grows with `recent_origins`, the number of providers, pool size and request
history length.

Experimental FASE retention is an exception: live calls reconstruct retained pools
from historical ledger evidence, so reads can grow with ledger age. Benchmark its
latency on your ledger before enabling it; offline replay uses incremental pools.
