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
rule over saved one-step folds from one or more series on a shared clock, using only
folds whose targets have passed, labelling regimes from history known at each origin,
and reports the router's score beside each fixed provider's, overall and per series. Folds come from a
`gnomon_evaluate` study (`origin`, target time, actual and each provider's point). Times
must carry a timezone and are compared as instants. Scores are mean per-fold losses
(`aggregation: mean_<metric>_over_folds`), matching the live per-origin score for
one-step folds; this is not an aggregate RMSLE over all folds. As in live comparison,
a fold with a negative RMSLE prediction or actual is excluded (`excluded_folds`), not
clipped.
Replay is simulation evidence; judge a deployed router on prospectively recorded
forecasts.

## Cost of routing

Each routed call reads at most the evidence window per series (its own plus pool
members), independent of how long the ledger has run: the ledger indexes forecasts by
series, horizon and origin, and routing decisions by kind and series. Ledgers created
before these indexes gain them the next time they are opened for writing. Per-call
work grows with `recent_origins`, the number of providers, pool size and request
history length.
