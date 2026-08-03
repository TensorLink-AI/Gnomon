# Results: the short-history news regime (against HYPOTHESIS.md)

Run 2026-08-03 at commit `d2c503a`, same session that registered the
hypotheses. Data: the 50-task surrogate described in HYPOTHESIS.md
(real S&P 500 daily closes, 30 in / 7 out, seed 20260803 — the official
MTBench sets are Hugging Face-only and that host is policy-denied here).
Every comparison to the official DeepSeek V4 Flash numbers is
cross-dataset and qualitative; those numbers are the recorded means
restated in HYPOTHESIS.md (control MSE ≈ 6.5 / MAPE ≈ 2.8%; Gnomon arm
≈ 10.9 / 3.8%).

Aggregates use the official metric block (per-task MSE/MAPE, mean over
tasks passing the official `mse > 100` filter). On this task set **no
task failed the filter and no task abstained** under any condition, so
filtered and unfiltered means coincide and every mean is over all 50.

## E1 — Martingale floor

| condition | mean MSE | mean MAPE |
|---|---|---|
| `last_value` floor | **2.56** | **1.61%** |
| gnomon pure (defaults) | 7.42 | 2.37% |

- **H-E1a holds**: floor MAPE 1.61% is inside the registered [1.5%,
  3.5%] band, at the low end. The regime is reproduced: a frozen number
  is close to the best anyone did officially (control 2.8%).
- **H-E1b is falsified, by its registered falsifier**: gnomon-pure is
  **+47% relative MAPE** (2.37 vs 1.61) and **2.9× MSE** against the
  floor, losing 34/50 tasks on MSE (winning 16). This exceeds the
  pre-registered 25% falsifier threshold, so the registered
  consequence applies: **the Gnomon arm's official gap is substantially
  selection-inflicted** — fold-starved selection (one selection fold at
  30/7) actively picks losers (theta 8, window_average 9, drift 7,
  ets 7, seasonal_naive 5, linear_trend 3; last_value only 11/50) —
  and is not primarily news-deprivation. Selection guardrails at short
  history outrank any news mechanism in the roadmap.
- Sizing the news edge: control 2.8% MAPE sits *above* this floor band,
  consistent with "the LLM is a decent martingale, and any real news
  edge is small in absolute terms" — but the dataset substitution means
  this is qualitative, not a measured decomposition.

## E2 — TSFM at 30 points

- **H-E2a confirmed by execution**: `eligible_tsfms(30, 7, "D")` →
  all seven adapters, `{}` exclusions. The capability gate carries no
  information at short history (`min_context_length` defaults to 1,
  never overridden — `src/gnomon/tsfm.py:124,137-168`).
- **H-E2b not executed**: sandbox workers fetch weights from
  huggingface.co at inference (`tsfm_sandbox.py:289-291` etc.), which
  this environment's proxy denies. The prediction stands registered for
  a machine with Hub access; recipe in HYPOTHESIS.md and the design doc.

## E3 — Volatility-capped tilt, oracle direction

Direction from actuals (once per task), magnitude k·σ of the input
window's daily diffs, constant shift on the `last_value` floor:

| k | mean MSE | Δ vs floor | tasks better/worse |
|---|---|---|---|
| 0.5σ | 1.99 | **−22.3%** | 40 / 10 |
| 1.0σ | 1.93 | **−24.7%** | 35 / 15 |
| 2.0σ | 3.32 | **+29.7%** (worse) | 24 / 26 |

- **H-E3a holds on the margin**: best k is 1.0 (registered set {1, 2}),
  reduction 24.7% is inside the registered [20%, 50%].
- **H-E3b build threshold passed** (24.7% ≥ 10%) — but barely two-fold
  clear of it after the registered oracle discount: a real proposer at
  p = 0.65 directional accuracy earns ≈ (2p−1) = 30% of the oracle gain
  at small k ⇒ ≈ **7–8% MSE**, *below* the 10% build bar; p = 0.75 ⇒
  ≈ 12%, just above. The ceiling is real but thin, and **k = 2σ
  actively destroys value** — any build must cap at ≤ 1σ and must
  measure the proposer's directional hit rate before granting influence.

## E4 — Interval honesty at 30 points

Empirical q10–q90 coverage, all 50 tasks × 7 steps (350 obs), nominal 80%:

- Pooled: **63.7%** → **H-E4 confirmed** (registered band [60%, 76%)).
- Per step: 82%, 76%, 64%, 64%, 60%, 56%, **44%**. Step 1 is honest;
  coverage decays monotonically to a coin-flip at step 7 — the
  constant-width interval the artifact itself discloses
  (`constant_interval_width`) is the visible mechanism.

## Post-hoc diagnostics (NOT pre-registered; motivated by E1's falsifier)

1. **Margin sweep** (`margin_sweep.json`): raising
   `minimum_baseline_improvement` monotonically closes the self-inflicted
   gap — 0.02 (default) → MSE 7.42; 0.10 → 6.99; 0.25 → 6.75; 0.50 →
   4.32 (with `last_value` selected 11 → 19 → 25 → 34 of 50). Even at
   0.50 the mean stays 69% above the floor: a margin alone shrinks but
   does not remove the pathology, because a single fold still lets a
   lucky candidate clear any margin.
2. **Bias-correction wobble is not the story**: on the 11 tasks where
   `last_value` won selection at default margin, gnomon's q50 path vs
   the raw floor is MSE ratio 1.018 (mean |q50₁ − last| ≈ 0.43σ). The
   damage is model choice, not the residual recentring.

## Files

- `summary.json` — E1/E2a/E3/E4 aggregates (versioned).
- `margin_sweep.json` — post-hoc diagnostic (versioned).
- `raw/` — 50 per-task records incl. quantile rows (untracked,
  regenerable via `scripts/build_tasks.py` + `scripts/run_experiments.py`).
