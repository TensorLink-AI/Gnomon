# Pre-registered hypotheses: the short-history news regime (MTBench-shaped)

Registered 2026-08-03, at commit `d2c503a` (branch
`claude/gnomon-news-driven-trust-rbgc17`), **before** any of the four
experiments below was run. `RESULTS.md` is written against these
predictions, not adjusted to fit what comes out. One mechanics smoke run
(a single 30-point task through `gnomon forecast`, to learn the artifact
shape for the audit) preceded this file; no aggregate number had been
computed when these predictions were frozen.

## Environment constraints, disclosed up front

This container has **no LLM API key** and its egress policy **denies
huggingface.co** (403 on CONNECT, recorded by the agent proxy — an
organization policy denial, not a transient failure). Consequences:

- The official MTBench processed datasets (Hugging Face-only) cannot be
  downloaded, and `results/deepseek-v4-flash/` (untracked, machine-local)
  does not exist here. The **official recorded numbers** used below are
  the ones restated in the task brief from that run: control (ungated
  DeepSeek V4 Flash) running means **MSE ≈ 6.5, MAPE ≈ 2.8%**; Gnomon
  arm **MSE ≈ 10.9, MAPE ≈ 3.8%**.
- All four experiments therefore run on a **surrogate task set**: 50
  tasks of 30 daily bars in / 7 out, real S&P 500 closes (2013–2018,
  plotly/datasets mirror of the Kaggle 5-year dump), tickers restricted
  to median price $30–200 so the MSE scale regime matches the official
  aggregates (`scripts/build_tasks.py`, seed 20260803). Same asset
  class, same bar convention, same window shape, no news text. Every
  cross-dataset comparison against the official numbers is **qualitative
  by construction** and is flagged as such in RESULTS.md.
- TSFM weights also come from the Hub at inference time
  (`tsfm_sandbox.py` workers call `from_pretrained` against
  huggingface.co), so **experiment 2 cannot produce forecast numbers in
  this environment**. Its eligibility-layer claims are still testable and
  are pre-registered; the forecast-quality prediction is registered now
  for a later machine.

Metrics follow the adapter's official block
(`benchmarks/mtbench/gnomon_forecaster.py`): per-task MSE/MAPE, means
over tasks passing the official `mse > 100` failure filter, with
abstained and filtered counts reported beside every mean. Coverage is
additionally reported unfiltered (the filter is about point error, not
intervals).

## E1 — Martingale floor

Run pure `last_value` (flat continuation of the final observed bar) and
`gnomon` pure mode (no LLM, no events) on the 50 tasks.

> **H-E1a.** `last_value` filtered mean MAPE lands in **[1.5%, 3.5%]** —
> the same band as the official control's recorded 2.8%. The regime is
> near-martingale: a frozen number is already close to the best
> anyone did.
>
> **H-E1b.** Gnomon-pure filtered mean MAPE is within **±20% relative**
> of `last_value`'s (selection at 2 folds can neither beat nor badly
> lose to the martingale on average).
>
> **Implication if both hold:** the LLM control's advantage over the
> Gnomon arm in the official run is mostly "the LLM is a decent
> martingale"; the genuine news edge is bounded by roughly the distance
> between the control's 2.8% and the martingale floor — small in
> absolute terms, and that bound sizes the whole opportunity.

Falsifiers:
- `last_value` MAPE > 5% or < 1% → the surrogate does not reproduce the
  regime; every cross-dataset statement in RESULTS.md must be voided,
  not weakened.
- Gnomon-pure > 25% relative **worse** than `last_value` → the official
  Gnomon-arm gap is substantially *selection-inflicted* (fold-starved
  selection actively picking losers), not news-deprivation — which
  re-orders Part 3: selection guardrails at short history would outrank
  any news mechanism.

## E2 — TSFM at 30 points

> **H-E2a** (testable here): `eligible_tsfms(history_length=30,
> horizon=7, frequency="D")` admits **all seven** registered adapters
> with zero exclusions — i.e. the capability gate carries no information
> at short history because `min_context_length` defaults to 1 and no
> adapter overrides it.
>
> **H-E2b** (registered for a machine with Hub access):
> `chronos_bolt_mini`, sandbox-installed and entering the same single
> selection fold, wins selection on **≥ 40%** of the 50 tasks and moves
> the filtered mean MAPE by **less than 0.5 absolute points** in either
> direction — a 30-point context is too short for a zero-shot tier to
> close a martingale-floor gap by itself.

Falsifiers: any adapter excluded at 30/7/D falsifies E2a. On the later
run: filtered mean MAPE improving by ≥ 0.5 points falsifies the
"zero design work closes the gap" skepticism and promotes the TSFM tier
in the roadmap; Chronos never winning a fold falsifies the "wins folds
but doesn't move means" shape.

## E3 — Volatility-capped tilt, oracle direction

Base path: `last_value` flat. Per task: σ = standard deviation of the 30
input bars' first differences; direction d = sign(mean(actuals) −
last_value) taken from the actuals — an **oracle for direction only,
once per task** (matching "LLM reads the news once and picks a sign");
magnitude is always k·σ, never from actuals. Tilted path = last_value +
d·k·σ at every step, k ∈ {0.5, 1, 2}. Report filtered mean MSE/MAPE vs
the untilted floor, plus per-task deltas.

> **H-E3a.** Best k ∈ {1, 2}, with filtered mean MSE reduction between
> **20% and 50%** vs the untilted floor. (Over 7 near-random-walk steps
> the mean displacement is ≈ 1.9σ, so a 1–2σ shift in the right
> direction removes a large share of squared error.)
>
> **H-E3b** (build threshold, frozen now): if even the best k yields
> **< 10%** mean MSE reduction, volatility-capped directional tilts are
> **not worth building** for this regime, whatever their aesthetics.

Falsifiers: reduction < 10% at every k → mechanism (c) is dropped for
this dataset and RESULTS.md says so. Reduction > 50% at k=2 →
the cap is leaving value on the table and a bucketed-k design (LLM picks
a coarse bucket) deserves the follow-up experiment. Note the oracle is
an upper bound: a real proposer with hit rate p < 1 on direction earns
roughly (2p−1) of the oracle's gain at small k, and the RESULTS
discussion must apply that discount before any build claim.

## E4 — Interval honesty at 30 points

From the Gnomon-pure run's artifacts: empirical coverage of q10–q90
against the 7 actuals per task, pooled over all non-abstained tasks
(primary, unfiltered; per-step secondary), vs the nominal central 80%.

> **H-E4.** Empirical coverage lands **below** nominal, in **[60%,
> 76%)** — the pooled residuals are selection-optimistic (the artifact
> itself discloses this) and 8 residuals cannot place an honest 90th
> percentile, and the constant-width interval under-covers the later
> steps of a 7-step horizon.

Falsifiers: coverage in [76%, 84%] → intervals are already
approximately honest at 30 points; mechanism (a)'s interval-side
urgency drops and the trust bug reduces to wording. Coverage < 60% →
worse than predicted; interval repair becomes the first roadmap item,
ahead of any news mechanism. Coverage > 84% → over-wide intervals;
different bug, same priority as the under-coverage case.

## Analysis plan

- All 50 tasks run under identical config (defaults, `--frequency D`,
  synthetic daily axis via the adapter's `write_bar_csv` convention);
  no post-hoc task filtering beyond the official `mse > 100` rule,
  whose exclusion count is always reported.
- Abstentions counted and reported beside every mean; an abstained task
  contributes to no mean.
- Per-task pairs (floor vs Gnomon vs each tilt) land in
  `RESULTS.md` as win/loss counts alongside means; means never stand
  alone.
- Seeds and scripts are versioned beside this file; raw per-task
  outputs stay untracked per the directory `.gitignore`.
