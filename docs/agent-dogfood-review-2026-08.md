# Agent dogfood review — an LLM agent uses Gnomon cold, 2026-08-07

An agent (Claude, running in Claude Code) was pointed at this repository with
no prior context and asked for its thoughts on using Gnomon as its tool. This
is the write-up: what the first-contact experience was like, what held up,
and two correctness findings in the threshold/exceedance path, both verified
by execution against the shipped examples.

Method: `pip install -e .`, then the README's own 60-second scenario —
forecast `examples/messy_requests.csv` 14 days ahead, investigate what
changed, and build an alert rule for crossing 340 at 20× miss cost — plus
the filthy-file repair path, an MCP stdio smoke test (`initialize` /
`tools/list`), and the full test suite (891 passed, 2 skipped, 29 s).

## Verdict

The contract holds where most tools break: every wrong invocation I made
came back as structured JSON with a usable repair option, degradation was
disclosed instead of hidden, and the messy file was refused until repair was
explicitly opted into — then completed with support downgraded. The
first-contact experience is genuinely better than any forecasting surface I
have been wired to.

But the flagship scenario the README tells an agent to run — monitor
`messy_requests.csv` for exceedance of 340 — publishes a crossing
probability of **0.61 per step** beside quantiles that imply **0.10–0.20**,
in the same artifact. The two numbers cannot both be true, and an agent
quoting the monitor's alert rule quotes the wrong one. The cause is a
consistency break introduced with recentring suppression (F1). A second,
smaller finding: the `evaluate_threshold_risk` plan operator crashes on
every invocation (F2).

## Findings

### F1 — Threshold probabilities are not recentred like the intervals they ship beside

**Where.** `pipeline.py` `threshold_analysis_stage`; consumed by
`macros.monitor` (`probability_above_per_step`), `macros.decide`
(`scenario_probabilities`), and the forecast artifact's `threshold` block.

**What happens.** Run the README scenario:

```
gnomon monitor examples/messy_requests.csv --target requests \
  --threshold 340 --horizon 14 --alert-cost 1 --miss-cost 20
```

The artifact publishes, for every step, `point 315.3`, `q80 321.4`,
`q90 351.8` — so by its own intervals, P(above 340) sits between 0.10 and
0.20 — and, in the same payload, `probability_above_per_step: 0.6071` with
`first_timestamp_point_above: null`. A probability of 0.61 that the value
exceeds 340, attached to a distribution whose 80th percentile is 321.4, is
self-contradictory on its face.

**Why.** On this fold-starved run `conformal_spreads(recentre=False)`
zeroes the median component: the published quantiles are centred on the
model's point path, discarding the median backtest residual (+27.9 here —
`last_value` under-predicts throughout because the series trends up). But
`threshold_analysis_stage` computes

```
point + centre_shift + (residual - centre_shift) * scale
```

with `centre_shift = 0`, which leaves the raw residual cloud *uncentred*:
its median lands at `point + 27.9`, so 17 of 28 residuals clear the
threshold and P = 17/28 = 0.6071. The docstring's claim that probabilities
are "recentred and scaled exactly like the published intervals" was true
when `centre_shift` carried the median; with recentring suppressed it is
false, and the crossing probability inherits the exact location bias the
suppression was measured to remove (H-G5). Instrumented reproduction:
residuals passed in are `[5.1 … 72.6]`, all positive, median 27.9;
`spreads[1] = (20.8, 0.0, 36.5)`.

**Blast radius.** Every fold-starved `monitor` and `decide` run — i.e. the
short-history regime Gnomon most expects agents to bring — publishes
exceedance probabilities biased in the direction of the model's mean
backtest error. With no costs supplied, the default 0.5 rule flips the
alert decision outright (0.61 alerts; the interval-implied ≤ 0.20 does
not). The suite's 891 passing tests do not cover the
consistency between `probability_above` and the published quantiles.

**Fix.** Recentre the cloud by the residual median unconditionally —
`point + centre_shift + (residual - residual_median) * scale` — so the
probability mass sits wherever the published intervals sit, under either
recentring policy. Add a consistency test: for any artifact, the empirical
P(above t) must agree with the published quantiles' bracketing of t at
every step.

### F2 — `evaluate_threshold_risk` operator crashes on every call

**Where.** `operators.py:466-489`; registered with a live runner in
`registry.py:114-118`; reachable through plan execution
(`execution.py:_execute_step`). Zero test coverage (no reference in
`tests/`).

**What happens.** The operator builds
`residual_quantiles = {0.1: …, 0.5: …, 0.9: …}` and passes it as the
`spreads` argument of `threshold_analysis_stage`, whose contract is
`dict[int, tuple[low, centre, high]]` keyed by lead step. `spreads[1]`
then raises `KeyError: 1` on the first loop iteration — every invocation,
any input. Through a plan the caller sees
`{"code": "OPERATOR_ERROR", "message": "1"}`, which is both a crash and a
contract violation (an unrepairable, meaningless error message).

**Fix.** Build real per-lead spreads (`conformal_spreads` on the pooled
residuals, or `{step: (q50-q10, q50, q90-q50) for step in 1..len(points)}`),
and add the missing test — the operator currently cannot have ever run
successfully, which is also a hint that no shipped path exercises it; if
none does, consider whether it should be registered at all.

## First-contact notes (agent UX)

Small things, listed because the repo's bar for this is high:

1. **Flag repair works.** `--column requests` → structured error with
   `{"unknown": "--column", "suggestion": "--target"}`. I applied the
   repair mechanically and succeeded. This is the contract doing its job.
2. **`--cost-ratio` gets no suggestion.** The README frames the scenario as
   "costs us 20x a false alarm", so a cost-ratio flag is the natural first
   guess; the error returns no `flag_suggestions` and no pointer to
   `--alert-cost`/`--miss-cost`. A repair option mapping a ratio to the
   cost pair would close the README-to-CLI gap.
3. **Time-column inference error under-explains.** On
   `filthy_requests.csv`, `--time` inference fails with "no column
   qualifies" and `candidates: []`, although `timestamp` exists and
   qualifies perfectly well once named explicitly (the *series* then fails
   validation for duplicates, with good repair options). The error should
   name near-miss columns and why each was rejected; "no column qualifies"
   reads as "your file has no time column", which is false here.
4. **Refuse-then-repair is the right shape.** Duplicate timestamps refused
   by default with two repair options; `--repair aggressive` completes with
   `support: weakly_supported`. Exactly the disclosed-cleaning behaviour
   the README promises.
5. **MCP surface checks out.** `gnomon mcp serve` over stdio answers
   `initialize` and lists 24 tools; the tool names match the README's
   claim.
6. **The self-critical docs are load-bearing.** `codebase-review-2026-08`,
   the kill memos, and the measurement docs materially changed how far I
   trusted the outputs — and let me confirm quickly that neither finding
   above was already known. Few repositories give an agent that.

## What was actually run

- Full suite: `python -m pytest tests -q` → 891 passed, 2 skipped, 29.09 s.
- README scenario end-to-end (forecast / investigate / monitor) on
  `messy_requests.csv`; repair path on `filthy_requests.csv`; contrast run
  on `daily_requests.csv` (P per step 0.0 — consistent there because the
  series sits far below the threshold, not because the F1 path is sound).
- F1 reproduction: monkeypatched `threshold_analysis_stage` to dump inputs
  under `macros.monitor`; residual cloud and spreads as quoted above.
- F2 reproduction: direct call with well-formed arguments →
  `KeyError: 1` from `pipeline.py:723`.
