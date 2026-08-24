# Governed breach decisions

Gnomon separates three claims that an agent will otherwise collapse:

1. the probability that any value crosses a threshold during the horizon;
2. whether that event is more likely than not; and
3. whether action is justified under the caller's costs.

The first is an engine estimate. The second is a description. The third is a
client-policy projection. None may rewrite another.

## Event executable

For a threshold job with enough history, evaluation reserves eight disjoint
rolling origins after candidate selection and before the final report-only
test fold. The selected executable predicts each reserved origin using only
the history available at its cutoff. Residuals remain aligned by origin and
lead, then are replayed around the published median path. A path contributes
one “any breach” outcome and, where applicable, its first crossing step.

This preserves measured cross-lead dependence. Independence composition of
per-step marginals is never a governed estimate.

Governed (`support: supported`) authority requires all of:

- at least eight complete post-selection replay paths;
- the fixed, scale-free early/late residual comparability gate passing;
- final-test interval coverage present and inside its verifiable band; and
- no enrichment or fallback that changed the executable without providing
  its own post-selection event residuals.

When any requirement fails, the estimate **degrades, it does not vanish**
— a missing estimate was measured to price as never-act under asymmetric
costs, the most expensive policy on the board. The ladder, each rung
disclosed in `method`, `residual_source`, and `reasons`:

1. `aligned_fold_residual_trajectory_replay_v1` — governed when all
   requirements hold.
2. `independence_composed_marginals_v1` (`support: best_effort`) — the
   published per-step marginals composed under a stated independence
   assumption. They carry the same conformal recentring and per-lead
   spread scaling as the published intervals, which measured better on
   decision cost than raw few-origin residual paths (1.40 vs 2.01 per
   case on the diagnostic corpus). A blocked residual bootstrap over the
   richest available residual source still runs, contributing the timing
   and maximum distributions and a disclosed
   `bootstrap_diagnostic_probability`.
3. `blocked_residual_bootstrap_v1` (`support: best_effort`) — when no
   marginals exist: synthesized trajectories preserving within-block
   dependence, with the interval's sample size pinned to the real origin
   count, never the synthetic path count.

Only when nothing can be estimated at all is the probability withheld
(`support: insufficient`).

## Single-shot policy

The caller supplies `action_cost`, `miss_cost`, and optionally mitigation
effectiveness `e` in `(0, 1]`:

```text
loss(act)     = action_cost + P(breach) × miss_cost × (1 - e)
loss(monitor) = P(breach) × miss_cost
break_even    = action_cost / (miss_cost × e)
```

Whenever a probability exists, the expected-loss comparison at the point
estimate yields a recommendation; what varies is its authority.
`decision_support: supported` requires a supported event estimate whose
90% interval clears break-even entirely on one side. A best-effort
estimate, or an interval straddling the boundary, publishes the same
expected-loss recommendation at `decision_support: best_effort` with the
demotion's reason attached. The recommendation is `null` only when no
probability could be formed (`event_probability_unavailable`): silence
defaulting to monitor was measured to invert the cost asymmetry it
exists to respect.

This policy assumes one irreversible choice now. It does not price the option
to observe another period and act later. `alert_cost` remains a separate
sequential false-alert policy and must not be supplied together with
`action_cost`.

The risk object is immutable. An LLM may explain it or record a separately
labelled synthesis receipt; it cannot replace the governed recommendation.
