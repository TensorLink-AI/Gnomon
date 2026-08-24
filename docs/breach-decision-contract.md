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

The event claim is withheld when any of these holds:

- fewer than eight complete post-selection paths;
- the fixed, scale-free early/late residual comparability gate fails;
- final-test interval coverage is absent or outside its verifiable band; or
- an enrichment or fallback changed the executable without providing its own
  post-selection event residuals.

The numerical estimate remains visible under `support: insufficient`; only
its authority to drive an action is withheld.

## Single-shot policy

The caller supplies `action_cost`, `miss_cost`, and optionally mitigation
effectiveness `e` in `(0, 1]`:

```text
loss(act)     = action_cost + P(breach) × miss_cost × (1 - e)
loss(monitor) = P(breach) × miss_cost
break_even    = action_cost / (miss_cost × e)
```

Gnomon recommends `act` only when the lower end of the event probability's
90% interval exceeds break-even, and `monitor` only when its upper end is
below break-even. If the interval crosses the boundary, the recommendation is
`null` with `decision_support: insufficient`.

This policy assumes one irreversible choice now. It does not price the option
to observe another period and act later. `alert_cost` remains a separate
sequential false-alert policy and must not be supplied together with
`action_cost`.

The risk object is immutable. An LLM may explain it or record a separately
labelled synthesis receipt; it cannot replace the governed recommendation.
