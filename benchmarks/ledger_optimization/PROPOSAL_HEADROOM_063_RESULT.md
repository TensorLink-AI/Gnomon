# Existing proposal selection cannot supply20% over the strong control

Even a perfect hindsight selector over the six saved prospective forecast arms
would improve the matched block-CV control by only7.9141%. Adding all six
individual providers as alternative choices raises that ceiling to18.0039%,
still below20%. This rules out a selector-only path over these exact saved
predictions on these416 development tasks. It does not establish a universal
ceiling for Gnomon, new causal mixture weights or model-configuration search.

| Hindsight choice class | Minimum mean RMSLE | Max gain vs matched block CV | Max gain vs single-provider CV |
|---|---:|---:|---:|
| Six individual providers |0.220504673|14.7680%|21.6910%|
| Six saved prospective proposals |0.238236305|7.9141%|15.3939%|
| Union of both sets |0.212133063|18.0039%|24.6640%|
| Independent per-step envelope |0.105495927|59.2225%|62.5347%|

The last row has a much larger action space than the current shared four-block
rule: it uses the future actual to choose each horizon's ideal convex weight
independently. It is an optimistic lower bound on error, not an available agent
policy or evidence that this gain is predictable. No such forecast was deployed.

The matched block-CV control has mean RMSLE0.258711066; current single-provider
CV selection has0.281582768. These are different controls. Switching to the latter
would make some ceilings exceed20%, but would not demonstrate improvement over
the stronger frozen comparator. Even against the weaker current-CV selector,
20% using only individual providers would require92.20% of the hindsight gain.
The actual Hermes baseline is a separate experiment and is not replaced here.

The union ceiling is below20% in both domains:18.5699% electricity,17.8554%
pedestrian. It is also below20% in early rounds0..7 (17.1889%,128 tasks) and later
rounds8..25 (18.3347%,288 tasks). All416 remain in the primary denominator; these
dependent subgroup summaries are descriptive, not independent confirmations.

The next development action should not be another gate choosing among these
already-saved proposals. The existing continuous mixture space has additional
hindsight headroom (046), but past retrieval experiments have not learned most
of it. The user's model-iteration task also permits a common configuration
search space; the six-recipe screen is narrower than that full workflow. Any
further candidate must demonstrate improved use of matured evidence with the
same actions and budget available to the control. Do not weaken that control,
choose easier source periods or expose hindsight choices to a learner.

Protocol/code frozen at4c3eaa3 before computation. Five synthetic tests passed
after correcting a pre-freeze test's exact floating-point equality assertion.
Independent array-based audit passed22,654 checks: forecast/actual identity,
RMSLE, minima/ties, envelope projection, all aggregates and scoped ceilings.
No new provider/API calls or weight fits;0.1345s diagnostic compute wall time.
Historical cost remains25,584 forecast computations/12,792 model estimator fits.

[Receipt and archive](evidence/broad-proposal-headroom-063.json). This is a
hindsight diagnostic on reused development data, not a statistical confidence
interval, held-out result or executable ledger gain. Main/PyPI unchanged;
validation055 and final reserves untouched. The20% matched-agent goal is unmet.
