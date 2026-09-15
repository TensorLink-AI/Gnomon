# Eval 3 registration: threshold arithmetic and wrong decisions

Inherits COMMON.md. Hypothesis: a computed per-step threshold reference reduces
wrong decision recommendations from probability conversion. This is prospective
prototype evidence, not a currently shipped Gnomon capability or calibration uplift.

80 cases/arm: two per shared window. Same supplied quantiles in every arm, derived
from that window's location and positive training scale. Quantile levels are
0, .01, .05, .25, .5, .75, .95, .99, 1 with strictly increasing non-uniform values.
The endpoints explicitly define finite support. The scenario distribution is
declared piecewise-linear CDF between knots. This makes reference probabilities
identified by assumption, not inferred from sparse real-provider quantiles.

One above and one below marginal event per window. Rotate false-alarm:miss ratios
20:1, 1:1, 1:20 across windows. An unnecessary action costs C_FA and missing a breach
costs C_M; correct decisions cost zero. Act iff p > C_FA/(C_FA+C_M); equality means
do not act. This convention matters: 20:1 gives a 20/21 cutoff, not 1/21.
Choose event probabilities by rotating cutoff offsets [-.04,-.01,0,.01,.04], clipped
to [.001,.999], and compute the threshold through the declared inverse CDF.
Agents receive threshold, costs and quantiles, never the precomputed probability.
Request probability and act/do_not_act for a single named time step.

Ordinary has Python and the full distribution specification. Lean adds unchanged
release Gnomon. Full adds a benchmark-only forecast-contract overload accepting
provider=benchmark_supplied_quantiles and threshold={level,direction}, reading
the identical public quantile fixture. It returns per-step marginal probability,
method and assumptions; no model is fitted and no existing provider is changed.
No additional MCP tool and no first-passage approximation. This overload must be
labelled in inventory and receipts; a future production implementation needs its
own API review. It is not a seventh shipped tool.

Primary numerical quantity: absolute submitted probability minus reference (also
signed error); material arithmetic error >0.01. Primary business binary: wrong
action relative to reference Bayes decision OR missing/invalid handoff, all 80 tasks.
Report wrong actions/80 and per 100, excess false alarms and missed-breach *decision
recommendations*, and expected excess cost under the declared distribution.
Do not equate Bayes disagreement with a realized loss or claim actual dispatches.
No invented realized outcomes or Brier calibration claim. Cost units are scenario
units, not measured customer dollars. Missing actions have unknown cost, not zero.

Success: corrected significant primary-composite reduction and no increase in
failures, with lower expected excess cost among complete paired answers. No interim
stopping. Report all 80/arm or explicitly incomplete denominator.

Reference behavior: reject nonfinite/crossed/tied quantiles, invalid directions,
unsupported tails, absent support endpoints and path-event requests. Strict above
and below agree at continuous knots. Production sparse quantiles should return
probability bounds or explicitly assumption-labelled interpolation, not an exact
calibrated probability. Discrete atoms require another declared contract.

Limitations: per-step marginals ONLY. At-any-time crossing, first passage and joint
path events are not identified and must be refused. Real forecast miscalibration
survives perfectly correct arithmetic. Synthetic known-distribution decisions show
mechanical utility, not field calibration or economic returns. If the control
computes correctly, publish the null rather than weaken its Python access.
