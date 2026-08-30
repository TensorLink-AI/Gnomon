# v0.6 loop I006: joint multi-horizon uncertainty

Decision: **no-build for cumulative totals; preserve existing breach risk.**

Gnomon's existing threshold engine already answers the operational any-step
breach question and keeps weak dependence assumptions explicit. On the frozen
60-window real-series replay, 45 cases carried best-effort event probabilities,
15 had no estimable event probability, and the governed policy cost 60 versus
90 for always-act and 140 for always-monitor. The corpus contained no
dependence-preserved supported rows, so this run makes no new supported-path
skill claim.

The bounded candidate reused its blocked-bootstrap paths to publish cumulative
horizon totals. It preserved every existing forecast, interval, probability,
support field, and decision exactly. Its 24 available total intervals were
coherent and averaged only 25.46% of the width obtained by summing marginal
bands, but covered just 25% of realised totals. The frozen 65–95% coverage gate
therefore failed decisively. No thresholds, bootstrap settings, quantiles, or
corpus choices were tuned after inspection, and the candidate was removed from
the product.

The rejected implementation remains reproducible at commit `54dab4d`; commit
`e61b293` removes it while retaining JointHorizonBench and the raw resumable
baseline, treatment, and final artifacts under `results/v06-p6-joint-horizon-*`.
The final local suite passed 2,560 tests with 11 skips. The competition-specific
`docs/astrid-btc-agent-plan.md` remains untracked and excluded.
