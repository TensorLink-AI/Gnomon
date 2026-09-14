# Adaptive memory strength074: negative development result

Frozen source runner ea5b863 follows073selection rules. All541tasks completed,
including125warm-up tasks and416scored tasks. This is a numerical experiment on
reused development evidence, not a Gnomon/Hermes agent or held-out result.

| Policy | Mean per-case RMSLE | Electricity | Pedestrian |
|---|---:|---:|---:|
| Matched current-only blend |0.2576259533432357|0.10690353854294049|0.4083483681435309|
| Adaptive ledger strength |0.2540764722213029|0.1069073802896316|0.40124556415297424|
| Fixed half-memory068 |0.2510782023798801|0.10483908831312373|0.3973173164466364|
| Strong fixed-six baseline050 |0.2587110657586681|0.10748648550251698|0.4099356460148192|
| Previous lifetime ledger061 |0.25198521475530405|0.10734528573584194|0.3966251437747662|

Adaptive strength improves1.37777%over matched control and1.79142%over strong050,
but worsens1.19416%versus068and0.82991%versus061. Electricity is marginally worse
than matched control; both domains worsen versus068.20%and incumbent/domain
gates fail. No paid or protected-data confirmation is justified.068remains the
lowest original-development mean, itself only2.54157%better than matched control.

All416scored decisions had complete mature candidate cohorts. Selected strengths:
0:36cases,.25:6,.5:42,.75:137,1:195. The rule frequently chose a larger history
weight from earlier outcomes, but those choices did not improve aggregate future
performance over fixed.5. This establishes a negative result for this frozen
selection rule; it does not establish that all adaptive memory is ineffective.
No current outcome was used to select its own strength. All five candidates
were executed before each decision batch's outcomes were exposed, and later
selection recomputed their risks on the same mature16-neighbor cohort.

Current-only forecasts and fixed.5ledger forecasts reproduce068pointwise within
1e-12, with identical aggregate means. Earlier warm-up anchors use the exact045
algorithm/tolerance. Candidate execution times are simulated historical origins;
the stored source/recording availability assumption remains nominal period end.

Costs:5,410logical blend requests,2,586physical blend fits,2,824exact-input cache
hits,125additional warm anchors.166,598blend iterations and2,007anchor iterations.
40.069seconds wall,37.303CPU. No new source-model forecasts or API calls.
Inherited costs remain49,616raw forecast computations,416anchors,11,902search
surrogate solves and31,378logical search attempts per arm. The extra5candidate
requests per task are equally available in both arms; control candidates often
coincide, so cache reuse does not require wasting computation to equalize costs.

Independent audit results and complete file inventory are in
`evidence/memory-strength-074.json`. Audit reconstructs source/configuration
bindings, temporal neighbors, matched candidate scores and ties, training masses,
all unique convex certificates including warm anchors, cache accounting, all
forecasts and score summaries. Full local archive includes every decision,
candidate forecast, fit, outcome and input-access hash. Main/PyPI unchanged.

Next work should diagnose why historical strength rankings transfer poorly before
adding another selector. The current evidence supports preserving the fixed068
incumbent rather than promoting this adaptive rule or claiming the20%objective.
