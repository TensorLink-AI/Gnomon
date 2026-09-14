# Development084: predecessor context did not improve the best mean

Frozen5e99bc8 applied083's additional prior-backtest error context to the same
six models and416 development tasks, with125available warm-ups. Both arms used
the same current CV,32-tree conditional-risk learner and24simplex weight solves;
ledger additionally used mature historical production episodes with their own
preceding CV2errors. Missing CV0predecessors were explicitly marked. Current
production outcomes were read for scoring after both forecasts were produced.

| Approach | Overall416 | Electricity208 | Pedestrian208 |
|---|---:|---:|---:|
| Current CV with prior-error context |0.2591598621124356|0.11411138899262907|0.4042083352322421|
| Ledger with prior-error context |0.2501629167711588|0.1083125374659629|0.3920132960763547|
| Strong block-CV050 |0.2587110657586681|0.10748648550251698|0.4099356460148192|
| Lifetime ledger061 |0.25198521475530405|0.10734528573584194|0.3966251437747662|
| Search/blend reference068 |0.2510782023798801|0.10483908831312373|0.3973173164466364|
| Previous conditional risk081 |0.24977022394038842|0.10591261650813541|0.39362783137264146|

Ledger gains3.47158% versus matched control and3.30413% versus strong050, but
is0.15722% worse than081 overall. Electricity worsens081by2.26594% and068by
3.31312%; it also worsens050and061. Pedestrian improves081by0.41017%. The higher
matched percentage than081does not establish a stronger method:084control also
degraded. Both20% requirements and incumbent/domain guards fail. No promotion.

Early128-case matched gain1.26985%; later288-case gain4.35782%. Early scores
worsen081by1.85877%, later scores improve0.53320%. Those descriptive splits
do not establish growing memory benefit, justify excluding cold starts or
support choosing different methods after observing the domain outcomes.

All416paired cases completed without a solver failure.832forests,26,624trees,
19,968weight fits,328,417iterations;150.535s wall,133.826s CPU. Zero new base
forecasts or API calls. Inherited preparation and comparison costs remain
49,616raw forecasts,416045anchor fits,832068blend fits,832081forests/19,968081
weight fits,11,902search surrogate solves and31,378logical search attempts per
arm. Previous unsuccessful experiments remain separately retained;150seconds
is only this source run's incremental cost.

Independent audit:1,816,368checks, zero failures,76.116s; these are assertions,
not independent statistical samples. Retained in evidence/lagged-risk-084.json.
It checks every
predecessor's task/fold/configuration identity,24-hour horizon and source/local
recording visibility; reconstructs all17features and36Gram targets; verifies
every tree node's support/weighted means, predicted PSD matrices,19,968convex
certificates, predictions, scores and costs. It does not refit the base providers
or establish that nominal period-end availability is real historical provenance.

Full evidence root results/lagged-risk-084-001, compressed archive, per-file
hashes and logs preserved.081 remains the lowest predeclared overall mean and
068 remains an unchanged comparison reference; neither passed the final objective.
No validation/final data were opened and no paid agent run was launched.
Main/PyPI unchanged. This is a numerical proxy, not a released ledger feature
or a matched1.2.0/DeepSeek result. The20%/95% held-out agent target remains unmet.

The added observable context did not solve transfer calibration. Another variant
must address a measured limitation of the estimate/learning objective, not
assume that more features or a domain-specific retrospective winner will help.
