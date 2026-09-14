# Development081: lower overall conditional-risk score, failed promotion gate

Frozen source driver50bf8c1 applied the080 recipe without changing its model,
features, regularization, solver or admission thresholds after scoring. Both arms
used the same six fixed forecasts and current CV evidence. Ledger additionally
used all strictly earlier, mature same-domain production errors. Forecast-step
features contained predictions only. Historical target, source and recording
availability were checked before outcome reads, under the disclosed nominal
period-end recording assumption. These are numerical development proxies, not
new Gnomon/Hermes executions or a shipped ledger change.

| Approach | Overall416 | Electricity208 | Pedestrian208 |
|---|---:|---:|---:|
| Current-CV conditional risk |0.25768712397413296|0.1102053144368317|0.4051689335114342|
| Ledger conditional risk |0.24977022394038842|0.10591261650813541|0.39362783137264146|
| Strong fixed block-CV050 |0.2587110657586681|0.10748648550251698|0.4099356460148192|
| Lifetime ledger061 |0.25198521475530405|0.10734528573584194|0.3966251437747662|
| Incumbent search/blend068 |0.2510782023798801|0.10483908831312373|0.3973173164466364|

Ledger gains3.07229% against its matched control,3.45592% against050 and
0.52094% against068 overall. It worsens068 electricity by1.02398%, while
improving pedestrian by0.92860%. Early128-case gain versus matched control is
2.30137%; later288-case gain is3.38164%. This descriptive split does not
establish increasing memory value or a reliable effect on new series.

The overall mean is lower than the previous predeclared development methods,
but both20% requirements and the domain incumbent guard fail.068 remains the
promotion reference;081 is retained as a failed candidate. No paid agent run,
protected validation or final confirmation is justified by this result. No
confirmatory uncertainty interval is claimed on repeatedly used development
data. The actual final-agent20%/95% objective remains unfulfilled.

All416paired cases completed:832forests,26,624trees,19,968quadratic weight fits,
313,736solver iterations;141.322s wall and126.652s CPU. Zero new base forecasts
or API calls. Inherited costs remain49,616raw forecast computations,416045anchor
fits,832068blend fits,11,902search surrogate solves and31,378logical search
attempts per arm. The141seconds are not the cost of creating the source evidence.

Independent audit:1,937,682checks, zero failures,70.126seconds. These checks
are structural/numerical assertions, not independent statistical samples.
Results are preserved in evidence/local-risk-081.json and
the source archive's verification.json. The verifier reconstructs training
features and all36Gram targets, every tree node's sample support and weighted
means, all24query matrices per arm, simplex objectives/gradients/gap bounds,
in-range forecasts, immutable comparisons, aggregate scores and recorded costs.
It reuses previously audited raw forecasts rather than refitting base models.

Evidence root: results/local-risk-081-001. The archive and per-file SHA-256
receipt preserve all model structures, training references, outputs and logs.
Main/PyPI are unchanged. Next work should diagnose which risk estimates fail to
transfer between CV and production, including the electricity regression,
before another source policy is frozen; do not select a domain-specific winner
after seeing these scores and call it a prospective result.
