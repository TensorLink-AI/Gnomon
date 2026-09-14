# Development086: norm scaling changes little and fails the gate

Frozen12879f3 applied085's case-norm-scaled training targets with081's original
features and action. Current CV and historical labels are scaled by their
smoothed RMSLE under the unchanged045anchor. The same transformation, learner,
forecast candidates and solver are available in both arms. Ledger additionally
uses only strictly earlier, mature same-domain production evidence. No current
production actuals enter training or query features.

| Approach | Overall416 | Electricity208 | Pedestrian208 |
|---|---:|---:|---:|
| Current CV, norm-scaled targets |0.25713977227119794|0.11090155030378321|0.40337799423861265|
| Ledger, norm-scaled targets |0.24961817893132554|0.10713164375598666|0.3921047141066644|
| Strong block-CV050 |0.2587110657586681|0.10748648550251698|0.4099356460148192|
| Lifetime ledger061 |0.25198521475530405|0.10734528573584194|0.3966251437747662|
| Search/blend reference068 |0.2510782023798801|0.10483908831312373|0.3973173164466364|
| Previous conditional risk081 |0.24977022394038842|0.10591261650813541|0.39362783137264146|

Ledger gains2.92510% versus its matched control and3.51469% versus strong050.
The0.06087% improvement over081overall is tiny and is not established as a
reliable difference. Electricity worsens081by1.15097% and068by2.18674%; pedestrian
improves081by0.38694%. Both20%requirements and domain guards fail. No promotion.

Early128-case gain versus matchedcontrol2.62877%; later288-case gain3.04453%.
Relative081, early improves0.23541% and later worsens0.00995%. Neither descriptive
split establishes growing accumulated-memory value or justifies excluding cases.
Do not infer that the small mean reduction proves the training objective was
the main cause of earlier failures. A learned query risk is not a guaranteed
upper bound on future loss;085's bound applies to complete training cases.

All416paired cases completed without a solver failure.832forests/26,624trees,
70,096case-scale evaluations and19,968weight fits/279,614iterations;
141.482s wall and128.961s CPU. Scales were neither clipped nor selected by outcome.
Zero new base forecasts or API calls. Inherited49,616raw computations,416045anchor
fits,832068blend fits,832081forests/19,968081weight fits,11,902search surrogate
solves and31,378logical search attempts per arm remain disclosed. Prior failed
experiments are retained separately; this runtime is not total project cost.

Independent audit:2,324,657checks, zero failures,85.172s. These are structural
and numerical assertions, not independent statistical samples. The audit in
evidence/norm-risk-086.json reconstructs every training
case norm/scale, verifies tangent value/gradient, all weighted tree node outputs
and sample supports, query matrices and PSD,19,968convex certificates, source
visibility and configuration identity, forecasts, unchanged comparisons, scores
and cost totals. Raw base forecasts are reused from previously audited evidence.

Full source artifacts, case scales, trees, solver traces and logs are retained
under results/norm-risk-086-001 with archive/member SHA-256 hashes. This is a
repeated-development numerical proxy, not new Gnomon/Hermes execution, held-out
evidence or a released ledger feature. Main/PyPI and protected data are unchanged.
No paid confirmation is justified. The actual1.2.0/DeepSeek20%/95% goal is unmet.

Changing the squared-loss training target to this RMSLE-related surrogate did
not materially change the outcome. This weighs against another arbitrary target
scale or feature sweep. Further work must identify a new source of transferable
decision evidence, while retaining the existing strong comparisons and scope.
