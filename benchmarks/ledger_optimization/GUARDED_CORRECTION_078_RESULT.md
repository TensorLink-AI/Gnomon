# Guarded accumulated-error078: negative development result

Source driver frozen88835b9after077synthetic preparation. All416paired tasks
completed, using six fixed model configurations and separate guard/production
cutoffs. Seventh-slot search selection was excluded to avoid using the held-back
CV fold through its configuration choice. No source forecasts/API calls or
protected validation/final access.

| Policy | Mean per-case RMSLE | Electricity | Pedestrian |
|---|---:|---:|---:|
| Guarded control |0.26104146582467785|0.11126401516698547|0.4108189164823703|
| Guarded ledger |0.2578173225002649|0.10933882526357587|0.40629581973695394|
| Uncorrected045 |0.26063159842148664|0.11122640184939629|0.410036794993577|
| Strong050 |0.2587110657586681|0.10748648550251698|0.4099356460148192|
| Lifetime061 |0.25198521475530405|0.10734528573584194|0.3966251437747662|
| Incumbent068 |0.2510782023798801|0.10483908831312373|0.3973173164466364|

Ledger improves1.23511%over matched guarded control,1.07979%over uncorrected045,
and0.34546%over strong050. It worsens2.68407%versus068and2.31446%versus061.
Both domains worsen versus068; electricity also worsens strong050.20%and
incumbent/domain gates fail. No promotion or paid/final confirmation justified.

The held-back guard enabled195/416control corrections and207/416ledger
corrections. Rejected cases retain the exact045baseline and skip production
forest fitting. The control overall is slightly worse than uncorrected045.
Thus a held-back improvement did not guarantee useful future corrections. This
experiment does not prove why generalization failed or that all guards fail.

Guard training uses first two current CV folds and ledger-only historical
production outcomes mature by t-24h. Guard baseline weights also exclude the
third fold. The third CV fold is scored only after both guard forecasts exist.
Production uses saved045three-fold weights and, when enabled, a new forest with
three current folds plus historical records mature by t. All-domain history
means all visible records in the same domain, not all domains mixed together.
The historical timing remains an explicit nominal period-end simulation.

Costs:416guard baseline solves/5,994iterations,832guard forests,402production
forests,39,488trees.57.447s wall/50.859CPU. Both arms have identical recipes and
maximum budgets; disabled production corrections do not waste extra forest fits.
Inherited costs49,616raw computations,416045anchors,832068blend fits,
11,902search surrogate solves and31,378logical search attempts per arm remain
reported. Forest construction is new numerical computation, not a source-model
forecast or API call. Prior unrelated experiment costs are not erased.

Audit incident: the first independent tree audit found a one-row node-support
mismatch. A diagnostic retry identified the case/tree. The independently
reconstructed float32 features were byte-equal to the learner's features;
NumPy's weak-scalar array comparison rounded a saved double split threshold to
float32. Widening the float32 values before comparing to the double threshold
reproduces the tree's actual comparison and restores node counts. The evaluator,
learner, predictions and scores were unchanged. Failed audit logs and a focused
reconstruction script/result are retained. Two boundary tests cover this audit
precision rule. Final full audit passed2,551,857checks over1,234forests/39,488trees and416
baseline certificates in44.871s. Most assertions are repeated tree-node checks,
not independent statistical evidence. Metrics are in the tracked receipt.

No new forecast superiority is established. This is repeated-development
numerical evidence, not an actual1.2.0/DeepSeekagent trial or untouched result.
The fixed068ledger remains incumbent. Main/PyPI unchanged;20%/95%final goal
unmet. The next step should identify a concrete remaining source of transferable
forecast error before adding further model/guard complexity.
