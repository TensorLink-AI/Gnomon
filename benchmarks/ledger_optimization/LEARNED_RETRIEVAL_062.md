# Development062: learn relevance from matured comparative outcomes

Previous turn was a status-only check, not experimental progress. Lifetime061
used older records but improved recent060 by only 0.1082%. The next structural
hypothesis replaces hand-written neighbor distance with supervised relevance.
Freeze this protocol, implementation and synthetic tests before scored fitting.

Use exactly the existing 1066 episodes and original 416 development tasks. No
new source observations, forecasting models, validation or final-reserve access.
Keep exact saved global CV045, matched block CV050, original ledger050, expanded
recent ledger060 and lifetime ledger061 predictions as all five comparators.

At each domain/origin, take all earlier episodes whose target period has closed
and whose outcome recording time is no later than the query origin. Same-origin
outcomes are excluded. Source and recording availability remain the disclosed
period-end assumptions of the source experiments, not actual vintage evidence.
Minimum support is 32 episodes and three origins. Otherwise use matched block CV.

Train one ExtraTreesRegressor per domain/origin on the same twelve predecision
features as043. Its six targets are matured production RMSLE minus historical
CV RMSLE for each provider, centered by the six-provider mean of that residual.
This removes shared difficulty and asks which contexts predict CV ranking errors.
No current production outcomes are supplied to training or retrieval. Use051's
fixed settings: 64 trees, depth4, leaf minimum8, all features, no bootstrap,
seed17, squared-error criterion, one thread. No hyperparameter sweep.

Each query receives uniform weight over the training cases in its leaf for each
tree, averaged across64 trees. Keep every positive-weight case; no top-k filter.
These learned nonnegative relevance weights sum to one. Preserve/export training
identities, features, targets, all tree structures, query leaves and record weights.
This differs from051, which predicted Gram matrices for a quadratic metric proxy.

Use050's unchanged four-block smoothed mean-case RMSLE objective,045 CV anchor,
penalty0.01, smoothing1e-6, solver and gap certificate<=1e-5. Current three CV
cases receive total mass0.5; all retrieved history receives total mass0.5 using
the learned weights. Forecasts and scores remain the same log-mixture definitions.
Failures stop and are retained; do not remove tasks or relax acceptance.

Gate: >=20% overall reduction against both CV controls; positive overall
reduction against all three ledger comparators; positive reduction against every
comparator in each domain. Report all 416 tasks, comparisons, support, training
forests/trees, mixture fits, iterations and elapsed costs. No confirmatory interval
on repeatedly used development cases. A failed gate forbids paid confirmation.

Inherited cost:25,584 forecast computations/12,792 model estimator fits. Additional
cost is evidence forest training/retrieval and mixture fitting only; zero original
forecast/API calls. This is a numerical development prototype, not a shipped agent
result. Main/PyPI unchanged; the final matched-agent 20% objective remains intact.
