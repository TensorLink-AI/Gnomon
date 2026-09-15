# Observed search ceiling — diagnostic 104

Freeze this calculation before computing its results. Use exactly the pilot and
candidate-100 continuation audit batches 001–004, after every batch passes its
independent audit. Authenticate the committed reports and snapshot inventories.
These are reused development cases, not an untouched confirmation sample.

For every complete three-arm case, recompute the selected forecast RMSLE and
the RMSLE of every successful current-origin production forecast in each arm's
retained numerical log against the same host actuals. Require exact task,
history, future timestamps, unit and series identity. Include the common
seasonal fallback as an explicitly labelled option in every arm, including
failed workflows. Preserve incomplete arm groups as pending; do not select
cases by success, improvement, history length or model family.

Report actual selected means, the hindsight minimum within each arm's explored
set plus fallback, and the hindsight minimum over the three arms' combined
explored sets plus fallback. Report configuration counts, all-case and per-series
means, and the relative reductions against the actual no-ledger mean. Use the
arithmetic mean of per-case RMSLE. Do not pool squared errors or introduce a
new cold/mature threshold. Zero control means make ratios undefined.

These hindsight selections are unattainable policies, not ledger recommendations
or alternative benchmark scores. A ceiling below 20% would limit selection-only
improvement within these observed sets, not prove that the model families or
ledger cannot help. A larger ceiling would establish opportunity only; it would
not establish that past-only evidence can identify the better forecast. The
union also combines different search trajectories and is not an arm with a fair
shared budget. Do not deploy it or compare its hypothetical costs as an agent.

This analysis may read already-used development targets for scoring, but must
not expose them to agents, update a live ledger, train a policy, open reserved
data or issue model/API calls. Preserve scripts, hashes, failures and outputs.
The actual experiment, 20% objective, final gate and main/PyPI remain unchanged.
