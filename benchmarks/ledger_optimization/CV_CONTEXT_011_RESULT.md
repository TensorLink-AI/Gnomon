# CV-context screen 011: do not promote

The training-selected retrieval rule did not improve the later development
slice. This screen made zero API/provider calls, changed no forecasts and opened
no new confirmation data. It does not establish the 20% forecast-error objective.

Protocol and implementation were committed as `bd9768f` before computation.
Input: the existing eight development series, 26 origins each, 208 cases.
The four variants were fixed in [the protocol](CV_CONTEXT_011.md). Selection used
origins 0–17 only; origins 18–25 are a previously used development-validation
slice, not an independent holdout. No policy was selected on confirmation 010.

The selected variant requires four past matched origins with the same historical
CV leader as today's, lower paired mean loss and wins on at least 75% of those
origins before switching providers.

| Policy | Training mean RMSLE | Later development mean RMSLE |
| --- | ---: | ---: |
| Current CV leader | 0.5747938171 | 0.5456698185 |
| Existing unconditioned support rule | 0.5624192321 | 0.5352551237 |
| Training-selected CV-context rule | 0.5735641932 | 0.5472044473 |

The new rule is 0.28% worse than current CV and 2.23% worse than the existing
support rule on the later slice. Retain all four variants in the result; do not
substitute another variant because its later score is better. No product change
or paid agent comparison is justified by this screen.

Five tests pass, covering visible-outcome cutoffs, exclusion of current/future
origins, context labels based on prior CV rather than eventual success,
future-outcome perturbations, deterministic input ordering, unchanged input
forecasts, duplicate identities, independent arithmetic and zero-loss ratios.
Additionally, all 208 reference-support choices reproduce the previously saved
screen-007 choices, and the input SHA-256 remains unchanged. Source availability
at valid time and local recording at horizon close remain disclosed replay
assumptions, not measured historical retail publication records.

[Full retained result](evidence/cv-context-011.json) includes each case's chosen
provider, retrieved origins, losses, all variant scores, and input/script/protocol
hashes. No monetary charge was incurred by this offline screen. Original live
experiment costs and negative confirmation remain in their own records.

Further work toward the objective must remain on development data. A future
live trial would need equal raw information for every arm, including these
historical CV labels. This result provides no basis to consume a new final cohort
or relax the 20% threshold, metric, forecast portfolio or uncertainty requirement.
