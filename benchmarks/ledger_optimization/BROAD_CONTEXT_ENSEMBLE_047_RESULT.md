# Context-matched ensemble 047: small gain, target not met

The frozen context-retrieval rule reduced mean case RMSLE **1.90%** against
the unchanged CV ensemble control on all 416 development tasks. Both domains
improved, but the >=20% development promotion gate failed. No paid agent
confirmation was launched and no reserved future observations were accessed.

| Domain | Cases | CV ensemble | Context-ledger ensemble | Error reduction |
| --- | ---: | ---: | ---: | ---: |
| Electricity | 208 | 0.111226402 | 0.110224520 | 0.90% |
| Pedestrian counts | 208 | 0.410036795 | 0.401129603 | 2.17% |
| Overall | 416 | 0.260631598 | 0.255677062 | 1.90% |

This changes only the retrieval rule relative to earlier ensemble work: match
six CV scores and six observed-history features against temporally available
same-domain episodes, then train combination weights using the sixteen nearest
contexts and current CV folds. Both arms retain the same six forecasts and
combination action. The control predictions are byte-identical to 045. Original
production outcomes are used for scoring only after the current weights exist.

The earlier recent-history ensemble gained 0.96% against this same control.
The new result suggests context matching can help this prototype, but this is
repeated development work on known cases. Neither difference establishes an
untouched-set benefit, an agent benefit, a shipped Gnomon change, or the user's
20% objective. The future-aware 24.11% opportunity from 046 was not used as
training data and remains a hindsight bound, not achieved performance.

All 416 weight fits completed, using 7,759 optimizer iterations and 2.794 seconds
for the measured processing loop (2.716 CPU seconds). Input loading and the
independent audit are outside that timer. No provider refits or API calls were
made. The shared inherited cost remains 12,984 forecast computations, including
warm-up and CV. Both available information and action space remain matched.

Eight synthetic retrieval/numerical tests passed before execution. The audit
passed **97,777 checks**, independently recalculating temporal filtering,
candidate scaling, distances and ranks; exact training-pair hashes; simplex
feasibility and convex objective bounds; every derived forecast and score; and
aggregates. It reuses the already audited 043 feature derivation and does not
refit original providers. Recording availability remains explicitly assumed at
period end, not reconstructed real-world recording history.

Protocol and implementation were frozen at `6b60292` before scoring. Raw cases,
all candidate distances, selected neighbors, weights, certificates, environment
manifest, rerun instructions and audit are retained in
`results/broad-context-ensemble-047-001/`. Archive hashes and per-file inventory
are in [the receipt](evidence/broad-context-ensemble-047.json).

This is a negative promotion result with a small useful signal. The next
development question is whether historical error relationships predict current
errors reliably enough to support larger corrections. Another arbitrary
neighbor-count sweep would not by itself answer that question. Main, PyPI,
reserved panels and the original agent evaluation remain unchanged.
