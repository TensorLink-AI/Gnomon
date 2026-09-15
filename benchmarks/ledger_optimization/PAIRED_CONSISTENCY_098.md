# Per-origin consistency — offline candidate

The completed 097 pilot contains a useful distinction between lower average
error and consistent wins. For `item_1047756_store_23`, the recording-visible
review at the September 13 origin compared these prior production forecasts:

| Prior origin | Ridge: window 365, lags 14, alpha 10 | Ridge: window 730, lags 28, alpha 10 |
|---|---:|---:|
| August 16 | 0.644845 | 0.652809 |
| August 30 | 0.530833 | 0.486254 |
| Mean RMSLE | 0.587839 | 0.569532 |

The longer-window configuration has a lower mean but wins only one of the two
origins. The agent's saved rationale said it had the lowest error in “both
matched windows.” That wording is ambiguous: at this early stage the recent
and lifetime summaries share the same origins. It does **not** establish a
confirmed false claim about individual origins. An earlier progress message
overstated that conclusion and was corrected. The interpretation risk remains:
the brief reports means but does not directly show per-origin wins and losses.

`paired_consistency_098.paired_consistency(review, full_evidence_bytes)` adds
left/right win counts, exact ties and a boolean identifying whether any
aggregate winner lost individual origins. Empty evidence has zero counts and
a null loss flag, rather than looking like a zero-error tie. The left/right
orientation follows the existing pair's configuration order.

The helper checks the full-evidence digest, source identities, per-origin
counts, explicit historical timestamps, matched horizons, aggregate scores,
window references and page navigation before rendering. It retains original
scores, exclusions, configuration identities, page order and full-evidence
references. Identical windows keep their existing references. It neither pools
unequal cohorts into global ranks nor makes a forecast choice.

`detailed=True` additionally exposes observed difference ranges, average paired
differences, latest-origin winners and configurations never worse within the
cohort. Ranges and counts are descriptive, not confidence intervals or causal
evidence. Upstream ledger retrieval remains responsible for recording visibility
and matching predictions to actuals: a digest alone does not prove correctness.

## Verification and cost

Ten tests cover a mean winner with losses, origin versus aggregate ties, empty
evidence, opposite recent/lifetime winners, partial pages, immutable inputs,
bad hashes, duplicate JSON keys, revision conflicts, nonhistorical origins,
inconsistent counts/scores and invalid reference chains.

All ten current-origin review pages from the completed pilot were replayed,
including repeated requests. Original scores, navigation and references were
preserved. There were 43 separately rendered windows after existing window
references were respected; one flagged an aggregate winner with an origin loss.
These overlapping page/window counts are not independent samples or a measured
rate of agent misunderstanding.

The initial detailed-default version increased compact UTF-8 bytes from 49,851
to 74,399 (49.25%). The counts-default version used 60,482 bytes (21.33% above
the original). Both iterations and source snapshots remain under
`results/paired-consistency-098-offline-*`. This is a byte-size measurement, not
token cost or evidence of improved decisions. Added context may outweigh the
benefit; a later prospective test must measure that tradeoff.

This module is not imported by the live 097 worker. Its frozen source inventory
was checked unchanged. No model fits, Engy calls or final-target access were
used. Finish and audit the current continuation before considering deployment
in a separately frozen candidate. The 20% objective remains unestablished.

Evidence: `evidence/paired-consistency-098-offline-001.json`.
