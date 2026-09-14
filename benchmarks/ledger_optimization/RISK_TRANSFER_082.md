# Diagnostic082: predicted versus realised risk transfer

Freeze before computing these diagnostics. Read only the416 already scored081
case artifacts, verified against its committed receipt. No raw model fitting,
new predictions, new source records, validation, final or API access. This is
post-result explanation, not a prospective selection policy or an accuracy gain.

For each case and each arm, compute the mean predicted squared-log loss for
its saved weights and the045 anchor using its own saved24conditional Gram
matrices. Independently recompute realised mean squared-log loss from saved
forecasts/actuals, including068. Compare predicted anchor improvement with
realised anchor improvement. Separately evaluate BOTH saved weight vectors
under each arm's risk matrices: predicted ledger-versus-control improvement
can be compared with the realised paired squared-log difference.

Report all cases, each domain and fixed early0-7/later8-25 phases: mean predicted
and realised losses/differences, counts of beneficial/harmful/tied changes,
Pearson correlation (null for constant inputs), sign agreement and conditional
counts when predicted ledger gain is positive. Exact sign comparisons; no
threshold selection, bins tuned on outcomes or alternate-policy scores. Keep
per-case values and the distinction between squared-log training loss and mean
per-case RMSLE evaluation. No causal or confirmatory uncertainty claim.

Audit source hashes, all risk quadratic forms and independently computed
actual losses, counts/denominators and summaries. Preserve code, full output,
archive, hashes, runtime and zero-fit/API costs.082cannot pass a promotion gate:
it can only identify whether the081risk estimates help distinguish good from
bad weight changes. Do not splice domain winners or access held-out evidence.
