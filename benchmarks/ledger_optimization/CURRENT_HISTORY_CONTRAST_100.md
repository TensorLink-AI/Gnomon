# Current backtests versus accumulated production evidence — candidate 100

Status: offline development candidate; not deployed in the running 097 experiment.

The observed 097 round-11 ledger session for item 1047756/store 23 compared
Ridge and random forest. Random forest had lower RMSLE on all three current CV
folds. The ledger's direct comparison favored Ridge on one much older production
origin, with no matches in the last four origins. The agent selected Ridge.
This motivates making these distinct evidence cohorts easier to compare. It
does not establish that the selection was wrong or that following current CV
would improve future forecasts.

`current_history_contrast_100.current_history_contrast` describes an exact pair
of complete, versioned, three-fold current runs alongside an already retrieved
historical pair. It reports mean RMSLE, fold wins/ties, sample counts, historical
recency and whether the cohort winners differ. It preserves full-evidence
references, makes no forecast selection, and performs no new query or fit.
Missing catalog entries or pages are unknown evidence, not zero errors or ties.
Means from CV and production are never directly compared with each other.

The historical review's hash, identities and paired aggregates are checked by
the existing 098 validator. Current inputs must come from authenticated execution
records; this helper checks their identity, matched fold endpoints, temporal
ordering, scores and references, but cannot authenticate a caller's numbers or
recording availability by itself. A future integration must use trusted records,
not agent-supplied scores. No change to the installed Gnomon package is proposed.

## Read-only replay plan

Before running the replay, fix its scope to every ledger session in the 097 pilot
and independently audited continuation batches 001–004 (30 sessions). Read only
task inputs, boundary logs, experiment logs and already returned review files.
Do not read host targets, calculate counterfactual forecast accuracy, fit models,
call Engy, or access final/protected evaluation data.

Process each boundary in original order. Retain only the latest successfully
returned historical review and explicitly returned complete backtest runs.
After a backtest, compare it with each distinct previously returned current run.
After a review, describe all pairs of current runs already returned. Default
`start` does not expose CV scores, so its internal folds are not eligible until
explicitly returned by a backtest. Never infer missing page content.

Recompute current fold RMSLE from logged CV pairs, verify each returned execution
reference, and require its targets to end no later than the task origin. Record
source hashes and verify source bytes remain unchanged. Measure standalone
contrast bytes against the original response bytes; this is payload overhead,
not a tokenizer or future API cost measurement. Preserve every emitted panel,
including unknown-history and agreeing-cohort cases. Report coverage and
disagreement counts, acknowledging overlapping pairs/windows are not independent.

This replay can validate correctness and characterize payload cost. It cannot
show agent benefit. Deployment requires a frozen integration, actual Hermes
boundary tests, fair unchanged evidence/budgets, and a prospective development
comparison after 097 completes. The 20% final target remains unchanged and unmet.
