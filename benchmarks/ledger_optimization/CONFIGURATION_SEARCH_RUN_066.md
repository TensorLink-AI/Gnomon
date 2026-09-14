# Development066: chronological shared-budget configuration search

065's shared acquisition and evidence access passed synthetic preflight at
aadab1c/cfe01bd. This runner applies those frozen rules to the already-reserved
original development cohort. Freeze runner/protocol/tests before any new source
configuration is fitted. No policy or hyperparameter change after scoring begins.

Use exactly416 scored tasks and125 complete original warm-ups (541 tasks/arm),
retaining the three known unavailable warm-ups. Source spans037 and042 and raw
original038/043 forecasts are verified by their existing SHA receipts. Slice
each task using its original history/target indices and verify the730-observation
history hash,24 targets, source-label phase, origin and target closure. Copy only
observed history/features/unit/source identity into the forecasting request.
Targets and fixed guard forecasts remain outside the search function's arguments.

Two workers, one per domain. Within a domain, process all series chronologically,
and finish both arms for every series at an origin before adding any of that
origin's studies to the available history. Each arm accumulates only its own
executed backtests; control acquisition receives no history, ledger acquisition
uses065's eligible historical backtests. Production actuals are scored after
the same-origin decision batch and stored separately with source/recording time
at the target-period end. They are not search inputs. All availability clocks
remain the disclosed nominal period-end replay assumptions, not real vintages.

Each arm executes six starter configurations at three backtest folds plus
production (24 logical attempts), then11 extra three-fold backtests (33), then
one final selected-config forecast:58/60 total. Maintain the reserved-final-fit
admission check. Original starter predictions may be inherited after verifying
their raw source hashes, CV actuals and064 numerical parity. Reused results still
count as logical attempts. Unselected extra configurations do not receive
production outcomes. Final selection uses only actual current CV as in065.

Share a per-task cache of exact requests between arms. Its identity binds the
complete observed task input hash, canonical config, implementation revision and
history endpoint. Validate cached identity and complete finite nonnegative points.
Alternate which arm executes first by round, so physical cache ownership is
arbitrary and cannot be interpreted as arm efficiency. Report combined physical
costs and each arm's logical attempts/reuse separately. Source-inherited results
carry their original filename/hash references. No historical source or completed
experiment is overwritten.

Write an attempt record before executing each numerical request; failed fits
remain charged. Persist each proposal, current backtest, final execution ID,
decision and cost. Stop on a numerical/integrity failure, retain it, notify the
peer to stop between requests, and do not report an incomplete subset as the
full result. Complete or terminate both worker handles before declaring a run
finished. Observation timeouts do not authorize a restart.

Primary comparison: ledger search versus matched current-only search. Also retain
the exact strong050 block-CV and061 lifetime-ledger predictions for all416 tasks.
Report arithmetic mean case RMSLE overall, per domain, early0..7 and later8..25,
completion, selected configs, logical/physical/reused attempts, estimator fits,
surrogate solves, elapsed/CPU time and all inherited costs. The common original
source generation cost is12,984 computations;061's additional historical source
cost is12,600. They do not disappear merely because cached outputs are reused.

Unchanged development gate:>=20% reduction versus both matched search and strong
block CV; positive gain over lifetime061 overall and all three comparisons in
each domain. No confirmatory interval on reused development. Passing only permits
further integration work, not a superiority claim. Failing forbids paid confirmation.
This is a standalone numerical proxy for reuse of tuning experiments, not an
actual Gnomon/Hermes run or a production-outcome calibration result. Final agent
comparison still requires corrected1.2.0/Engy deepseek-v4.1-flash settings and
untouched final outcomes. Main/PyPI, validation055 and all final reserves unchanged.
