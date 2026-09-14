# Shared sequential search065: frozen policy and evidence-access contract

064 prepared the common78 configurations without scoring source data. This
protocol freezes the subsequent numerical proxy's search algorithm before those
configurations are evaluated. It tests reuse of earlier tuning experiments,
not a new forecasting model, production-error calibration or an agent claim.

## Same current task and budget

Use all416 original development tasks and125 complete original warm-ups. No
added057-series experiments, validation or final tasks are used for tuning.
Retain the original failed three warm-up cases as unavailable. Same nominal
hourly coordinates,730 history,24 horizon, three CV endpoints and source hashes.

Both arms begin with exactly the original six configurations and their current
three backtests plus six production forecasts, charged24 numerical attempts.
Those six forecasts maintain the same common comparison cohort as038, whether
selected or not. Then run11 sequential extra configuration backtests (three
attempts each) and one final selected-config forecast:58 logical attempts/arm/task.
The last two slots cannot admit another three-fold batch plus reserved final fit.
Every attempt counts, including deterministic baselines and failed numerical
fits. Physically reused identical computations are charged to each logical arm,
and physical/reused counts are reported separately. Do not silently drop failures.

Common original starter evidence already exists and may be reused only after
hash/request/parity verification. Its inherited generation cost is12,984
computations for541 cases. The separate061 guard also has12,600 extra historical
computations; retain those comparison costs separately. No new configuration
gets a production forecast unless selected; its completed CV folds are backtest
evidence only. Historical production scores are recorded after maturation, but
do not enter this search policy. This isolates reuse of completed tuning work.

## Common acquisition rule

Both arms use the same weighted kernel regression and lower-confidence-style
acquisition rule, with fixed parameters, not an outcome-selected search sweep.
Encode configuration with four kind indicators and normalized window, log-lag,
log-alpha, depth and log-season (nine coordinates total). Context is the same
twelve predecision features from043, computed from the original six CV scores
and current observed history. No current production actual enters any proposal.

For each study independently, standardize log1p CV RMSLE using the mean/std of
its original six CV results; std floor0.01. This makes prior tuning results
comparable without treating a difficult series as a universally bad configuration.
The current arm uses all its completed current backtests as observed targets.

For the ledger arm only, retrieve16 prior same-domain, same-arm studies by
standardized twelve-feature distance, using all earlier eligible studies,
population feature std floor0.1 and ties by origin/series. Minimum16 studies and
three distinct origins. Earlier study origin must be strictly before the current
origin; its backtest source availability and recording time must be at/before
now. Source availability cannot exceed that study origin, and recording cannot
precede it. Revision and executed-backtest kind must match. Complete all
same-origin arm/series decisions before exposing those studies to later queries.
These are backtest recording clocks, not production-horizon maturity clocks.

No-memory training mass is1 over its current completed configurations. With
eligible history, current configurations receive total0.5 and retrieved studies
total0.5, split equally by study and then by that study's executed configurations.
Do not invent results for unexecuted configurations or mix the other arm's trials.
At cold start the same code reduces exactly to no-memory behavior.

Kernel: exp(-2*squared configuration distance -0.5*mean squared standardized
context distance). Regression solves (K+diag(0.05/sample_mass)). Posterior kernel
spread is sqrt(1-k^T inverse(K+noise) k), with tiny numerical clipping only; it
is not calibrated statistical uncertainty. Propose the untested configuration
minimizing predicted standardized CV minus0.5*spread, ties by064 catalogue order.
Repeat after each three-fold result. No production forecasts or scores influence
the acquisition. Finally select the smallest observed current mean CV RMSLE,
ties by catalogue order, in both arms. Never substitute the surrogate's expected
score for the actual current backtest result.

## Evidence, verification and scope

Persist every tested configuration/revision, fold endpoints, points/actuals,
source/recording clock, attempt/cost and failure. Proposals retain their complete
untested ranking, neighbor identities/distances, training hash/counts, standardizer
and acquisition diagnostics. Production outcomes stay separate and are not
silently treated as backtest labels. Historical selection comparisons are based
on actually executed configs, not an all-configurations intersection.

Tests before source execution must cover canonical identities, unchanged cold
start, future/same-origin/late-recording exclusion, irrelevant production values,
independent kernel solve arithmetic, neighbor support/masses and budget admission.
Freeze the task runner separately before source scoring; preserve numerical
failures and require an independent full audit. Initial synthetic preparation
is not a forecast-accuracy result and does not justify paid confirmation.

Report all416 tasks, domain/early/later summaries, chosen configs, completion,
logical/physical attempts, reused computations, historical cost, surrogate solves,
time and all fixed guards. Development gate:>=20% lower mean case RMSLE against
both matched current-only search and fixed050 strong block CV; positive gain
against061 incumbent overall; positive gain against all three in each domain.
No confirmatory interval on reused development. This tests numerical reuse of
past backtests, not production-outcome-based correction or a unique storage benefit.
Eventual Hermes arms retain common raw historical opportunities and1.2.0 runtime.
Main/PyPI and the final matched-agent20%/95% objective remain unchanged.
