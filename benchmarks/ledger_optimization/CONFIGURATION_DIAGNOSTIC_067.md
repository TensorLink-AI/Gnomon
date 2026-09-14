# Diagnostic067: distinguish proposal quality from selection quality

Freeze this diagnostic before reading new comparisons from completed066. It is
hindsight analysis of original development only, never an executable selector.
No new forecasts, candidate tuning, protected final or validation access, API
calls or modification of066 decisions. The failed066 gate remains failed.

For all416 scored cases, reconstruct each arm's available production forecasts:
original six plus that arm's final selected configuration if different. The
other ten extra backtested configurations have no production forecasts and
must never be silently scored, imputed, or counted as available alternatives.
Keep this incomplete coverage explicit: at most7 of17 tested configurations
have production outcomes per arm. Missing production forecasts cannot support
claims about the full78-configuration catalogue or all17 tested configurations.

Compute: six-provider current-CV selection; each actual066 selected forecast;
best production forecast among original six; best among each arm's available
production forecasts; best among both arms' union of produced forecasts. Also
retain exact strong block-CV and incumbent lifetime061 scores. Report arithmetic
mean case RMSLE overall, per domain, early0..7 and later8..25. For hindsight
choices report ties, exact candidate IDs and gap from prospective choices.
Compare20% ceiling against strong block-CV without weakening that guard.

For each arm, count original-vs-extra selections, CV improvement over original
six selection, production improvement/worsening/ties relative to it, selected
configuration frequencies, and oracle selection regret. These diagnose whether
search's current-CV criterion translates to the future period; they are not
statistical guarantees or causal explanations. Report same-selection frequency
between arms and preserve the paired per-case inputs to every aggregate.

Read completed066 sources through the archived inventory hash, bind decision
and forecast IDs to the audited source and verify all reported values. Keep
costs inherited:06624,032 new forecast fits plus original12,984 and guard12,600
forecast computations, excluding separately disclosed prior/audit costs. This
read-only diagnostic incurs zero new forecast or weight fits. All metadata and
findings are retained with a new receipt; do not replace066 results.

Next-step interpretation is fixed: if hindsight single-model selection among
the produced forecasts cannot beat strong block-CV by20%, do not optimize a
selector over those same saved outputs expecting20%. Investigate a common
ensemble or different justified model capability before any new paid trial.
If a ceiling permits20%, that only establishes numerical headroom; prospective
and untouched final evidence would still be required.
