# Development091: remove ineligible late runs before ambiguity detection

090reproduced a public1.2.0comparison failure on original development evidence:
retrospective backtests recorded after the first target can make otherwise valid
prospective forecasts ambiguous. Freeze a narrow fix before rerunning that
cohort. Filter each recording-visible execution with a resolvable target grid
whose first target is at or before execution recording time. Report its own
exclusion and recording/target times. It must not enter fingerprint ambiguity,
duplicate counts, provider identity comparisons or scoring. Do not rely solely
on caller-supplied retrospective metadata.

Keep ambiguity rejection when two eligible prospective runs have conflicting
inputs or provider identities. Identical eligible retries remain duplicates,
not independent samples. Preserve missing/invalid timestamp diagnostics and all
source/recording cutoffs, task matching, units and revision requirements. Queries
make zero forecasts and writes. No original data or saved studies are edited.

Test this change in the clean tracked product module on the development branch.
For original-corpus comparison, preserve a narrowly modified copy of the pinned
1.2.0history module as separately hashed development code. Its storage, request
execution and actual lookup still use the installed1.2.0runtime. This is NOT an
unmodified published comparison or a released1.2.0feature. Never edit the pinned
installation or pretend the new comparison has the original source fingerprint.

Unit/regression cases: differing late backtests, identical late retry, late-only
evidence, exact target-boundary exclusion, genuinely conflicting prospective
inputs, genuinely changed prospective provider identity, valid duplicate retry,
early query cutoffs and normal matched comparisons. Test product ledger/context
consumers as appropriate. Run no agent inference in this experiment.

After source freeze, repeat the sixteen090queries against copied original030
ledgers. This is a new experiment, not a relaxed090gate. Preserve old outputs
and the090failure. Compare baseline public and modified results under identical
queries and verify any recovered origin uses only eligible original production
executions and cutoff-visible actuals. Recompute their RMSLE independently from
saved predictions/actuals. Previously scored valid origins and predictions must
remain intact. Report recovered/lost pair-origins, exclusions and query costs;
do not infer an agent accuracy gain from additional evidence coverage.

All files and expected comparisons are development evidence. Protected/final
data stay closed. Main/PyPI unchanged. The20%/95%goal remains unmet until a real
matched agent and untouched final evaluation establishes it.
