# Development057: more accumulated episodes, unchanged retrieval rule

056's richer query features did not improve the incumbent. Test a different
infrastructure hypothesis: the same index/rule may be limited by too few prior
comparable series. Increase the prior episode pool from eight to sixteen series
per domain while keeping scored tasks, forecasts and retrieval parameters fixed.
This is a development test of experience breadth, not a release or agent claim.

## Prospective identity selection

Use only037 eligibility metadata. Reconstruct its fixed hash order. Verify
positions0:8 are original development,8:24 final-reserved,24:32 the locked052
validation identities. Choose positions32:40 as **additional memory-training
series**, eight per domain. Require40 unique eligible IDs. No replacement,
seed changes, favorable subgroups or additional scored series. Commit the named
manifest before interpreting any new observation values. The eligibility
prefixes were already examined in037; subsequent outcomes were not used.

Never read validation outcomes or final-reserved counts for this experiment.
Keep the original416 development tasks/16 scored series. The new16 memory
series are never scored as evaluation tasks and never join the denominator.

## Historical evidence geometry and source preparation

Use the original dates,730-hour histories,24-hour forecasts,26 weekly origins,
three CV folds and six recipes. For new memory series, only original development
origins0 through24 can mature before a later scored origin; do not generate
origin25 or parse its unneeded final168-hour tail. Additionally attempt the same
eight earlier weekly warm-up origins,−8 through−1. Maximum528 new historical
cases (16×33), before missing warm-up exclusions. Source/recording availability
is the same explicit nominal period-end assumption, not measured publication.

The combined source span is6130 hours. Concatenate the eight-week warm-up offset1344 with the4786-hour
main memory span (730+24×168+24). These share the initial730 observed hours.
Record both lengths and all overlap hashes; the combined length is6130.
Retain missing early positions. A warm-up case with missing history/actuals is
unavailable in its entirety, without imputation. Missing main-span observations,
duplicates or source-identity mismatch stop preparation; no identity replacement.
Use exact archive hashes and initial-prefix hashes from the frozen receipts.
Interpret counts only for the exact selected IDs and bounded ranges.

## Fixed evidence use and common cost

Forecast-generation code must be frozen before computations. Keep six hourly
recipes exactly as038 and context features/retrieval exactly as043/047. No056
error-profile features, learned covariance, new models or parameter search.
Pool original125 warm-up plus original416 development episodes with the new
training episodes. At each scored origin expose only closed/recorded earlier
outcomes from the same domain; retain latest eight distinct origin instants and
sixteen nearest standardized contexts. Same scale floor0.1, twelve features,
minimum16 contexts/three origins, half CV/half history fitting mass. Use050's
original045 anchors, block objective, solver and certificate threshold.

Preserve exact original050 matched-CV/ledger and045 global-CV forecasts as
comparators. Only expanded-memory ledger mixtures are newly fitted. Include all
416 cases and any cold-start fallback. Failures are retained and stop execution.
No validation/final tuning or paid confirmation of a failed development gate.

Count every additional historical forecast computation and estimator fit; this
is not free evidence. Retain source hashes, all generated forecasts and their
backtests, contexts, maturity cutoffs, retrieved records, fits and scores. Any
future matched-agent test must give both arms the same raw new episode records
and the same generation/execution budgets; the index is the intended treatment,
not privileged data. This offline comparison cannot establish that agent effect.

## Reporting and gate

Report overall/domain mean per-case RMSLE versus matched CV, global guard and
old050 ledger, including count/runtime costs. Require>=20% gain versus both
controls, positive gain versus old ledger overall, and positive gain versus all
three comparators in each domain. Original development cases are reused; do not
claim confirmatory confidence from them. Preserve negative results. API calls0,
no main/PyPI changes, no final-reserved access. The20% final-agent goal is unchanged.
