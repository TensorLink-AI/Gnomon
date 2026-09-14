# Error-profile retrieval056: no improvement over the incumbent

The frozen development hypothesis added48 observed backtest-error features to
the evidence index. It completed all416 original development tasks but did not
beat050's simpler retrieval rule. Preserve the negative result; do not promote
it or test it repeatedly on the separate validation series.

| Method | Overall mean case RMSLE |
|---|---:|
| Global CV guard |0.260631598421|
| Matched intraday CV |0.258711065759|
| Incumbent ledger050 |0.253424725902|
| Error-profile ledger056 |0.253806508720|

New-rule gain versus matched CV:1.8958%; versus global guard:2.6187%. It is
**0.1506% worse than the incumbent**. Domain gains versus matched CV are0.0390%
electricity and2.3826% pedestrian counts. Both domains are slightly worse than
the incumbent ledger. The20% development gate is not met.

All comparators retain their exact saved predictions. Only the retrieved
historical cohorts change; forecast recipes, anchors, training masses and
mixture objectives stay fixed. Queries use exclusively predecision CV targets;
production outcomes are excluded from query features and cannot affect neighbor
selection until they have matured as historical evidence.

Freeze6d9fa45 preceded scoring. Four new synthetic tests and four inherited
retrieval tests pass. Independent audit:207,907 checks, zero failures, covering
profile arithmetic, scales/component distances, all candidates/neighbors,
training hashes, numerical certificates, every derived forecast/score and
comparator identity. These checks are not independent statistical samples.

Cost:416 additional weight fits,23,908 optimizer iterations,4.224s measured run;
zero new original forecast computations or API calls. Shared inherited costs
remain12,984 forecast computations/6,492 estimator fits. Full context profiles,
weights, forecasts, outcomes, hashes and archive are retained in
[evidence/broad-error-profile-056.json](evidence/broad-error-profile-056.json).

This used only the original development panel. No validation outcomes were read
for this experiment; final reserves remain untouched. The separate055 result
stays5.17% versus its matched control, with95% interval[1.32%,8.80%]; that is a
different panel and cannot be compared as if this rule lost3.27 percentage
points on the same tasks. The actual Hermes result also remains unchanged.
No promotion, paid follow-up, main merge or PyPI release. Goal remains unmet.
