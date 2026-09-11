# Contextual ledger development, equal historical information

Registered before running the contextual policy screen or its live agent test.
The eight forecasting recipes remain fixed. No ensembles or bias correction.
The 24-series confirmation partition remains unscored and unavailable here.

## Equal information and tools

Every arm receives the same current request, current CV results, provider list,
forecast-time condition labels, and complete matched historical per-origin MAE
and RMSLE rows available at the present source/recording cutoffs. Thus the raw
history control can see exactly the evidence underlying a ledger summary.
The treatment is organization/retrieval of accumulated evidence, not privileged
access to observations. Equal exposure over time and equal current historical
information are separate questions; this phase explicitly tests the latter.

`no_ledger` receives those raw rows without computed historical summaries.
`ledger_119` receives the same rows plus original MAE summary cards.
`ledger_context` receives the same rows plus calculated RMSLE summaries and
context retrieval. All use the same forecast/selection tools, model, budget,
fallback and typed completion policy. Same maximum output budget; actual input
tokens may differ and must be reported. Existing experiments had a different
control information contract and must not be pooled with this phase.

## Context definition frozen for the first experiment

Conditions come from each original origin-bounded request only:

- Last-28-observation zero fraction: low (<0.1), intermittent (<0.5), high.
- Last-14 versus preceding-14 mean: falling (<0.75 ratio), stable (0.75–4/3),
  rising (>4/3). Both zero means stable; positive after zero means rising.
- Last-28 coefficient of variation: stable (<=1), volatile (>1); all-zero
  history is stable. These are statistical labels, not business explanations.
- Known future promotion covariate: none, some, all; missing covariate unknown.
  Availability of the promotion plan is the same source assumption as the
  original recipes. No future demand values are used.

Record labels in a **copy** of the development ledger, tied to an existing
execution with synthetic replay clock at its forecast origin. This is a
reconstruction of mechanically observable labels from frozen past inputs, not
evidence that a real-time agent recorded them in 2016. Preserve original ledgers.

Retrieval order: all four labels; sparsity/trend/promotion; sparsity/trend;
sparsity; unfiltered history. Select the first cohort with at least four complete
matched origins. This rule is fixed before scores are inspected. Count eligibility
is not confidence or a reason to ignore the current CV. Conflicting labels,
late availability, missing candidates and incomplete actuals retain existing
ledger exclusions. No pooling across series in this first iteration.

## Evaluation

Offline development screening can inspect all eight development series × 26
origins, retaining each outcome and automatic selection rule. It cannot establish
an agent benefit or the final target. The first live context test uses all eight
series × origins 0,8,17,25 × requested seeds 7,19 × the three arms above:
192 decisions, 64 matched case/seed pairs. Preserve all failures and costs.
Do not choose a favorable subset after observing the run. Report cold-start and
mature samples separately; all remain in the primary mean per-case RMSLE.

The 20% final objective remains unchanged. Any confirmation freeze must explicitly
carry this equal-information contract forward; the earlier final-arm descriptions
are insufficient by themselves to specify the control after this clarification.
