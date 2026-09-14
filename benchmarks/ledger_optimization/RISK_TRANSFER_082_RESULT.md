# Diagnostic082: estimated gains are optimistic and poorly calibrated

Frozen16c1457 diagnostic reads only the416 already scored081case artifacts;
no new fitting, source observations, selection rule, final or API access.
All differences below are mean squared-log loss differences, not mean-case
RMSLE percentages. They cannot replace the primary081evaluation metric.

The ledger risk model prefers its saved weights to control weights on all416
cases. Realised ledger improvements occur in272cases, with144regressions.
The current-CV model instead prefers control weights in all416cases. These
opposing signs are largely a consequence of optimizing each action against
its own risk estimate; they are not independent evidence of reliable selection.

| Cohort | Ledger predicted gain over control | Realised gain | Pearson correlation | Realised wins |
|---|---:|---:|---:|---:|
| All416 |0.0126838179|0.0039667890|-0.126728|272/416|
| Electricity208 |0.0041490960|0.0004179087|-0.648541|137/208|
| Pedestrian208 |0.0212185397|0.0075156694|0.007825|135/208|

Thus the size of the ledger-estimated benefit does not reliably predict the
size of the observed benefit here. Electricity is particularly problematic:
the estimated average benefit is almost ten times the observed squared-loss
benefit, and the correlation is negative. Correlation is descriptive, may be
sensitive to large errors, and is not a causal claim or a generalization bound.
Do not turn its sign into an inverted selector after observing these outcomes.

Against unchanged045anchor weights, ledger predicts positive gain in416cases,
but improves260and worsens156. Estimated mean gain0.0129780416 versus realised
0.0069804512. Electricity's predicted gain0.0048028966 versus realised0.0006934023,
correlation-0.401385. Pedestrian's corresponding values are0.0211531866 versus
0.0132675002, correlation0.238833. The correction is useful on average but its
own risk reduction is not a calibrated measure of the benefit in a given task.

This diagnoses a limitation of081's conditional-risk estimator, not a numerical
ledger defect. Neither more trees nor a larger adjustment is justified by these
results. Future work needs a prospective check of evidence transfer and the
mismatch between per-step squared-log training loss and mean-case RMSLE. Earlier
078guard and074strength selection already failed; repeating those policies or
choosing domain winners retrospectively would not establish new progress.

Runtime0.879s; independent8,554checks, zero failures,0.905s. The audit reconstructs
quadratic forms with array algebra and realised losses from forecast/actual
vectors, then verifies all counts, correlations and source hashes. Three unit
tests covered quadratic forms, ties/conditional denominators and correlations.
No new forecasts, weight fits or API calls. Full inherited081costs remain in its
receipt. Source rows, manifest, report, audit and logs are archived with hashes
in evidence/risk-transfer-082.json. No promotion or final-data access.
