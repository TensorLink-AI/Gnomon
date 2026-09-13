# Evidence predictive-skill audit 048

Freeze implementation before running. This is a diagnostic of the completed
047 development policy, not another accuracy candidate, final test, or policy
search. All prior development results are known. Read only frozen 038/043/045/
047 artifacts; do not access 046 hindsight weights or any reserved observations.

Question: does the retrieved historical evidence predict the current relative
performance of models better than the three current backtests? Good optimizer
certificates do not prove that the fitted objective predicts future loss.

For all 416 existing tasks, retain all six models and the exact sixteen visible
neighbors selected by 047. Define three prospective estimates of each model's
24-step RMSLE: current CV mean, retrieved production mean, and their equal
blend. Only the scoring stage sees the current realized RMSLE. Do not use this
audit to retrospectively change any forecast, weight, neighbor or exclusion.

Primary diagnostic: mean squared error of the predicted model-to-model RMSLE
differences over all fifteen unordered model pairs, first averaged within case
then across cases. This tests relative model evidence, removing shared errors
in forecasting the overall difficulty of a case. Report skill relative to CV as
1 - MSE_estimate/MSE_CV, with an explicit undefined value if CV MSE is zero.
Also report uncentered six-model risk MSE and mean absolute error, and counts
of correct, opposite, predicted-tied and actual-tied pair orderings. Tolerance
1e-12 defines diagnostic ties only; original scores/ties remain unchanged.

Separately assess the two actual ensemble proposals from 047. Apply their saved
weights to each of the three CV folds and sixteen retrieved prior forecast/
actual pairs. Predict the RMSLE gain from control to ledger using CV, historical
and equal-blend evidence. Compare with the actual current gain, reporting gain
prediction MSE, sign counts and mean predicted/realized gain. These are training
losses of the fitted weights, not independently held-out validation scores.
Explicitly disclose that in-sample objective improvement is not evidence of
future benefit. Count helped/hurt/tied tasks and total RMSLE gained/lost.

Report all diagnostics overall and separately for electricity/pedestrian counts.
Fifteen contrasts from a case and multiple origins from a series are dependent;
do not call them independent samples or compute a confidence interval treating
them as independent. There is no accuracy promotion gate in this diagnostic.
The goal's >=20% held-out matched-agent bar remains unchanged. No paid follow-up
or final access is authorized by a positive diagnostic alone.

Preserve per-case estimates, realized risks, contrast errors, ensemble gain
estimates, neighbor references, source hashes, runtime and explicit zero new
provider/API calls. Check temporal visibility against the archived episode
recording fields again. Independently verify formulas and aggregates. Outcome
availability remains the inherited period-end assumption, not real store times.
