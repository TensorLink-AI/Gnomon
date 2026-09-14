# Preparation085: case-norm-scaled conditional risk targets

Previous turn preserved negative084 atfbfc6ee.082showed optimistic conditional
risk estimates;084's extra prior-error context did not improve081. This change
addresses a different measured mismatch:081 averages squared-log errors whereas
evaluation averages each case's RMSLE. Do not include084's extra features or
retrospectively gate predictions using the observed082correlations.

Keep081's six forecasts, ten forecast-only features,32-tree/depth4/leaf24 joint
36-output learner, sample masses,045anchor and per-lead simplex solver. For each
visible training pair let E[h,m]=log1p(point[m,h])-log1p(actual[h]). With query
anchor a, compute n=sqrt(mean_h((E[h]@a)^2)+epsilon^2), epsilon=1e-6 (the same
smoothing scale used by050). Replace the raw target E[h]E[h]^T with
E[h]E[h]^T/(2*n). Store every training case norm and scale. No clipping, norm
quantile choice, target winsorization, hyperparameter search or future label.

For any24simplex weights W on this PARTICULAR complete training case, define
Q(W)=mean_h(W[h]^T G[h] W[h])+n/2+epsilon^2/(2*n).
The concavity of sqrt gives smoothed_RMSLE(W)<=Q(W), with equality and matching
gradient at W[h]=a. This is a case-level tangent quadratic upper bound; its
maximum smoothing difference from exact RMSLE is1e-6. PSD is preserved by the
positive scale. Cases with near-zero anchor risk get large scale, so retain
those values/costs/failures explicitly rather than secretly capping them.

This identity does NOT prove that a forest prediction is an upper bound on an
unseen query loss. Conditional averaging and query distribution shift remain
estimation assumptions. The fixed learner may still be poorly calibrated.
The experiment trains an RMSLE-related local surrogate, not an exact solver
for expected future case RMSLE. Final scoring remains unchanged mean-case RMSLE.

Preparation uses synthetic pairs only. Test the bound, tangency/gradient,
near-zero norms, finite PSD scaled targets, input rejection, complete model
support and simplex certificates. Retain scale values, trees, hashes and costs.
Freeze a separate086source driver before any source-data accuracy calculation.

Future source scope remains416scored/125warm-up tasks; control3currentCVcases at
mass1/3; ledger currentCV total0.5 plus all eligible same-domain historical
production cases total0.5. Both arms use the same scaled-target primitive.
Metadata origin/target/source/recording gates precede historical outcome reads.
Keep050/061/068/081comparisons and the failed084 result unchanged.
Gate:20% below matchedcontrol and050, positive versus061/068/081overall and all
five comparisons positive in both domains. No final/API confirmation after a
failed gate; no protected data, new base models or main/PyPI changes in085.
The final1.2.0/DeepSeek matched-agent20%/95% requirement remains unmet.
