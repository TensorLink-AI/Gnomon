# Conditional error correction076

Freeze before source execution.075showed no selector among its five recorded
strength outputs can reach20%on these development cases. Expand the common
forecast action instead. Both arms use an identical regularized correction
learner, with the frozen068blend as their respective uncorrected prediction.
Control sees three current CV pairs; ledger sees the same current pairs plus
sixteen mature historical pairs at the original068half/half masses. All source
configuration identities, neighbors, temporal bounds and416tasks remain068.
No075hindsight candidate selection, new forecasts or protected-data access.

Apply the current arm's four6-hour/seven-slot weights to every training forecast.
For each lead, let b be that weighted log1p prediction, and d_m=log1p(p_m)-b.
The eight raw predictors are [b,d_1,...,d_7]. Compute their means and population
standard deviations from training predictions only, weighted by case mass/24;
scale each by max(std,.1). Current production predictions do not fit scaling.
The11design columns are intercept, eight standardized predictors, sin(2pi*h/24)
and cos(2pi*h/24), h=0..23. These are relative lead positions, not an inferred
source calendar or DST claim. Historical/current CV phase alignment is inherited
and rechecked. Target residual is log1p(actual)-b.

Fit coefficient theta by minimizing sum_i q_i*sqrt(mean_h((X_ih theta-r_ih)^2)
+1e-12)+.1*||theta||^2. Equal fixed penalty on all11coefficients, no sweep or
selection. This smoothed mean-case RMSLE objective differs from049's squared
surrogate and uses forecast-level/disagreement information rather than a fixed
24-lead residual mean. Solve from zero with analytic gradient, L-BFGS-B,
maxiter500,ftol1e-14,gtol1e-9. The objective is at least.2-strongly convex;
||gradient||^2/.4 bounds suboptimality. Require solver success, finite values,
bound<=1e-8and objective<=zero-correction objective+1e-10; preserve failures and
stop if these checks fail. Do not alter tolerances after seeing source outcomes.

Apply correction using frozen training scaling: point=expm1(max(0,b+Xtheta)).
Retain raw log prediction, correction, clipped leads and any excursions beyond
the range of the seven raw model forecasts. Output is a derived forecast with
source executions/task identity retained, not a claim a provider executed it.
Both arms complete fit/application before current actuals enter scoring.

Report416cases,832new correction fits, solver iterations, wall/CPU, source hashes,
all coefficient/scaler/certificate records, scores, clipping/extrapolation counts.
Preserve inherited068and066costs (49,616raw computations,416045anchors,
832068blend fits,11,902search surrogate solves,31,378logical search attempts/arm).
No new API/provider calls. Gate:>=20%over matched corrected control ANDstrong050;
positive over uncorrected068ledger AND061overall, and all four gains positive
within both domains. Also disclose uncorrected068control to catch degradation.
No paid/final confirmation after a failed development gate. This is a numerical
prototype, not evidence of actual1.2.0/DeepSeekv4.1-flash agent performance. Final
objective still requires untouched matched-agent evaluation and95%uncertainty
excluding zero. Main/PyPI unchanged.
