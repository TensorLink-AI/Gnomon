# Certified development ensemble opportunity bound 046

After 045's 0.96% gain, determine whether the shared ensemble action space can
possibly support 20% on these development tasks. This deliberately uses the
current future actuals: **hindsight diagnostic only**, never a deployable policy.
Use exactly the 416 source forecasts from 038 and the stronger CV-ensemble
baseline from 045. No new data, provider forecasts or reserved target access.

For each task, fit the same six-weight simplex ensemble to that task's actual
24-step outcome using the uniform-initialized, ftol=1e-12 refined solver. Keep
its regularizer 1e-6 * ||w-uniform||² and the 1e-5 convex gap criterion. Preserve
every solver failure; do not select tasks or silently relax tolerances.

Report an interval bounding the true unregularized hindsight minimum RMSLE:

- Upper bound: RMSLE of the feasible fitted weights (without the regularizer).
- Lower bound: max(0, regularized_objective - certified_gap - (5/6)*1e-6).
  The subtracted term is the maximum regularizer on the six-simplex.

The gap certificate is a global convex suboptimality bound from 044/045. These
bounds do not assume the numerical optimizer attained the exact optimum. Average
per-case lower and upper bounds to bound mean-case RMSLE and implied improvement
against the frozen CV ensemble. Report overall and both domains. If the lower
bound exceeds 0.8 times control RMSLE, 20% is impossible **within this action
space on these development cases**, even with future information. If not, do not
claim feasibility for a practical learner merely because hindsight permits it.

No policy variant, model, metric, cutoff, scored case or final-set access is
changed. Count all 416 diagnostic weight fits and iterations separately. These
are additional analysis costs, not agent or provider calls. Main/PyPI unchanged.
