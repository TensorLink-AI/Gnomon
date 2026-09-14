# Preparation077: chronological validation of an accumulated error model

076expanded the forecast action but worsened both arms' previous predictions.
Next test a guarded nonlinear correction, with identical fitting/validation
capabilities in both arms. This preparation is synthetic only; freeze the
source runner separately before any source outcomes are evaluated.

Use six fixed038/066configurations, excluding the search-selected seventh slot:
its selection used all three current CV folds, so it cannot be used to claim
an independent third-fold correction-validation check. This also removes066's
search-memory treatment from the new prediction path. Retain all its incurred
costs and the earlier strong050/061/068comparators, unchanged. Original541task
population,416scored,125available warm-up source records, no validation/final access.

At current origin t, the three archived fixed-configuration CV origins are
[t-72h,t-48h,t-24h], each predicting24hours. Use the first two to fit a common
six-model baseline with the045global-CV algorithm (ftol1e-12,maxiter500,
gap<=1e-5). No third-fold outcomes fit that baseline. Correction training at
validation origin t-24h also uses only those first two folds, plus for ledger
all same-domain archived production pairs whose origin is strictly earlier and
whose last target, source availability and local recording are<=t-24h. Control
has no historical production pairs. No context-neighbor or current-outcome
selection of historical records. Both stages preserve source/config identities.

Correction primitive: ExtraTreesRegressor32trees,max_depth4,min_samples_leaf24,
max_features1.0,bootstrapfalse,random_state17,n_jobs1,criterion squared_error;
all remaining defaults are recorded from the installed version. Same learner
in both arms; no parameter search. Residual target=log1p(actual)-baseline_log.
Each24-step training forecast supplies ten features per step: six model log
forecasts centered on that forecast window's baseline-log mean and divided by
max(window population std,.1), baseline log at the step, that scale, sin/cos
of relative lead2pi*h/24. Feature construction uses only model predictions and
current stage's baseline weights, never actuals. Relative lead is not an
inferred wall-clock/calendar label. All labels/units/phases remain original.
Current training cases total mass1when no historical pairs exist; otherwise
current mass.5and historical mass.5, equal within each group and each24leads.

Evaluate corrected versus uncorrected baseline on the third current CV fold.
Enable correction only when its RMSLE is strictly lower by more than1e-12;
otherwise retain baseline, with both validation scores and decision cause saved.
This is an observed backtest choice, not a guarantee about future performance.
At production t, reuse the frozen045full-three-fold baseline weights, fit a
fresh correction on all three current folds plus ledger's all-domain historical
pairs mature by t, only if correction was enabled. This difference in cutoff
is explicit. Saved historical forecasts remain unchanged. Floor corrected
log prediction at zero and disclose clipping. No predicted validation or
production target becomes eligible historical evidence before its closure.

Required source run records416validation baseline solves, up to832validation
forests and832production forests, actual trees/fits/logical attempts, all
training identities and sample weights, tree structures and predictions,
validation decisions, baseline/guard scores, runtime and inherited costs.
Both arms have the same upper budget; do not fit a production corrector that
was rejected by validation solely to equalize cost. No new base-model/API calls.
Horizon stays24 with416scored tasks; do not alter the cohort.

Promotion requires>=20%lower mean per-case RMSLE than matched guarded control
ANDstrong050, positive over061AND068overall, all four positive in both domains.
Retain uncorrected045baseline and no-correction selections. No paid/protected
confirmation after a failed development gate. Actual final objective remains
matched1.2.0/DeepSeekv4.1-flash agents, untouched final cases and95%uncertainty
excluding zero. Main/PyPI unchanged; this is a separate numerical prototype.

Synthetic preparation tests temporal filtering before payload access, complete
case shapes, fixed-feature/target separation, tree recipe, zero-residual behavior,
validation preference/ties and training mass accounting. Preparation results
must not be reported as an accuracy improvement.
