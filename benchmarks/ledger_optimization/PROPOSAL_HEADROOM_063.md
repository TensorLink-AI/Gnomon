# Development063: distinguish evidence selection from proposal headroom

062 completed and was audited/pushed at6b83b3e: progress, with a negative result.
Before another retrieval change, measure whether choosing among the existing
forecast proposals could possibly close the20% gap. This is a hindsight
diagnostic, not a candidate policy, evaluation redesign or weakened objective.
Freeze protocol/code/tests before computation. No numerical model or weight fits.

Use all416 original development cases from038 and the six saved prospective
forecast arms in062: global CV, matched block CV, original050 ledger, recent060
ledger, lifetime061 ledger, learned062 ledger. Recheck source hashes and exact
case/actual identities. None of these forecasts is changed. Do not read any
reserved data or fit a selector using the diagnostic's labels.

Compute per-case RMSLE directly from saved points/actuals. Report:

1. Actual current-CV provider choice from038, and each of the six recent arms.
2. Best single-provider hindsight choice among038's six model predictions.
3. Best saved-proposal hindsight choice among the six062 arms.
4. Best hindsight choice among their union (six providers plus six proposals).
5. An intentionally looser pointwise envelope bound: for each horizon step,
   project log1p(actual) onto the minimum/maximum log1p of the six038 model
   predictions. Its RMSLE is the smallest possible with independent per-step
   convex weights. This is outside the shared four-block action space and cannot
   be treated as an available policy. It only lower-bounds error for any convex
   mixture of those six models. No extrapolation, residual correction or new
   forecasts are included in this bound.

For each hindsight class report mean case RMSLE and maximum possible relative
improvement against both actual current-CV provider selection and matched block
CV. Keep those baselines explicitly named; never replace the stronger comparator
with the weaker one to claim20%. Report overall, domain and the frozen early
rounds0..7 versus later rounds8..25, all retained in the primary denominator.

For each oracle/baseline pair, label20% numerically impossible in that stated
class only when its error exceeds0.8*baseline. Otherwise label it merely not
ruled out. Show what fraction of the oracle's potential improvement20% would
require, or null if it has no positive improvement. These are deterministic
headroom calculations, not confidence intervals or evidence of predictability.

Preserve each case's scores, hindsight selected IDs/ties, envelope residuals,
source hashes, aggregates, runtime and costs. Independently recompute metrics,
minima, envelope projection and aggregates. Add tests with known arithmetic,
ties, invalid inputs and the envelope's lower-bound property. Zero API/provider
calls and no paid confirmation or final access from a favorable bound alone.
Main/PyPI and the final matched-agent20% goal remain unchanged.
