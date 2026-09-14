# Preparation083: bind prior backtest errors to forecast-time risk context

081improves mean development RMSLE but fails the20% and electricity guard;
082shows its estimated benefit magnitudes are poorly calibrated. Do not reverse
the observed correlations or learn a gate on current outcomes. This preparation
tests a distinct input representation using only information already observable
at forecast time. It is synthetic preparation, not another source accuracy claim.

Keep080's32-tree/depth4/leaf24 joint36-output error-Gram learner and per-lead
simplex solver unchanged. Append seven features to its ten forecast-only
features: the six model signed log errors from the immediately preceding
available24-hour backtest, at the same relative lead, and a presence indicator.
For absent predecessor use six zeros and indicator0; otherwise indicator1.
This is observed prior error, not any part of the target being predicted.
Do not standardize, clip or infer missing predecessor outcomes.

For the source preparation expected in084, retain all three current CV pairs.
CV0has missing predecessor; CV1uses CV0; CV2uses CV1. The production query uses
CV2. Historical production training rows use that historical task's own CV2,
never the query's CV or newly recomputed historical predictions. The current CV
origins are current-72h/-48h/-24h, with each24h outcome ending before or at the
next forecast origin. Check target/source/recording visibility, model/config
identity and the relative-lead phase before consuming predecessor values.
The nominal period-end availability assumption remains explicit.

Control has current3CV case masses1/3. Ledger has current3CV total0.5 and all
eligible same-domain historical production pairs total0.5. Each24step case has
equal step mass. Same045anchor, six fixed066models and API/forecast budgets in
both arms. This differs from079's two-variable linear signed-error correction:
it learns joint model error relationships with nonlinear prior-error context,
then retains nonnegative model weights. It differs from081only in input context.
Keep050/061/068 and081as fixed comparisons, with no post-result domain switching.

Synthetic checks: prior errors and missingness, malformed/nonfinite rejection,
no current target in query feature signature, PSD leaf mixtures, temporal
predecessor rejection before feature construction, and simplex certificates.
Retain tree structures, input hashes,17features,36targets and all solver costs.
No source records, protected validation, final outcomes, provider/API calls or
accuracy claim in083. Freeze a separate084source driver before scoring.

The future gate remains20% below both matchedcontrol and strong050; improvements
over061/068/081overall and all comparisons positive in each domain. No paid or
protected confirmation after a failed gate. This common numerical prototype
does not establish a shipped ledger improvement or a matched-agent benefit.
Main/PyPI remain unchanged; the1.2.0/DeepSeek final20%/95% goal is still unmet.
