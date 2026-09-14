# Error-transition079: does recent signed error transfer to the next day?

Freeze this rule before new source fits. Earlier039/043estimated aggregate
production loss from CV loss;049used a fixed mean signed error;076/078conditioned
corrections on raw forecast levels/disagreement. This tests a distinct predictor:
each fixed model's signed24-step error in the immediately preceding daily CV
forecast. That error is known at current origin and may identify persistent bias.
No claim of predictability follows merely from writing this rule.

Use fixed six038/066models only, excluding search-selected configurations, all
416scored tasks and125available warm-up histories. Same forecast data and
corrector primitive for both arms. Baseline weights are the unchanged045full-CV
six-model weights. Strong050, lifetime061and068remain comparison guards.

For each model, define signed log error e=log1p(actual)-log1p(prediction), aligned
by relative lead h=0..23. Current training consists of two daily transitions:
CV0error->CV1error and CV1error->CV2error. For ledger, add every same-domain
historical task mature by current origin; each supplies last-CV error->production
error. Forecast origin gap must be24hours for every transition, with full fixed
config identity and lead phase. Historical task origin strictly precedes current;
last target/source/local recording must all be<=current. Metadata is filtered
before reading historical outcome payloads. The current production error is
never a training target. Simulated period-end recording remains disclosed.

Fit separate intercept a and slope b for each of six models by minimizing
sum_i q_i mean_h((e_next_ih-a-b*e_prev_ih)^2)+.1*(a^2+b^2).
Control: current transition masses.5/.5. Ledger: current total mass.5, historical
total mass.5, equal within groups; without past evidence use control masses.
The regularization is fixed in advance, not chosen using development outcomes.
Solve each2x2normal equation exactly; save coefficients, normalized sufficient
statistics, objective/gradient, positive-definiteness and training-input hash.
This squared-log surrogate is not the primary mean-case RMSLE metric. Failure
or nonfinite inputs stop the experiment and preserve all preceding costs/results.

Predict current model error a+b*last_CV_error[h]. Correct each raw model log
forecast with that amount and floor it at zero; retain all corrected model
points/log corrections/clipped lead indices. Combine corrected models using
unchanged045weights in log space. Both arms finish fitting and application
before current production actuals enter scoring. This is a derived forecast,
not a new source-provider execution. No training on an inferred business cause.

Report416paired cases,832transition regressions (six2x2systems each), all source
config/reference identities, masses, coefficients/statistics, predicted errors,
clipping and output scores, domains/early-late phases and costs. No new base
forecasts/API. Inherited raw forecasts49,616,045anchors416,068blend fits832,
search surrogate solves11,902and31,378logical search attempts/arm remain logged.
No tree fits, new guard or data selection from079results.

Gate:>=20%gain over matched corrected control ANDstrong050, positive over061
AND068overall, all four positive in both domains. Also report uncorrected045
so weak-control degradation is visible. No paid/protected confirmation after a
failed gate. No validation/final data access or main/PyPI changes. This numerical
development test cannot establish the actual1.2.0/DeepSeekv4.1-flash final-agent
objective or95%uncertainty requirement.
