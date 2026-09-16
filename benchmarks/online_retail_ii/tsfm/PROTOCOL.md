# Additional Gnomon + ledger + TSFM arm

User-requested augmentation of the running 14-day-start Online Retail II replay.
The existing three arms and their immutable bundle remain unchanged. This new
arm starts with fresh native memory at the same first origin, on the same 48
products, 51 horizons, two seeds and 2,448 tasks: 4,896 agent sessions.

## Choice and evidence

Keep the ten original numerical providers and all existing history, CV, matured
outcome, ledger, typed execution/selection and native memory tools. Add two
selectable registered providers: `paracast_route` and `paracast_ensemble2`.
The latter explicitly sends `mode=ensemble, top_k=2`; route delegates underlying
foundation-model selection to the service. Neither policy is forced on the agent.
The agent remains DeepSeek v4.1 Flash through Engy, with Gnomon 1.2.0, four
numerical selection attempts, 12 model requests, 3,072 output tokens/request,
480 seconds and at most two correction turns within those bounds.

API forecasts use one univariate series per request, D frequency, 14-day horizon,
and quantiles 0.1/0.5/0.9. The point is the 0.5 quantile, with negative values
clipped to zero and clipping disclosed. No target observations, customer IDs,
future covariates or feedback are sent. The source data are the already approved
public UK product-sales panel, restricted to history at the task origin.

Before each chronological stage, retain each policy's live response once per
task. Both seeds and later CV folds reuse that exact response. Forecast/selection
still goes through a registered, typed Gnomon execution and consumes the same
four-call budget. Remote preprocessing is separately metered: planned 4,896
forecast requests total, plus bounded metadata reads. This is not a claim that
the HTTP request occurs inside each agent tool invocation or is free.

Use only completed prior 14-day horizons for CV; zero CV folds initially. Append
past API predictions to the same matched numerical ledger, only once their
outcomes mature. Failed TSFM calls are retained as disclosed weekly-naive control
fallbacks, excluded from TSFM evidence, and unavailable for agent selection.
Stop further paid dispatch if over 10% of a preparation batch fails. No automatic
request retries or accuracy-based case selection.

## Service and credential handling

Use the user-designated Chutes endpoint directly, billed to their account.
Credentials are read only by the host controller, never placed in prompts,
worker environments, bundles, argv, request logs or public artifacts. Retain
request/response hashes, model identities, router confidence, credits and latency.
Capture health/catalog snapshots; reject an image change. Do not call feedback.
Catalog reads are advisory: retain their failure receipts and continue when
health and forecast endpoints work. Never repeat a forecast because a catalog
read failed. A preparation-only startup may be reused after verifying its task,
policy, revision and point hashes, provided no agent session was attempted.
No external weight immutability or training cutoff is attested by this interface;
model parameters or shared router state may change independently. Record that
limitation rather than fabricate a historical model revision or training cutoff.

This is a retrospective capability comparison. Foundation-model training may
overlap this old retail dataset. Recorded-time replay is simulated, as in the
original experiment. Results cannot establish that these models were deployable
in 2009, or isolate ledger benefit: the new arm has two additional predictors
and their CV/evidence. Standalone route/ensemble controls are scored too.

## Admission and reporting

Four real-Hermes synthetic-Engy preflight sessions exercise both TSFM policies,
native memory, Gnomon task binding and a matured ledger. Live API predictions
used by that check are retained for the real arm. Then run the accuracy-blind
12-session pilot: three fixed products, first two origins and both seeds.
Require zero worker failures and at least 90% resolved selections. Include the
pilot and all later failures in the final denominator. Run two workers so the
existing six-worker experiment can continue.

Report only matched case/seed subsets when comparing arms. Include TSFM-alone
baselines, chosen-policy frequency, actual API calls/credits/latency and cached
reuse separately from LLM requests/tokens. Do not compare all-period model
scores with partial agent results. No retrospective protocol tuning from scores.

## Direct Chronos-2 control

The additional user-requested control sends `mode=explicit, model=chronos2`
directly to the same HTTP endpoint. It makes one forecast for each of the same
2,448 task histories and 14-day horizons, beginning with only 14 observed days.
There is no agent, Gnomon engine, ledger, native memory or model selection.
Seeds do not apply to this fixed control. Compare its frozen point forecast
against each matched agent seed without charging for duplicate API calls.

Use identical median/clipping, metrics and disclosed weekly-naive failure policy.
Retain all request/response receipts and the actual returned model identity,
reject other model identities, and stop if more than 10% of a period fails.
Track 2,448 planned direct forecast requests separately from the 4,896 router/
ensemble preparation calls. Check service health/image each period. Shared
service-state changes and unknown pretraining overlap remain limitations.
