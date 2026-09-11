# M5 development candidate preparation and first screen 015

Freeze before fitting on M5. Use only the development CSV and manifest from 014,
whose SHA-256 values are checked by the runner. The reserved source data is not
an input to this command. No agent/API calls or confirmation evaluation.

Use the unmodified `arena/retail_providers.py` from Arena commit
`b600eaa2c2691bebed926dac996d4b03e0c216e9`, SHA-256
`74f987cd9b260900795c3c8e1f6697a76ea87415f597a0d825b7c8bbc20d49eb`.
Retain the source in ignored experiment storage, with its origin and hash in the
manifest; do not substitute a different installed module. Numerical package
pins are in m5-numerical-requirements.txt and match the original preparation's
numerical stack. Capture the complete runtime package inventory before dispatch.

Eight recipes unchanged: seasonal naive 7, window average 28, AutoETS 7,
AutoARIMA 7, AutoARIMA with covariates 7 (legacy identifier contains `promo`),
additive AutoTheta 7, CrostonOptimized, and MSTL 7 with AutoTheta trend. Preserve
the original clipping and seasonal fallback, and report every fallback. Neither
invalid forecasts nor duplicate predictions may be silently deleted.

Eight development series, all 26 registered origins, 14-day horizon. History
expands from the registered initial cutoff, capped at the original 730 days.
Two CV origins, 28 and 14 days before each production origin. Report mean of
their per-fold RMSLE (not pooled RMSLE). All CV actuals must be available by the
current origin; no production future targets enter a CV card or request.

Requests use Gnomon's typed ForecastRequest and preserve history, dates, units,
series, season, frequency and all three cutoff meanings. Period-end UTC labels
are those frozen in 014. Day-of-week covariates use the original reporting date,
not the following midnight that labels its end. Promotion feature is the
registered constant-zero unavailable-data convention. Future dates and calendar
covariates are generated from the origin; the provider never receives future
sales. Source and recording availability are the already-disclosed replay
assumptions. Do not claim observed publication vintages.

Freeze every original request/point/metadata and both CV folds. Reuse the pinned
provider's exact-request cache; distinguish forecast requests, cache hits and
actual recipe cache misses. Use two CPU workers and single-thread numerical
libraries. No alternate fits after inspecting a candidate's error. Retain failed
preparation artifacts and do not report partial case grids as complete.

First offline screen is fixed, without tuning: current-CV leader; original
support rule (minimum four matured same-series origins, lower mean RMSLE and
at least half paired wins); and lifetime-MAE selection after four origins,
otherwise CV. Lifetime-MAE is a deterministic diagnostic, not the 1.1.9 agent.
Include all cold starts. Report all, cold 0–3, mature 4–25, training 0–17 and
already-designated later development 18–25. No variant is chosen after scoring.

Recompute candidate scores from saved points and actuals independently. Compute
the future-aware per-case minimum as a non-deployable opportunity diagnostic.
If it cannot beat current CV by 20%, do not run paid selection trials merely to
hope for a weaker agent control. Passing that diagnostic is not sufficient:
the fair agent comparison, 1.1.9 comparison and untouched final gates remain
required. Do not treat development or deterministic policy comparisons as proof.

Before dispatch require tests for cutoff-bounded request construction, unchanged
requests after perturbing future targets, CV maturity, identity preservation,
strict invalid-result rejection and rejection of an unregistered input manifest.
No main, tag or PyPI changes.
