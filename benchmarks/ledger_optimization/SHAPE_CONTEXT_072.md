# Development072: retrieve evidence using observed calendar shape

071's exact-series priority did not improve068. Test an alternative explicit
comparison criterion: hourly and weekday shape of observed demand, alongside
existing coarse context. Freeze before source scoring. Do not use069hindsight
projections, production residuals or062learned labels as query features.

For each current/historical730-hour request, use its last672hours (four weeks),
with nominal aware hourly labels ending one hour before origin. Compute log1p
observations. For each24clock hours and7weekdays compute its mean log level,
subtract the overall672-hour log mean and divide by max(.1, population log std).
This yields31shape components. Each hour group contains28observations; each
weekday96. Preserve source-label phase and disclose nominal timezone assumptions.
Reject malformed/history-size/timezone inputs; no future values or imputation.

Use061coarse filter first: same domain, strictly earlier origin, mature source
and recording boundaries, unique identities. Only then fetch profiles for eligible
episodes. Standardize each shape component across all eligible episodes using
population std floored at.1. New distance = existing coarse squared distance/12
+ shape squared distance/31. Equal per-dimension group normalization is frozen,
not selected by observing outcomes. Tie by distance,origin,series_id. Same quota
16and readiness>=16episodes/>=3origins. No exact-series priority in this test.
Keep profile hashes, per-component scales, both distances and selected identities.

All other068learner behavior unchanged:066search choices, source forecasts,
three current CV folds and16mature production pairs, masses, anchors, four
six-hour simplexes, seven slots, .01/4penalty, SLSQPftol1e-14/max500/refine32,
convex gap<=1e-5,832fits across416tasks,125warm-ups/three unavailable. The control
must reproduce068exactly. Preserve source/config identities and stop on failures.
No new provider/API calls or055validation/final access. Profiles are cached by
exact task ID and their input hashes; all preparation costs/bytes remain visible.

Execute isolated068runner with only retrieval replaced; preserve base report,
072amendment, profiles and source accesses. Retain unchanged strong050,061,
068control and068ledger comparisons. The20%gate remains versus matched control
ANDstrong050, positive versus061AND068overall, and all four positive in both
domains. No candidate promotion or paid/final confirmation after failed gate.

This is reused-development numerical evidence for one retrieval mechanism,
not a new Hermes/Gnomon trial or held-out superiority result. The final goal
still requires1.2.0/Engydeepseek-v4.1-flash matched agents, untouched cases and
paired95%uncertainty excluding zero. Main/PyPI remain unchanged.
