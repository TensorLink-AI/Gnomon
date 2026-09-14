# Development061: retain older comparable episodes

060's broader pool helped overall only modestly. Its index still discards every
record older than eight origin instants, before considering contextual similarity.
Test whether that truncation prevents accumulation of useful long-term experience.
Change only the eligible time window to **all mature visible origins**. This is
one prospective structural retention hypothesis, not a window-length sweep.

Use the same1066 episodes and416 original development tasks as060. No new source
values, historical forecasts, validation observations or final targets. Preserve
all four comparators:045 global CV,050 matched intraday CV,050 original ledger,
and060 expanded-pool recent ledger. Keep their exact saved predictions.

Retain same-domain and source/recording-maturity checks, twelve predecision
features, population standardization with floor0.1, sixteen nearest contexts,
minimum16 candidates/three origins, and tie-breaking. Standardization is now
computed on the entire visible pool; that is an explicit consequence of removing
the age filter. Do not include current production actuals in features or queries.
Record counts of eligible/selected records outside the old eight-origin window.

Fit new lifetime-memory mixtures using the same045 anchors and050 block objective,
half CV/half history mass, solver,1e-5 certificate and fallback rules. No model,
neighbor count, penalty or forecasting-budget changes. Numerical failure stops
and is retained; no task removal or weakened acceptance threshold.

Freeze before fitting. Report overall/domain mean case RMSLE versus all four
comparators, older-neighbor use and costs. Gate:>=20% gain versus both CV controls,
positive gain versus both ledger comparators overall and in each domain, and
positive gain versus both CV controls in each domain. Reused development results
are not confirmatory confidence evidence. No paid confirmation on a failed gate.

Shared inherited forecast cost remains25,584 computations/12,792 estimator fits.
This comparison adds weight fitting/retrieval only. Zero API calls, no validation
or final-reserved access, main merge or PyPI release. The final matched-agent
objective remains unchanged and unachieved.
