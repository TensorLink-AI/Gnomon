# Locked disjoint-series development validation 052

The strongest mechanism tested so far is050's intraday ledger mixture. Its
matched-control improvement on repeatedly used development cases is2.04%, not
the20% goal. Stop choosing another rule on that same panel for this validation.
Lock050, then measure it once on additional series whose scored outcomes have
not been parsed in these experiments. This remains development validation:
the previously reserved final panel stays untouched, and a numerical result
cannot establish a matched-agent benefit by itself.

## Identity selection before outcomes

Use the frozen eligibility metadata from037. Reconstruct the original hash order
`sha256('20260914:panel035:<domain>:<series_name>')`. Verify that the first eight
IDs equal037 development and the following sixteen equal037 reserved, exactly.
Use positions24 through31 (eight further IDs per domain) for this validation.
Do not change seed, choose a favorable subset, replace failed identities, or
alter any reserved membership. Eligibility was based on timestamp coverage and
the initial730 observed hours; those prefixes were examined in037. Subsequent
scored outcomes for these new IDs were not numerically parsed there. This is
not a claim that their bytes were absent from the downloaded archives.

Produce and commit the full named ID manifest from metadata before loading new
count values. If fewer than32 eligible IDs exist, or the previous partition does
not match, stop with the failure retained. Source archives, eligibility files
and previous selection hashes must match the frozen receipts.

## Fixed tasks and availability

Same26 weekly origins,24-hour horizon,730-hour rolling input, CV ends658/682/706,
and dates as037/038: electricity ends2015-01-01 with the original one-second
phase; pedestrian ends2020-05-01 with exact publisher hourly labels. Same six
recipes and feature definitions. Eight earlier weekly warm-up attempts per
series, as042/043. Missing earlier hours make the entire corresponding warm-up
cohort unavailable; no interpolation, no series replacement, no scored-task
removal. Duplicates or inconsistent source identities reject preparation.
Empty earlier coverage means no available warm-up evidence, not invented data.

All selected scored spans must retain full finite nonnegative observations and
match the already frozen initial-prefix hashes. Preserve missing warm-up positions
and every inclusion/exclusion. Do not numerically parse future observations for
any037 reserved series or any unselected series. Raw CSV/TSF transit is allowed;
interpret count values only after selecting the exact permitted identity/range.
Both arms use the same sources. Nominal hourly time coordinates and assumed
period-end source/recording availability retain the original limitations.

## Locked inference and evidence rule

Use the exact six recipes in hourly_numerical.py as frozen for038, and the exact
045 global CV ensemble implementation (ftol1e-12). Both arms get050's four
six-hour mixtures,0.01 anchor penalty,1e-6 smoothing,ftol1e-14,500 SLSQP iterations,
and at most32 certificate refinements with unchanged1e-5 acceptance bound.
The anchor is each new task's global CV mixture, computed from its own CV folds.

Historical retrieval uses047's twelve predecision features, latest eight visible
origin instants within the same domain, scale floor0.1, sixteen nearest contexts,
and half CV/half history training mass. Use **only this validation panel's own
past/warm-up episodes**, not future or earlier development-panel outcomes.
This matches the earlier eight-series-per-domain pool. Require16 eligible
contexts across at least three distinct origins. If unavailable, the ledger arm
uses its matched intraday CV control, and reports insufficient evidence. This
explicit cold-start fallback is specified before new coverage/outcomes are read;
050's original completed panel never exercised it. Do not drop such cases.

No051 learned matrices,049 corrections,046 hindsight weights, or other policy
variants enter the run. Mechanical adapters may change file locations/cohort
names and add this explicit cold-start handling; they must not change numerical
recipes, parameters, retrieval or scores. Freeze those adapters before computing
the validation forecasts. A numerical failure halts the run with preserved
partial evidence; any correction requires a separately recorded uniform rerun.

## Fixed reporting and uncertainty

All416 cases remain in the denominator. Primary metric: mean per-case RMSLE.
Report ledger versus matched intraday CV and versus global CV guard, overall
and per domain. Freeze10,000 paired bootstrap replicates, RNG seed20260914:
within each domain, sample eight series clusters with replacement; independently
sample seven circular four-origin blocks, truncate to26 origins, and apply that
same origin sequence across all sampled series in that domain. Use identical
indices for every arm. Compute each replicate's relative mean-RMSLE reduction;
report2.5th/97.5th percentiles. This retains within-series blocks and shared
calendar shocks within each domain; sixteen series are still a small panel.
Do not treat416 tasks or repeated seed draws as independent series.

Passing this validation requires >=20% overall reduction against both controls,
95% intervals excluding zero against both, and positive point improvement against
both in each domain. Report failures and all results unchanged. No best-subset,
metric substitution, or final-set access after seeing a near miss. A pass would
justify planning a frozen matched-agent final test, not declare the goal met.

## Retention and boundaries

Keep protocols, source/implementation hashes, IDs, fits, forecasts, scores,
interval resamples, exclusions and all costs on dev/ledger-optimization. API
calls are zero for numerical preparation/validation. Forecast computations and
evidence/weight fits must be counted separately; record skipped warm-up cohorts.
The earlier negative experiments remain retained. Main and PyPI unchanged.
