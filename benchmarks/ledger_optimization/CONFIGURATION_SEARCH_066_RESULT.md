# Development066: prior-backtest-guided configuration search did not pass

Frozen runner30e4a40 applied the shared065 policy to416 original development
cases plus125 available warm-ups, using a common78-configuration catalogue.
Both arms tested17 configurations per task with58 logical attempts. Ledger
search retrieved only earlier, available same-arm/domain backtests; control
used current backtests alone. No agent or API was run. This tests accumulated
backtest reuse, not production-outcome calibration or general ledger usefulness.

| Approach | Overall416 | Electricity208 | Pedestrian208 |
|---|---:|---:|---:|
| Current-only search |0.27513740193270925|0.11737963050054014|0.4328951733648783|
| Prior-backtest ledger search |0.2740828435805278|0.11642005103522045|0.43174563612583516|
| Existing strong block-CV |0.2587110657586681|0.10748648550251698|0.40993564601481913|
| Previous lifetime ledger061 |0.25198521475530405|0.10734528573584194|0.3966251437747662|

Ledger search improved over its matched control by0.38328%, but was5.94168%
worse than strong block-CV and8.76941% worse than incumbent061. Early128-case
gain1.21120%; later288-case gain0.03846%. No increasing accumulated-benefit
pattern was established. The frozen development gate failed. No paid or final
confirmation of this candidate is warranted. No confirmatory interval is reported
on these repeatedly used development data. Main/PyPI and final reserves unchanged.

All1,082 arm cases completed. Costs including warm-ups:31,378 logical attempts
and5,951 surrogate solves per arm;24,032 combined new physical forecast fits.
Shared-cache ownership is arbitrary and cannot support a per-arm efficiency
claim. Original source preparation12,984 forecasts/6,492 estimator fits and
incumbent extra history12,600 forecasts/6,300 fits remain disclosed inherited
costs. Wall time1,353.04s, about22.6minutes;0 API calls. No new Hermes result.

Independent post-run audit passed2,404,902 assertions: all inputs,541 outcome
pairs,1,082 decisions,62,756 attempts,11,902 acquisition reconstructions,37,016
cached requests and aggregate scores. These counts include repeated structural
checks, not independent statistical samples. All78 used configurations received
one deterministic request replay (78 audit forecasts/75 estimator fits), with
60.83s total audit time. The audit does not independently implement sklearn or
recompute every physical fit. Helper first passed3,250 assertions on22 saved
synthetic065 proposals. No audit result alters the source trace or policy.

Evidence: results/configuration-search-066-001, complete archive and file-hash
inventory linked by evidence/configuration-search-066.json. Source predictions,
current/past training identities, attempts, proposals and outcomes are retained.
Verifier: configuration_search_run_verify.py and search_schur_audit.py.

Next decision: do not promote this search policy or claim a20% benefit. Before
another expensive candidate, diagnose whether selecting one model from three
current backtests is losing the benefit of the stronger existing ensemble, and
whether historical experiments can improve a common ensemble action under an
unchanged matched budget. Any new learner/action must be frozen before its next
run, preserve the current strong guard and keep final evaluation untouched.
