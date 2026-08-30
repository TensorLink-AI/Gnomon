# v0.7 loop I013: exact-candidate release gate

Decision: **release recommended.** The exact product and release-boundary head
`fa59f2b` passed CI, benchmark validation, package smoke, and the container
build. This metadata-only completion record must pass the same workflows on
its exact head before the annotated `v0.7.0` tag is created.

The forecast product candidate was `77ffb5a`; `7352f91` refreshed five
byte-exact goldens whose only changes were the 0.7.0 runtime stamp and
version-derived forecast IDs. The first exact-head CI run then exposed a
general checkout-path-dependent overflow in the compact capabilities response
and a release validator that understood curated manifests but not loop
checkpoint manifests. Commit `58a395f` made the workspace brief relative and
bounded, added strict checkpoint path/byte/digest validation, and retained all
checkpoint metadata. These changes do not touch forecasts, context admission,
selection, or the Evidence profile used by the small shards.

The untouched four-case ContextBench confirmation completed 4/4 with zero
errors, retries, or leakage; exact publication parity; and one Gnomon call per
case. The engine exposed two `no_distinct_numeric_path` relationships and the
final agent preserved both. In each, the agent explained that the immutable
primary already lacked the claimed continuing trend, so applying
`trend_ceases` could not produce a defensibly distinct path. Context
automation remained false.

The matched TemporalBench shard accounted for two T2 and two T4 rows in both
arms. Evidence preserved 12/12 canonical typed answers, kept all four primary
forecasts unchanged, and used one Gnomon call per row. It answered 6/12 local
choices versus the direct control's 2/12; that tiny result is descriptive, not
a performance claim. One T4 provider 502 required a visible row-level retry.
Support remained disclosed as 20 degraded and 4 best-effort channel results.

The untouched seed-7044 CiK shard scored 3/3 with zero errors, abstentions,
primary-immutability failures, or automation-eligible recommendations. Mean
selected RCRPS was 0.02144 versus 0.05205 for each run's own immutable primary.
The spike task correctly retained its supported primary while preserving two
better-in-hindsight priors as ineligible. The sampled prior retained typed
reason `sampled_prior_has_no_historical_skill` and
`candidate_preserved: true`. These three cases do not support a universal CiK
claim.

An isolated, non-editable Python 3.12 wheel install reported 0.7.0 consistently
from distribution metadata, CLI, runtime artifacts, content IDs, and MCP
server info. Twelve installed-wheel behavioral smokes passed. A real 69-step
threshold artifact and its human brief were inspected: the first 33 evaluated
steps were separated from the extrapolated remainder, breach probability was
withheld as uncalibrated, automation stayed false, and the immutable primary
was preserved beside a labelled prior-assisted path. The full TSFM-isolated
repository suite passed **2,653 with 11 skipped** after the CI revision.

Raw responses, attempts, receipts, traces, and artifacts remain in the local
paths in `validation.json`. The external-evaluation intake remains unchanged.
`docs/astrid-btc-agent-plan.md` and every other pre-existing untracked file
remain outside this release.
