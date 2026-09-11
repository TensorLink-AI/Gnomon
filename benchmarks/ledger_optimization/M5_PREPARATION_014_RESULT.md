# M5 source preparation 014: development panel ready, no performance result

Prepared a new retail development panel under source amendment 014. No models
were fitted, no scores computed and no Engy calls made. The 20% target remains
unachieved. This is preparation for a possible new evaluation, not confirmation.

The [protocol](M5_PROTOCOL_014.md), source pin, selection code and six initial
tests were committed as `ce5c03c` before downloading M5. Source bytes match the
pinned Nixtla Git blob and SHA-256
`cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774`.
Original data attribution is the M5 forecasting competition / Walmart; access
was through the [Nixtla mirror](https://nixtlaverse.nixtla.io/datasetsforecast/m5.html).
Raw data stays in ignored local storage, not in this commit.

| Preparation item | Result |
| --- | --- |
| Source store-item series inspected through initial cutoff only | 30,490 |
| Eligible under fixed initial-history rule | 27,380 |
| Excluded: fewer than 28 nonzero initial observations | 3,110 |
| Development | 8 series, 4 each in CA_3 and WI_2 |
| Reserved | 24 series, 3 each in the remaining 8 stores |
| Store or item overlap between splits | None |
| Development rows | 5,840: 730 daily rows per series |
| Reporting dates in the window | 2014-05-24 through 2016-05-22 |
| First / last forecast origins under period-end convention | 2015-05-25 / 2016-05-09 UTC |
| Planned origin/horizon pairs per split series | 26 origins, 14 days each |

Selection used identities and the initial 366-day history only. All splits use
the preregistered SHA-256 ordering; no rerolled seed or score-based eligibility.
Only development targets were exported as numbers. Reserved output contains
identities and initial-history metadata; later reserved targets remain in the
source archive and were not scored, summarized or used to select the panel.
This is a statement about this preparation, not proof that a public benchmark
never appeared in any model's training data.

The initial attempt stopped before target export because the mirror calendar
has no `d` column. Its published loader derives d_1, d_2, ... from row order.
Correction `0cce4b4` adds that mapping with daily-order validation, a seventh
test and a documented amendment before the successful retry. The failed attempt
is retained. It did not change sampling, cutoffs, data bytes or model recipes.

Two clean preparations produced identical development bytes, selected identities
and prefix eligibility counts. Independent checks cover all 5,840 reporting-date
to period-end mappings, 208 complete development horizons, hashes and disjoint
stores/items. Development CSV SHA-256:
`db247708b5511ebad809739fcddb98074f44fca33e72f53e6ce8628b3697ff5a`.

Two limitations are fixed before scoring. Availability is assumed at each daily
period-end under a uniform UTC replay convention; these are not measured store
publication timestamps. Promotion data is unavailable, so the unchanged recipes
receive an explicitly labeled zero placeholder and day-of-week features. This
does not establish that promotions were absent. Future prices/weather are not
introduced as known information. Every arm must receive the same convention,
inputs and numerical fallback disclosure.

Next gate: freeze the candidate-preparation adapter and verify the eight pinned
StatsForecast recipes on development data, then measure headroom and past-only
ledger policies before any paid comparison. A negative result must remain
negative. Do not reopen or resample the reserved stores to manufacture 20%.
Final inference, if eventually justified, must cluster the three items within
each of eight reserved stores and preserve shared temporal dependence.

Evidence and reproduction:

- [Source and split manifest](evidence/m5-panel-014.json).
- [Download command, timing and complete output](evidence/m5-fetch-014.json).
- [Failed attempt](evidence/m5-prepare-014-failed.json), [successful retry](evidence/m5-prepare-014-recovery.json),
  and [second clean preparation](evidence/m5-prepare-014-reproduction.json).
- [Validation record](evidence/m5-preparation-014-audit.json).

```sh
.venv/bin/python -m benchmarks.ledger_optimization.m5_prepare prepare \
  --archive results/ledger-optimization/m5-source-014/m5.zip \
  --output results/ledger-optimization/m5-panel-NEW
```

The output directory must not exist. Use the logged fetch command with a new
archive path if the source is absent; downloads are pinned and verified. Main,
tags, PyPI, prior Favorita evidence and the separate FreshRetailNet ROI work
remain unchanged.
