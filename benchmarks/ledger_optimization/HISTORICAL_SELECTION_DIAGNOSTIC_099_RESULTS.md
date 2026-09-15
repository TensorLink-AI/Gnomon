# Fixed historical selection replay — no improvement over actual decisions

Development diagnostic only: 68 already audited sessions, with the same 22 cases
completed in all three arms. The plan fixed CV, last-four, last-twelve and lifetime
rules before their scores were computed. Each historical rule requires at least
two origins shared by every currently eligible configuration; otherwise it uses
CV. All choices were serialized and hashed before the separate scoring pass.

| Within-arm rule | Hermes RMSLE | Hermes + Gnomon RMSLE | Hermes + Gnomon + ledger RMSLE |
|---|---:|---:|---:|
| Actual agent decision | 0.492300 | 0.492123 | 0.493495 |
| Current three-fold CV minimum | 0.505337 | 0.506622 | 0.507994 |
| Last four global past origins, with CV fallback | 0.505337 | 0.500823 | 0.507994 |
| Last twelve global past origins, with CV fallback | 0.505337 | 0.500823 | 0.507994 |
| Lifetime matched origins, with CV fallback | 0.505337 | 0.500823 | 0.507994 |

All three historical windows produce the same selected forecasts in this
limited cohort, although support counts differ. In the ledger arm, history was
admissible on 7/22 cases and fell back on 15/22; none of its supported choices
differed from the CV minimum. Hermes has the same 7/22 coverage and no selection
changes. Gnomon without ledger has 7/22 last-four coverage, 8/22 longer-window
coverage and two changed choices: one improves future error and one worsens it.

The two no-ledger changes illustrate the limitation. On item 1372862/store 12,
round 2, historical means select Random Forest and worsen RMSLE from 0.241676
to 0.433079. On item 1047756/store 23, round 4, they select Ridge and improve
RMSLE from 1.192336 to 0.873364. The actual agent already chose Ridge on the
second case. Replacing the actual agent with these fixed rules does not improve
any arm's average in this snapshot. Preserve both outcomes; do not present only
the favorable example.

This is a restrictive **all-current-candidates common-cohort** diagnostic. A new
configuration without matured outcomes can force a fallback even when some
candidate pairs have useful evidence. Zero changed ledger choices does not mean
the ledger contains no useful pair comparisons, nor does it establish that
context-sensitive retrieval or broader exploration cannot help. It does argue
against deploying these particular historical-average selectors based on the
current evidence.

Five targeted tests passed: matched cohorts and source immutability, missing
candidate fallback, global recent windows, exact ties, and rejection of future,
duplicate/naive timestamps or invalid scores. The real-data selection pass also
checked current three-fold endpoints, production execution identity, versioned
configuration identity, series/unit/horizon, complete historical timestamp ranges,
recording visibility and recomputed historical/CV metrics. Scoring reproduced
every original submitted score within 1e-12.

The selection function has no current-target or forecast-value argument. Its
driver never loads host target files. However, completed development outcomes
had already been observed during monitoring/auditing, and a metadata discovery
read loaded host-job JSON before the selector was run. This is not a blinded or
prospective experiment. Replayed choices hold observed exploration and later
history fixed; actual earlier choices could change later agent behavior.

No new fits or agent/API requests were made. Reuse in a live agent would still
have retrieval and reasoning costs that this diagnostic does not estimate.
No live code was changed and no final target was accessed. The 20% objective
remains unestablished.

Plan: `HISTORICAL_SELECTION_DIAGNOSTIC_099.md`. Receipt:
`evidence/historical-selection-099-offline-001.json`. Selection SHA-256:
`220a212539b3a814d4890e938dc75bf60b31542611277a2b692dce04ce40a1dd`.
The complete per-session choices, predictions, scores, scripts and source hashes
are retained in `results/historical-selection-099-offline-001/`.
