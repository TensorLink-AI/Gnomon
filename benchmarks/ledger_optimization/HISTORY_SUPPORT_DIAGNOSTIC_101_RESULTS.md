# Historical evidence available before selection

The frozen 67-session scan completed with unchanged source hashes and no agent
requests, model fits, additional ledger queries, or task-outcome scoring. It uses
all audited ledger sessions through 097 batch 008, including four cold starts.
Two series have 26 origins each; the other series have nine and six. This uneven
interim coverage is descriptive, not the completed experiment or a final sample.

Every session received a review before its final selection: 71 review calls in
67 sessions. Among 177 pairs of completed current configurations, 150 appeared
in returned history pages. Of the other 27, 26 contained an identity absent from
the latest returned catalog; one pair was not in the returned pages. These are
unknown comparisons, not zero-error records or evidence of a failed ledger.
Ten sessions had at least one such unknown pair.

| Historical window | Known pairs with zero origins | One origin | At least two origins | Sessions with an at-least-two-origin pair | Sessions with a differing CV/history winner on such a pair |
|---|---:|---:|---:|---:|---:|
| Last four origins | 18 | 43 | 89 | 58/67 | 7/67 |
| Last twelve origins | 6 | 27 | 117 | 58/67 | 11/67 |
| Lifetime | 3 | 25 | 122 | 58/67 | 12/67 |

The columns describing pairs partition the same 150 known comparisons. Windows,
pairs and sessions share origins and are not independent replicates. Two origins
is a count threshold, not evidence of statistical reliability. Disagreement means
the CV and historical sets of lowest-error configurations have no member in
common, within that exact pair; it is not a global model ranking.

There were 119 pairs involving the selected configuration, of which 104 had known
history. Considering only these pairs, disagreement with at least two historical
origins occurred in seven recent-window sessions, ten last-twelve sessions and
twelve lifetime sessions. All counts were recomputed from complete saved summary
evidence, authenticated by its hash. This does not imply the agent fetched every
full-evidence file or read every completed CV result.

The median newest matched origin was 14 days old in each window. The oldest
newest-origin age reached 56 days for recent, 154 for last-twelve and 238 for
lifetime comparisons. Those extrema disclose stale comparisons; they are not
thresholds selected for excluding observations or tuning a policy.

## Implication for the next experiment

Unlike the earlier optional-ledger run, this trial is exercising retrieval.
Matching history is often present, so a broad claim that the ledger is simply
empty would be wrong. Current CV and historical rankings usually agree among
the configurations these agents tested. Improving presentation may help in the
disagreement cases, but these counts do not predict an accuracy gain. History
could also affect exploration; this scan does not measure that causal effect.

Keep the already frozen candidate 100 unchanged. Its prospective comparison
tests whether showing current CV beside historical support and recency helps
agents interpret the evidence. This diagnostic neither promotes a candidate nor
opens the final set. Missing identities and unreturned comparisons remain visible
as unknown; they must not be filled with scores from different configurations.

Evidence: `results/history-support-101-offline-001/` contains the frozen plan and
67-session source list, analyzer, summary reconciliation, full per-session/pair
report, source hashes and stdout/stderr. The committed receipt is
`evidence/history-support-101-offline-001.json`. The 20% objective is unestablished.
