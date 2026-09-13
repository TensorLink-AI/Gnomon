# Panel preparation 035: failed coverage gate

No forecast or agent run was performed. The prospective two-source panel was
not exported, and no reserved later values were parsed or scored.

| Source | Eligible under frozen rules | Rejected | Requirement |
| --- | ---: | ---: | ---: |
| Electricity | 320 | 1: too few nonzero initial observations | 24 |
| Pedestrian counts | 5 | 61: insufficient position coverage | 24 |

Attempt 001 rejected a one-second start-label phase. That format issue was
corrected explicitly at c1d7b8d before attempt 002, preserving all source dates,
sample sizes, eligibility conditions and the hash seed. Attempt 002 then stopped
at the coverage gate. A separate read-only eligibility review retained per-ID
start labels, lengths, nominal endpoints and reasons; it did not inspect later
observation values.

The publisher describes the pedestrian collection as extending to April 2020,
but that does not imply that every sensor covers that endpoint on the TSF
regular-grid representation. These metadata do not establish why a specific
sensor ends earlier: it may reflect actual availability, preprocessing or both.
Do not invent missing rows, select a more favorable time window, reduce the
reserved population or treat these five sensors as the intended 24.

Next preparation work needs a source with adequate individual-series coverage
and explicit timestamp semantics, or a substantively justified, prospectively
recorded source amendment. The electricity source passed the initial coverage
gate but has not been scored. The two-source protocol remains failed rather
than silently becoming an electricity-only experiment.

Exact failures, eligibility counts and metadata hashes are retained in
`evidence/broad-panel-035.json`. Eight tests passed for source metadata handling,
selection invariance under changed later values, identity disjointness, period
slicing and overwrite rejection. These are preparation checks, not evidence
that ledger improves accuracy. The 20% target remains unmet; all previous
results, main, PyPI and existing reserved M5 outcomes remain unchanged.
