# Override-outcome trust screen 012: do not promote

Requiring evidence that earlier ledger overrides helped did not beat the
existing support rule. This offline development screen made zero API/provider
calls, changed no forecasts and opened no new confirmation data. It does not
establish the 20% forecast-error objective.

Protocol, implementation and tests were committed as `b3bfe38` before scoring.
The [registered protocol](OVERRIDE_TRUST_012.md) fixes four variants and uses
only the same 208 cases from eight development series. Origins 0–17 select the
variant; origins 18–25 are an already-used development slice, not a fresh holdout.

Training selected `override_trust_2_4`: retrieve up to four earlier matured
override decisions, require at least two, and allow a new support override only
when their mean paired loss is lower than their contemporaneous CV selections
and at least half are strict wins. Each historical proposal is reconstructed
using only evidence visible at that historical origin. Unknown support remains
distinct from observed evidence against an override.

| Policy | Training mean RMSLE | Later development mean RMSLE |
| --- | ---: | ---: |
| Current CV leader | 0.5747938171 | 0.5456698185 |
| Existing support rule | 0.5624192321 | 0.5352551237 |
| Training-selected override-trust rule | 0.5708318650 | 0.5426050080 |

The selected rule is 0.56% better than current CV but 1.37% worse than existing
support on the later slice. It rejects 17 incumbent overrides: seven become
lower-error choices, nine become higher-error choices and one ties. In the
remaining cases, 23 overrides pass the gate and 24 cases propose no override.
These are descriptive development counts, not confidence or causal evidence.
No variant is substituted based on its later score. Do not promote this rule
or spend on a live agent trial of it based on this screen.

The five new tests and five shared CV-context tests passed. They cover future
and late-outcome exclusion, past-only reconstruction, insufficient versus
negative evidence, deterministic ordering, immutable forecast inputs and
rejection of duplicate or numerically inconsistent records. Every cached loss
was independently recomputed. All 208 incumbent choices reproduce screen 007.
The input SHA-256 is unchanged, and the retained report is byte-identical to
the original local output. Source availability at valid time and recording at
horizon close are inherited replay assumptions, not observed retail publication
timestamps.

[Complete retained result](evidence/override-trust-012.json) contains every
variant, case choice, retrieved origin, paired loss, input hash and source hash.
No monetary charge was incurred. Main, release tags and PyPI remain unchanged.
The original negative confirmation and its costs remain separately preserved.

This is the second unsuccessful follow-up retrieval screen after confirmation.
It offers no reason to reopen that spent cohort, weaken the control, change the
metric or reduce the target. The 12.49% hindsight ceiling on confirmation 010
still rules out 20% there with its fixed forecasts; it is not a universal bound
on other datasets. A new final evaluation would require a separately frozen,
fair design and a plausible development result before spending its holdout.
