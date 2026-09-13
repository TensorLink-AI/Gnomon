# Evidence predictive-skill audit 048

Historical context improved how often the evidence ranked models correctly,
but did not consistently improve its estimates of the size of their differences.
This helps explain why the completed 047 ensemble gain was only **1.90%**.
No forecasts or policies changed in this diagnostic.

| Evidence estimate | Correct pair ordering | Pair-difference MSE | MSE change versus CV |
| --- | ---: | ---: | ---: |
| Current backtests | 60.21% | 0.037828968 | Reference |
| Retrieved history | 61.04% | 0.061691674 | 63.08% worse |
| Equal CV/history blend | 62.29% | 0.039980619 | 5.69% worse |

These are fifteen dependent model contrasts per task, across the same 416
development tasks. They are not 6,240 independent samples. Overall RMSLE risk
MAE improved from 0.12327 for CV to 0.10956 for the blend; predicting absolute risk
and predicting useful model differences are separate questions.

The domains disagree: blended pair-difference MSE was **66.79% worse** than CV
for electricity, but **11.48% better** for pedestrian counts. A common historical
weight can therefore conceal different calibration behavior. This is descriptive
development evidence, not justification for an outcome-selected domain policy.

The actual ensemble changes helped **278 tasks**, hurt **137**, and tied one.
Across the 416 tasks they saved 4.57963 summed case RMSLE and added 2.51854, for
a net mean gain 0.00495454. Historical and blended training evidence predicted
an improvement on every non-tied task. Their correct-sign counts merely equal
the number of helpful changes; these estimates do not identify the harmful ones.
They are losses on records used to fit the weights, not an independent test of
whether a proposed change will work.

The mean blended predicted gain 0.00527711 is close to the mean realized
gain 0.00495454, but this aggregate agreement should not be interpreted as reliable
per-task guidance. Gain-prediction MSE remains 0.00144527, with the 137 wrong-sign
cases retained.

As a post-hoc arithmetic implication of the frozen help/loss counts, a perfect
hindsight filter that removed every harmful 047 change could improve the fixed
CV ensemble by only **4.22%**:
`4.5796296654 / 416 / 0.2606315984`. This uses future outcomes and is not a
deployable policy. Improving the accept/reject gate alone cannot turn these
particular proposals into the 20% target. A subsequent development experiment
needs materially stronger forecast proposals or corrections, with the same
actions available to the no-ledger control, rather than another confidence label
on the current proposals.

Implementation/protocol frozen at `4bcf7eb`. Four synthetic tests passed before
execution; independent audit passed **120,453 checks**. The audit reproduced
all model contrasts, ensemble losses, visibility checks and aggregates using
separate array arithmetic. Diagnostic processing took 0.9683 seconds wall and
0.9414 CPU seconds, excluding load/audit. Zero provider calls, API calls or weight fits.
No future-aware 046 weights or reserved future observations were accessed.

Raw estimates, contrasts, neighbor references, report, manifest, audit and rerun
instructions are retained in `results/broad-evidence-skill-048-001/`; see
[the receipt and archive hashes](evidence/broad-evidence-skill-048.json).
Repeated development data and synthetic period-end availability assumptions
remain limitations. This does not satisfy the held-out matched-agent objective.
Main and PyPI remain unchanged.
