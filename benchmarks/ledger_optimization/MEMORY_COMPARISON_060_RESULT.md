# Broader historical memory: small gain, development gate unmet

Doubling the available series pool from eight to sixteen per domain improved
overall RMSLE slightly on the original416 development tasks. The expanded
ledger is **2.49% better than matched CV** and **0.46% better than the incumbent
ledger**. Electricity worsens slightly. The20% and positive-per-domain gates
are not met. Preserve this result; no promotion or paid confirmation.

| Method | Overall mean case RMSLE |
|---|---:|
| Global CV guard |0.260631598421|
| Matched intraday CV |0.258711065759|
| Original ledger050 |0.253424725902|
| Expanded-memory ledger060 |0.252258171733|

| Domain | Gain vs matched CV | Gain vs original ledger |
|---|---:|---:|
| Electricity |−0.0636%|−0.1965%|
| Pedestrian counts |+3.1649%|+0.6368%|
| Overall |+2.4942%|+0.4603%|

The new pool contributed at least one neighbor to408 of416 decisions, averaging
5.976 of16 selected neighbors. Thus the index actually used the added episodes;
usage alone did not establish a large or uniform accuracy benefit. Overall gain
against the global CV guard is3.2127%.

## Fixed tasks, rule and additional cost

057 froze sixteen additional training IDs before count reads, excluding original
scored, validation and final-reserved series.059 generated525 extra historical
cases; they remain training-only and never entered the scored denominator.
At each scored origin,047's unchanged retrieval used only earlier closed/recorded
same-domain outcomes. The12 feature definition, latest-eight-origin window,
sixteen-neighbor limit, standardization and half-history mass stayed fixed.
All three comparator forecasts remain exact saved predictions; only the
expanded-memory mixtures were newly fitted.

Added experience required12,600 forecast computations and6,300 estimator fits,
567.97s wall and1081.06 summed worker CPU seconds. No API calls. Those costs
remain visible in [059](MEMORY_FORECASTS_059_RESULT.md). Common evidence cost is
25,584 computations/12,792 estimator fits including the original data. The060
comparison itself added416 weight fits/23,862 optimizer iterations,4.738s, with
zero original-model calls. Additional experience is not free or a private budget
advantage; a later matched-agent design must expose the same raw records/budgets
to both arms.

## Verification and interpretation

Code/protocol frozen at e83e009 before comparative scoring. Two new merge tests
and four inherited retrieval tests passed. The059 source/forecast audit passed
190,074 checks;060's independent retrieval/fit/score audit passed92,935 checks,
with zero failures. These validate arithmetic, identities and artifacts, not
independent statistical samples. No cases failed or were removed.

This uses the original repeatedly examined development panel. No confirmatory
interval is claimed. The separate055 new-series result remains5.17% against its
matched control; it uses different series and does not validate060. No052
validation outcomes or final-reserved counts were used in this experiment.
The actual Hermes result remains the earlier1.14% gain with uncertainty crossing
zero. The20% final matched-agent objective remains unachieved.

Full forecasts, references, costs, hashes and archive:
[evidence/broad-memory-breadth-060.json](evidence/broad-memory-breadth-060.json).
Main/PyPI unchanged. Both numerical generation and comparison have finished;
no paid agent evaluation is running.
