# Development089: equal evidence-reading accuracy, half the input tokens

The frozen paired Engy experiment completed all64sessions using
deepseek-v4.1-flash: sixteen existing030review packets, original versus087brief
format, seeds7and19. Both formats gave the same facts and typed submission tool.
No forecasts were run and no original result was rescored.

| Observed endpoint | Original format | Brief format |
|---|---:|---:|
| Sessions with every requested fact correct |32/32|32/32|
| Schema-complete submissions |32/32|32/32|
| Correct field checks |1,496/1,496|1,496/1,496|
| Upstream requests |32|32|
| Syntax corrections |0|0|
| Service errors |0|0|
| Reported input tokens |216,640|108,312|
| Reported output tokens |17,845|17,574|
| Reported total tokens |234,485|125,886|
| Mean request wall seconds |18.645|17.709|

The ratio of paired mean input-token use fell50.0037%; averaging each pair's
fraction separately gives47.7629%. Total reported tokens fell46.3138%. Token
usage was present in every response, including cached-token details in the raw
receipts. The requested model was reported in all64responses. These are reported
tokens, not billed cost; no billing receipt was available. Wall time is descriptive
and does not establish a reliable latency improvement.

The prespecified interface adoption screen passed: complete cohort, no integrity
failures, equal all-facts-correct rate, and at least25%less paired mean prompt
use. The result supports using this smaller format in a future development
workflow. It does not prove better forecasting, better model selection, a20%
RMSLEgain, a95%accuracy interval, or readiness for final confirmation.

## What was and was not tested

Questions covered the first and last returned pair, last4/lifetime windows,
RMSLEscores and exact-tie winners, matched counts, date bounds, recent/lifetime
disagreement, page coverage/next offset and provider-call count. The system
instruction explicitly disallowed global ranking across different pair cohorts;
that answer is instruction-following, not an independent discovery of the rule.
Agents submitted structured facts; there was no requirement for redundant final
chat JSON. No agent fetched additional evidence pages, iterated models or made
a forecast decision. This is direct Engy tool use, not a full Hermes run.

The sixteen reviews were selected by fixed round numbers1,9,17,25 across all
four reused series, not by score. Round0had no saved full reviews and is outside
this interface cohort; no forecast failures or cold-start cases were removed
from the original accuracy results. Both formats already achieved100%on this
bounded test, so the run demonstrates token reduction without an observed
accuracy loss, not improved reasoning or broad robustness. Generalization to
other models, tasks and larger reports remains untested.

## Audit and preservation

Protocol/runner/scorer frozen ate9cab63; prepared-input and schedule receipts
at1e50c55before any request. The runner's frozen code and inputs stayed unchanged
throughout. Independent raw-response audit1d4e9c1passed1,288checks: every sent
packet, question, model/settings/tool, budget, raw submitted object, usage and
field grade. An independent mapping-based verdict agreed with every scored
answer. The original-evidence answer-key audit735b46apassed712checks including
reconstruction of aggregate RMSLEfrom recorded per-origin metrics. It did not
refit original forecasts. Ten runner/grader/auditor tests passed.

All64sessions used one request each. There were no readiness calls, hidden
retries, service errors or unaccounted started requests. The process exited0.
Original source hashes and all prepared/raw-response files remained unchanged
during audit. Full inputs, responses, usage, timings, references, audit results
and frozen source are retained in the089bundle with per-file/archive hashes.
No API key or authorization header is in the saved requests. Original paid030
costs remain separate from this360,371reported-token test.

The real-ledger integration gate is separate:090found and corrected session
envelope incompatibility, then its copied-original-ledger replay exposed a
retrospective-backtest ambiguity problem. That gate has not passed. Do not
treat089's presentation result as approval to deploy the complete review path.
Main/PyPI and protected data unchanged. The latest numerical development gain
is still2.93%, and the actual-agent20%/95%objective remains unmet.
