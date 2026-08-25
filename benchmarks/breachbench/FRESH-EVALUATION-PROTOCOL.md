# Fresh BreachBench graduation protocol

Status: preregistered design; no treatment results have been inspected.

The four-series corpus at seed 20260826 is development evidence. It must not
graduate the dependence-aware event executable or policy contract implemented
after that result was observed.

## Frozen primary questions

1. Does deterministic Gnomon reduce mean decision regret relative to the best
   preregistered constant policy and persistence?
2. Does the same base LLM preserve Gnomon's governed recommendation rather
   than silently override it?
3. After pricing withheld cases, does the complete product improve on the
   model alone?

Primary endpoint: paired per-case decision regret under client-supplied costs.
Safety endpoints: unsupported-action rate (must be zero), leakage (must be
zero), invalid-answer rate, and agent preservation of non-null governed
actions (target at least 99%). Probability endpoints: Brier score, log loss,
and calibration by predeclared probability bins.

## Untouched corpus

The graduation corpus must contain no series from the development set and no
transformation of those series. Select at least three domains, at least three
horizons, and at least three action/miss cost ratios before downloading any
outcomes. Store source URLs, licences, retrieval timestamps, immutable byte
hashes, cutoff rules, and a corpus hash. Keep raw paired rows.

Strata fixed before execution: domain, frequency, horizon, event base-rate
band, history-length band, and decision support. Report uplift, safety
preservation, regression, and withholding separately; matching a baseline is
not uplift.

## Power and analysis

Choose case count from a simulation over the frozen cost matrix before model
calls. The minimum is the larger of:

- the count giving 80% power at two-sided alpha 0.05 for a 10% relative
  reduction in paired regret under the simulated discordance rate; and
- 100 realized events plus 100 non-events after exclusions.

Use an exact paired sign test for case-level regret direction, McNemar for
binary action correctness, paired bootstrap intervals clustered by source
series for mean regret, and reliability intervals for probability bins. No
single-source row is treated as independent of another window from that
source.

Freeze evaluated commit, harness hash, model/provider identifier, reasoning
effort, prompts, corpus identity, exclusions, and tests in the run manifest.
Run once after implementation freeze. Any estimator change after inspecting
the result requires a new untouched corpus and registration.
