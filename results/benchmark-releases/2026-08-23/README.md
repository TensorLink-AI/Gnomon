# Defensible temporal-reasoning validation — 2026-08-23

This release tests the product claim that Gnomon improves the same LLM's
temporal reasoning without silently changing its governed forecast. It uses
DeepSeek V4 Flash 0731 through Engy for both matched arms.

## Decision

The current Evidence surface passes the reasoning graduation criterion on the
matched 80-case TemporalBench sample:

- field-level temporal-choice accuracy: **35.76% Gnomon vs 28.48% control**;
- paired outcomes: 64 questions fixed, 42 broken, exact McNemar **p=0.0409**;
- 80/80 tasks completed, with one observed tool call at both median and p95;
- the immutable-primary check passed and no supplied context covariate was
  admitted into the primary forecast.

Forecast quality is **not claimed to be superior**. Mean sMAPE was 10.90 for
Gnomon and 12.31 for control (21 paired wins, 19 losses, p=0.875). Across 240
channel forecasts, MASE had 87 wins, 108 losses, and 45 ties (p=0.152). These
results show no statistically demonstrated overall regression or advantage.
SpO2 was nominally weaker but does not survive correction across six channels;
it remains a product-monitoring item.

Independent deterministic validation also shows:

- CompilerBench: 96.25% overall, 100% ambiguity/refusal accuracy, no invented
  targets accepted;
- PropertyBench future-process volatility: 56.81% balanced accuracy across
  2,016 cases and three independent seeds, above the precommitted 55% gate;
- TransitionBench: all five gates passed across 1,296 cases;
- LeakTrap: 0/40 leak flags, 0/40 future transcriptions, 40/40 structural
  access proofs.

Volatility direction remains diagnostic and is explicitly **not graduated**:
its scale estimates are useful, but direction did not beat the majority-class
baseline. The withdrawn answer-bearing ReasoningBench is not used here.

Every JSON file contains its evaluated and harness revisions, dataset identity,
configuration identity, source digest, and explicit scope. The raw task-level
runs remain benchmark artifacts rather than committed product evidence.
