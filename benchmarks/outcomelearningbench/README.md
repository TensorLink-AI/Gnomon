# OutcomeLearningBench

OutcomeLearningBench tests whether Gnomon's human-facing candidate policy can
learn prospectively from realised outcomes without contaminating the immutable
primary or automation authority. It uses the real publication and tracking
code in chronological order.

The suite covers a stable useful prior, a stable harmful prior, a regime
reversal, outcomes unavailable at the current cutoff, and strong outcomes from
an unrelated series in the same project. It measures policy behavior and
safety, not LLM forecasting skill. No generated outcome is passed into the
forecast that it scores.

```bash
uv run python benchmarks/outcomelearningbench/run_outcomelearningbench.py \
  --output-dir results/outcomelearningbench/latest
```

The benchmark fails unless useful same-series history is eventually used,
harmful or future-known history is not used, cross-series transfer is absent,
an abrupt reversal demotes the prior within two newly resolved losses, and
every primary/automation invariant holds. Multiple same-class candidates from
one forecast origin are conservatively collapsed and cannot inflate the
evidence count. Skill earned by one compiler/model identity cannot transfer to
a replacement proposer without that proposer earning its own outcomes.

A 20-seed moderate-noise sweep checks that useful priors graduate across at
least 80% of independent streams while a matched no-skill placebo falsely
graduates in at most 10%. The raw per-seed streams are retained in the summary;
the aggregate is never substituted for the chronological records.
