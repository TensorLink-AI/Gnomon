# ReasoningBench — retired historical instrument

> **Withdrawn and off by default.** The original treatment exposed
> answer-bearing fields, so its published uplift measured transcription rather
> than temporal reasoning. It must not support product claims. The code is
> retained to reproduce and audit that negative finding; use DossierBench for
> packet-design uplift and DiscriminationBench for engine-side evidence
> quality. Running this historical instrument requires the explicit
> `--acknowledge-retired` flag.

ReasoningBench is a matched generated evaluation of whether Gnomon's computed
temporal evidence improves an LLM's reasoning, rather than merely its tool use.
It complements TemporalBench, whose canonical choice projection intentionally
prevents an LLM from changing an engine answer.

Each arm receives the same 192-point history, narrative claim, historical
episodes, question, and output vocabulary. The `control` arm gets no computed
answer. The `evidence` arm additionally gets bounded numeric measurements,
their provenance, identifiability, assumptions, Gnomon's calibrated support
status, its production `automation_eligible` flag, and neutral measurement
semantics such as “recent residual scale divided by reference residual scale;
1.0 is neutral.” It does **not** receive a canonical diagnosis, analogue
consensus, or recommended action. Support is reported separately as a grounded
engine field; diagnosis plus useful next action is the primary reasoning
endpoint. Generator truth stays exclusively in the scorer. Cases sweep six
temporal properties and easy, moderate, and marginal
effect sizes using fresh seeded noise.

The benchmark reports diagnosis, confidence, historical-analogue use, useful
follow-up choice, their conjunction, property/difficulty/claim splits, paired
exact McNemar tests, and tokens by arm. Directly grounded fields and synthesized
next actions are reported separately. Analogue accuracy is a shared prompt
comprehension diagnostic, not a Gnomon treatment effect. This benchmark
measures tool-conditioned synthesis, not a general increase in model reasoning.
Full evidence is not repeated in the prompt: production keeps it in the
immutable `temporal_answers.json` receipt.

```bash
uv run python -m benchmarks.reasoningbench.run_reasoningbench \
  --cases 72 --seed 99173 --concurrency 8 \
  --acknowledge-retired \
  --output-dir results/reasoningbench-heldout
```

For Engy, the defaults use `deepseek-v4-flash-0731` at
`https://api.engy.ai/v1` and read `ENGY_API_KEY` from `.env`. Use `--resume` to
score saved rows without repeating successful calls.

Version 0.5 decision runs use the pre-committed, previously unseen generator
seeds `282843`, `316228`, and `331663`; do not select or discard a seed after
seeing its result. Provider
replicates use `--replicate N`. Every summary binds the evaluated/harness commit, exact
harness digest, generator version, seed, case count, model, and temperature.
Never merge rows whose provenance differs except for the declared seed or
replicate dimension.
