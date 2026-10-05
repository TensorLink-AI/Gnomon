# Memory release regression

See [results and recommendation](RESULTS.md).

Compare the integrated memory-v2 / FASE options with the pre-integration 1.4.0
candidate (`b741aa68bd86d534753ba3b8e5788d720c8c3543`). This reuses the existing
crypto, alpha and Favorita evaluations. Their dates have already been inspected:
this is regression evidence, not a fresh holdout or an evaluation of full FASE.

The input archive is the existing trusted local `gnomon-fase/experiments` directory,
containing `crypto_broad/out/*.pkl`, `cross_domain/out/alpha.pkl`, its protocol and
`cross_domain/data/favorita`. Raw data is not shipped in the wheel or this repository.
Do not load untrusted pickle files. Each result hashes the source inputs and runtime;
previous result files are never overwritten. Matching completed runs are reused
only after input/source/policy checks; a POSIX lock prevents duplicate workers. Install `.[replay]` from the checkout for NumPy replay acceleration
and bootstrap summaries; it makes no model, LLM or network calls.

```bash
PYTHONPATH=src python benchmarks/memory_release_eval/run.py \
  --inputs /absolute/path/experiments --output /tmp/memory-eval \
  --task rv_1d --arm current
```

Repeat for the tasks and arms in [protocol.json](protocol.json), then run
`PYTHONPATH=src python benchmarks/memory_release_eval/summarize.py /tmp/memory-eval`.
Crypto `v2_profile=levels` is exactly the current feature set, so its redundant run
is omitted. `v2_profile` changes features for alpha and Favorita only. The runner
checks sampled current-arm decisions against the original release implementation.
The summarizer refuses mismatched input cohorts or losses and reports paired time
blocks (all assets together); Favorita additionally has a paired series interval.
It does not choose production settings from observed winners.

The FASE arm changes features, distance and retention together. The bounded-only
arm isolates distance; the existing experiments contain finer ablations. Other
memory-v2 selection options are functionally tested but remain disabled here.

A no-memory pooled selector is also replayed as a stronger comparator; it was
added after the run began, without changing any candidate settings.
