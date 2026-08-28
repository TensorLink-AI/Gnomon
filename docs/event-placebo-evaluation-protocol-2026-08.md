# Recurring-event placebo-separation protocol

Status: preregistered before fresh evaluation on 2026-08-28.

## Change under test

For the recurring-event executable, the episode-level 95% lower bound of the
absolute effect must exceed the strongest of the twelve fixed displaced
schedule placebos. The immutable history-only primary, known-at filtering,
fold construction, minimum-improvement threshold, shrinkage, and publication
policy are unchanged.

The previously inspected standard corpus with seed
`5045431918660058516` is diagnostic only. It cannot graduate the change.

## Frozen validation

Generate three standard corpora with 20 cases per family using seeds
`829001`, `829002`, and `829003`. Generate three production-stress corpora
with eight cases per stratum using seeds `829101`, `829102`, and `829103`.
Do not replace a failed seed.

Run the repository's unchanged ContextBench scorer and retain every case and
oracle file. The change graduates only if:

- all six runs complete with zero temporal leakage and publication parity;
- the pooled standard false-influence rate is below 1%;
- standard admission precision is at least 90% and recall at least 80%;
- both true standard influence families improve mean sMAPE;
- every stress run retains all required strata and default-policy immutability;
- pooled high-SNR stress recall is at least 80%; and
- no false numeric or structural assertion changes the immutable primary
  under the default policy.

Per-run confidence intervals and failures remain visible. No family, seed,
case identifier, oracle field, or benchmark label may enter runtime logic.

