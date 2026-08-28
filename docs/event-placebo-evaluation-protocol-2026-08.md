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

Six corpora were generated once with the benchmark's cryptographically fresh
seed mode and frozen before executing the estimator. Do not replace a failed
corpus:

| Corpus | Seed | Cases SHA-256 | Oracle SHA-256 |
| --- | ---: | --- | --- |
| standard-1 | 886503434522399952 | `957eba9b3f9291a6b721460c7198fc6e1d9953a10a61b4ed39f598f04c61287b` | `a64d221064c8ee701d3482a2929f7a230adf04ce26106df4e86b9517ed952f0a` |
| standard-2 | 7097755664058623658 | `dbfbdcab40f7fe7deadb8520d0d3943ad36f9c080304558a8c101df256f6c71a` | `6315a42028b3ef1f870823f46856c0957a46aa51d2883559f412d689821d53de` |
| standard-3 | 8950528693820448630 | `9dbbc6f795d42445dafe22ddaabca0e9fec6e8eaa602d1f0acbb3a04880a227e` | `75f568ac90bce0bb661c9d33c3c131424c8bcf5f66e3b3f4b95ca3945b9eddcf` |
| stress-1 | 1005055899785588237 | `1926293e9cbd9672d34807b6a2c0a70ba9936e3379ef46c299d17e571904131b` | `121a2ad434d23cac6d37b2bdae643486d8be8ba5d011d2c2d8fc227d0f03ca66` |
| stress-2 | 1660064188514182809 | `79dc85df07b917d265cd6036966cd060b2ab3dd1438eaea4e35c9ac0bc46e0a0` | `05614ff3f13164956f6094a07dd3e6ac77471808c73b1aa729033428c8c30ac4` |
| stress-3 | 8049010174415117375 | `c4b00266e1c26489a918fc32e547687201830f8fcb4521882308cc8156ded152` | `d0beb44959656d53c5fde5505834c0bc827837382ae60db607abc45594d09c1d` |

Standard corpora contain 20 cases per family; stress corpora contain eight
cases per stratum.

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
