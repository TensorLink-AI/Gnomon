# Gnomon external benchmark evidence — 2026-08-22

This release contains aggregate results produced by Gnomon commit `1f37307`
and audited on PR #77. The report/release harness was subsequently hardened at
`b45a69c`. Raw prompts, responses, per-case rows, credentials, and caches are
excluded. Every retained JSON names the evaluated code, harness, dataset, and
configuration; `manifest.json` hashes the curated files.

## What the runs show

| Evaluation | Scope | Result | Defensible interpretation |
| --- | ---: | --- | --- |
| CompilerBench | 80/80 cases | 100% exact intent accuracy; 100% refusal and ambiguity accuracy; zero invented targets | The latest compiler completed this generated corpus. The earlier truncation failure was fixed with a general output-completeness check and sufficient compile budget. One seed is not a universal language-understanding claim. |
| AdapterBench | Full conformance set | All seven statistical adapters and installed Toto2-4M adapter conformed; all three adversarial adapters were rejected | The adapter protocol graduates for the implementations actually installed and tested. |
| AdmissionBench | 5,000 cases | Evidence-weighted loss 0.9251 vs 0.9768 always-candidate and 1.0007 always-baseline | The model-admission policy passed its independent synthetic gates. This validates policy behavior, not Toto accuracy on a real domain. |
| Toto2-4M model comparison | 80 held-out synthetic cases | sMAPE 2.8463 classical-only vs 2.3082 with Toto; Toto arm won 61, lost 16, tied 3; paired sign p=2.42e-7 | Toto added value on this pinned synthetic corpus. It is not evidence that Toto wins across real-world domains or against other TSFMs. |
| AnomLLM trend | 400 official synthetic trend series | F1 0.2645 DeepSeek control vs 0.5791 Gnomon; affiliation F1 0.3214 vs 0.5888 | Strong positive anomaly-detection evidence on the **trend family only**. The other official families were absent locally and are not claimed. |
| Context-is-Key sensor | 4 official tasks, one seed | capped/imputed RCRPS 0.1403 direct DeepSeek vs 0.3439 Gnomon | Negative result: the direct control was better on this very small sensor subset. No broad CiK conclusion is justified. |
| TimeSage-MT | 12 cross-tier tasks; 27 mechanically scored turns | raw mechanical pass 55.6% direct vs 37.0% Gnomon; 3 fixed, 8 broken, McNemar p=0.2266. Numerosity-robust accuracy was 47.1% for both | Negative/inconclusive result. Gnomon used 141 requests vs 64 and did not improve robust accuracy. Thirty-seven turns were unscored, six CSVs were truncated, and no judge model was used, so this is not leaderboard-comparable. |
| MTBench matched | 20 finance cases | MSE 25.67 direct vs 22.51 Gnomon; MAPE 4.176% vs 4.077%; Gnomon won 10 and lost 10 for both metrics, sign p=1.0 | Neutral small-sample result. Lower means do not establish superiority; Gnomon used 86 requests vs 20. |
| MTBench Gnomon pure | Full 525-case finance set | 461 passed the official filter, 64 failed it; filtered MSE 11.12, unfiltered MSE 84.33; no abstentions/errors | Engine coverage only, without a matched LLM control. The large filtered/unfiltered gap is disclosed because quoting only the official filtered mean would hide the worst cases. |

## Important audit corrections

The MTBench parquet bridge formerly renamed canonical cases to positional
`sample-N` aliases, which made matched reporting refuse the comparison. The
bridge now preserves `shard#row` identity. Existing control predictions were
not regenerated: each identity was restored only after its stored
ground-truth vector exactly matched the corresponding official parquet row.

Comparison reports now include costs only for the named arms and can recover
official-control usage from the run manifest. Aggregate releases also remove
generic per-case `results` arrays, even when a small run would fit under the
size limit.

## Coverage limits

- AnomLLM covers only `synthetic/trend`.
- CiK covers only four sensor tasks and one seed.
- TimeSage-MT covers 12 of 240 tasks and only mechanically scoreable turns.
- The matched MTBench comparison covers a 20-case prefix of one finance task.
- The Toto comparison uses one held-out synthetic corpus and one installed
  model adapter.
- ReasoningBench remains withdrawn; no answer-bearing historical result was
  republished.

## Reproduce and validate

```bash
python -m benchmarks.release build benchmarks/releases/2026-08-22.json
python -m benchmarks.release validate results/benchmark-releases/2026-08-22
```

Rebuilding requires the ignored raw run directories and external benchmark
checkouts. Validation requires only the tracked aggregate release.
