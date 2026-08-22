# Gnomon benchmark evidence — 2026-08-21

This is the latest curated aggregate release, not a dump of local runs. Each
JSON file is traceable to an ignored raw summary through the source digest in
`release_metadata`; `manifest.json` hashes the curated files. Per-case rows,
prompts, responses, receipts, credentials, and caches are excluded.

## Historical results (not current-code validation)

Every retained run in this release predates the release hardening and lacks a
recorded evaluated/harness commit. They are preserved as historical aggregate
observations, not as evidence that commit `6d7d41f` or later passes a product
gate. A future release may use `complete` or `graduated` only when both commits,
the dataset identity, and the configuration identity are recorded.

| Evaluation | Scope | Result | Interpretation |
| --- | ---: | --- | --- |
| PropertyBench | 520 cases | Historical single-seed pass; **not graduated** | Current-code seeds 777 and 555 fail different gates, so the retained run cannot support a graduation claim. |
| TransitionBench | 1,620 cases | Historical unverified | The aggregate is retained, but the evaluated and harness commits were not recorded. |
| AdapterBench | Full conformance set | Historical unverified | The aggregate is retained, but the evaluated and harness commits were not recorded. |
| VolatilityBench | 180 primary cases plus balanced suite | Direction not graduated | Scale error improves over the constant baseline in the balanced suite, but held-out balanced direction accuracy is 51.4%; direction remains weak/diagnostic. |
| ContextBench | 80 cases per included arm | Evidence: 100% disposition accuracy, 93.8% admission precision, 75% recall, 5% false influence, zero leakage | Context admission is useful but not perfect; false influence and missed influence remain product risks. One Evidence replicate is included here and identified as such. |
| TemporalBench base LLM | 80 T2/T4 rows | T2 choice 34.2%, T4 27.5%, OW-sMAPE 11.478 | Same DeepSeek control used for the matched product comparison. Choice scoring is the repository's disclosed local exact match. |
| TemporalBench Evidence | 80 T2/T4 rows | T2 choice 32.5%, T4 32.5%, OW-sMAPE 10.756; one call median/p95 | Latest complete Evidence run. It predates the newest answer-contract smoke test and therefore does not prove that the newest code improves the full result. |

OW-MASE and OW-RMSSE in these TemporalBench summaries can be dominated by
near-zero scaling denominators. OW-sMAPE and per-channel scaled errors should
be read alongside them. Summary subtraction is not a paired significance test;
use the raw CI artifact with `benchmarks.report` for matched inference.

The newer `temporalbench-product-contract-preflight-20260821` run contained
only eight cases. It is deliberately excluded rather than being presented as a
full TemporalBench result.

The former ReasoningBench headline was withdrawn. Its evidence packet exposed
answer-valued direction, analogue, and action fields, so the paired statistic
primarily measured transcription. The corrected harness rejects any packet
that directly contains scorer-answer values and requires a fresh matched run.

## Coverage gaps

This release does not invent “latest” numbers for adapters without a retained,
complete, release-grade run. LeakTrap evidence is already tracked separately
under `results/leaktrap/`. ContextCacheBench and Workflow Bench continue to run
as CI contract checks. CompilerBench, CiK, AnomLLM, MTBench, and TimeSage-MT
need new complete matched runs before they can enter a dated evidence release.
These omissions and reasons are machine-readable in `manifest.json`.

## Reproduce and validate

```bash
python -m benchmarks.release build benchmarks/releases/2026-08-21.json
python -m benchmarks.release validate results/benchmark-releases/2026-08-21
```

Rebuilding requires the corresponding ignored raw run directories. Validation
requires only this release and runs in CI on every pull request. Scheduled and
manual benchmark workflows upload raw outputs as expiring GitHub artifacts.
