# Gnomon benchmark evidence — 2026-08-21

This is the latest curated aggregate release, not a dump of local runs. Each
JSON file is traceable to an ignored raw summary through the source digest in
`release_metadata`; `manifest.json` hashes the curated files. Per-case rows,
prompts, responses, receipts, credentials, and caches are excluded.

> **Evidence provenance (revised 2026-08-21).** Every run in this release was
> evaluated at a commit **at or before `038291e`**, whose exact revision was
> not recorded. These results are historical evidence for that code; they do
> **not** validate the temporal-reasoning and admission hardening merged in
> PR #74 or anything after it. Each result row now carries a `provenance`
> block — evaluated commit, harness commit, dataset identity, provider/model,
> configuration, and explicit validity limitations — and `manifest.json`
> carries the same caveat machine-readably. The ReasoningBench result
> originally published here has been **withdrawn** (see below).

## Headline results

| Evaluation | Scope | Result | Interpretation |
| --- | ---: | --- | --- |
| PropertyBench | 520 cases | Every declared gate passed at the pinned release seed (9100) | Not seed-robust: alternate seeds 777 and 555 fail declared gates at the same revision. Cross-seed robustness is tracked by the scheduled multi-seed CI job; the core ground truth is the engine's own property functional (self-consistency, not external validity). |
| TransitionBench | 1,620 cases | Graduated | Easy accuracy, moderate accuracy, and supported-claim precision gates passed. |
| AdapterBench | Full conformance set | Graduated | Statistical adapters conformed and adversarial adapters were rejected. |
| VolatilityBench | 180 primary cases plus balanced suite | Direction not graduated | Scale error improves over the constant baseline in the balanced suite, but held-out balanced direction accuracy is 51.4%; direction remains weak/diagnostic. |
| ReasoningBench | — | **Withdrawn 2026-08-21** | The previously published 18.1% → 65.3% uplift is retracted: the evidence packet handed to the treatment arm directly encoded three of the four scored expected answers, so the result primarily measured transcription of the packet. The harness has been redesigned; no ReasoningBench number is citable until a fresh matched run is committed. |
| ContextBench | 80 cases per included arm | Evidence: 100% disposition accuracy, 93.8% admission precision, 75% recall, 5% false influence, zero leakage | Context admission is useful but not perfect; false influence and missed influence remain product risks. One Evidence replicate is included here and identified as such. |
| TemporalBench base LLM | 80 T2/T4 rows | T2 choice 34.2%, T4 27.5%, OW-sMAPE 11.478 | Same DeepSeek control used for the matched product comparison. Choice scoring is the repository's disclosed local exact match. |
| TemporalBench Evidence | 80 T2/T4 rows | T2 choice 32.5%, T4 32.5%, OW-sMAPE 10.756; one call median/p95 | Latest complete Evidence run. All 240 typed choice answers were engine-authored (preservation rate 1.0), so choice accuracy measures the engine's projection, not model reasoning — and T2 is below the base arm. T1/T3 are absent because that condition crashed on those tiers at run time (since fixed). Predates the newest answer-contract changes. |

OW-MASE and OW-RMSSE in these TemporalBench summaries can be dominated by
near-zero scaling denominators. OW-sMAPE and per-channel scaled errors should
be read alongside them. Summary subtraction is not a paired significance test;
use the raw CI artifact with `benchmarks.report` for matched inference.

The newer `temporalbench-product-contract-preflight-20260821` run contained
only eight cases. It is deliberately excluded rather than being presented as a
full TemporalBench result.

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
