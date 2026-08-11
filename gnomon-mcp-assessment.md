# Gnomon-MCP TemporalBench Assessment

## Context

We evaluated **TemporalBench** (arXiv:2602.13272) using the Gnomon benchmark suite at `/root/Gnomon` on branch `claude/gnomon-harness-issues-hgerrj`. The LLM used was `deepseek-v4-flash-0731` via a custom OpenAI-compatible endpoint at `https://api.engy.ai/v1` (key in `$HERMES_CUSTOM_API_ENGY_AI_API_KEY`).

Three conditions were run across all tiers (T1–T4, ~764 rows):

| Condition | Description |
|-----------|-------------|
| **Control** | Prompt sent verbatim to the LLM; model returns JSON directly |
| **Gnomon-Agent** | Gnomon computes evidence (forecasts, anomalies, stats); evidence is injected into the prompt; LLM answers based on it |
| **Gnomon-MCP** | Model gets real `gnomon mcp serve` tool surface + `submit_answer` harness tool; multi-turn agent loop per row |

## Results Summary

### Choice Accuracy (MCQ)

| Tier | Control | Gnomon-Agent | Gnomon-MCP (partial) |
|------|:-------:|:------------:|:--------------------:|
| **T1** | 43.4% | 48.9% | 32.7% (13 rows) |
| **T2** | 34.5% | 31.9% | 0.0%* (12 rows) |
| **T3** | 23.2% | 20.4% | 28.6% (11 rows) |
| **T4** | 44.4% | 43.1% | 0.0%* (12 rows) |

\* T2/T4 rows all hit the token cap (250K) before completing. One recent fix test showed a successful T2 row with SMAPE 9.92% and 1/3 MCQ correct (route: "informed-direct", model wrote own values).

### Forecast Metrics

| Metric | Control | Gnomon-Agent | Gnomon-MCP |
|--------|:-------:|:------------:|:----------:|
| Forecast rows scored | 281 | 2 | 1 (test) |
| SMAPE / OW_sMAPE | 0.46% / 12.1% | n/a (2 rows) | 9.92% (1 row) |

Gnomon-Agent had high forecast abstention: 362 channels "degraded", 124 abstained.

## Known Issues

### 1. Token Cap Kills T2/T4 Submissions (partially fixed)

The token cap in `mcp_agent.py` (`MAX_RUN_TOKENS = 250_000`) was hit on every T2/T4 row after 5–8 tool calls. The loop checked the cap **before** checking whether `submit_answer` had been called, so completed submissions were voided.

**Fix applied:** Doubled cap to 500K and added a submission check before the cap check.

**Remaining concern:** Even 500K gets tight with multi-turn conversations + long tool results (full forecast arrays, full data dumps).

### 2. Jail Violations on gnomon_forecast

The model calls `gnomon_forecast` with `output_dir='/tmp/gnomon-*'` which is outside the run's jail directory. This wastes a round-trip on the error response.

**Root cause:** The jail path is set per-run (`tempfile.mkdtemp`) but the model doesn't know what it is. It needs to be surfaced in the system prompt or tool description.

### 3. T1/T3 Tool Distraction

Pure MCQ rows (no forecast channels) still get the full Gnomon tool surface + a system prompt saying forecasting tools are available. T1 accuracy dropped (32.7% vs 43.4% control). T3 improved (28.6% vs 23.2% control), likely because T3's complex question packs benefit from structured reasoning.

### 4. Per-Channel Forecasting is Token-Inefficient

The model calls `gnomon_forecast` once per channel (6 calls for 6 vitals) instead of batching. Gnomon already has `forecast_multi()` for multi-channel forecasting, and the gnomon runner has `forecast_channels()` which does batched evaluation — but there's no MCP tool that exposes it.

### 5. Tool Result Verbosity

`gnomon_forecast` returns full forecast arrays (29 values per channel). `gnomon_inspect` returns full data summaries. These long tool results compound the token problem with every turn.

## Patches Applied

| File | Change |
|------|--------|
| `benchmarks/common/openrouter.py` | `OPENROUTER_BASE_URL` reads from env var, falls back to openrouter.ai |
| `benchmarks/temporalbench/run_temporalbench.py` | Added `--base-url` CLI arg; removed T2/T4-only restriction for gnomon-mcp when all tiers specified |
| `benchmarks/temporalbench/mcp_agent.py` | Raised `MAX_RUN_TOKENS` 250K→500K; added submission check before cap check; added `_mcp_mcq_only()` for T1/T3 MCP handling with correct answer format per tier |
| `benchmarks/tests/test_temporalbench_mcp_agent.py` | Updated tier restriction test to match new behaviour; updated token cap test |

## Proposed Improvements

### 1. Per-Condition System Prompts

Separate MCP system prompts for MCQ-only tiers vs forecast tiers:

- **T1/T3:** Skip tool introduction entirely. Just say "Answer via submit_answer."
- **T2/T4:** Explicit instructions about jail path, batch forecasting, and always including MCQ in `submit_answer`.

### 2. Batch Forecast MCP Tool

Add a `gnomon_forecast_all` tool that accepts multiple channels at once, wrapping the existing `forecast_multi()` function. Cuts tool calls from ~6 to ~1 per row.

### 3. Jail Path Disclosure

Expose the run jail directory in the system prompt template so the model uses the correct `output_dir` on the first try. This prevents the jail violation round-trip.

### 4. Tool Result Trimming

Truncate long forecast arrays in tool responses to summary stats + first/last few values, keeping results token-efficient.

### 5. Softer Cap Handling

Instead of a hard cap that voids the row, cap per-tool-message size so earlier rounds don't bloat the context. Or allow partial results when a cap is hit (submit whatever forecast/MCQ was completed).

## Reproduction

```bash
cd /root/Gnomon
source .venv/bin/activate

# Test a small MCP run on T2/T4
OPENROUTER_BASE_URL=https://api.engy.ai/v1 \
OPENROUTER_API_KEY=$HERMES_CUSTOM_API_ENGY_AI_API_KEY \
python3 -m benchmarks.temporalbench.run_temporalbench \
  --data-dir ~/temporalbench \
  --condition gnomon-mcp \
  --model deepseek-v4-flash-0731 \
  --tiers T2,T4 --limit 4 \
  --output-dir results/tb-mcp-test

# Verify no regressions
python3 -m pytest benchmarks/tests/
```

Results appear in `results/<output-dir>/gnomonbench.jsonl` and `results/<output-dir>/details/`.

## File Layout

| Path | Purpose |
|------|---------|
| `benchmarks/temporalbench/run_temporalbench.py` | Main entrypoint; argument parsing, row iteration, scoring dispatch |
| `benchmarks/temporalbench/mcp_agent.py` | `_Run` class (T2/T4 MCP agent loop); `_mcp_mcq_only` (T1/T3 MCQ handler) |
| `benchmarks/temporalbench/gnomon_runner.py` | Gnomon engine integration; `forecast_channels()` for batched forecasting |
| `benchmarks/common/openrouter.py` | OpenAI-compatible chat client; works with any base URL |
| `benchmarks/temporalbench/scoring.py` | Choice and forecast scoring |
| `benchmarks/temporalbench/tasks.py` | Dataset loading, prompt parsing, JSON extraction |