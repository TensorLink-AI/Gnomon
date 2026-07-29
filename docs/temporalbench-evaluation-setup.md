# Robust TemporalBench Evaluation Setup for Hermes / LLM ± Aion

## The problem

We want to rigorously test how different configurations perform on TemporalBench's
4 task tiers (T1-T4), not just T2 forecasting. The benchmark has:

- **T1** — Historical Structure Interpretation (MCQ: trend, volatility, seasonality, outliers)
- **T2** — Context-Free Forecasting (numerical + qualitative MCQ)
- **T3** — Contextual Temporal Reasoning (MCQ grounded in domain context)
- **T4** — Event-Informed Prediction (numerical + event-conditioned MCQ)

Each tier tests a different capability. Our initial benchmark only tested T2
numerical accuracy on 20 sampled tasks. A robust evaluation needs all 4 tiers,
all 191 tasks, and multiple model configurations.

## Configurations to test

We want a matrix of configurations to isolate what each layer contributes:

### Group A: Raw LLM (no Aion, no tools)

| Config | Model | Tools | Aion | Description |
|---|---|---|---|---|
| A1 | GPT-4o | None | ❌ | Baseline LLM, prompted directly with TemporalBench tasks |
| A2 | Claude Sonnet 4 | None | ❌ | Different frontier model |
| A3 | GLM-5.2 (current) | None | ❌ | This model |
| A4 | DeepSeek-V3 | None | ❌ | Open model |

These establish the "LLM alone" floor — can the model reason about time
series without any tools?

### Group B: LLM + tools (no Aion)

| Config | Model | Tools | Aion | Description |
|---|---|---|---|---|
| B1 | GPT-4o | web_search | ❌ | LLM can look up domain info |
| B2 | Claude Sonnet 4 | web_search + code_exec | ❌ | LLM can compute statistics |
| B3 | GLM-5.2 | web_search + code_exec | ❌ | Current model with tools |
| B4 | GPT-4o | code_exec (pandas/statsmodels) | ❌ | LLM can use statistical libraries |

These test whether tool access (especially code execution for computing
forecasts) helps the LLM beyond raw reasoning.

### Group C: LLM + Aion (no TSFMs)

| Config | Model | Tools | Aion | Description |
|---|---|---|---|---|
| C1 | GPT-4o | aion_forecast | ✅ (baselines+stat) | LLM calls Aion for T2 forecasts |
| C2 | Claude Sonnet 4 | aion_forecast | ✅ (baselines+stat) | Same, different model |
| C3 | GLM-5.2 | aion_forecast | ✅ (baselines+stat) | Current model + Aion |

These test whether Aion's evaluation layer improves T2 performance vs
the LLM computing forecasts itself (Group B4).

### Group D: LLM + Aion + TSFMs

| Config | Model | Tools | Aion | TSFM | Description |
|---|---|---|---|---|---|
| D1 | GPT-4o | aion_forecast | ✅ (full) | Chronos-Bolt Mini | Aion + foundation model |
| D2 | Claude Sonnet 4 | aion_forecast | ✅ (full) | Chronos-Bolt Mini | Same, different LLM |
| D3 | GPT-4o | aion_forecast | ✅ (full) | Toto-2.0-22m | Different TSFM |
| D4 | GPT-4o | aion_forecast | ✅ (full) + ensemble | Chronos + Toto | Ensemble of TSFMs |

These test the full stack — does the TSFM layer add value on top of Aion?

### Group E: LLM + Aion + TSFMs + Covariates (future)

| Config | Model | Tools | Aion | TSFM | Covariates | Description |
|---|---|---|---|---|---|---|
| E1 | GPT-4o | aion_forecast + web | ✅ (full) | Toto-2.0 | ✅ | LLM fetches covariates |
| E2 | Claude Sonnet 4 | aion_forecast + web | ✅ (full) | Chronos-2 | ✅ | Covariate-aware TSFM |

These test the covariate enrichment pipeline (when implemented).

## How to run each configuration

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  Benchmark Runner (Python script)                   │
│                                                      │
│  1. Load TemporalBench dev set (191 series, T1-T4)  │
│  2. For each config:                                 │
│     a. Build the prompt for each task                │
│     b. Call the LLM (API or Hermes)                  │
│     c. Parse the LLM's response                      │
│     d. Score against ground truth                    │
│  3. Aggregate per-config, per-tier, per-domain       │
│  4. Output comparison table                          │
│                                                      │
├─────────────────────────────────────────────────────┤
│  LLM Backends:                                       │
│  - Direct API (OpenAI, Anthropic, etc.)             │
│  - Hermes chat -q (spawns Hermes with tools)         │
│  - Hermes delegate_task (parallel subagents)        │
│  - Hermes cron (scheduled batches)                   │
├─────────────────────────────────────────────────────┤
│  Aion Backend:                                       │
│  - CLI: aion forecast/inspect/capabilities           │
│  - MCP: aion mcp serve                               │
│  - Python: from aion.runtime import forecast         │
├─────────────────────────────────────────────────────┤
│  Scoring:                                            │
│  - TemporalBench's forecast_metrics_utils.py         │
│  - T1/T3/T4: MCQ accuracy                            │
│  - T2/T4: MAE, sMAPE, OW_sMAPE (MIMIC)               │
└─────────────────────────────────────────────────────┘
```

### Prompt construction per tier

Each tier needs a different prompt strategy:

#### T1 (Historical Structure Interpretation)
- **Group A (raw LLM):** Feed the raw time series + questions directly
- **Group B (LLM + code):** LLM can compute autocorrelation, trend tests, etc.
- **Group C/D (Aion):** Feed Aion's `inspect` output (which includes detected
  seasonality, frequency, diagnostics) alongside the questions. Aion doesn't
  answer MCQs, but its diagnostics can inform the LLM's interpretation.

#### T2 (Context-Free Forecasting)
- **Group A (raw LLM):** Feed the history, ask for numerical forecast + MCQ
- **Group B (LLM + code):** LLM uses pandas/statsmodels to compute a forecast
- **Group C/D (Aion):** LLM calls `aion_forecast`, gets the artifact, then
  answers the qualitative MCQs based on Aion's output (selected model,
  support, warnings, scores)

#### T3 (Contextual Temporal Reasoning)
- **Group A (raw LLM):** Feed history + domain context + MCQ
- **Group B (LLM + web):** LLM can search for domain-specific info
- **Group C/D (Aion):** Same as A, but LLM has Aion's diagnostics as context

#### T4 (Event-Informed Prediction)
- **Group A (raw LLM):** Feed history + event description + MCQ
- **Group C/D (Aion):** LLM uses Aion's context event workflow — propose
  events, validate, forecast with context. This is where Aion's context
  ablation should shine.

### Hermes-specific execution

For configurations that use Aion (Groups C, D, E), the most robust approach
is to use Hermes as the agent runtime:

```bash
# For each task, spawn Hermes with the appropriate config
hermes chat -q "You are forecasting on TemporalBench. Here is the task:

[prompt with time series data]

Use the aion_forecast tool to produce a forecast, then answer the
qualitative questions based on Aion's output." \
  --model gpt-4o \
  --toolsets terminal,file,web \
  -Q
```

Or for batch processing:

```python
from hermes_tools import terminal

# For each task in TemporalBench:
for task in tasks:
    prompt = build_prompt(task, config_group="C1")
    result = terminal(f'hermes chat -q "{prompt}" --model gpt-4o -Q')
    # Parse result, score against ground truth
```

For parallel execution:

```python
# Use delegate_task to run multiple tasks concurrently
from hermes_tools import delegate_task

tasks_batch = [
    {"goal": build_prompt(task, group="C1"), "context": f"Task {task['id']}"}
    for task in tasks[:20]  # first 20
]
results = delegate_task(tasks=tasks_batch)
```

### Direct API execution (for Group A — raw LLM, no tools)

For configurations without tools (Group A), call the LLM API directly:

```python
import openai

client = openai.OpenAI()

for task in tasks:
    prompt = build_raw_prompt(task)  # T1/T2/T3/T4 prompt with data inline
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    answer = response.choices[0].message.content
    score = score_task(task, answer)
```

### Scoring

Use TemporalBench's own `forecast_metrics_utils.py` for consistency:

```python
from forecast_metrics_utils import compute_forecast_metrics

# For T2/T4 numerical tasks:
metrics = compute_forecast_metrics(
    ground_truth=task["ground_truth"],
    prediction=llm_or_aion_forecast,
    dataset=task["domain"],  # "mimic" uses OW_sMAPE
)

# For T1/T3/T4 MCQ tasks:
accuracy = sum(1 for q in task["mcqs"] if q["predicted"] == q["label"]) / len(task["mcqs"])
```

### Output format

```json
{
  "config": "C1",
  "model": "gpt-4o",
  "aion": true,
  "tsfm": null,
  "results": {
    "T1": {"accuracy": 0.65, "per_domain": {"MIMIC": 0.70, "PSML": 0.62, ...}},
    "T2": {"mae": 7.31, "smape": 8.5, "mcq_accuracy": 0.55, "per_domain": {...}},
    "T3": {"accuracy": 0.12, "per_domain": {...}},
    "T4": {"mae": 8.2, "mcq_accuracy": 0.30, "per_domain": {...}}
  },
  "overall_score": 0.45
}
```

## Recommended execution plan

### Phase 1: Quick validation (this session)
- [ ] Run Group A3 (current model, raw) on 20 sampled tasks, all 4 tiers
- [ ] Run Group C3 (current model + Aion) on same 20 tasks, T2 only
- [ ] Compare: does Aion improve T2 numerical accuracy?

### Phase 2: Full benchmark (dedicated session)
- [ ] Run Group A1-A4 (4 raw LLMs) on all 191 tasks, all 4 tiers
- [ ] Run Group B3-B4 (LLM + tools) on all 191 tasks
- [ ] Run Group C1-C3 (LLM + Aion) on all 191 tasks, T2
- [ ] Run Group D1-D4 (LLM + Aion + TSFMs) on 20 sampled tasks, T2

### Phase 3: Submit to leaderboard
- [ ] Best config → format for TemporalBench leaderboard
- [ ] Submit to https://huggingface.co/spaces/Melady/TemporalBench_Leaderboard

## Key considerations

1. **Cost control:** Running 191 tasks × 4 tiers × 10 configs = 7,640 LLM calls.
   Use sampling for expensive configs. T2 numerical only (no MCQs) for TSFM configs
   since TSFMs don't answer qualitative questions.

2. **Model API access:** Need API keys for GPT-4o, Claude Sonnet 4, DeepSeek-V3.
   Current model (GLM-5.2) is available via Nous.

3. **Hermes vs direct API:** Use direct API for Group A (no tools). Use Hermes
   for Groups C/D (Aion tool calls). Use Hermes with web toolsets for Group B/E.

4. **Determinism:** Set temperature=0 for all LLM calls. Aion is deterministic.
   Run each config twice and check variance for LLM configs.

5. **T1/T3 MCQ scoring:** These are qualitative — the LLM answers multiple-choice.
   Aion doesn't help directly with MCQs, but its diagnostics (inspect output:
   seasonality, trend, frequency) can inform the LLM's interpretation.

6. **T4 event tasks:** This is where Aion's context event workflow should provide
   the biggest advantage — the LLM proposes events, Aion validates and evaluates
   them, and the event-conditioned forecast is evidence-backed.
