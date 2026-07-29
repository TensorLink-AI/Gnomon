# Aion — Covariate Enrichment & TSFM Capability Matrix

**Status:** Design document  
**Date:** 2026-07-29  
**Authors:** Testing session feedback

---

## Problem

Aion today is univariate-only. It takes `timestamp + target` and forecasts.
There's no way for an agent to:

1. Discover relevant external data (weather, holidays, market data, events)
2. Fetch it from the web
3. Structure it as covariates aligned to the time grid
4. Feed it into models that can use it
5. Have Aion evaluate whether the covariates actually help

The `context_events` system is the closest analog — discrete events from
user-supplied files, tested via ablation. But it's not the same as continuous
covariates from web sources, ingested into multivariate models.

Meanwhile, TSFMs have different and evolving capabilities:

- Some support exogenous covariates (Chronos-2, Toto-2, Moirai-2)
- Some only do univariate (Chronos-Bolt, FlowState, TTM, MOMENT)
- Some provide quantile forecasts natively, others only point forecasts
- Some support variable context lengths, others have fixed windows
- Some can do few-shot fine-tuning, others are zero-shot only

Aion needs a **covariate enrichment pipeline** and a **TSFM capability
matrix** so the agent and the evaluation layer know which models can use
which features.

---

## Part 1: Covariate Enrichment Pipeline

### Overview

```
User: "Forecast retail sales for next 14 days"
  │
  ▼
LLM reasons about the data and domain:
  "This is daily US retail sales. Relevant covariates:
   - US federal holidays (binary)
   - Day of week (categorical)
   - Weather forecast for store location (temperature, precipitation)
   - Nearby competitor promotion events"
  │
  ▼
LLM fetches web data (via existing Hermes tools):
  web_search("US federal holidays August 2026")
  web_extract("https://...holiday-calendar...")
  web_search("weather forecast New York August 2026")
  web_extract("https://...weather-api...")
  │
  ▼
LLM structures covariates as a CSV:
  timestamp,is_holiday,is_weekend,temperature,precipitation
  2026-08-01,0,1,28.5,0.0
  2026-08-02,0,1,27.3,1.2
  ...
  │
  ▼
LLM calls aion_forecast with enriched dataset:
  aion forecast sales.csv \
    --covariates covariates.csv \
    --covariate-mapping is_holiday:binary,is_weekend:binary,temperature:continuous,precipitation:continuous \
    --time timestamp --target sales --horizon 14 --frequency D
  │
  ▼
Aion evaluates covariate value:
  1. Aligns covariates to the time grid (same timestamps)
  2. Checks temporal availability (covariates must be known at each fold cutoff)
  3. Runs ablation: model WITH covariates vs WITHOUT on identical folds
  4. If stable lift → covariates admitted
  5. If not → covariates rejected, univariate model retained
  │
  ▼
LLM interprets:
  "Aion admitted is_holiday (8% lift on backtest) and is_weekend (5% lift).
   Temperature and precipitation showed no stable effect and were excluded.
   Selected model: chronos_bolt_mini (supports exogenous covariates).
   Forecast: 2,450 units, with a spike predicted for the federal holiday."
```

### Components

#### 1. CLI: `--covariates` flag

```bash
aion forecast data.csv \
  --covariates covariates.csv \
  --covariate-mapping is_holiday:binary,temperature:continuous \
  --time timestamp --target sales --horizon 14 --frequency D
```

The covariates CSV has the same timestamp column as the main data, plus
one column per covariate. Aion aligns them by timestamp.

#### 2. Data layer: covariate loading and alignment

```python
# data.py — new function
def load_covariates(
    covariate_path: str,
    time_column: str,
    covariate_mapping: dict[str, str],
) -> dict[str, list[float]]:
    """Load and validate covariates aligned to the time grid.

    Args:
        covariate_path: path to covariates CSV
        time_column: timestamp column name
        covariate_mapping: {column_name: type} where type is "binary", "continuous", or "categorical"

    Returns:
        {covariate_name: [values]} aligned to the same timestamps as the main data
    """
```

#### 3. Temporal alignment: covariate availability check

Critical invariant: **covariates must be known at each historical fold cutoff.**

If the covariate is a weather *forecast*, it must have been available at
that point in time. If it's a holiday calendar, it must have been known.
Aion checks this by requiring covariates to have a `known_at` column or
by assuming they were known at their timestamp.

```python
# temporal.py — new function
def validate_covariate_availability(
    covariates: dict[str, list[float]],
    timestamps: list[datetime],
    fold_origins: list[int],
) -> list[str]:
    """Check that each covariate is available at every fold cutoff.

    Returns list of warnings for covariates with availability gaps.
    """
```

#### 4. Covariate ablation: does it help?

Extends the existing `context_eval.py` pattern:

```python
# covariate_eval.py — new module
def assess_covariates(
    values: list[float],
    covariates: dict[str, list[float]],
    horizon: int,
    season: int,
    minimum_improvement: float,
) -> CovariateAssessment:
    """Compare model WITH vs WITHOUT each covariate on identical folds.

    For each covariate:
    1. Run the model with all covariates → baseline_with score
    2. Run the model minus one covariate → ablated score
    3. If ablated score is worse → covariate contributes
    4. If ablated score is better or same → covariate excluded

    Returns per-covariate lift, retained/rejected status, and the
    reduced covariate set to use for the final forecast.
    """
```

#### 5. LLM enrichment workflow

New MCP tool: `aion_propose_covariates`

```json
{
  "name": "aion_propose_covariates",
  "description": "Propose covariates for a forecasting task. The agent describes what external data it found and how it maps to the time grid. Aion validates temporal alignment and returns the covariate set for use in aion_forecast.",
  "inputSchema": {
    "properties": {
      "target_description": {"type": "string", "description": "What the target variable represents (e.g. 'daily retail sales in the US')"},
      "proposed_covariates": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ["binary", "continuous", "categorical"]},
            "source": {"type": "string", "description": "Where the data came from (URL, API, document)"},
            "description": {"type": "string", "description": "Why this covariate is relevant"},
            "values": {"type": "array", "description": "Aligned values or URL to fetch them"}
          }
        }
      }
    }
  }
}
```

The LLM proposes covariates with their source and rationale. Aion validates
alignment and availability. The evaluation layer decides whether they help.

#### 5a. Covariate discovery guidance — `aion_suggest_covariates`

The problem with `aion_propose_covariates` is that it assumes the LLM already
knows what to fetch. In practice, the LLM needs **guidance from Aion** about
what covariates are likely relevant, what format Aion expects, and what the
temporal constraints are.

A new tool: `aion_suggest_covariates` — Aion inspects the dataset and returns
a structured guide for the LLM:

```json
{
  "name": "aion_suggest_covariates",
  "description": "Inspect a dataset and suggest what external covariates might improve the forecast. Returns domain-aware suggestions, temporal constraints, and a structured format guide. The LLM uses this to decide what to fetch from the web.",
  "inputSchema": {
    "properties": {
      "input": {"type": "string", "description": "Path to the data CSV"},
      "time_column": {"type": "string"},
      "target_column": {"type": "string"},
      "frequency": {"type": "string"},
      "horizon": {"type": "integer"},
      "series_column": {"type": "string"}
    }
  }
}
```

**What Aion returns:**

```json
{
  "dataset_summary": {
    "target": "sales",
    "frequency": "D",
    "history_start": "2026-01-01",
    "history_end": "2026-07-28",
    "horizon": 14,
    "forecast_period": "2026-07-29 to 2026-08-11",
    "observations": 209,
    "series_count": 1,
    "detected_seasonality": "weekly (lag-7 autocorrelation: 0.71)",
    "detected_trend": "upward (last 30d mean 12% above first 30d mean)",
    "volatility": "moderate (CV=0.34)",
    "domain_hint": "retail (based on target name 'sales' and daily frequency)"
  },
  "covariate_suggestions": [
    {
      "name": "is_holiday",
      "type": "binary",
      "relevance": "high",
      "reason": "Retail sales typically spike on holidays. Daily frequency + retail domain.",
      "fetch_hint": "Search for: 'US federal holidays 2026' or use a holiday API. Return 1 for holiday dates, 0 otherwise.",
      "temporal_requirement": "Must cover 2026-01-01 to 2026-08-11 (history + horizon)",
      "known_at_requirement": "Holidays are known in advance — safe to use at all fold cutoffs"
    },
    {
      "name": "is_weekend",
      "type": "binary",
      "relevance": "high",
      "reason": "Strong weekly seasonality detected (autocorrelation 0.71 at lag 7). Weekend flag captures this.",
      "fetch_hint": "Derive from timestamps: Saturday/Sunday = 1, else 0. No web fetch needed.",
      "temporal_requirement": "All timestamps",
      "known_at_requirement": "Always known"
    },
    {
      "name": "temperature",
      "type": "continuous",
      "relevance": "medium",
      "reason": "Weather can affect retail foot traffic. No strong evidence in the data itself.",
      "fetch_hint": "Search for: 'weather history [location] 2026' or use a weather API. Need daily average temperature.",
      "temporal_requirement": "Must cover 2026-01-01 to 2026-08-11. For the forecast period, weather forecasts are needed.",
      "known_at_requirement": "Historical weather is known. Forecast-period weather must be from a forecast that was available at the cutoff time — or use climatology as an approximation."
    },
    {
      "name": "promotion_flag",
      "type": "binary",
      "relevance": "unknown",
      "reason": "If promotions were run during the history, they would explain variance spikes.",
      "fetch_hint": "Check internal promotion calendar or sales records. If unavailable, skip.",
      "temporal_requirement": "Must cover full history if available",
      "known_at_requirement": "Only known if the promotion was planned before it ran"
    }
  ],
  "format_guide": {
    "csv_format": "timestamp,is_holiday,is_weekend,temperature,promotion_flag",
    "alignment": "One row per timestamp, matching the main data's timestamp column",
    "missing_values": "Use null for unknown dates — Aion will handle gaps",
    "known_at_column": "Optional: add a 'known_at' column if covariates have different availability dates"
  },
  "temporal_constraints": {
    "history_coverage": "2026-01-01 to 2026-07-28",
    "forecast_coverage": "2026-07-29 to 2026-08-11",
    "fold_cutoffs": ["2026-03-15", "2026-04-15", "2026-05-15", "2026-06-15"],
    "warning": "Covariates must be available at each fold cutoff. If a covariate was only known after a cutoff date, it cannot be used for folds before that date."
  },
  "eligible_models_with_covariates": [
    {"model": "toto2_22m", "reason": "Supports multivariate + covariates"},
    {"model": "moirai2_small", "reason": "Supports feat_dynamic_real"},
    {"model": "ttm", "reason": "Supports exogenous infusion"}
  ],
  "eligible_models_without_covariates": [
    {"model": "theta", "reason": "Univariate — will serve as baseline"},
    {"model": "seasonal_naive", "reason": "Mandatory baseline"},
    {"model": "chronos_bolt_mini", "reason": "Univariate TSFM — baseline candidate"}
  ]
}
```

**How the LLM uses this:**

1. Calls `aion_suggest_covariates` before fetching anything
2. Gets domain-aware suggestions with fetch hints (what to search for)
3. Gets temporal constraints (what date range to cover, fold cutoff dates)
4. Gets the CSV format Aion expects
5. Gets a list of which models can use covariates vs which are univariate-only
6. Decides what to fetch, fetches it via `web_search` / `web_extract`
7. Structures as CSV in the specified format
8. Calls `aion_forecast --covariates covariates.csv`

This is the guidance layer — Aion doesn't fetch data itself, but it tells
the LLM **what to look for, where to look, and how to format it**.

#### 5b. Covariate validation — `aion_validate_covariates`

Before running the full forecast, the LLM can validate its covariates:

```json
{
  "name": "aion_validate_covariates",
  "description": "Validate covariates before forecasting: check temporal alignment, coverage gaps, and known-at availability. Returns warnings about missing dates, future leakage risks, and format issues.",
  "inputSchema": {
    "properties": {
      "covariates_file": {"type": "string"},
      "time_column": {"type": "string"},
      "main_data_file": {"type": "string"},
      "frequency": {"type": "string"}
    }
  }
}
```

Returns:
- Coverage report (which dates have covariates, which are missing)
- Alignment check (do timestamps match the main data?)
- Leakage warning (any covariate values that wouldn't have been known at historical cutoffs?)
- Format validation (are values numeric/binary as expected?)
- Suggested fixes (interpolate gaps, extend coverage, etc.)

#### 6. Models that support covariates

| Model | Covariate Support | How |
|---|---|---|
| linear_trend | ✅ (add as features) | OLS with extra columns |
| ets | ❌ | Univariate only |
| theta | ❌ | Univariate only |
| Chronos-Bolt | ❌ | Univariate only |
| Chronos-2 | ✅ | Exogenous covariates via `future_df` parameter |
| Toto-2.0 | ✅ | Multivariate via alternating attention |
| Moirai-2.0 | ✅ | `feat_dynamic_real` parameter |
| FlowState | ❌ | Univariate only |
| TTM | ✅ | Exogenous infusion support |

Aion's evaluation pipeline should filter model eligibility based on
whether the model supports covariates and whether covariates are provided.

### Covariate admission rules (extending the context event pattern)

1. **Temporal validity**: The covariate must be known at each historical fold cutoff (no future leakage)
2. **Stable lift**: The covariate model must beat the univariate model by ≥ minimum_improvement on more than half of valid folds
3. **Not confined to one fold**: The lift is not from a single anomalous period
4. **Calibration check**: Adding the covariate doesn't degrade interval coverage beyond policy limits
5. **Complexity justified**: The extra data and model complexity are worth the improvement

If any rule fails, the covariate is excluded and the univariate model is used.

---

## Part 2: TSFM Capability Matrix

Different TSFMs have different capabilities. Aion needs to track these
so the evaluation pipeline knows:

- Which models can use covariates (if provided)
- Which models provide native quantile forecasts
- Which models support variable context lengths
- Which models support few-shot fine-tuning
- What the minimum history requirement is
- What frequencies the model supports

### Capability registry

```python
# tsfm.py — extend each adapter with capability metadata

@dataclass(frozen=True)
class TSFMCapabilities:
    """Declares what a TSFM adapter can actually do."""

    # Forecasting
    supports_univariate: bool = True
    supports_multivariate: bool = False
    supports_covariates: bool = False
    supports_exogenous: bool = False  # future-known covariates

    # Uncertainty
    supports_quantiles: bool = True
    supports_sampling: bool = False  # probabilistic samples

    # Input flexibility
    min_context_length: int = 1
    max_context_length: int = 512
    variable_context_length: bool = True
    min_history: int = 10

    # Frequency support
    supported_frequencies: list[str] | None = None  # None = any
    requires_frequency: bool = True

    # Fine-tuning
    supports_few_shot: bool = False
    supports_fine_tuning: bool = False

    # Output
    max_horizon: int | None = None  # None = unlimited
    point_only: bool = False  # no native quantiles

    # Performance
    cpu_inference: bool = True
    gpu_recommended: bool = False
    approximate_latency_ms: int = 100  # per forecast on CPU
```

### Capability matrix (current TSFMs)

| Capability | Chronos-Bolt Mini | Chronos-2 | Toto-2.0-22m | FlowState R1.1 | TTM R2 | Moirai-2.0-Small | MOMENT-1-Small |
|---|---|---|---|---|---|---|---|
| **Univariate** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Multivariate** | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Covariates** | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Exogenous** | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Quantiles** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Sampling** | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Min context** | 1 | 1 | 1 | 1 | 1 | 1 | 512 |
| **Max context** | 2048 | 2048 | 4096 | 4096 | 512 | 1000 | 512 |
| **Variable ctx** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Min history** | 10 | 10 | 10 | 10 | 10 | 10 | 512 |
| **Frequencies** | any | any | any | any (scale-adjusted) | min→hourly | any | any |
| **Few-shot** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| **Fine-tune** | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ |
| **Max horizon** | 64* | 256 | 1000+ | 2880 | 720 | 1000+ | 96 |
| **CPU inference** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **GPU recommended** | ❌ | optional | optional | ❌ | ❌ | optional | optional |
| **Params** | 21M | ~1B | 22M | 18.5M | 1-3M | 14M | 38M |
| **Sandbox deps** | chronos-forecasting | chronos-forecasting≥2.0 | toto-models | granite-tsfm | granite-tsfm | uni2ts | momentfm |

*Chronos-Bolt recommends horizon ≤ 64; quality degrades beyond.

### How the evaluation pipeline uses the capability matrix

When Aion runs `aion_forecast` with covariates:

1. **Filter eligible models**: Only models where `supports_covariates == True` are considered for the covariate-enhanced forecast. Univariate models still run as baselines.

2. **Check context length**: If the history is 800 points and the model's `max_context_length` is 512, Aion truncates or skips the model with a warning.

3. **Check frequency**: If the data is 5-minute frequency and the model only supports hourly+, the model is skipped.

4. **Check horizon**: If the horizon is 168 and the model's `max_horizon` is 64, Aion either chunks the forecast or skips the model.

5. **Report in capabilities**: `aion capabilities` shows the full capability matrix so agents know what's available.

### Updated capabilities output

```json
{
  "models": {
    "tsfm": ["chronos_bolt_mini", "toto2_22m"],
    "tsfm_available": ["chronos_bolt_mini", "chronos_bolt_small", "toto2_22m", ...],
    "tsfm_capabilities": {
      "chronos_bolt_mini": {
        "covariates": false,
        "quantiles": true,
        "max_context": 2048,
        "max_horizon": 64,
        "multivariate": false,
        "params_m": 21
      },
      "toto2_22m": {
        "covariates": true,
        "quantiles": true,
        "max_context": 4096,
        "max_horizon": null,
        "multivariate": true,
        "params_m": 22
      }
    }
  }
}
```

### How the LLM uses the capability matrix

The agent calls `aion_capabilities` and sees which TSFMs support covariates.
It then knows:

- "I have covariates (weather, holidays) → I should use Toto-2 or Moirai-2, not Chronos-Bolt"
- "I have 800 history points → MOMENT won't work (needs exactly 512), but Chronos will"
- "I need 168-step forecast → Chronos-Bolt recommends ≤64, so I should use Toto or Moirai instead"

This lets the LLM reason about **model selection strategy** without
making the actual selection — Aion's evaluation layer still picks the
winner based on backtest.

---

## Part 3: Implementation plan

### Phase 1: TSFM Capability Matrix (foundation)

- Add `TSFMCapabilities` dataclass to `tsfm.py`
- Populate capabilities for all 7 registered adapters
- Expose in `aion capabilities` output
- Filter model eligibility in `evaluation.py` based on capabilities
- Tests

### Phase 2: Covariate Ingestion

- `--covariates` and `--covariate-mapping` flags in CLI
- `load_covariates()` in `data.py`
- Temporal alignment in `temporal.py`
- Pass covariates through `runtime.py` → `evaluation.py`
- Models that support covariates receive them; univariate models ignore them
- Tests

### Phase 3: Covariate Ablation

- `covariate_eval.py` — ablation: WITH vs WITHOUT each covariate on identical folds
- Per-covariate lift reporting
- Retained/rejected status with reasons
- Evidence records for covariate decisions
- Tests

### Phase 4: LLM Enrichment Workflow

- `aion_suggest_covariates` MCP tool — Aion inspects data, returns domain-aware
  suggestions with fetch hints, temporal constraints, format guide, and
  eligible model lists
- `aion_validate_covariates` MCP tool — validates temporal alignment,
  coverage gaps, leakage risks, format before forecasting
- `aion_propose_covariates` MCP tool — LLM proposes covariates, Aion validates
- Agent workflow: suggest → fetch (web_search) → validate → forecast
- Aion validates temporal alignment and availability
- Aion evaluates covariate value via ablation
- Tests

### Phase 5: Covariate-Aware TSFM Adapters

- Update Toto-2 adapter to pass covariates via multivariate input
- Update Moirai-2 adapter to pass `feat_dynamic_real`
- Update Chronos-2 adapter (when available) to pass `future_df`
- Update TTM adapter to use exogenous infusion
- Tests

---

## Design invariants (preserved from existing architecture)

1. **Covariates must earn admission** — like context events, they enter the model only after demonstrating stable lift on identical folds
2. **No future leakage** — covariates must be known at each historical fold cutoff
3. **Univariate baseline is mandatory** — even with covariates, the univariate baseline always runs and must be beaten
4. **LLM proposes, Aion evaluates** — the LLM can fetch and propose covariates, but Aion's backtest decides whether they help
5. **Capability-driven model selection** — Aion filters which models can run based on declared capabilities (covariates, context length, frequency, horizon)
6. **Evidence travels with the result** — covariate decisions (admitted/rejected, lift, ablation scores) appear in the evidence ledger

---

## What this enables

```
Agent: "I'm forecasting electricity demand for next week."
  → searches web for weather forecast
  → fetches holiday calendar
  → structures as covariates
  → calls aion_forecast with covariates
  → Aion evaluates: weather helps (12% lift), holidays don't (0.3% lift, rejected)
  → Aion selects Toto-2 (supports covariates, wins on backtest)
  → Agent: "Aion admitted temperature as a covariate (12% backtest improvement).
    Selected model: Toto-2 (22M, supports multivariate). Holiday flag was
    rejected — no stable lift. Forecast: 4.2 GWh, with a dip predicted
    for the cooler weekend."
```

This is the vision: an agent that can **enrich** its own forecasting data
from the web, but where every enrichment is **tested** before it's trusted.
