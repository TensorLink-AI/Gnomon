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

#### 5a. Covariate format guidance — `aion_covariate_guide`

The LLM decides what covariates are relevant — it has domain knowledge,
web access, and reasoning ability. Aion's job is to tell the LLM **how to
format whatever it finds** so the data is temporally valid and usable by
Aion's evaluation pipeline.

Aion does NOT suggest *what* to fetch. It tells the LLM:

- The temporal constraints (date range, fold cutoffs, known-at requirements)
- The CSV format it expects
- Which models can use covariates (so the LLM knows the capability ceiling)
- What validation checks will be applied (so the LLM can self-check)

```json
{
  "name": "aion_covariate_guide",
  "description": "Get format and temporal constraints for enriching a dataset with covariates. Aion tells you the date ranges, fold cutoffs, CSV format, and which models can use covariates. You decide what covariates to fetch — Aion tells you how to structure them correctly.",
  "inputSchema": {
    "properties": {
      "input": {"type": "string", "description": "Path to the main data CSV"},
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
  "temporal_constraints": {
    "history_start": "2026-01-01",
    "history_end": "2026-07-28",
    "forecast_period": "2026-07-29 to 2026-08-11",
    "fold_cutoffs": ["2026-03-15", "2026-04-15", "2026-05-15", "2026-06-15"],
    "critical_rule": "Covariate values must be known at each fold cutoff. If a covariate was only knowable after a cutoff date, it cannot be used for evaluation folds before that date. This prevents future leakage."
  },
  "format_guide": {
    "csv_format": "timestamp,covariate_1,covariate_2,...",
    "alignment": "One row per timestamp, matching the main data's timestamp column exactly",
    "column_naming": "Use descriptive names (e.g. is_holiday, temperature, promo_flag)",
    "value_types": "Numeric for continuous, 0/1 for binary, integer codes for categorical",
    "missing_values": "Leave empty or use null for unknown dates — Aion will handle gaps with a configurable policy",
    "optional_known_at_column": "If covariates have different availability dates (e.g. a weather forecast issued at a specific time), add a 'known_at' column with the date the value became known"
  },
  "model_eligibility": {
    "with_covariates": [
      {"model": "toto2_22m", "how": "Multivariate input — covariates as additional channels"},
      {"model": "moirai2_small", "how": "feat_dynamic_real parameter"},
      {"model": "ttm", "how": "Exogenous infusion"},
      {"model": "linear_trend", "how": "OLS with covariate columns as features"}
    ],
    "univariate_only": [
      {"model": "theta", "note": "Will serve as univariate baseline"},
      {"model": "seasonal_naive", "note": "Mandatory baseline"},
      {"model": "chronos_bolt_mini", "note": "Univariate TSFM"},
      {"model": "ets", "note": "Univariate"},
      {"model": "flowstate", "note": "Univariate"},
      {"model": "moment_small", "note": "Univariate"}
    ]
  },
  "validation_checks": [
    "Timestamps must match the main data exactly (one row per observation)",
    "Covariate values must be numeric (binary 0/1, continuous float, or integer codes)",
    "No future leakage: covariate must have been knowable at each fold cutoff",
    "Coverage: ideally covers the full history + forecast period. Gaps are handled but reduce effectiveness.",
    "If using a 'known_at' column: known_at date must be <= the corresponding timestamp for historical data"
  ],
  "dataset_context": {
    "target": "sales",
    "frequency": "D",
    "observations": 209,
    "series_count": 1,
    "horizon": 14,
    "detected_properties": {
      "seasonality_lag_7_autocorr": 0.71,
      "trend_direction": "upward",
      "volatility_cv": 0.34
    }
  }
}
```

**How the LLM uses this:**

1. The LLM looks at the data and reasons: "This is retail sales, daily,
   with weekly seasonality. I should check for holidays, weather, and
   promotions."
2. The LLM calls `aion_covariate_guide` to learn the **format and
   temporal rules**
3. The LLM fetches whatever it thinks is relevant via `web_search` /
   `web_extract` — Aion doesn't tell it what to fetch
4. The LLM structures the data as a CSV following Aion's format guide
5. The LLM calls `aion_forecast --covariates covariates.csv`
6. Aion validates alignment, checks for leakage, runs ablation, and
   decides whether the covariates actually help

**The key principle:** Aion provides the **rails** (format, temporal
constraints, validation rules, model capabilities). The LLM provides the
**reasoning** (what's relevant, where to find it, how to interpret it).
The LLM decides what to fetch; Aion decides whether it helps.

#### 5b. Covariate validation — `aion_validate_covariates`

After the LLM fetches and structures covariates, it can validate before
the full forecast run:

```json
{
  "name": "aion_validate_covariates",
  "description": "Check covariates for temporal alignment, coverage gaps, and leakage risks before forecasting. Returns specific issues and suggested fixes.",
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

This is a pre-flight check — the LLM can fix issues before committing
to a full forecast run.

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

- `aion_covariate_guide` MCP tool — Aion provides format constraints,
  temporal rules, model capabilities, and validation checks. The LLM
  decides what to fetch; Aion tells it how to structure it.
- `aion_validate_covariates` MCP tool — pre-flight validation of
  alignment, coverage, leakage, format
- `aion_propose_covariates` MCP tool — LLM proposes covariates, Aion
  validates and evaluates via ablation
- Agent workflow: LLM reasons about data → calls guide for format →
  fetches via web_search → validates → forecasts
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
  → LLM reasons: "Electricity demand is affected by weather, day-of-week,
    and holidays. Let me fetch weather and holiday data."
  → calls aion_covariate_guide (gets format, date ranges, fold cutoffs,
    model capabilities — knows Toto-2 and Moirai-2 can use covariates)
  → web_search("weather forecast for grid location August 2026")
  → web_search("UK bank holidays August 2026")
  → structures as covariates CSV following Aion's format
  → calls aion_validate_covariates (passes — aligned, no leakage)
  → calls aion_forecast --covariates covariates.csv
  → Aion: weather admitted (12% lift), holidays rejected (0.3% lift)
  → Aion: selects Toto-2 (supports covariates, wins on backtest)
  → Agent: "Temperature admitted as covariate. Forecast: 4.2 GWh,
    dip predicted for cooler weekend. Holidays showed no effect."
```

This is the vision: an agent that uses its own reasoning to **decide**
what external data matters, fetches it from the web, and feeds it to
Aion — where every covariate is **tested** before it's trusted.

The LLM owns the "what" and the "where." Aion owns the "how" (format,
temporal validity) and the "whether" (does it actually help on backtest).
