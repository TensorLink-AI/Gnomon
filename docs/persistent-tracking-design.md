# Aion Persistent Forecast Tracking — Design

## The problem

Aion today is stateless. Every forecast is a one-shot: run it, get the result, move on. There's no way to:
- Remember what was predicted last week and compare it to what actually happened
- Track which models win on which datasets over time
- Alert an agent when a threshold crossing was predicted and occurred
- Adjust model weights based on realised performance
- Notify when a forecast has gone stale and needs re-running

This is the gap between "forecasting tool" and "forecasting system."

## The design

Three layers, each independently useful:

```
┌─────────────────────────────────────────────────────┐
│  Layer 3: Agent Integration                         │
│  (Memory, reminders, cron, MCP events)              │
│  Hermes memory · any agent via MCP · webhooks        │
├─────────────────────────────────────────────────────┤
│  Layer 2: Project & Scoring                          │
│  (aion track · aion score · aion compare)            │
│  Forecast registry · actual submission · scoring    │
│  Model performance tracking · drift detection        │
├─────────────────────────────────────────────────────┤
│  Layer 1: Artifact Store                             │
│  (Already exists — immutable forecast directories)  │
│  artifact.json · forecast.csv · evidence.jsonl       │
└─────────────────────────────────────────────────────┘
```

---

## Layer 2: Project & Scoring (`aion track`)

### Forecast registry

Every forecast already writes an immutable artifact directory. The registry is a lightweight index that tracks:

```json
{
  "forecast_id": "forecast_abc123",
  "project": "api-capacity",
  "series": "__default__",
  "cutoff_time": "2026-07-29T04:00:00Z",
  "horizon": 24,
  "frequency": "h",
  "selected_model": "seasonal_naive",
  "support": "supported",
  "threshold": 5000,
  "threshold_peak_probability": 0.239,
  "created_at": "2026-07-29T05:00:00Z",
  "actuals_submitted": false,
  "score": null,
  "artifact_path": "/opt/data/aion-output/forecast_abc123"
}
```

Storage: a single SQLite database at `~/.local/share/aion/registry.db`. Zero dependencies — Python's `sqlite3` is stdlib. No daemon, no server, no external state.

### CLI commands

```bash
# Register a forecast (called automatically by `aion forecast` when --project is set)
aion track register --project api-capacity --forecast-id forecast_abc123

# List forecasts for a project
aion track list --project api-capacity
# → shows forecast_id, cutoff, model, support, score (if scored)

# Submit actuals (what actually happened)
aion track actuals --project api-capacity --file actuals.csv
# → matches actuals to forecasts by timestamp, computes scores

# Score a specific forecast
aion track score --forecast-id forecast_abc123 --file actuals.csv

# Compare two forecasts
aion track compare --forecast-id forecast_abc123 --forecast-id forecast_def456

# Show model performance over time
aion track performance --project api-capacity --model seasonal_naive
# → shows MASE, MAPE, bias, coverage for each model across all scored forecasts

# Show model win rate
aion track leaderboard --project api-capacity
# → ranked table of models by average score across all forecasts
```

### How scoring works

When actuals are submitted, Aion:
1. Loads the forecast artifact (point, q10, q50, q90 per step)
2. Loads the actual values for the same timestamps
3. Computes per-step and aggregate metrics:
   - **MASE** (Mean Absolute Scaled Error) — scale-free, comparable across datasets
   - **MAPE** (Mean Absolute Percentage Error) — intuitive
   - **Bias** (mean(actual - forecast)) — systematic over/under-prediction
   - **Interval coverage** — what fraction of actuals fell within q10-q90
   - **Threshold accuracy** — was the threshold crossing predicted correctly?
4. Writes the score to the registry
5. Updates the model performance table

### Model performance tracking

```sqlite
CREATE TABLE model_performance (
    project TEXT,
    model TEXT,
    forecast_id TEXT,
    mase REAL,
    mape REAL,
    bias REAL,
    coverage REAL,
    threshold_accuracy REAL,
    scored_at TEXT,
    PRIMARY KEY (project, model, forecast_id)
);
```

This enables the leaderboard: "across all forecasts in project `api-capacity`, `ets` has an average MASE of 0.72, `seasonal_naive` has 0.81, `drift` has 0.95."

### Drift detection

When a new score comes in, Aion compares it to the model's historical average:
- If MASE increased by >50% → "Model performance degraded on last forecast"
- If bias changed sign → "Model flipped from over to under-prediction"
- If coverage dropped below 50% → "Intervals no longer reliable"

These are warnings, not errors — they show up in `aion track list` and in the MCP tool output.

---

## Layer 3: Agent Integration

### Hermes-specific (memory + cron + reminders)

**Memory:** The agent stores a compact summary of each forecast in Hermes memory:

```
memory add "Aion forecast api-capacity 2026-07-29: model=seasonal_naive, support=supported, threshold=5000 peak_prob=24%, score=0.73 MASE (submitted 2026-08-01)"
```

The agent can then recall: "When did we last forecast API capacity, and how accurate was it?"

**Cron:** Recurring forecast that runs, scores, and alerts:

```bash
# Daily forecast + score previous + alert on threshold
hermes cron create "0 8 * * *" \
  "1. Score yesterday's Aion forecast for api-capacity using actuals from ~/metrics/api_traffic.csv.
   2. Run a new 24h forecast with --threshold 5000.
   3. If threshold peak probability > 70%, alert that capacity breach is likely.
   4. If yesterday's MASE > 1.0, note that the model is performing worse than a naive baseline.
   5. Save a summary to memory." \
  --name daily-capacity-forecast --deliver telegram
```

**Reminders:** The agent can set a reminder for when the forecast horizon ends:

```
"Remind me at 2026-07-30T05:00 to submit actuals for the api-capacity forecast"
```

When the reminder fires, the agent:
1. Reads the forecast artifact
2. Pulls fresh data for the forecast period
3. Calls `aion track actuals` to score it
4. Reports the result

### Agent-agnostic (MCP events + webhooks)

The MCP server can expose new tools for persistent tracking:

```json
{
  "name": "aion_track_forecast",
  "description": "Register a forecast in a project for ongoing tracking and scoring",
  "inputSchema": {
    "properties": {
      "forecast_id": {"type": "string"},
      "project": {"type": "string"},
      "threshold": {"type": "number"}
    }
  }
}
```

```json
{
  "name": "aion_submit_actuals",
  "description": "Submit actual values to score previous forecasts",
  "inputSchema": {
    "properties": {
      "project": {"type": "string"},
      "actuals_file": {"type": "string"}
    }
  }
}
```

```json
{
  "name": "aion_model_performance",
  "description": "Show which models perform best on a project over time",
  "inputSchema": {
    "properties": {
      "project": {"type": "string"}
    }
  }
}
```

Any MCP-capable agent (Claude Desktop, Cursor, another Hermes instance, a custom agent) can:
1. Run a forecast (`aion_forecast`)
2. Track it (`aion_track_forecast`)
3. Later submit actuals (`aion_submit_actuals`)
4. See which models won over time (`aion_model_performance`)

### Webhook events (for external systems)

Aion can emit events when scoring completes:

```json
{
  "event": "forecast_scored",
  "project": "api-capacity",
  "forecast_id": "forecast_abc123",
  "model": "seasonal_naive",
  "mase": 0.73,
  "threshold_accuracy": 0.92,
  "drift_detected": false
}
```

These can trigger:
- PagerDuty alerts when threshold accuracy drops
- Slack notifications when a model degrades
- Auto-scaling decisions when threshold probability is high
- Dashboards that show forecast accuracy over time

### How this generalises beyond Hermes

The key insight: **Aion's persistence layer is agent-agnostic.** The registry is a SQLite file. The MCP tools are a protocol. The webhooks are HTTP. Any agent framework can integrate:

| Framework | Integration |
|---|---|
| Hermes Agent | Memory + cron + reminders (deepest integration) |
| Claude Desktop | MCP tools (`aion_track_forecast`, `aion_submit_actuals`) |
| Cursor / VS Code | MCP tools (same protocol) |
| Custom Python agent | Python API (`from aion.tracking import Project`) |
| Shell scripts / CI | CLI commands (`aion track *`) |
| External systems | Webhook events on scoring |
| Dashboards / BI | SQLite queries against the registry |

---

## The flywheel effect

```
Forecast → Track → Wait → Submit Actuals → Score → Learn
    ↑                                                    │
    └────────── Adjust model weights from scores ────────┘
```

1. **Week 1:** Run forecast with `seasonal_naive`. Track it. Score: MASE 0.81.
2. **Week 2:** Run with `ets` enabled. Score: MASE 0.72. `ets` wins.
3. **Week 3:** Install `chronos_bolt_mini`. Score: MASE 0.68. TSFM wins.
4. **Week 4:** Enable ensemble. Score: MASE 0.65. Ensemble wins.
5. **Week 5:** `chronos` starts degrading (concept drift). Score: MASE 0.90. Aion flags drift.
6. **Week 6:** Aion's leaderboard shows `ets` is now the most reliable. Agent switches back.

**The agent doesn't need to be a data scientist.** It runs forecasts, submits actuals, and lets Aion's scoring tell it which model is winning. Over time, the model selection improves automatically.

---

## Implementation plan

### Phase 1: Registry + Scoring (CLI only)
- `src/aion/tracking.py` — SQLite registry, scoring, model performance
- `aion track register/list/actuals/score/compare/performance/leaderboard`
- Automatic registration when `--project` flag is passed to `aion forecast`
- Tests

### Phase 2: MCP Tools
- `aion_track_forecast`, `aion_submit_actuals`, `aion_model_performance`
- Added to `toolspec.py` and `mcp_server.py`
- Hermes plugin updated to expose them

### Phase 3: Drift Detection + Alerts
- Compare new scores to historical averages
- Emit warnings in `aion track list` output
- Webhook events on drift detection

### Phase 4: Auto-Adjust
- Use model performance history to bias model selection
- "Chronos has 0.90 MASE on this project — downweight it in selection"
- Config: `evaluation.auto_adjust_weights: true`

### Phase 5: Agent Memory Integration
- Hermes memory entries on score completion
- Cron job templates for recurring forecast + score workflows
- Reminder system for actual submission

---

## What makes this different

| Feature | Aion | Generic forecasting libs | LLM-based "forecasting" |
|---|---|---|---|
| Evidence-backed | Every number traces to backtest | Some | Never |
| Abstention | First-class | Rare | Never |
| Model tracking over time | Planned (this design) | Manual | Never |
| Agent-safe protocol | MCP + CLI + Python | CLI only | N/A |
| Dependency isolation | Per-model sandboxes | Shared env | N/A |
| Ensemble + meta-model | Competes as a candidate | Separate step | N/A |
| Realised scoring | Planned (this design) | External | Never |
| Drift detection | Planned (this design) | Enterprise-only | Never |

The persistence layer turns Aion from a tool into a system — one that gets better the more you use it.
