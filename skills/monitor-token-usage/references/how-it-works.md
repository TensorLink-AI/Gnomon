# How monitor-token-usage works

## What each run does

- **Daily totals** add input, output, cache-read and cache-write tokens from
  `session_model_usage`. Each usage row is spread over that session's assistant
  messages, so sessions crossing midnight or switching models land on the right
  days. Today is partial and is never used as a forecast input.
- **Hourly** (optional `token-tracker` plugin): the plugin records the token
  counts of every main-agent LLM call as it happens (no prompts or replies).
  `hours` shows exact per-hour tokens, calls and tokens per call, plus the
  biggest single calls; `report` adds a 24-hour sparkline. Side tasks
  (compression, titles, web extraction) do not pass through the plugin's hook
  and stay daily-only. Per-call history starts when the plugin is enabled.
- **Hourly forecasts and alerts** (with the plugin) start as soon as 24 complete
  hours of history exist. Hours before the plugin was enabled are estimated
  from `state.db`, so an existing install starts immediately. Every 6 hours
  (`--hourly-every`) `check` records a 24-hour model forecast and a
  `seasonal_naive` baseline (same hour yesterday, or last week once two weeks
  exist) and scores them like the daily ones. A finished hour above its 90%
  range raises HOURLY SURGE, naming its call count, tokens per call and the
  session that used most of it; an expected-busy hour that went quiet raises
  HOURLY DIP. Hours under `--hourly-floor` (200K tokens) never alert.
  Hourly alerts use only the plugin's live records, because `state.db` session
  totals can lag until a session is saved.
- **Coverage self-check** compares the plugin's tokens with Hermes's main-agent
  totals in `state.db` over 24 complete hours ending 2 hours ago. `check` alerts
  once a day (TRACKER GAP) when it captured less than 90% (`--min-coverage`).
- **Call mix** splits usage into calls per day (`api_call_count`), tokens per
  call and cache-read share, comparing the last 7 days with the 7 before. More
  calls means more agent activity; bigger calls usually mean longer contexts
  being resent. A falling cache-read share raises cost at the same token count.
- **Cost** uses Hermes's own cost records (actual, else estimated). The blended
  rate comes from the last 28 days of priced usage. The report says how many
  tokens had no cost record.
- **check** (daily part once per day; repeats are no-ops) records a 7-day forecast from the
  model (with a 90% range) and from `seasonal_naive` into a Gnomon ledger at
  `$HERMES_HOME/data/token-usage/ledger.db` (default `~/.hermes`). It appends completed days as actuals,
  scores forecasts whose 7 days have passed, and pairs model against baseline.
- **Alerts** (each fires once): a completed day above or below the 90% range of
  the forecast made the day before (SURGE / DIP), with the call-mix change that
  drove it; today already above today's forecast high; the 7-day median
  projection above `--weekly-budget`; hourly surges and dips; a failed forecast;
  a tracker gap.

