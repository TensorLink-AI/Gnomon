---
name: monitor-token-usage
description: Forecast and monitor the tokens and cost of Hermes's LLM calls with Gnomon (Ephemeris models optional), alert when a day or hour lands outside its forecast range and say what caused it (more calls, bigger calls, one runaway session), and report whether the model beats a free baseline.
---

# Monitor Hermes token usage

Use when the user asks about LLM token usage, spend, cost projections, budgets,
why usage changed, or wants alerts on unusual usage. One script does the work; it reads Hermes
`state.db` read-only and runs Gnomon as a subprocess. No LLM calls.

```sh
S="${HERMES_HOME:-$HOME/.hermes}/skills/monitor-token-usage/scripts"
python3 "$S/token_usage.py" report            # history, call mix, 7/30-day projection, track record
python3 "$S/token_usage.py" report --json     # same, machine-readable
python3 "$S/token_usage.py" check             # record forecasts, score, alert (daily and hourly)
python3 "$S/token_usage.py" hours             # exact hourly usage, forecast range, biggest calls (plugin)
```

Common options: `--tz Australia/Brisbane` (day boundaries; default UTC),
`--provider ephemeris/chronos2` (the default model), `--weekly-budget 5` (USD),
`--base-url-filter engy` (count one billing endpoint only). `--help` lists all.

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
  `~/.hermes/data/token-usage/ledger.db`. It appends completed days as actuals,
  scores forecasts whose 7 days have passed, and pairs model against baseline.
- **Alerts** (each fires once): a completed day above or below the 90% range of
  the forecast made the day before (SURGE / DIP), with the call-mix change that
  drove it; today already above today's forecast high; the 7-day median
  projection above `--weekly-budget`; hourly surges and dips; a failed forecast;
  a tracker gap.

## Setup

1. Copy this skill directory to `~/.hermes/skills/monitor-token-usage/`.
2. Daily forecasting starts after 14 complete days of Hermes usage (until then
   `check` says `WAITING`); hourly forecasting after 24 complete hours.
3. For Ephemeris models, the user must opt in; use the `connect-ephemeris` skill.
   Without it, pass `--provider seasonal_naive` or another local provider. The
   script never falls back silently: a failed model forecast raises an alert.
4. For hourly tracking, install and enable the plugin, then restart Hermes
   (gateway included). General plugins load only when listed in `plugins.enabled`:

```sh
cp -r "${HERMES_HOME:-$HOME/.hermes}/skills/monitor-token-usage/hermes-plugin/token-tracker" \
      "${HERMES_HOME:-$HOME/.hermes}/plugins/"
hermes plugins enable token-tracker
```

   Records go to `~/.hermes/plugin-data/token-tracker/calls.db` (per profile).
5. Before scheduling, check the two things Telegram delivery depends on.
   Otherwise the jobs are created but nothing reaches the user:
   - **The gateway is running.** Hermes cron runs inside the gateway daemon
     (`hermes gateway install` for a service, or `hermes gateway`).
   - **Telegram is connected and has a home channel.** `deliver="telegram"`
     sends to the home channel: the user sends `/sethome` in the chat that
     should receive alerts (or sets `TELEGRAM_HOME_CHANNEL`).

   If either is missing, ask the user to connect Telegram first, or schedule with
   `deliver="local"` (output kept in `~/.hermes/cron/output/`) and switch later
   with `cronjob(action="update", job_id=..., deliver="telegram")`.
6. Schedule the two no-agent cron scripts. Hermes only runs scripts from
   `~/.hermes/scripts/`, so copy them there first:

```sh
cp "$S"/token-usage-alert.sh "$S"/token-usage-daily.sh ~/.hermes/scripts/
```

```python
cronjob(action="create", name="token-usage-alert", schedule="every 15m",
        script="token-usage-alert.sh", no_agent=True, deliver="telegram")
cronjob(action="create", name="token-usage-daily", schedule="0 9 * * *",
        script="token-usage-daily.sh", no_agent=True, deliver="telegram")
```

Then run the daily job once now (`cronjob(action="run", job_id=...)`) and confirm
the summary arrives in Telegram. Hermes reports a failed send as `delivery_failed`
in `hermes cron list` and `hermes cron doctor`, and marks a job whose Telegram
credentials are missing `blocked_config`. The alert job has no test message: it
prints nothing unless something fires. Every 15 minutes catches a runaway loop
within about an hour; without the plugin, every 6h is enough. Add options such as
`--tz` by editing the `check` line in the copied scripts.

## Reporting results honestly

- Quote the provider name with every projection. Multi-day ranges add daily
  bounds, so they are wider than a true 90% interval; say so if asked.
- The track record is the evidence that the model is useful. Report it as
  written, including when the model loses to `seasonal_naive`. Before any pairs
  have matured (7 days after the first `check`), say there is no track record yet.
- Ephemeris does not attest model revisions. The track record assumes the served
  model was stable over the window; mention this if a model change is suspected.
- The alert range is the forecast's uncertainty, not a budget. A SURGE means
  "unusual for this history", not "too expensive". Use `--weekly-budget` for cost.
- Hourly numbers are exact for main-agent calls only. Quote the coverage line
  with them; below ~90% they undercount. Hours before the plugin was enabled
  are estimates and only feed the forecast history.
- When explaining a change, use the call-mix line: it compares with the previous
  7 days, which may mix weekdays and weekends. Today's call count is partial, so
  only its tokens per call is compared.
- Forecasting with Ephemeris may incur a small charge: `check` makes one daily
  model call per day plus, with the plugin, one hourly call every 6 hours; each
  `report` makes one.
