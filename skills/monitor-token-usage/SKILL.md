---
name: monitor-token-usage
description: Forecast and monitor the tokens and cost of Hermes's LLM calls with Gnomon (Ephemeris models optional), alert when a day or hour lands outside its forecast range and say what caused it (more calls, bigger calls, one runaway session), and report whether the model beats a free baseline. Use when the user wants Hermes token or cost usage forecast, monitored or explained.
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
`--provider NAME` (default `ephemeris/chronos2`), `--weekly-budget 5` (USD),
`--base-url-filter engy` (count one billing endpoint only). `--help` lists all.
Check `gnomon capabilities` for the discovered provider names and pass `--provider`
explicitly when the default is not listed, or `--provider seasonal_naive` without
Ephemeris.

## What it does

`report` shows daily history, the call mix (calls per day, tokens per call, cache
share), cost, and 7/30-day projections with the model's track record against a free
`seasonal_naive` baseline. `check` records daily (and, with the plugin, hourly)
forecasts in a Gnomon ledger, scores them as outcomes arrive, and prints alerts:
SURGE/DIP days and hours, a projection over `--weekly-budget`, a failed forecast, or
a TRACKER GAP. Details: [how it works](references/how-it-works.md).

## Setup

Ask before each setup step that changes the user's system: installing or enabling
the plugin, restarting Hermes, creating cron jobs, and sending messages. Do only the
steps the user approves.

1. Copy this skill directory to `$HERMES_HOME/skills/monitor-token-usage/`
   (`HERMES_HOME` defaults to `~/.hermes`).
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

   Records go to `$HERMES_HOME/plugin-data/token-tracker/calls.db` (per profile).
5. Before scheduling, check the two things Telegram delivery depends on.
   Otherwise the jobs are created but nothing reaches the user:
   - **The gateway is running.** Hermes cron runs inside the gateway daemon
     (`hermes gateway install` for a service, or `hermes gateway`).
   - **Telegram is connected and has a home channel.** `deliver="telegram"`
     sends to the home channel: the user sends `/sethome` in the chat that
     should receive alerts (or sets `TELEGRAM_HOME_CHANNEL`).

   If either is missing, ask the user to connect Telegram first, or schedule with
   `deliver="local"` (output kept in `$HERMES_HOME/cron/output/`) and switch later
   with `cronjob(action="update", job_id=..., deliver="telegram")`.
6. When the user wants scheduled alerts, schedule the two no-agent cron scripts.
   Hermes only runs scripts from `$HERMES_HOME/scripts/`, so copy them there first:

```sh
H="${HERMES_HOME:-$HOME/.hermes}"
mkdir -p "$H/scripts"
cp "$H/skills/monitor-token-usage/scripts/token-usage-alert.sh" \
   "$H/skills/monitor-token-usage/scripts/token-usage-daily.sh" "$H/scripts/"
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

## If something fails

- `WAITING`: fewer than 14 complete days (daily) or 24 complete hours (hourly) of
  history; report when forecasting will start instead of forcing a forecast.
- Failed forecast alert or provider not found: list providers with
  `gnomon capabilities` and pass a listed `--provider`; never switch models silently.
- `blocked_config` or `delivery_failed` in `hermes cron list`/`hermes cron doctor`:
  Telegram is not connected or has no home channel; ask the user to fix it, or
  deliver locally meanwhile.
- `TRACKER GAP`: the plugin captured under 90% of Hermes's totals; check that it is
  enabled in every profile and that Hermes was restarted, and quote coverage.

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
