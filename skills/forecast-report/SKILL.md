---
name: forecast-report
description: In Hermes, forecast any time series (sales, demand, load, revenue, traffic) with Gnomon — the Ephemeris ensemble when connected, or a chosen or local model — backtest it against a seasonal-naive baseline, and send a short summary plus a seaborn chart of actuals, forecast and 50%/90% ranges, for example to Telegram. Use when a Hermes user wants a series forecast reported with a chart, once or on a schedule; for connected gnomon_* tools use use-gnomon, and for Python use forecast-with-gnomon.
---

# Forecast report

Use when the user gives a forecasting task (a CSV, a file sent in chat, or numbers
pasted into the conversation) and wants the forecast explained and charted. One
script does the work. It runs Gnomon as a subprocess and makes no LLM calls.

```sh
S="${HERMES_HOME:-$HOME/.hermes}/skills/forecast-report/scripts"
python3 "$S/forecast_report.py" data.csv --title "Store A daily units" --unit units
```

The output is a few lines of summary. Its last line is `MEDIA:<path to chart.png>`;
Hermes turns that line into an image attachment.

## Steps

1. **Get the data into a CSV** with one timestamp column (ISO 8601, e.g.
   `2026-09-28` or `2026-09-28T14:00`) and one numeric column. Other columns are
   ignored; the script picks the first timestamp-like and first numeric column,
   or pass `--time-col` / `--value-col`. For pasted numbers, write the CSV
   yourself, with real dates from the user; never invent dates or values.
   Rows that share a timestamp are refused until you choose `--agg sum` or
   `--agg mean` (e.g. several stores per day).
2. **Pick the options** that fit the task:
   - `--horizon N` steps ahead (default 48 hours, 14 days, 8 weeks, 6 months,
     4 quarters, 3 years).
   - `--freq h|D|W|MS|QS|YS` only if the inferred spacing is wrong.
   - `--provider` picks the model (see below); the default is the ensemble.
   - `--stat mean` for levels (price, temperature, inventory); the default
     `sum` suits flows (sales, visits, tokens).
   - `--title` and `--unit` label the chart and summary.
3. **Run it** and read the summary before sending anything. `--json` gives the
   full numbers (every step's median and quantiles) for follow-up questions.
4. **Deliver it**:
   - In a Telegram conversation, reply with the script's output, keeping the
     `MEDIA:` line exactly as printed and on its own line. You may shorten or
     reword the summary lines above it; keep the provider name and the backtest.
   - From another surface (CLI, another chat), send it with
     `send_message(target="telegram", message=<output>)`. The `MEDIA:` line is
     delivered as a photo.
   - Only when the user asks for a recurring report on a CSV that something else
     keeps up to date: put the command in a script in `$HERMES_HOME/scripts/`
     (`HERMES_HOME` defaults to `~/.hermes`) and schedule it with
     `cronjob(action="create", schedule=..., script=..., no_agent=True,
     deliver="telegram")`. The script's output, chart included, is the message.

## Choosing the model

`--provider` takes any name from `python3 "$S/forecast_report.py" --list-models`:

- `ephemeris/ensemble` (default): every Ephemeris model forecasts and the service
  combines them by averaging each quantile across models. The summary and chart
  name the models it used.
- `ephemeris`: the Ephemeris router picks one model for this series.
- `ephemeris/<model>` (e.g. `ephemeris/chronos2`): one specific model.
- A local provider such as `historical_mean`, when Ephemeris is not connected.

If the user names a model or mode, use it. Ephemeris runs may incur charges: unless
the user asked for Ephemeris or already approved paid forecasts, confirm before the
first ensemble run, or use a local provider. `--providers-config PATH` uses an
operator TOML instead of the saved connection (the ensemble then needs a provider
with `mode = "ensemble"`). To compare
models, run the script once per provider and compare the backtest lines (each run
is one holdout window, so small differences are not meaningful).

## What the script does

- Forecasts the next `--horizon` steps with the model (`--provider`, with
  5/25/50/75/95% quantiles) and with Gnomon's
  `seasonal_naive` baseline (same hour yesterday, same weekday last week, same
  month last year; last value when history is shorter than two seasons).
- Backtests both: it hides the last `--horizon` observed points, forecasts them
  from the history before, and reports each MAE and how many actuals fell in the
  model's 90% range. Skipped when history is too short (`--no-backtest` to skip).
- Records the two live forecasts in a Gnomon ledger
  (`$HERMES_HOME/data/forecast-report/ledger.db`, series `forecast-report/<slug of
  title>`),
  so they can be scored when the actuals arrive (see `use-gnomon-ledger`).
  Backtest forecasts are not recorded. `--no-ledger` records nothing.
- Draws the chart (seaborn): recent actuals, the backtest window with its
  forecast, the forecast median with 50% and 90% ranges, and the baseline. It is
  saved under `$HERMES_HOME/cache/images/forecast-report/`, which Hermes always
  allows as an attachment source. `--out` chooses another path.
- Never falls back silently: a failed model call is printed as a ⚠️ line under
  the title, and the report then shows the baseline only. Gaps in the timestamp
  grid are counted and flagged. A chart failure still prints the summary.

## Setup

1. Copy this skill directory to `$HERMES_HOME/skills/forecast-report/`.
2. Install seaborn for the `python3` that runs the script:
   `python3 -m pip install seaborn`. Check with `python3 -c "import seaborn"`.
   Gnomon itself does not need seaborn; the script finds the `gnomon` command on
   `PATH` (or `$GNOMON_CMD`, or `--gnomon`).
3. Ephemeris models need the user's opt-in; use the `connect-ephemeris` skill.
   Without it, pass `--provider historical_mean` or another provider from
   `gnomon capabilities`, and say the model is a simple baseline.
4. For Telegram delivery, the gateway must be running and Telegram connected with
   a home channel (`/sethome` in the target chat). Hermes reports failed sends as
   `delivery_failed` in `hermes cron list` / `hermes cron doctor`.

## If something fails

- `gnomon` not found: install Gnomon for the `python3` running the script, or pass
  `--gnomon PATH` / set `$GNOMON_CMD`.
- Provider not found: run `--list-models` and use a listed name; never switch model
  without saying so.
- ⚠️ model-failure line: report it with the baseline-only result; do not rerun a
  paid call repeatedly.
- `chart not drawn`: install seaborn (`python3 -m pip install seaborn`); the summary
  is still valid.
- Duplicate timestamps refused: ask whether to `--agg sum` or `--agg mean`.

## Reporting results honestly

- Name the provider with every number. The chart subtitle and summary do.
- The backtest is one window of `--horizon` points: evidence, not proof. Report
  it as printed, including when the model loses to the baseline, and do not
  claim accuracy beyond it.
- The ranges are the model's uncertainty, not guarantees. A backtest where the
  90% range held far fewer than 90% of actuals means the ranges are too narrow.
- A total over the horizon sums step medians; it has no range of its own.
- Mention flagged gaps, and ask the user about them rather than filling them in.
- Each run makes two Ephemeris calls (forecast and backtest), which may incur a
  charge; the ensemble runs several models per call and may cost more than a
  single model. `--no-backtest` makes one call.
