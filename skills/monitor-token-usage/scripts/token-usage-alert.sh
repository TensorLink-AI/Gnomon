#!/usr/bin/env bash
# Hermes no-agent cron script. Copy to ~/.hermes/scripts/ (cron scripts must live there).
# Silent unless an alert fires; schedule every 15 minutes (every 6h without the plugin).
# Add options after "check", e.g. --tz Australia/Brisbane --weekly-budget 5.
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
exec python3 "$HERMES_HOME/skills/monitor-token-usage/scripts/token_usage.py" check --quiet "$@"
