# Agent skill

The [packaged skill](../skills/use-gnomon/SKILL.md) explains the current execution
tools, model choice, explicit cutoffs and bounded result retrieval. The wheel
installs it under `share/gnomon/skills/use-gnomon` in its environment prefix.
Make that skill directory available through your agent host's skill mechanism.

For recurring outcome review, the additional
[`use-gnomon-ledger` skill](../skills/use-gnomon-ledger/SKILL.md) connects recorded
evidence with the host's ordinary memory. See the [Hermes setup](hermes-ledger.md).

The skill does not install models, select credentials, grant spending permission
or execute recorded decisions.

## All packaged skills

Every skill below ships in the wheel under `share/gnomon/skills/<name>`. Install a
skill's whole directory (some include scripts or references).

| Skill | Use when |
|---|---|
| [use-gnomon](../skills/use-gnomon/SKILL.md) | an agent has `gnomon_*` MCP/CLI tools: inspect, forecast, backtest, read results, date calculations |
| [forecast-with-gnomon](../skills/forecast-with-gnomon/SKILL.md) | forecasting in Python with `GnomonSession`, including your own registered models |
| [use-gnomon-ledger](../skills/use-gnomon-ledger/SKILL.md) | comparing models on recorded history, reviewing decisions, keeping lessons |
| [route-with-gnomon](../skills/route-with-gnomon/SKILL.md) | choosing among models per forecast from recorded outcomes, with episodic memory across related series; replay first |
| [trade-with-gnomon](../skills/trade-with-gnomon/SKILL.md) | forecast-led trading: backtest, paper, then user-approved live, every decision recorded |
| [setup-gnomon-ephemeris](../skills/setup-gnomon-ephemeris/SKILL.md) | installing Gnomon, configuring Ephemeris (saved connection or TOML), connecting MCP, troubleshooting |
| [connect-ephemeris](../skills/connect-ephemeris/SKILL.md) | a Hermes user opted in and enters or rotates an Ephemeris key through secure entry |
| [forecast-report](../skills/forecast-report/SKILL.md) | Hermes: forecast a series and send a summary with a chart, once or on a schedule |
| [monitor-token-usage](../skills/monitor-token-usage/SKILL.md) | Hermes: forecast, monitor and explain LLM token and cost usage |

For optional Ephemeris signup in Hermes, run
`gnomon connect ephemeris --install-hermes-skill`. This installs `use-gnomon` and
`connect-ephemeris` into the active Hermes profile. The companion skill uses
Hermes native secure credential capture only after opt-in; it is separate so
local forecasting never requires a key. See [secure setup](ephemeris-onboarding.md#native-hermes-setup).
