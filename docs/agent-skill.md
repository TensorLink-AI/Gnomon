# Agent skill

The [packaged skill](../skills/use-gnomon/SKILL.md) is a short instruction set:
inspect once, forecast with an explicit provider, evaluate only when asked, page
large results with `gnomon_read`, and use ledger/route/temporal only when exposed.
It tells the agent to prefer a configured Ephemeris provider, then the user's
registered model, and to label a built-in baseline as a reference. Every answer
keeps the provider and revision, execution ID, snapshot ID with `as_of`, and
quantiles when present.

The wheel installs the skill under `share/gnomon/skills/use-gnomon` in its
environment prefix. Make that directory available through your agent host's
skill mechanism. The skill does not install models, select credentials, grant
spending permission or execute recorded decisions.
