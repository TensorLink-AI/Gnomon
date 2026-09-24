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
