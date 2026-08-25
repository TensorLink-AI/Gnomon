# Publication modes

Gnomon separates the immutable numeric paths from the recommendation a human
reads first. This lets new qualitative information be useful without claiming
that it was historically validated.

`strict` is the governed default. Only the history-only primary or a candidate
that passed Gnomon's fold-safe admission may be recommended. An LLM ranking is
ignored in this mode.

`best_effort` may put a sealed `prior_assisted` conditional path first. The
history-only primary remains present, the weaker support label remains attached,
and the recommendation is never automation-eligible. Use this mode for a human
who wants the best bounded answer available from new information.

`scenario` returns the primary and each bounded conditional path. A governed
LLM selection may rank their immutable IDs and recommend one, citing verified
claims, counterevidence, confidence, and what would change its selection. The
selection cannot contain forecast numbers, change support, conceal the primary,
or authorize automation.

Recommendation and automation are deliberately separate. Automation requires
an explicit policy with a `policy_id`, `authorize: true`, and a supported
minimum tier; a prior-assisted scenario is never automation-eligible regardless
of that policy.

CLI example:

```bash
gnomon forecast data.csv --time timestamp --target value --horizon 7 \
  --publication-mode best_effort --dossier dossier.json
```

The canonical artifact is not rewritten. The projection is persisted beside it
as `<artifact-id>.<publication-seal>.publication.json`, with seals over the publication and every
scenario. When `--project` is supplied, the recommendation reuses Gnomon's
synthesis receipts and is scored against actuals independently of the primary.

The MCP `gnomon_forecast` tool exposes the same `publication_mode`,
`temporal_dossiers`, `scenario_selection`, and `automation_policy` fields.
