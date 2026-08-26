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

The preferred prior-assisted path is `effect_composed`: an LLM or human cites
the supplied text and proposes a typed effect (shape, timing, scope, magnitude
distribution, and uncertainty basis). Gnomon—not the model—composes that effect
over the sealed primary. Supported shapes cover pulses, level and trend changes,
variance, ramps/recovery, bounds, seasonal amplitude/phase and cross-series
relationships. A full model-authored quantile path remains a compatibility lane,
is labelled `model_authored`, and has no additional authority.

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

Agents can use the same single forecast call without implementing Gnomon's
sealing format. Pass `--context-proposal proposal.json`, `--context-text`, and
`--context-known-at`; the proposal contains verified claims and an
`effect_proposal`. Invalid proposals receive typed violations and at most one
repair attempt (`effect_proposal_repair`). Context is therefore always used,
represented as a labelled scenario, or rejected with a reason.

Every projection includes a compact `temporal_state` (level, trend, interval
width, seasonality/regime receipts, path shape, analogues, cross-series state,
conflicts and sufficiency). The governed selector ranks sealed candidates from
this state and cited claims; it cannot edit their numbers. The sidecar retains
the complete sealed candidate portfolio so all alternatives—not only the
winner—can be scored when outcomes arrive.

The canonical artifact is not rewritten. The projection is persisted beside it
as `<artifact-id>.<publication-seal>.publication.json`, with seals over the publication and every
scenario. When `--project` is supplied, the recommendation reuses Gnomon's
synthesis receipts and is scored against actuals independently of the primary.

The MCP `gnomon_forecast` tool exposes the same `publication_mode`,
`temporal_dossiers`, a compact raw `context_submission` object,
`scenario_selection`, and `automation_policy` fields.
