# Agent skill and feedback receipts

Gnomon ships a thin agent workflow in [`skills/use-gnomon`](../skills/use-gnomon/).
It teaches an agent how to use the 10-tool default `core` surface efficiently,
when the compact three-tool `evidence` profile is sufficient, how to preserve
support and provenance, and how to recover from an abstention. It contains no
forecasting logic: Gnomon remains the only component that computes governed
primary temporal facts.

## Local feedback by default

Feedback is explicit, not ambient telemetry. Create a private local receipt:

```bash
gnomon-feedback create \
  --category unnecessary_calls \
  --context surface=evidence \
  --context verb=gnomon_forecast \
  --context call_count=3 \
  --context minimum_call_count=1 \
  --private-note "The host inspected and fetched the artifact first."
```

The command returns an ID but sends nothing. Each receipt is stored under
`.gnomon/feedback` with private file permissions. Structured context uses a
small allowlist; raw series, prompts, messages, credentials, paths, and tool
arguments have no fields in the schema.

`--private-note` stays only in the local receipt. `--public-summary` enters the
shareable projection after best-effort redaction, so it should contain only
text the user is willing to share.

Preview the exact shareable bundle before doing anything else:

```bash
gnomon-feedback preview fb_REPLACE_WITH_ID
```

Export requires a separate consent action:

```bash
gnomon-feedback export fb_REPLACE_WITH_ID \
  --output feedback-bundle.json \
  --consent
```

Submission is also explicit and requires a configured HTTPS receiver. Gnomon
does not ship a hidden or default collection endpoint:

```bash
gnomon-feedback submit fb_REPLACE_WITH_ID \
  --endpoint https://feedback.example/v1/receipts \
  --consent
```

If authentication is required, place the bearer token in
`GNOMON_FEEDBACK_TOKEN`; do not put it in a command argument. Local HTTP is
accepted only for `localhost` test receivers.

## Learning without behavioral surveillance

`gnomon-feedback summarize` reports category, product surface, task-kind, and
call-efficiency counts. It never returns free text, receipt IDs, or artifact
IDs. These aggregates can reveal recurring repair loops, redundant calls,
confusing abstentions, and missing capabilities without reconstructing a
person's prompts or data.

Every shareable receipt contains a reproduction hash and an anonymous claim
digest. The corresponding claim secret stays local. A future receiver can
award API credits, bounties, or recognition only after a report is confirmed,
deduplicated, and judged actionable; the contributor can then prove the claim
without attaching an identity to the report. Calls, tokens, and submission
volume must never earn rewards, because that would incentivize artificial use
and poison the product signal.
