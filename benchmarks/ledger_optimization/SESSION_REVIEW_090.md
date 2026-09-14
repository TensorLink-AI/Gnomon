# Development090: actual session-envelope integration

Before synthetic execution: the frozen v4 core stores GnomonSession.forecast
responses in event.execution. The088synthetic probe used
InferenceEngine.forecast(...).to_dict(), which includes a full request. The
visible-record filter currently requires that request inside the envelope.
Check both public response forms with an identical synthetic request and ledger
before proposing this adapter for the actual agent workflow. Do not change089's
running experiment, its frozen code, questions or dispatch.

Pinned public1.2.0; new local synthetic ledger only. Original engine probe and
visible filter stay preserved in Git. Record exact response keys, rejected
fields, before/after outcomes and forecast/query counts. A missing optional
request member in a session response must not cause loss of a recorded execution.
Authority remains ledger.execution: require matching provider/revision/fingerprint
and result; validate all supplied request fields against the canonical recorded
request, including covariates/season/cutoffs. If a response includes a request,
validate it too. Preserve the current required event task identity fields and
do not silently admit contradictory metadata. Check stored configuration metadata
when it is available. Query calls must not execute forecasts or mutate the ledger.

This is compatibility for an already-authorized development workflow, not a new
forecasting method, paid experiment or superiority claim. Main/PyPI, original
agent results and protected targets remain unchanged. The20%goal remains unmet.
