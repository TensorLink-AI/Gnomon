# Prospective paired collection pilot 096 — running; continuation threshold unreachable

At **2026-09-15 00:10:19 UTC**, the same controller was verified live: 17/36
sessions completed, all forecasts valid, zero API errors. Full workflows were
4/6 plain Hermes, 5/6 Gnomon, and 5/5 ledger. Two plain sessions already missed
the full-workflow requirement, so that arm can reach at most 10/12. The frozen
11/12 threshold is therefore unreachable. **Do not launch the 276-session
continuation**, regardless of the eventual accuracy. Finish and audit all 36
pilot sessions, retaining failures and costs. The terminal gate has not yet run.

Five matched cases currently have RMSLE 0.652261 / 0.595800 / 0.585080 for
plain / Gnomon / ledger. These early, reused-development scores are monitoring,
not efficacy evidence or a basis for altering the gate. Full terminal analysis
will supersede this snapshot.

The two inspected plain failures used all 12 exploration requests on summaries
and file reads, then reached the protected selection phase with no ML backtest.
Both finished with a valid seasonal baseline, four numerical fits and 16 API
requests, without an API error. One read several source files; the other mostly
inspected history windows. They were not numerical-budget failures. Their full
returned model messages are retained in
`results/collection-096-failure-monitor-002/stdout.json`; exact commands and
stderr accompany them. The independent completed-session audit is still pending.

A continuation runner and exact-source synthetic resume proof were prepared
locally, but remain undispatched. A separate, undeployed common progress-reminder
prototype is described in `COMMON_WORKFLOW_PROGRESS_097.md`. No live worker,
prompt, budget or gate was changed.

The frozen 36-session pilot launched at **2026-09-14 23:53:33 UTC** after the
093 controller finished cleanly, all 43,356 final evidence files were verified,
both synthetic seed integrations passed, and the remote read-only launch
checks passed. The first paid requests and running controller were verified.
No completed pilot outcome was available at initial verification.

## Question and treatment

The completed 093 run showed a 2.61% ledger RMSLE reduction with an exploratory
95% interval crossing zero, essentially no mature-history accuracy gain, and
sparse comparable production forecasts. This pilot tests prospective evidence
collection: every backtested configuration also produces one current forecast
in **all three arms**, charged to the same numerical budget. Unselected
forecasts mature only after their full actual horizon becomes visible.

The three arms are plain Hermes, Hermes + published Gnomon **1.2.0**, and
Hermes + Gnomon 1.2.0 + development ledger. All start with fresh state. The
model is Engy **deepseek-v4.1-flash**, requested seed 7, temperature 0.2, maximum
3,072 output tokens. The same raw information, configuration space, native
memory availability, 16-request limit, 60-fit limit and 480-second agent budget
apply to every arm. The host limit is 520 seconds. Two series chains run in
parallel, with arm order rotated and all arms completing an origin before the
next origin is released.

This changes collection for every arm; it is not an isolated change in ledger
selection policy. Compare the freshly matched arms within 096. Do not reuse
093 controls or interpret a between-run difference as a causal ledger effect.
No missing historical production forecasts were manufactured retrospectively.

## Gate and evidence

The pilot is four reused development series × three origins × three arms.
Continuation requires all 12 forecasts valid per arm, at least 11/12 full
workflows per arm, and zero independent audit failures. Accuracy is not a
pilot admission criterion. If that gate passes, preserve these 36 outcomes
exactly once before any separately launched 276-session continuation. This
controller does not launch continuation automatically.

- Controller PID: `4166086`; start ticks: `1557442121`.
- Pilot child PID: `4166118`; start ticks: `1557442135`.
- Boot ID: `998193f3-2771-4162-80e8-1a4887370f60`.
- Run: `/root/gnomon-ledger-ml-v3/code/results/collection-096-pilot-001`.
- Controller records: sibling `collection-096-pilot-launch-001`.
- Tested bundle: sibling `collection-096-pilot-bundle-001/payload`.
- Bundle SHA-256: `e8dff0dc8a2f9de2428d347396e6ddec6c66a3e10c4c0c3ff88bc617947ff254`.

Recheck PID/start ticks and boot ID; a status file alone does not prove liveness.
At termination inspect `pilot-exit.json`, `AUDITED.json`, `INCOMPLETE.json`,
`FINISHED.json`, the complete report, and the gate. Preserve any failure; do not
restart because a monitor times out or a result is unfavorable.

Monitor using:

```sh
python3 -m benchmarks.ledger_optimization.pod_guarded_093 \
  --run collection-096-pilot-001 \
  --launch collection-096-pilot-launch-001
```

Receipts: `evidence/collection-096-pilot-launch-001.json`,
`evidence/collection-096-pilot-bundle-001.json`, and
`evidence/seed-integration-093-completed-001.json`.

The **20% objective remains unmet**. The untouched final holdout remains
closed; main and PyPI are unchanged. This is a development pilot, not final
efficacy evidence.
