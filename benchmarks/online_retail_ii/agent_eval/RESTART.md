# September 16 continuation after calendar-history rejection

The original run stopped after 2,592 aggregated sessions. Another 192 direct /
Gnomon sessions completed in already-dispatched futures and were retained as
`result.json`, but the controller had stopped adding their scores. All 2,784
completed sessions must survive continuation, including failures and native
memory. None is rerun. The 96 ledger attempts at the next origin failed during
local evidence preparation, before any API request.

Cause: published Gnomon 1.2.0's history comparison uses elapsed origin lag as a
task dimension. A UK calendar day across the spring clock change has one less
elapsed hour. Requests used local daily timestamps and end-of-day origins;
the raw comparison returned `incompatible_evidence` after still admitting each
individual origin. Numerical forecasts and actuals were unchanged.

`calendar_history.py` retains the original whole-window query. Only for this
specific rejection does it check identical provider identities, all original
task dimensions, the same Europe/London daily grid and local origin lag. It
then reruns each origin through public `compare_history` with unchanged evidence
cutoffs and requires exact equality of every admitted origin. Genuine identity
changes still reject. No package implementation is patched or guard removed.

`agent_eval.resume` copies the stopped results and native homes to a fresh
output directory, verifies original source/runtime/input hashes, recomputes
saved metrics, and imports unaggregated completed results. It only reopens
preparation failures with no worker dispatch or API attempt; their files are
archived. The original failed run and code bundle remain unchanged. The same
providers, prompts, budgets, model, seeds, original pilot and cohort continue.

Before paid resumption, six real-Hermes sessions with synthetic transport run
on March 27 and April 10. They verify direct/Gnomon parity, native memory, typed
completion and ledger RMSLE against the original matured pairs. Unit tests
cover both spring and autumn calendar changes, a real task mismatch, missing
aggregate recovery, duplicate rejection and retention of failed sessions.

Remote continuation: `/root/online-retail-agent-resume-001`.
The separate 14-day-start queue is reissued at `/root/online-retail-two-week-002`
and waits for this continuation to finish all 3,744 sessions successfully.
The original roots retain failure/cancellation evidence and continuation links.
