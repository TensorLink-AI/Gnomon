# ROI run failure diagnosis — 2026-09-16

The main reproducible defect is the agent loop's recovery from malformed tool
arguments. This is a harness failure, not evidence of a Gnomon forecasting defect.
The interrupted ROI run has 140 submitted answers, seven transport failures,
two token-cap failures, and 631 rows blocked by unknown spending. Another 84
planned cases have no completed observation. Preserve all 864 in the denominator;
the blocked rows are not independent model attempts. An in-flight task was
interrupted when diagnosis began; its journal remains authoritative.

Five of the seven transport failures immediately follow JSONDecodeError while
parsing submit_answer. The loop appended the original malformed assistant tool
call to conversation history, then sent an error tool response and tried another
model request. That next request still contained invalid JSON arguments.

An excluded two-request synthetic probe on the same Engy endpoint/model returned:

| Prior tool-call arguments | HTTP status | Outcome |
| --- | --- | --- |
| Valid JSON | 200 | Accepted |
| Malformed JSON | 400 | Rejected: assistant tool-call arguments must be valid JSON |

This reproduces the mechanism implicated in five historical failures. The old
logs retain only OpenRouterError, so the exact historical HTTP status cannot be
proved retrospectively. Two other failures follow ordinary Python results and
cannot be attributed from the retained logs. Malformed output may be truncated,
but historical finish reasons/raw malformed arguments were not retained; the
2048-token output limit is a hypothesis, not an established cause.

Every transport attempt increments the accounting attempt count; a rejected or
failed request has no usage receipt. Complete charge becomes unknown. The
per-arm cost guard then correctly refuses further calls, producing blocked rows.
The supervisor nevertheless treated the resulting exit 2 as an ordinary graded
failure and moved to another arm. This obscured an infrastructure failure behind
hundreds of error rows. CPU/RAM guards did not stop the run; observed destination
memory headroom was about 41 GiB.

Changes implemented in an isolated checkout, without mutating the stopped run:

- Stop a task on malformed arguments, retaining its known charge and failure.
  Do not repair the answer or replay malformed history. Distinguish an explicit
  length finish reason as model_output_truncated.
- Preserve safe transport categories and HTTP status in the trace, without raw
  provider bodies or credentials. Retain malformed-argument size/hash and finish
  reason for diagnosis.
- Stop dispatching later arms when accounting is incomplete. Preserve all
  original observations and unknown charges; never substitute zero.

127 tests passed remotely on the verified Gnomon 1.2.0 source, including the
malformed-history path, known-charge preservation, safe HTTP diagnostics, and
whole-run accounting stop. One initial regression expectation encoded the old
retry behavior; it was updated to require one request and a retained failure.
No new full run has been started. Regression tests and the synthetic HTTP probe
are integration evidence, not business-utility or agent-uplift evidence.

Remote evidence: /root/gnomon-business-utility-20260916/transport-diagnostic-001,
transport-diagnosis-summary.json, transport-fix-tests.log,
transport-fix-tests-final.log; preserved trial: roi-followup-001.
