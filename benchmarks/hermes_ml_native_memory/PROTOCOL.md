# Hermes native-memory control follow-up

Run only the Hermes-without-Gnomon arm on the same 104 Favorita tasks (four series,
26 consecutive origins each), starting with fresh isolated homes and projects.
Do not rerun the pilot or either Gnomon arm. Preserve the existing three-arm run.
Compare to its original Hermes control and Gnomon arms on identical task keys;
this later, explicitly prompted follow-up is not a contemporaneous randomized
comparison and does not replace the original results.

Use the frozen checkpoint-v4 numerical lab, native Hermes, DeepSeek
deepseek-v4.1-flash through Engy, 16 requests/480 seconds/60 numerical attempts,
two bounded continuations, atomic forecast selection, identical visible outcomes
and task source. Keep forecasting completion criteria and fallback unchanged.
The only agent-facing change is MEMORY_WORKFLOW.md appended to TASK.md.

Ask the agent to maintain a compact native-memory pointer and a task-specific
skill with evidence references, following Hermes's own tool guidance. Revisit
and revise lessons as outcomes mature. Do not populate lessons, choose forecasts,
or supply conclusions on the agent's behalf. Native memory and skill writes,
reads, survival into later prompts, and update adoption are measured separately
from forecasting completion. Background review remains disabled so there are no
unmetered extra inference calls. This tests explicit in-budget native memory.

Queue the follow-up after the existing run terminates to retain two series workers
on the same pod without changing the existing trial's load. Waits use no API.
Use the same service-admission probes before each session, with all costs retained.
Require an offline native-tool persistence/reload test before inference. Reuse the
unchanged numerical/budget preflight by verifying exact frozen source hashes.

Report all 104 outcomes, including incomplete workflows and fallback; matched
RMSLE, completion, tokens, requests, fits, runtime, memory/skill calls, persisted
state and read/update adoption. Snapshot native memory/skills before and after
each session. Audit forecast evidence with the existing independent analyzer.
Keep every attempted session; no automatic rerolls. Preserve cost uncertainty.
These reused development series cannot establish the held-out 20% target.
