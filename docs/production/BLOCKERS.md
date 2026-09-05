# External acceptance dependencies

Consecutive blocked audits: **3**. The goal is now **blocked**, not complete.
The same external dependency persisted for three consecutive goal turns.
Iteration 20 was the latest concrete implementation progress.
This audit is **no progress**, not another implementation iteration or a verified
wait. Bookkeeping does not earn acceptance points.

Current verification: progress validator reports 97/100. Workspace discovery found
only the experiment/provider templates and the local provider-plugin example;
the experiment still has model, interpreter and spending placeholders, and its
model endpoint is `https://model-service.invalid/v1`. No user reply has supplied
the missing choices. All previously reported test handles are terminal.

Second audit rechecked the same files and progress validator: model ID and cost
placeholders remain, the model endpoint is still deliberately invalid, and no user
approval/configuration has arrived. No new service request or implementation was
made; the same external dependency is the blocker.

Third audit again found only unfilled templates, unchanged model/cost placeholders
and the invalid placeholder endpoint. No approval or configuration was supplied.
The goal was marked blocked after this verification. Score remains 97/100; the two
acceptance gates below have not been waived or relabelled complete.

The two original pending gates remain:

- Actual ordinary/lean/full agent comparison: choose the model and endpoint,
  credential environment-variable name, and approved spending limit. The existing
  driver, isolated software backends, cost controls and 11-task cohort are ready.
- Live Ephemeris contract (Paracast backend): confirm the deployment base URL, provide
  the credential environment-variable name, and authorize billable wake-up/testing
  with a spending limit. Earlier unauthenticated health/models probes returned401;
  that is not a served forecast or model-version check.

Do not extract a key from a repository instruction, wake a paid service, choose a
paid agent model or replace these gates with scripted tests. No additional default
framework, dataset or repeated full-suite run is needed to resolve these external
dependencies. When the user resumes, revalidate configuration and authorization and
start a fresh blocked audit if the same dependency is still missing. Continue the
actual acceptance runs when the required user/external input arrives.
