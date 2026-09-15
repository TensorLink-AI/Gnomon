# Targon preparation, 2026-09-16

No live agent evaluation or model API request has started.

The pinned Gnomon 1.2.0 wheel was located on the original accessible Gnomon pod,
SSH workload ID `wrk-kadzj08j3t1o`. Its wheel and extracted Python source hashes
match the registered A7 values. The earlier assertion that the runtime could not
be located was incomplete; remote evaluation assets contain it.

Initial preparation used the separately recorded `gnomon-arena-roi-117` deployment,
ID `wrk-tjdrfztojlsn`, under `/root/gnomon-business-utility-20260916/`. It has a
4-CPU quota and 49,999,998,976-byte memory ceiling; physical-host resources shown
by /proc are not its allocation. No new billable instance was provisioned.

User subsequently identified the intended deployment as **gnomon-arena**. The
original accessible pod's friendly name is not independently available in the
local records or SSH environment. Confirmation that it is `wrk-kadzj08j3t1o` is
pending; do not dispatch live trials on the ROI pod as a silent substitute.

Remote validation against verified **1.2.0**: **86 tests passed in 10.12 seconds**.
The initial remote batch had 68 passes and 18 packaging failures because the
transfer omitted pyproject.toml, which the matched identity hashes. After adding
the committed project descriptor, the complete batch passed. Both logs remain
on the pod; no task outcomes or protocol thresholds changed in response.

The ordinary and service images are being built sequentially from the existing
locked Dockerfiles on the initially selected ROI pod. This is preparation only.
The controller is `/root/gnomon-business-utility-20260916/build-images.py`; progress
and failures are in image-build-status.json and image-build-*.log. No model trial
automatically follows image building.

The memory guard now respects cgroup memory.max minus memory.current, conservatively
including charged cache. The original pod had approximately 49.3 GB charged
against its 50 GB ceiling, mostly file cache, at inspection. Its anonymous memory
was only about 50 MB. Do not conflate charged cache with active agent memory or
use the physical host's available memory to bypass the guard.

The matched runner's model transport remains to be selected. The original pod
has an existing Engy credential; the ROI pod has no configured OpenRouter, Engy
or OpenAI credential in its inspected task environment. No credentials were copied.
Native Codex-account transport is not implemented by this matched runner.
