# Targon preparation, 2026-09-16

No live agent evaluation has started at this preparation checkpoint. One excluded,
task-free model-readiness request succeeded (12 tokens, reported charge 1 micro-USD).

The pinned Gnomon 1.2.0 wheel was located on the original accessible Gnomon pod,
SSH workload ID `wrk-kadzj08j3t1o`. Its wheel and extracted Python source hashes
match the registered A7 values. The earlier assertion that the runtime could not
be located was incomplete; remote evaluation assets contain it.

Initial preparation used the separately recorded `gnomon-arena-roi-117` deployment,
ID `wrk-tjdrfztojlsn`, under `/root/gnomon-business-utility-20260916/`. It has a
4-CPU quota and 49,999,998,976-byte memory ceiling; physical-host resources shown
by /proc are not its allocation. No new billable instance was provisioned.

User confirmed the intended deployment is **gnomon-arena**, `wrk-kadzj08j3t1o`,
distinct from the ROI deployment. The verified source and harness have been moved
to `/root/gnomon-business-utility-20260916/` on the confirmed pod.

Remote validation against verified **1.2.0**: **86 tests passed in 10.12 seconds**.
The initial remote batch had 68 passes and 18 packaging failures because the
transfer omitted pyproject.toml, which the matched identity hashes. After adding
the committed project descriptor, the complete batch passed. Both logs remain
on the pod; no task outcomes or protocol thresholds changed in response.

The cross-pod image stream ended unexpectedly. Both images were rebuilt on the
confirmed pod from the existing locked Dockerfiles and verified wheel. Actual IDs:

- Ordinary: `sha256:54a1488536809589bb2deb255406a4672d034ba4b6ee2616892495fed3848c0d`.
- Service: `sha256:cc762d9406093da19a4ad888eb14266f7160340018ad3403ac06539402724090`.

The controller is `/root/gnomon-business-utility-20260916/build-images.py`; retained
build logs are image-build-*.log. The expanded harness/transport preflight passed
116 tests in 11.42 seconds on the confirmed pod. The subsequent container/MCP
and dispatcher batch passed 52 tests, with one lifecycle fixture timing out
during Docker startup. A9 extends only that fixture's startup/lifetime window;
its pinned-image recheck passed in 15.92 seconds. An intervening recheck used
an image tag and correctly failed the immutable-image validation before startup.
All logs are retained; these were excluded infrastructure checks, not agent tasks.

The memory guard now respects cgroup memory.max minus memory.current, conservatively
including charged cache. The original pod had approximately 49.3 GB charged
against its 50 GB ceiling, mostly file cache, at inspection. Its anonymous memory
was only about 50 MB. Do not conflate charged cache with active agent memory or
use the physical host's available memory to bypass the guard. Direct cgroup
reclaim was denied by the container; read-only file-descriptor cache advice then
released clean cached pages without modifying/deleting files. After preparation,
the pod had approximately 18 GiB of measured uncharged headroom. Thresholds were
not relaxed to admit the workload.

Use the existing Engy credential on the confirmed pod and deepseek-v4.1-flash in
the existing matched loop. No credentials were copied between pods. The nine
main passes retain a $5 reported-charge stop each, 32,000 cumulative tokens/task,
12 rounds/tools, 180 seconds/task, one worker and zero retries. A provider-reported
stop can overshoot by one operation; it is not a hard account cap. The common
transport accepts Engy's explicit charged_micro field under its documented
[micro-USD convention](https://engy.ai/docs/agent-api), with tests for missing,
invalid and wrong-origin charges. No token-price estimate substitutes for charges.
Native Codex-account transport is not used by this matched runner.

`dispatch.py` supervises the registered nine calls to the existing monitored
workflow CLI, writes durable status, continues after graded failures with intact
summaries, and stops on infrastructure/resource failure without retry. This is
sequential orchestration, not a new agent loop or a forced-tool adapter. Status
and results are written outside the frozen input/source directories.
